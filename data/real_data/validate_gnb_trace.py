"""
用模拟试验线路 5G-R 切换数据验证模型性能（支持单流 serving 测量）

输入：
- data/real_data/nr_gnb_rsrp_sinr_km.xlsx

数据特点：
- 每个时间点只有一个 serving 小区（PCI）对应的 RSRP/SINR
- 没有同一时刻的邻区测量

本脚本做的事情：
1) 读取并清洗 Excel（将乱码列名按列序映射成可读字段）
2) 以“位置”为索引，为每个 PCI 拟合/插值一条 RSRP 曲线（coverage profile）
3) 对每个采样点，基于所有 PCI 的 profile 在当前位置估计“最强邻区”（neighbor）
4) 生成与本项目模型一致的观测输入（ObservationWindow: 15×10）
5) 跑模型推理，记录模型选择的 (Hys, TTT)
6) 用 HandoverLogic 复现 A3+TTT+保护窗口的触发逻辑，得到“模型预测的切换事件”，并与真实 PCI 变化对比

输出：
- 控制台：统计摘要（真实/预测切换次数、事件匹配等）
- 可选：保存处理后的轨迹 CSV
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# 注意：本脚本支持“仅预处理”模式（不依赖 torch）。
# 与模型推理相关的 import 会在需要时再做延迟导入。


@dataclass(frozen=True)
class TraceColumns:
    time_iso: str
    km_mark: str
    rel_km: str
    gnb_id: str
    cell_id: str
    pci: str
    rsrp_dbm: str
    sinr_db: str
    direction: str
    speed_kmh: str


def _guess_columns_by_position(df: pd.DataFrame) -> TraceColumns:
    """
    该 Excel 文件的列名存在乱码风险，因此这里按列序做兼容映射。
    期望列数=10，顺序对应：
      0: time_iso
      1: km_mark
      2: rel_km
      3: gNodeB ID
      4: Cell ID
      5: PCI
      6: SSS RSRP(dBm)
      7: SSS SINR(dB)
      8: direction
      9: speed(km/h)
    """
    cols = list(df.columns)
    if len(cols) < 8:
        raise ValueError(f"Excel 列数过少（{len(cols)}），无法解析。列名={cols}")
    # 允许实际列数>10（但仍使用前10列）
    return TraceColumns(
        time_iso=cols[0],
        km_mark=cols[1],
        rel_km=cols[2],
        gnb_id=cols[3],
        cell_id=cols[4],
        pci=cols[5],
        rsrp_dbm=cols[6],
        sinr_db=cols[7],
        direction=cols[8] if len(cols) > 8 else cols[0],
        speed_kmh=cols[9] if len(cols) > 9 else cols[0],
    )


def load_trace_xlsx(path: str) -> Tuple[pd.DataFrame, TraceColumns]:
    df = pd.read_excel(path)
    cols = _guess_columns_by_position(df)

    df = df.copy()
    df.rename(
        columns={
            cols.time_iso: "time_iso",
            cols.km_mark: "km_mark",
            cols.rel_km: "rel_km",
            cols.gnb_id: "gnb_id",
            cols.cell_id: "cell_id",
            cols.pci: "pci",
            cols.rsrp_dbm: "rsrp_dbm",
            cols.sinr_db: "sinr_db",
            cols.direction: "direction",
            cols.speed_kmh: "speed_kmh",
        },
        inplace=True,
    )

    # 类型清洗
    df["time_iso"] = pd.to_datetime(df["time_iso"], errors="coerce")
    for c in ["km_mark", "rel_km", "rsrp_dbm", "sinr_db", "speed_kmh", "pci"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["time_iso", "rel_km", "pci", "rsrp_dbm", "sinr_db"]).reset_index(drop=True)
    df = df.sort_values("time_iso").reset_index(drop=True)

    # rel_km 从 0 起更好用
    rel0 = float(df["rel_km"].min())
    df["rel_km0"] = df["rel_km"] - rel0
    df["pos_m"] = df["rel_km0"] * 1000.0

    # 时间增量
    t = df["time_iso"].values.astype("datetime64[ns]")
    dt_s = (t[1:] - t[:-1]).astype("timedelta64[ns]").astype(np.int64) / 1e9
    dt_s = dt_s[np.isfinite(dt_s) & (dt_s > 0)]
    df.attrs["dt_s_median"] = float(np.median(dt_s)) if len(dt_s) else 0.05

    return df, cols


def build_pci_profiles(df: pd.DataFrame, bin_m: float = 10.0) -> Dict[int, Dict[str, np.ndarray]]:
    """
    为每个 PCI 建立位置->RSRP/SINR 的 profile，用于在任意位置插值估计“邻区”。
    """
    pos = df["pos_m"].to_numpy()
    pos_max = float(np.nanmax(pos))
    grid = np.arange(0.0, pos_max + bin_m, bin_m)

    profiles: Dict[int, Dict[str, np.ndarray]] = {}
    for pci, g in df.groupby("pci"):
        g = g.sort_values("pos_m")
        # 对原始点先按 bin 聚合（均值）以增强稳定性
        b = np.floor(g["pos_m"].to_numpy() / bin_m).astype(int)
        rsrp = g["rsrp_dbm"].to_numpy()
        sinr = g["sinr_db"].to_numpy()

        agg = {}
        for name, arr in [("rsrp_dbm", rsrp), ("sinr_db", sinr)]:
            s = pd.Series(arr).groupby(b).mean()
            agg[name] = (s.index.to_numpy(dtype=float) * bin_m, s.to_numpy(dtype=float))

        # 插值到统一 grid；profile 之外的位置填 -inf（保证不会被选成最强邻区）
        rsrp_grid = np.full_like(grid, fill_value=-np.inf, dtype=float)
        sinr_grid = np.full_like(grid, fill_value=np.nan, dtype=float)

        x_r, y_r = agg["rsrp_dbm"]
        if len(x_r) >= 2:
            rsrp_grid[:] = np.interp(grid, x_r, y_r, left=-np.inf, right=-np.inf)
            x_s, y_s = agg["sinr_db"]
            if len(x_s) >= 2:
                sinr_grid[:] = np.interp(grid, x_s, y_s, left=np.nan, right=np.nan)

        profiles[int(pci)] = {"grid_m": grid, "rsrp_dbm": rsrp_grid, "sinr_db": sinr_grid}

    return profiles


def estimate_neighbor_at_positions(
    serving_pci: np.ndarray,
    pos_m: np.ndarray,
    profiles: Dict[int, Dict[str, np.ndarray]],
    bin_m: float = 10.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    对每个采样点，基于所有 PCI profile 在当前位置的 RSRP，估计最强邻区。
    返回 (neighbor_pci, neighbor_rsrp_dbm)
    """
    all_pcis = np.array(sorted(profiles.keys()), dtype=int)
    grid = next(iter(profiles.values()))["grid_m"]
    max_idx = len(grid) - 1

    # 位置映射到 grid index
    idx = np.clip(np.round(pos_m / bin_m).astype(int), 0, max_idx)

    # 收集所有 PCI 在该位置的 RSRP（矩阵：[num_pci, num_samples]）
    rsrp_mat = np.vstack([profiles[p]["rsrp_dbm"][idx] for p in all_pcis])

    # serving PCI 对应行置为 -inf，避免被选成 neighbor
    serving_mask = (all_pcis[:, None] == serving_pci[None, :])
    rsrp_mat = np.where(serving_mask, -np.inf, rsrp_mat)

    best_row = np.argmax(rsrp_mat, axis=0)
    neighbor_pci = all_pcis[best_row]
    neighbor_rsrp = rsrp_mat[best_row, np.arange(rsrp_mat.shape[1])]

    # 若某些位置所有 neighbor 都是 -inf（coverage profile 不覆盖），降级为“无邻区”
    invalid = ~np.isfinite(neighbor_rsrp)
    neighbor_pci = neighbor_pci.astype(float)
    neighbor_pci[invalid] = np.nan
    neighbor_rsrp[invalid] = np.nan

    return neighbor_pci, neighbor_rsrp


def normalize(x: float, x_min: float, x_max: float) -> float:
    return float(np.clip((x - x_min) / (x_max - x_min + 1e-8), 0.0, 1.0))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--xlsx", type=str, default="data/real_data/nr_gnb_rsrp_sinr_km.xlsx")
    parser.add_argument("--model", type=str, default="checkpoints/rainbow_offline_final.pth")
    parser.add_argument("--env_config", type=str, default="configs/default_env_config.yaml")
    parser.add_argument("--model_config", type=str, default="configs/model_config.yaml")
    parser.add_argument("--bin_m", type=float, default=10.0)
    parser.add_argument("--save_csv", action="store_true")
    parser.add_argument(
        "--preprocess_only",
        action="store_true",
        help="只做数据清洗+邻区估计+导出，不运行模型推理（不需要torch）",
    )
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(base_dir, "../../"))

    xlsx_path = os.path.join(project_root, args.xlsx) if not os.path.isabs(args.xlsx) else args.xlsx
    model_path = os.path.join(project_root, args.model) if not os.path.isabs(args.model) else args.model
    env_cfg_path = os.path.join(project_root, args.env_config) if not os.path.isabs(args.env_config) else args.env_config
    model_cfg_path = os.path.join(project_root, args.model_config) if not os.path.isabs(args.model_config) else args.model_config

    if not os.path.exists(xlsx_path):
        raise FileNotFoundError(f"找不到数据文件: {xlsx_path}")

    # 读取轨迹
    df, _ = load_trace_xlsx(xlsx_path)
    dt_s = float(df.attrs.get("dt_s_median", 0.05))
    track_length_m = float(df["pos_m"].max() - df["pos_m"].min())

    # 基于位置拟合 PCI profiles，并估计邻区
    profiles = build_pci_profiles(df, bin_m=args.bin_m)
    neighbor_pci, neighbor_rsrp = estimate_neighbor_at_positions(
        serving_pci=df["pci"].to_numpy(dtype=int),
        pos_m=df["pos_m"].to_numpy(dtype=float),
        profiles=profiles,
        bin_m=args.bin_m,
    )
    df["neighbor_pci_est"] = neighbor_pci
    df["rsrp_neig_dbm_est"] = neighbor_rsrp
    df["delta_rsrp_dbm_est"] = df["rsrp_neig_dbm_est"] - df["rsrp_dbm"]

    # 真实切换事件：PCI 变化
    df["ho_true"] = df["pci"].ne(df["pci"].shift(1)).fillna(False)
    true_ho_count = int(df["ho_true"].sum())

    print("=" * 80)
    print("真实轨迹预处理摘要")
    print("=" * 80)
    print(f"数据行数: {len(df)}")
    print(f"PCI 数量: {df['pci'].nunique()}")
    print(f"位置范围: 0 ~ {track_length_m:.1f} m")
    print(f"采样间隔(估计): {dt_s:.3f} s")
    print("-" * 80)
    print(f"真实切换次数(PCI变化): {true_ho_count}")
    print("说明：邻区 RSRP 来自基于位置的 PCI coverage profile 插值估计；用于缺少同一时刻邻区测量的场景。")
    print("=" * 80)

    if args.save_csv:
        out_csv = os.path.join(base_dir, "gnb_trace_processed.csv")
        df.to_csv(out_csv, index=False, encoding="utf-8-sig")
        print(f"已保存处理后的轨迹: {out_csv}")

    if args.preprocess_only:
        print("已按 --preprocess_only 结束。若要运行模型推理与切换事件对比，请确保环境安装了 torch 后去掉该参数。")
        return

    # 以下为“带模型推理”的验证流程（需要 torch + 本项目 models/envs）
    try:
        import torch  # type: ignore
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(
            "当前环境缺少 torch，无法运行模型推理。你可以先用 --preprocess_only 导出处理后的 CSV。"
        ) from e

    # 从子目录运行本脚本时，sys.path 不含项目根，会导致 import envs/models 失败
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    import yaml
    from envs.ho_logic import HandoverLogic
    from models import RainbowWithForecast, ActionSpace, ObservationWindow

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"找不到模型文件: {model_path}")

    with open(env_cfg_path, "r", encoding="utf-8") as f:
        env_cfg = yaml.safe_load(f)
    with open(model_cfg_path, "r", encoding="utf-8") as f:
        model_cfg = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(model_path, map_location=device)
    hys_set = model_cfg["action_space"]["hys_set"]
    ttt_set = model_cfg["action_space"]["ttt_set"]
    num_actions = len(hys_set) * len(ttt_set)

    obs_dim = model_cfg["observation"]["obs_dim"]
    window_size = model_cfg["observation"]["window_size"]

    use_noisy = checkpoint.get("config", {}).get("use_noisy", False)
    if "use_cql" in checkpoint or "offline" in os.path.basename(model_path):
        use_noisy = False

    model = RainbowWithForecast(
        obs_dim=obs_dim,
        num_actions=num_actions,
        n_steps=window_size,
        feature_hidden=model_cfg["network"]["shared"]["hidden_dim"],
        encoder_hidden=model_cfg["network"]["encoder"]["hidden_dim"],
        num_atoms=model_cfg["network"]["rainbow"]["num_atoms"],
        v_min=model_cfg["network"]["rainbow"]["v_min"],
        v_max=model_cfg["network"]["rainbow"]["v_max"],
        use_noisy=use_noisy,
    ).to(device)
    model.load_state_dict(checkpoint["online_net_state_dict"])
    model.eval()

    action_space = ActionSpace()
    obs_window = ObservationWindow(window_size=window_size, obs_dim=obs_dim)
    obs_window.reset()

    ho_logic = HandoverLogic(dict(env_cfg))
    ho_logic.reset()

    rsrp_min, rsrp_max = float(env_cfg["rsrp_min_dbm"]), float(env_cfg["rsrp_max_dbm"])
    sinr_min, sinr_max = float(env_cfg["sinr_min_db"]), float(env_cfg["sinr_max_db"])

    predicted_ho_flags: List[bool] = []
    chosen_params: List[Tuple[float, float]] = []

    current_time = 0.0
    for i in range(len(df)):
        row = df.iloc[i]
        rsrp_serv = float(row["rsrp_dbm"])
        rsrp_neig = float(row["rsrp_neig_dbm_est"]) if np.isfinite(row["rsrp_neig_dbm_est"]) else rsrp_serv
        sinr_serv = float(row["sinr_db"])
        pos_norm = float(row["pos_m"] / (track_length_m + 1e-8))

        obs_raw = np.array(
            [
                normalize(rsrp_serv, rsrp_min, rsrp_max),
                normalize(rsrp_neig, rsrp_min, rsrp_max),
                normalize(sinr_serv, sinr_min, sinr_max),
                np.clip(pos_norm, 0.0, 1.0),
                0.5,
                0.5,
                0.5,
            ],
            dtype=np.float32,
        )
        info = {"rsrp_serv_dbm": rsrp_serv, "rsrp_neig_dbm": rsrp_neig}

        obs_window.update_time(current_time)
        if bool(row["ho_true"]):
            obs_window.update_ho_time(current_time)

        obs_window.build_observation(
            obs_raw,
            info,
            velocity_mps=float(row["speed_kmh"]) / 3.6 if np.isfinite(row["speed_kmh"]) else 0.0,
            track_length_m=track_length_m,
        )

        if i < window_size - 1:
            predicted_ho_flags.append(False)
            chosen_params.append((np.nan, np.nan))
            current_time += dt_s
            continue

        window = obs_window.get_window()
        window_tensor = torch.from_numpy(window).float().unsqueeze(0).to(device)
        with torch.no_grad():
            action = model.act(window_tensor, epsilon=0.0)

        hys, ttt = action_space.action_to_hys_ttt(int(action))
        chosen_params.append((hys, ttt))

        ho_logic.update_hys_ttt(hys, ttt)
        delta_rsrp = rsrp_neig - rsrp_serv
        _, ho_pred = ho_logic.execute_handover_with_a3(
            current_time=current_time,
            serving_cell=0,
            delta_rsrp=delta_rsrp,
            dt=dt_s,
            current_position=float(row["pos_m"]),
        )
        predicted_ho_flags.append(bool(ho_pred))
        if ho_pred:
            obs_window.update_ho_time(current_time)

        current_time += dt_s

    df["ho_pred"] = predicted_ho_flags
    df["hys_pred"] = [p[0] for p in chosen_params]
    df["ttt_pred"] = [p[1] for p in chosen_params]

    pred_ho_count = int(df["ho_pred"].sum())

    tol_s = max(0.2, 4 * dt_s)
    true_times = df.loc[df["ho_true"], "time_iso"].to_numpy()
    pred_times = df.loc[df["ho_pred"], "time_iso"].to_numpy()

    def _match_events(a: np.ndarray, b: np.ndarray, tol_seconds: float) -> int:
        if len(a) == 0 or len(b) == 0:
            return 0
        b_used = np.zeros(len(b), dtype=bool)
        match = 0
        tol = np.timedelta64(int(tol_seconds * 1e9), "ns")
        for t in a:
            diffs = np.abs(b - t)
            idx = np.argmin(np.where(b_used, np.timedelta64(10**18, "ns"), diffs))
            if not b_used[idx] and diffs[idx] <= tol:
                b_used[idx] = True
                match += 1
        return match

    matched = _match_events(true_times, pred_times, tol_s)
    precision = matched / pred_ho_count if pred_ho_count > 0 else 0.0
    recall = matched / true_ho_count if true_ho_count > 0 else 0.0

    print("=" * 80)
    print("真实轨迹模型验证摘要（含模型推理）")
    print("=" * 80)
    print(f"真实切换次数(PCI变化): {true_ho_count}")
    print(f"模型预测切换次数(A3+TTT触发): {pred_ho_count}")
    print(f"事件匹配(±{tol_s:.2f}s): matched={matched} | precision={precision:.3f} | recall={recall:.3f}")
    print("=" * 80)


if __name__ == "__main__":
    main()

