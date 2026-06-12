"""
将训练好的 Rainbow DQN 切换策略导出为 Simu5G 可查表使用的 policy_table.csv。

第一阶段采用 repeat_current 方式构造 GRU 历史窗口：
把当前离散观测重复 window_size 次，然后批量前向推理得到 action_id、Hys、TTT。
这种方式用于先打通 Simu5G -> 观测量 -> 查表 -> A3/TTT 参数控制的链路。
"""

import argparse
import csv
import glob
import json
import os
import sys
from datetime import datetime
from itertools import product
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.late_handover_guard import (  # noqa: E402
    LateHandoverGuardConfig,
    select_guarded_action as select_late_guarded_action,
)


DEFAULT_MODEL_CONFIG: Dict[str, Any] = {
    "action_space": {
        "hys_set": [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0],
        "ttt_set": [0, 50, 100, 150, 300, 650],
    },
    "observation": {
        "window_size": 15,
        "obs_dim": 7,
    },
    "network": {
        "encoder": {"hidden_dim": 128},
        "shared": {"hidden_dim": 256},
        "rainbow": {"num_atoms": 51, "v_min": -30.0, "v_max": 10.0},
    },
    "normalization": {
        "delta_rsrp_min": -30.0,
        "delta_rsrp_max": 30.0,
        "velocity_max": 100.0,
        "time_since_ho_max": 10.0,
    },
}

DEFAULT_ENV_CONFIG: Dict[str, Any] = {
    "track_length_m": 3000.0,
    "rsrp_min_dbm": -120.0,
    "rsrp_max_dbm": -60.0,
    "sinr_min_db": -10.0,
    "sinr_max_db": 20.0,
}


class ActionMapper:
    """本地动作映射，避免脚本帮助信息阶段依赖 torch。"""

    def __init__(self, hys_set: Sequence[float], ttt_set: Sequence[float]):
        self.hys_set = [float(v) for v in hys_set]
        self.ttt_set = [float(v) for v in ttt_set]
        self.num_ttt = len(self.ttt_set)
        self.actions = [(hys, ttt) for hys in self.hys_set for ttt in self.ttt_set]

    def action_to_hys_ttt(self, action_id: int) -> Tuple[float, float]:
        if action_id < 0 or action_id >= len(self.actions):
            raise ValueError(f"action_id 超出范围: {action_id}")
        return self.actions[action_id]


def deep_update(base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
    """递归合并配置，缺失字段保留默认值。"""
    merged = dict(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_update(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_yaml_config(path: str, defaults: Dict[str, Any]) -> Dict[str, Any]:
    """读取 YAML 配置；若未安装 PyYAML，则退回到脚本内默认配置。"""
    if not path:
        return defaults
    abs_path = resolve_path(path)
    if not os.path.exists(abs_path):
        print(f"[WARN] 配置文件不存在，使用默认值: {abs_path}")
        return defaults
    try:
        import yaml  # type: ignore
    except ImportError:
        print("[WARN] 未安装 PyYAML，使用脚本内默认配置。")
        return defaults
    with open(abs_path, "r", encoding="utf-8") as f:
        loaded = yaml.safe_load(f) or {}
    return deep_update(defaults, loaded)


def resolve_path(path: str) -> str:
    """把相对路径解析到项目根目录下。"""
    if os.path.isabs(path):
        return os.path.abspath(path)
    return os.path.abspath(os.path.join(PROJECT_ROOT, path))


def resolve_checkpoint_path(checkpoint_path: Optional[str]) -> str:
    """未指定 checkpoint 时，优先选择最新非 legacy 的 rainbow_offline_best.pth。"""
    if checkpoint_path:
        return resolve_path(checkpoint_path)

    patterns = [
        os.path.join(PROJECT_ROOT, "experiments", "runs", "*", "checkpoints", "rainbow_offline_best.pth"),
        os.path.join(PROJECT_ROOT, "experiments", "runs", "*", "checkpoints", "rainbow_offline_final.pth"),
    ]
    candidates: List[str] = []
    for pattern in patterns:
        candidates.extend(glob.glob(pattern))

    if not candidates:
        raise FileNotFoundError(
            "未找到 checkpoint。请通过 --checkpoint_path 指定模型文件，"
            "例如 experiments/runs/.../checkpoints/rainbow_offline_best.pth"
        )

    def rank(path: str) -> Tuple[int, int, float]:
        norm = path.replace("\\", "/")
        is_legacy = 1 if "legacy_20260416_offline_rainbow" in norm else 0
        is_final = 1 if path.endswith("rainbow_offline_final.pth") else 0
        return (is_legacy, is_final, -os.path.getmtime(path))

    return os.path.abspath(sorted(candidates, key=rank)[0])


def parse_float_list(value: str) -> List[float]:
    """解析逗号分隔浮点数列表。"""
    if value is None or value.strip() == "":
        return []
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def frange_inclusive(start: float, stop: float, step: float) -> List[float]:
    """生成包含右端点的浮点序列。"""
    values: List[float] = []
    cur = start
    eps = abs(step) * 1e-6
    while cur <= stop + eps:
        values.append(round(cur, 6))
        cur += step
    return values


def parse_range(value: str) -> List[float]:
    """解析 start:stop:step 或逗号列表。"""
    if ":" not in value:
        return parse_float_list(value)
    parts = [float(item.strip()) for item in value.split(":")]
    if len(parts) != 3:
        raise ValueError(f"范围格式应为 start:stop:step，当前为: {value}")
    start, stop, step = parts
    if step <= 0:
        raise ValueError("step 必须大于 0")
    return frange_inclusive(start, stop, step)


def clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def normalize(value: float, min_value: float, max_value: float) -> float:
    if max_value <= min_value:
        return 0.0
    return clip01((value - min_value) / (max_value - min_value))


def build_observation(
    rsrp_serv_dbm: float,
    rsrp_neig_dbm: float,
    sinr_db: float,
    speed_kmh: float,
    position_m: float,
    time_since_ho_s: float,
    model_config: Dict[str, Any],
    env_config: Dict[str, Any],
) -> List[float]:
    """按训练环境一致的 7 维特征顺序构造归一化观测。"""
    norm_model = model_config["normalization"]
    delta_rsrp_db = rsrp_neig_dbm - rsrp_serv_dbm
    speed_mps = speed_kmh / 3.6

    return [
        normalize(rsrp_serv_dbm, env_config["rsrp_min_dbm"], env_config["rsrp_max_dbm"]),
        normalize(rsrp_neig_dbm, env_config["rsrp_min_dbm"], env_config["rsrp_max_dbm"]),
        normalize(delta_rsrp_db, norm_model["delta_rsrp_min"], norm_model["delta_rsrp_max"]),
        normalize(sinr_db, env_config["sinr_min_db"], env_config["sinr_max_db"]),
        normalize(speed_mps, 0.0, norm_model["velocity_max"]),
        normalize(position_m, 0.0, env_config["track_length_m"]),
        normalize(time_since_ho_s, 0.0, norm_model["time_since_ho_max"]),
    ]


def infer_use_noisy(checkpoint: Dict[str, Any], checkpoint_path: str) -> bool:
    """根据 checkpoint 内容推断模型是否使用 NoisyLinear。"""
    state = checkpoint.get("online_net_state_dict", checkpoint)
    has_noisy_key = any("weight_sigma" in key or "bias_sigma" in key for key in state.keys())
    if has_noisy_key:
        return True
    configured = bool(checkpoint.get("config", {}).get("use_noisy", False))
    if "rainbow_offline" in checkpoint_path or "use_cql" in checkpoint:
        return False
    return configured


def load_model(checkpoint_path: str, model_config: Dict[str, Any], device: str):
    """加载 RainbowWithForecast 或 RainbowWithPhysicsRisk 模型。"""
    import torch
    from models.rainbow_model import RainbowWithForecast, RainbowWithPhysicsRisk

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    action_cfg = model_config["action_space"]
    obs_cfg = model_config["observation"]
    net_cfg = model_config["network"]
    num_actions = len(action_cfg["hys_set"]) * len(action_cfg["ttt_set"])
    use_noisy = infer_use_noisy(checkpoint, checkpoint_path)
    checkpoint_config = checkpoint.get("config", {}) or {}
    has_physics_risk = bool(checkpoint.get("physics_risk_enabled", False))
    risk_cfg = checkpoint_config.get("physics_risk", {}) or model_config.get("physics_risk", {}) or {}

    common_kwargs = dict(
        obs_dim=int(obs_cfg["obs_dim"]),
        num_actions=int(num_actions),
        n_steps=int(obs_cfg["window_size"]),
        feature_hidden=int(net_cfg["shared"]["hidden_dim"]),
        encoder_hidden=int(net_cfg["encoder"]["hidden_dim"]),
        num_atoms=int(net_cfg["rainbow"]["num_atoms"]),
        v_min=float(net_cfg["rainbow"]["v_min"]),
        v_max=float(net_cfg["rainbow"]["v_max"]),
        use_noisy=use_noisy,
    )
    if has_physics_risk:
        num_horizons = int(risk_cfg.get("num_horizons", len(risk_cfg.get("horizons_ms", [100, 200, 300]))))
        model = RainbowWithPhysicsRisk(
            **common_kwargs,
            num_risk_horizons=num_horizons,
            risk_hidden_dim=int(risk_cfg.get("risk_hidden_dim", 256)),
            physics_short_window_steps=int(risk_cfg.get("short_window_steps", 5)),
            physics_delta_t_s=float(risk_cfg.get("delta_t_s", 0.05)),
            physics_l3_alpha=float(risk_cfg.get("l3_alpha", 0.7)),
        ).to(device)
    else:
        model = RainbowWithForecast(**common_kwargs).to(device)

    state_dict = checkpoint.get("online_net_state_dict", checkpoint.get("state_dict", checkpoint))
    model.load_state_dict(state_dict)
    model.eval()
    return model, checkpoint, use_noisy


def iter_states(
    positions_m: Sequence[float],
    speed_bins_kmh: Sequence[float],
    rsrp_serv_bins_dbm: Sequence[float],
    delta_rsrp_bins_db: Sequence[float],
    sinr_bins_db: Sequence[float],
    time_since_ho_bins_s: Sequence[float],
) -> Iterable[Tuple[float, float, float, float, float, float]]:
    """枚举表格状态。"""
    return product(
        positions_m,
        speed_bins_kmh,
        rsrp_serv_bins_dbm,
        delta_rsrp_bins_db,
        sinr_bins_db,
        time_since_ho_bins_s,
    )


def export_policy_table(args: argparse.Namespace) -> Dict[str, Any]:
    import torch

    model_config = load_yaml_config(args.model_config, DEFAULT_MODEL_CONFIG)
    env_config = load_yaml_config(args.env_config, DEFAULT_ENV_CONFIG)
    checkpoint_path = resolve_checkpoint_path(args.checkpoint_path)
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model, checkpoint, use_noisy = load_model(checkpoint_path, model_config, device)
    action_space = ActionMapper(
        model_config["action_space"]["hys_set"],
        model_config["action_space"]["ttt_set"],
    )
    output_path = resolve_path(args.output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    positions_m = parse_range(args.positions_m)
    speed_bins_kmh = parse_range(args.speed_bins_kmh)
    rsrp_serv_bins_dbm = parse_range(args.rsrp_serv_bins_dbm)
    delta_rsrp_bins_db = parse_range(args.delta_rsrp_bins_db)
    sinr_bins_db = parse_range(args.sinr_bins_db)
    time_since_ho_bins_s = parse_range(args.time_since_ho_bins_s)

    obs_cfg = model_config["observation"]
    window_size = int(obs_cfg["window_size"])
    obs_dim = int(obs_cfg["obs_dim"])
    if obs_dim != 7:
        raise ValueError(f"当前导出脚本按 7 维观测编写，但配置 obs_dim={obs_dim}")

    guard_cfg = LateHandoverGuardConfig(
        enabled=bool(args.late_ho_guard),
        low_sinr_db=float(args.guard_low_sinr_db),
        critical_sinr_db=float(args.guard_critical_sinr_db),
        fast_sinr_db=float(args.guard_fast_sinr_db),
        moderate_sinr_db=float(args.guard_moderate_sinr_db),
        min_delta_db=float(args.guard_min_delta_db),
        strong_delta_db=float(args.guard_strong_delta_db),
        fast_speed_kmh=float(args.guard_fast_speed_kmh),
        min_time_since_ho_s=float(args.guard_min_time_since_ho_s),
        low_max_ttt_ms=float(args.guard_low_max_ttt_ms),
        critical_max_ttt_ms=float(args.guard_critical_max_ttt_ms),
        fast_max_ttt_ms=float(args.guard_fast_max_ttt_ms),
        hys_cap_db=float(args.guard_hys_cap_db),
        critical_hys_cap_db=float(args.guard_critical_hys_cap_db),
        delta3_hys_cap_db=float(args.guard_delta3_hys_cap_db),
    )

    total_rows = (
        len(positions_m)
        * len(speed_bins_kmh)
        * len(rsrp_serv_bins_dbm)
        * len(delta_rsrp_bins_db)
        * len(sinr_bins_db)
        * len(time_since_ho_bins_s)
    )
    if args.max_rows and args.max_rows > 0:
        total_rows = min(total_rows, args.max_rows)

    print(f"[INFO] checkpoint: {checkpoint_path}")
    print(f"[INFO] device: {device}, use_noisy={use_noisy}")
    print(f"[INFO] output: {output_path}")
    print(f"[INFO] rows: {total_rows}")

    fieldnames = [
        "position_m",
        "speed_kmh",
        "speed_mps",
        "rsrp_serv_dbm",
        "rsrp_neig_dbm",
        "delta_rsrp_db",
        "sinr_db",
        "time_since_ho_s",
        "obs_rsrp_serv_norm",
        "obs_rsrp_neig_norm",
        "obs_delta_rsrp_norm",
        "obs_sinr_norm",
        "obs_speed_norm",
        "obs_position_norm",
        "obs_time_since_ho_norm",
        "action_id",
        "hys_db",
        "ttt_ms",
        "q_value",
        "q_margin",
        "risk_value",
        "risk_penalty",
        "risk_adjusted_score",
        "raw_action_id",
        "raw_hys_db",
        "raw_ttt_ms",
        "guard_applied",
        "guard_reason",
        "guard_hys_cap_db",
        "guard_ttt_cap_ms",
        "top_actions",
    ]

    batch_obs: List[List[float]] = []
    batch_meta: List[Tuple[float, float, float, float, float, float, float]] = []
    rows_written = 0
    guard_applied_count = 0
    guard_reason_counts: Dict[str, int] = {}
    progress_every = max(1, min(50000, total_rows // 10 if total_rows else 1))

    def flush_batch(writer: csv.DictWriter) -> None:
        nonlocal batch_obs, batch_meta, rows_written, guard_applied_count
        if not batch_obs:
            return

        arr = torch.tensor(batch_obs, dtype=torch.float32, device=device)
        arr = arr.unsqueeze(1).repeat(1, window_size, 1)
        with torch.no_grad():
            if bool(args.use_physics_risk):
                if not hasattr(model, "get_q_and_risk"):
                    raise ValueError("--use_physics_risk 需要包含 risk head 的 checkpoint")
                q_values, risk_prob = model.get_q_and_risk(arr)
                h_idx = int(args.risk_horizon_index)
                if h_idx < 0:
                    h_idx = risk_prob.shape[2] + h_idx
                h_idx = max(0, min(h_idx, risk_prob.shape[2] - 1))
                risk_for_action = risk_prob[:, :, h_idx]
                score_values = q_values - float(args.risk_penalty) * risk_for_action
            else:
                q_values = model.get_q_values(arr)
                risk_for_action = torch.zeros_like(q_values)
                score_values = q_values
            top_k = min(int(args.top_k), q_values.shape[1])
            top_values, top_indices = torch.topk(score_values, k=top_k, dim=1)
            actions = top_indices[:, 0].cpu().tolist()
            q_values_cpu = q_values.cpu().tolist()
            risk_values_cpu = risk_for_action.cpu().tolist()
            score_values_cpu = score_values.cpu().tolist()
            top_values_cpu = top_values.cpu().tolist()
            top_indices_cpu = top_indices.cpu().tolist()

        for obs, meta, raw_action_id, q_values_row, risk_values_row, score_values_row, top_q_row, top_idx_row in zip(
            batch_obs, batch_meta, actions, q_values_cpu, risk_values_cpu, score_values_cpu, top_values_cpu, top_indices_cpu
        ):
            position_m, speed_kmh, rsrp_serv_dbm, rsrp_neig_dbm, delta_rsrp_db, sinr_db, time_since_ho_s = meta
            action_id, guard_info = select_late_guarded_action(
                action_space=action_space,
                q_values=q_values_row,
                raw_action_id=int(raw_action_id),
                speed_kmh=speed_kmh,
                delta_rsrp_db=delta_rsrp_db,
                sinr_db=sinr_db,
                time_since_ho_s=time_since_ho_s,
                cfg=guard_cfg,
            )
            guard_applied = bool(guard_info["guard_applied"])
            guard_reason = str(guard_info["guard_reason"])
            guard_hys_cap = guard_info["guard_hys_cap_db"]
            guard_ttt_cap = guard_info["guard_ttt_cap_ms"]
            hys_db, ttt_ms = action_space.action_to_hys_ttt(int(action_id))
            raw_hys_db, raw_ttt_ms = action_space.action_to_hys_ttt(int(raw_action_id))
            q_value = float(q_values_row[int(action_id)])
            risk_value = float(risk_values_row[int(action_id)])
            adjusted_score = float(score_values_row[int(action_id)])
            q_margin = float(top_q_row[0] - top_q_row[1]) if len(top_q_row) > 1 else 0.0
            top_actions = []
            for idx, value in zip(top_idx_row, top_q_row):
                top_hys, top_ttt = action_space.action_to_hys_ttt(int(idx))
                top_actions.append(f"{int(idx)}:{top_hys:.1f}/{top_ttt:.0f}:{float(value):.4f}")

            if guard_applied:
                guard_applied_count += 1
                guard_reason_counts[guard_reason] = guard_reason_counts.get(guard_reason, 0) + 1

            writer.writerow(
                {
                    "position_m": f"{position_m:.3f}",
                    "speed_kmh": f"{speed_kmh:.3f}",
                    "speed_mps": f"{speed_kmh / 3.6:.6f}",
                    "rsrp_serv_dbm": f"{rsrp_serv_dbm:.3f}",
                    "rsrp_neig_dbm": f"{rsrp_neig_dbm:.3f}",
                    "delta_rsrp_db": f"{delta_rsrp_db:.3f}",
                    "sinr_db": f"{sinr_db:.3f}",
                    "time_since_ho_s": f"{time_since_ho_s:.3f}",
                    "obs_rsrp_serv_norm": f"{obs[0]:.6f}",
                    "obs_rsrp_neig_norm": f"{obs[1]:.6f}",
                    "obs_delta_rsrp_norm": f"{obs[2]:.6f}",
                    "obs_sinr_norm": f"{obs[3]:.6f}",
                    "obs_speed_norm": f"{obs[4]:.6f}",
                    "obs_position_norm": f"{obs[5]:.6f}",
                    "obs_time_since_ho_norm": f"{obs[6]:.6f}",
                    "action_id": int(action_id),
                    "hys_db": f"{hys_db:.3f}",
                    "ttt_ms": f"{ttt_ms:.3f}",
                    "q_value": f"{q_value:.6f}",
                    "q_margin": f"{q_margin:.6f}",
                    "risk_value": f"{risk_value:.6f}",
                    "risk_penalty": f"{float(args.risk_penalty):.6f}" if args.use_physics_risk else "",
                    "risk_adjusted_score": f"{adjusted_score:.6f}",
                    "raw_action_id": int(raw_action_id),
                    "raw_hys_db": f"{raw_hys_db:.3f}",
                    "raw_ttt_ms": f"{raw_ttt_ms:.3f}",
                    "guard_applied": int(guard_applied),
                    "guard_reason": guard_reason,
                    "guard_hys_cap_db": "" if guard_hys_cap is None else f"{guard_hys_cap:.3f}",
                    "guard_ttt_cap_ms": "" if guard_ttt_cap is None else f"{guard_ttt_cap:.3f}",
                    "top_actions": ";".join(top_actions),
                }
            )
            rows_written += 1
            if rows_written % progress_every == 0:
                print(f"[INFO] written {rows_written}/{total_rows} rows")

        batch_obs = []
        batch_meta = []

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for idx, (position_m, speed_kmh, rsrp_serv_dbm, delta_rsrp_db, sinr_db, time_since_ho_s) in enumerate(
            iter_states(
                positions_m,
                speed_bins_kmh,
                rsrp_serv_bins_dbm,
                delta_rsrp_bins_db,
                sinr_bins_db,
                time_since_ho_bins_s,
            )
        ):
            if args.max_rows and args.max_rows > 0 and idx >= args.max_rows:
                break

            rsrp_neig_dbm = rsrp_serv_dbm + delta_rsrp_db
            obs = build_observation(
                rsrp_serv_dbm=rsrp_serv_dbm,
                rsrp_neig_dbm=rsrp_neig_dbm,
                sinr_db=sinr_db,
                speed_kmh=speed_kmh,
                position_m=position_m,
                time_since_ho_s=time_since_ho_s,
                model_config=model_config,
                env_config=env_config,
            )
            batch_obs.append(obs)
            batch_meta.append(
                (position_m, speed_kmh, rsrp_serv_dbm, rsrp_neig_dbm, delta_rsrp_db, sinr_db, time_since_ho_s)
            )

            if len(batch_obs) >= args.batch_size:
                flush_batch(writer)

        flush_batch(writer)

    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "checkpoint_path": checkpoint_path,
        "checkpoint_keys": sorted(list(checkpoint.keys())) if isinstance(checkpoint, dict) else [],
        "output_path": output_path,
        "rows": rows_written,
        "history_mode": "repeat_current",
        "window_size": window_size,
        "obs_dim": obs_dim,
        "device": device,
        "use_noisy": use_noisy,
        "grid": {
            "positions_m": positions_m,
            "speed_bins_kmh": speed_bins_kmh,
            "rsrp_serv_bins_dbm": rsrp_serv_bins_dbm,
            "delta_rsrp_bins_db": delta_rsrp_bins_db,
            "sinr_bins_db": sinr_bins_db,
            "time_since_ho_bins_s": time_since_ho_bins_s,
        },
        "normalization": {
            "rsrp_min_dbm": env_config["rsrp_min_dbm"],
            "rsrp_max_dbm": env_config["rsrp_max_dbm"],
            "sinr_min_db": env_config["sinr_min_db"],
            "sinr_max_db": env_config["sinr_max_db"],
            "delta_rsrp_min": model_config["normalization"]["delta_rsrp_min"],
            "delta_rsrp_max": model_config["normalization"]["delta_rsrp_max"],
            "velocity_max": model_config["normalization"]["velocity_max"],
            "track_length_m": env_config["track_length_m"],
            "time_since_ho_max": model_config["normalization"]["time_since_ho_max"],
        },
        "late_handover_guard": {
            "enabled": bool(args.late_ho_guard),
            "applied_rows": int(guard_applied_count),
            "applied_ratio": float(guard_applied_count / rows_written) if rows_written else 0.0,
            "reason_counts": guard_reason_counts,
            "params": guard_cfg.to_dict(),
        },
        "physics_risk": {
            "enabled": bool(args.use_physics_risk),
            "risk_penalty": float(args.risk_penalty),
            "risk_horizon_index": int(args.risk_horizon_index),
            "checkpoint_has_risk": bool(checkpoint.get("physics_risk_enabled", False)) if isinstance(checkpoint, dict) else False,
            "config": (checkpoint.get("config", {}) or {}).get("physics_risk", {}) if isinstance(checkpoint, dict) else {},
        },
    }
    metadata_path = args.metadata_path
    if metadata_path:
        metadata_path = resolve_path(metadata_path)
    else:
        metadata_path = os.path.splitext(output_path)[0] + ".metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"[INFO] done, rows={rows_written}")
    print(f"[INFO] metadata: {metadata_path}")
    return metadata


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="导出 Rainbow DQN policy_table.csv，供 Simu5G 查表选择 Hys/TTT。"
    )
    parser.add_argument("--checkpoint_path", type=str, default=None, help="模型 checkpoint；不填则自动找最新 best。")
    parser.add_argument("--model_config", type=str, default="configs/model_config.yaml", help="模型配置路径。")
    parser.add_argument("--env_config", type=str, default="configs/default_env_config.yaml", help="环境配置路径。")
    parser.add_argument(
        "--output_path",
        type=str,
        default="results/policy_tables/policy_table.csv",
        help="输出 CSV 路径。",
    )
    parser.add_argument("--metadata_path", type=str, default=None, help="metadata JSON 路径；默认与 CSV 同名。")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"], help="推理设备。")
    parser.add_argument("--batch_size", type=int, default=4096, help="批量推理大小。")
    parser.add_argument("--top_k", type=int, default=3, help="记录前 K 个动作及 Q 值。")
    parser.add_argument("--max_rows", type=int, default=0, help="仅导出前 N 行，用于快速测试；0 表示不限制。")
    parser.add_argument("--use_physics_risk", action="store_true",
                        help="若 checkpoint 包含风险头，则用 Q-risk 蒸馏 policy table。")
    parser.add_argument("--risk_penalty", type=float, default=2.0,
                        help="Q-risk 导出中的风险惩罚系数。")
    parser.add_argument("--risk_horizon_index", type=int, default=-1,
                        help="Q-risk 使用的预测窗口索引，默认 -1 表示最后一个窗口。")

    parser.add_argument("--positions_m", type=str, default="0:3000:100", help="位置 bin，格式 start:stop:step 或逗号列表。")
    parser.add_argument("--speed_bins_kmh", type=str, default="300", help="速度 bin，格式 start:stop:step 或逗号列表。")
    parser.add_argument(
        "--rsrp_serv_bins_dbm",
        type=str,
        default="-115:-65:5",
        help="服务小区 RSRP bin，单位 dBm。",
    )
    parser.add_argument(
        "--delta_rsrp_bins_db",
        type=str,
        default="-20,-15,-10,-7.5,-5,-3,-1.5,0,1.5,3,5,7.5,10,15,20",
        help="邻区 RSRP - 服务区 RSRP 的 bin，单位 dB。",
    )
    parser.add_argument(
        "--sinr_bins_db",
        type=str,
        default="-10,-7.5,-5,-3,0,3,5,8,10,15,20",
        help="SINR bin，单位 dB。",
    )
    parser.add_argument(
        "--time_since_ho_bins_s",
        type=str,
        default="0,0.5,1,2,5,10",
        help="距离上次切换时间 bin，单位秒。",
    )
    parser.add_argument("--late_ho_guard", action="store_true",
                        help="启用低 SINR/快速退化晚切安全层，在导出阶段限制过长 TTT 和过大 Hys。")
    parser.add_argument("--guard_low_sinr_db", type=float, default=-3.0,
                        help="低 SINR 晚切风险阈值，低于该值时限制 TTT。")
    parser.add_argument("--guard_critical_sinr_db", type=float, default=-6.0,
                        help="严重低 SINR 阈值，低于该值时采用更激进的 TTT/Hys 上限。")
    parser.add_argument("--guard_fast_sinr_db", type=float, default=0.0,
                        help="高速场景的 SINR 风险阈值。")
    parser.add_argument("--guard_moderate_sinr_db", type=float, default=3.0,
                        help="邻区明显更强时的中等 SINR 风险阈值。")
    parser.add_argument("--guard_min_delta_db", type=float, default=1.5,
                        help="触发晚切保护所需的最小 Delta RSRP，单位 dB。")
    parser.add_argument("--guard_strong_delta_db", type=float, default=6.0,
                        help="邻区明显更强的 Delta RSRP 阈值，单位 dB。")
    parser.add_argument("--guard_fast_speed_kmh", type=float, default=350.0,
                        help="高速晚切风险阈值，单位 km/h。")
    parser.add_argument("--guard_min_time_since_ho_s", type=float, default=0.5,
                        help="距上次切换时间低于该值时不启用保护，避免刚切后立刻触发。")
    parser.add_argument("--guard_low_max_ttt_ms", type=float, default=150.0,
                        help="低 SINR 或强邻区风险状态下的最大 TTT。")
    parser.add_argument("--guard_critical_max_ttt_ms", type=float, default=100.0,
                        help="严重低 SINR 状态下的最大 TTT。")
    parser.add_argument("--guard_fast_max_ttt_ms", type=float, default=150.0,
                        help="高速退化风险状态下的最大 TTT。")
    parser.add_argument("--guard_hys_cap_db", type=float, default=3.0,
                        help="晚切保护状态下的 Hys 上限。")
    parser.add_argument("--guard_critical_hys_cap_db", type=float, default=2.5,
                        help="严重低 SINR 状态下的 Hys 上限。")
    parser.add_argument("--guard_delta3_hys_cap_db", type=float, default=2.5,
                        help="Delta RSRP 小于 4 dB 时使用的更低 Hys 上限。")
    return parser


def normalize_cli_grid_args(argv: Sequence[str]) -> List[str]:
    """
    允许用户写 --xxx -10,0,10 这种负数列表。

    argparse 对逗号负数列表会误判为选项，本函数把它改写成
    --xxx=-10,0,10 再交给 argparse。
    """
    options_with_possible_negative_values = {
        "--positions_m",
        "--speed_bins_kmh",
        "--rsrp_serv_bins_dbm",
        "--delta_rsrp_bins_db",
        "--sinr_bins_db",
        "--time_since_ho_bins_s",
    }
    normalized: List[str] = []
    i = 0
    while i < len(argv):
        token = argv[i]
        if (
            token in options_with_possible_negative_values
            and i + 1 < len(argv)
            and argv[i + 1].startswith("-")
            and not argv[i + 1].startswith("--")
        ):
            normalized.append(f"{token}={argv[i + 1]}")
            i += 2
            continue
        normalized.append(token)
        i += 1
    return normalized


def main() -> None:
    parser = build_parser()
    args = parser.parse_args(normalize_cli_grid_args(sys.argv[1:]))
    export_policy_table(args)


if __name__ == "__main__":
    main()
