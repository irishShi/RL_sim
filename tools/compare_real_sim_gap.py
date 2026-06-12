"""对比实测轨迹与当前仿真 profile 的统计差距。

用途：
- 先不校正参数，只做同口径体检；
- 输出可复用的 CSV/JSON，供后续 real-to-sim calibration 使用。

注意：
- 实测数据是长线路、多 PCI serving-only 测量；
- 当前仿真是单 episode 双小区 A3 切换场景；
- 因此这里同时输出“全线路宏观统计”和“按真实切换间隔划分的局部片段统计”。
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from envs.train_ho_env import TrainHandoverEnv
from models import ActionSpace
from scripts.eval.test_simple import TraditionalA3Policy, run_episode
from utils.scenario_generator import ScenarioGenerator
from utils.scenario_profiles import ScenarioProfileSampler


def _q(values: Iterable[float], p: float) -> float:
    arr = np.asarray(list(values), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(np.percentile(arr, p))


def _mean(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(np.mean(arr))


def _std(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(np.std(arr))


def _metric_block(prefix: str, values: Iterable[float]) -> Dict[str, float]:
    arr = np.asarray(list(values), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {
            f"{prefix}_count": 0,
            f"{prefix}_mean": float("nan"),
            f"{prefix}_std": float("nan"),
            f"{prefix}_p5": float("nan"),
            f"{prefix}_p25": float("nan"),
            f"{prefix}_p50": float("nan"),
            f"{prefix}_p75": float("nan"),
            f"{prefix}_p95": float("nan"),
            f"{prefix}_min": float("nan"),
            f"{prefix}_max": float("nan"),
        }
    return {
        f"{prefix}_count": int(arr.size),
        f"{prefix}_mean": float(np.mean(arr)),
        f"{prefix}_std": float(np.std(arr)),
        f"{prefix}_p5": float(np.percentile(arr, 5)),
        f"{prefix}_p25": float(np.percentile(arr, 25)),
        f"{prefix}_p50": float(np.percentile(arr, 50)),
        f"{prefix}_p75": float(np.percentile(arr, 75)),
        f"{prefix}_p95": float(np.percentile(arr, 95)),
        f"{prefix}_min": float(np.min(arr)),
        f"{prefix}_max": float(np.max(arr)),
    }


def load_real_trace(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name="data")
    cols = list(df.columns)
    if len(cols) < 10:
        raise ValueError(f"实测轻量表列数不足: {len(cols)}")

    out = pd.DataFrame(
        {
            "time": pd.to_datetime(df.iloc[:, 0], errors="coerce"),
            "km_mark": pd.to_numeric(df.iloc[:, 1], errors="coerce"),
            "rel_km": pd.to_numeric(df.iloc[:, 2], errors="coerce"),
            "gnb_id": pd.to_numeric(df.iloc[:, 3], errors="coerce"),
            "cell_id": pd.to_numeric(df.iloc[:, 4], errors="coerce"),
            "pci": pd.to_numeric(df.iloc[:, 5], errors="coerce"),
            "rsrp_serv_dbm": pd.to_numeric(df.iloc[:, 6], errors="coerce"),
            "sinr_serv_db": pd.to_numeric(df.iloc[:, 7], errors="coerce"),
            "direction": df.iloc[:, 8],
            "speed_kmh": pd.to_numeric(df.iloc[:, 9], errors="coerce"),
        }
    )
    out = out.dropna(subset=["time", "km_mark", "pci", "rsrp_serv_dbm", "sinr_serv_db"]).copy()
    out = out.sort_values("time").reset_index(drop=True)
    out["pos_m"] = (out["km_mark"] - out["km_mark"].min()) * 1000.0
    out["dt_s"] = out["time"].diff().dt.total_seconds()
    out["ho_true"] = out["pci"].ne(out["pci"].shift())
    if len(out) > 0:
        out.loc[out.index[0], "ho_true"] = False
    return out


def summarize_real_global(df: pd.DataFrame) -> Dict[str, Any]:
    duration_s = (df["time"].max() - df["time"].min()).total_seconds() if len(df) else 0.0
    span_km = float(df["km_mark"].max() - df["km_mark"].min()) if len(df) else 0.0
    ho_count = int(df["ho_true"].sum())
    out: Dict[str, Any] = {
        "source": "real_global",
        "rows": int(len(df)),
        "duration_s": float(duration_s),
        "track_span_km": span_km,
        "unique_pci": int(df["pci"].nunique()),
        "unique_gnb": int(df["gnb_id"].nunique()),
        "ho_count": ho_count,
        "ho_per_km": float(ho_count / span_km) if span_km > 0 else float("nan"),
        "dt_median_s": float(df["dt_s"].dropna().median()) if df["dt_s"].notna().any() else float("nan"),
        "sinr_lt_0_ratio": float((df["sinr_serv_db"] < 0.0).mean()),
        "sinr_lt_5_ratio": float((df["sinr_serv_db"] < 5.0).mean()),
        "sinr_lt_10_ratio": float((df["sinr_serv_db"] < 10.0).mean()),
        "rsrp_lt_minus100_ratio": float((df["rsrp_serv_dbm"] < -100.0).mean()),
    }
    out.update(_metric_block("rsrp_serv_dbm", df["rsrp_serv_dbm"]))
    out.update(_metric_block("sinr_serv_db", df["sinr_serv_db"]))
    out.update(_metric_block("speed_kmh", df["speed_kmh"]))
    return out


def summarize_real_segments(df: pd.DataFrame, min_rows: int = 8) -> Dict[str, Any]:
    if len(df) == 0:
        return {"source": "real_segments", "segments": 0}

    seg_id = df["ho_true"].cumsum()
    rows: List[Dict[str, float]] = []
    for _, g in df.groupby(seg_id):
        if len(g) < min_rows:
            continue
        span_km = float(g["km_mark"].max() - g["km_mark"].min())
        duration_s = (g["time"].max() - g["time"].min()).total_seconds()
        rows.append(
            {
                "rows": float(len(g)),
                "span_km": span_km,
                "duration_s": float(duration_s),
                "rsrp_mean": _mean(g["rsrp_serv_dbm"]),
                "rsrp_p5": _q(g["rsrp_serv_dbm"], 5),
                "sinr_mean": _mean(g["sinr_serv_db"]),
                "sinr_p5": _q(g["sinr_serv_db"], 5),
                "speed_mean": _mean(g["speed_kmh"]),
            }
        )

    out: Dict[str, Any] = {
        "source": "real_segments",
        "segments": int(len(rows)),
        "min_rows": int(min_rows),
    }
    for key in ["rows", "span_km", "duration_s", "rsrp_mean", "rsrp_p5", "sinr_mean", "sinr_p5", "speed_mean"]:
        out.update(_metric_block(key, [r[key] for r in rows]))
    return out


def _trajectory_stats(trajectory: List[Dict[str, Any]], cfg: Dict[str, Any]) -> Dict[str, Any]:
    rsrp = [float(p["rsrp_serv_dbm"]) for p in trajectory]
    sinr = [float(p["sinr_serv_db"]) for p in trajectory]
    rsrp_a = np.asarray([float(p.get("rsrp_A_dbm", np.nan)) for p in trajectory], dtype=np.float64)
    rsrp_b = np.asarray([float(p.get("rsrp_B_dbm", np.nan)) for p in trajectory], dtype=np.float64)
    pos = np.asarray([float(p.get("x", np.nan)) for p in trajectory], dtype=np.float64)
    ho_count = int(sum(1 for p in trajectory if p.get("ho_executed", False)))
    track_km = float(cfg.get("track_length_m", 3000.0)) / 1000.0

    out: Dict[str, Any] = {
        "steps": int(len(trajectory)),
        "track_length_km": track_km,
        "speed_kmh": float(cfg.get("v_default_kmh", np.nan)),
        "ho_count": ho_count,
        "ho_per_km": float(ho_count / track_km) if track_km > 0 else float("nan"),
        "sinr_lt_0_ratio": float(np.mean(np.asarray(sinr) < 0.0)) if sinr else float("nan"),
        "sinr_lt_5_ratio": float(np.mean(np.asarray(sinr) < 5.0)) if sinr else float("nan"),
        "sinr_lt_10_ratio": float(np.mean(np.asarray(sinr) < 10.0)) if sinr else float("nan"),
        "rsrp_lt_minus100_ratio": float(np.mean(np.asarray(rsrp) < -100.0)) if rsrp else float("nan"),
    }
    out.update(_metric_block("rsrp_serv_dbm", rsrp))
    out.update(_metric_block("sinr_serv_db", sinr))

    if len(pos) > 0 and np.isfinite(rsrp_a).any() and np.isfinite(rsrp_b).any():
        delta = rsrp_b - rsrp_a
        mask = np.abs(delta) <= 5.0
        if np.any(mask):
            out["delta5_overlap_span_m"] = float(np.nanmax(pos[mask]) - np.nanmin(pos[mask]))
            out["delta5_overlap_ratio"] = float(np.mean(mask))
        else:
            out["delta5_overlap_span_m"] = 0.0
            out["delta5_overlap_ratio"] = 0.0
    return out


def run_sim_profile_samples(
    base_env_config: Dict[str, Any],
    sampler: ScenarioProfileSampler,
    profile_name: str,
    seeds: Iterable[int],
    a3_hys: float,
    a3_ttt: float,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    action_space = ActionSpace()
    for seed in seeds:
        env_config, meta = sampler.build_config(base_env_config, seed=int(seed) + 1000000, profile_name=profile_name)
        scenario_data = ScenarioGenerator(env_config).generate_scenario(seed=int(seed), position_resolution_m=1.0)
        env = TrainHandoverEnv(config=env_config)
        policy = TraditionalA3Policy(a3_hys, a3_ttt, action_space)
        policy.reset()
        traj = run_episode(
            env,
            lambda obs, info, dt, p=policy: p.decide(obs, info, dt),
            policy_name=f"A3_Hys{a3_hys}_TTT{a3_ttt}",
            scenario_data=scenario_data,
            a3_policy=policy,
        )
        stats = _trajectory_stats(traj["trajectory"], env_config)
        stats.update(
            {
                "source": "sim_profile",
                "profile": profile_name,
                "seed": int(seed),
                "resolved_track_length_m": float(env_config.get("track_length_m", np.nan)),
                "resolved_Ptx_A_dbm": float(env_config.get("Ptx_A_dbm", np.nan)),
                "resolved_Ptx_B_dbm": float(env_config.get("Ptx_B_dbm", np.nan)),
                "resolved_pathloss_exp_A": float(env_config.get("pathloss_exp_A", np.nan)),
                "resolved_pathloss_exp_B": float(env_config.get("pathloss_exp_B", np.nan)),
                "resolved_shadow_sigma_A": float(env_config.get("shadow_sigma_A", np.nan)),
                "resolved_shadow_sigma_B": float(env_config.get("shadow_sigma_B", np.nan)),
                "resolved_noise_dbm": float(env_config.get("noise_dbm", np.nan)),
                "enable_extra_interference": bool(env_config.get("enable_extra_interference", False)),
                "enable_fast_fading": bool(env_config.get("enable_fast_fading", False)),
            }
        )
        if "velocity_mps" in scenario_data:
            stats["speed_kmh"] = float(scenario_data["velocity_mps"]) * 3.6
        stats["profile_description"] = meta.get("profile_description", "")
        rows.append(stats)
    return rows


def summarize_sim_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    summaries: List[Dict[str, Any]] = []
    profiles = sorted({str(r["profile"]) for r in rows})
    metric_keys = [
        "rsrp_serv_dbm_mean",
        "rsrp_serv_dbm_p5",
        "rsrp_serv_dbm_p50",
        "rsrp_serv_dbm_p95",
        "sinr_serv_db_mean",
        "sinr_serv_db_p5",
        "sinr_serv_db_p50",
        "sinr_serv_db_p95",
        "sinr_lt_0_ratio",
        "sinr_lt_5_ratio",
        "sinr_lt_10_ratio",
        "rsrp_lt_minus100_ratio",
        "speed_kmh",
        "ho_count",
        "ho_per_km",
        "delta5_overlap_span_m",
        "delta5_overlap_ratio",
    ]
    for profile in profiles:
        subset = [r for r in rows if str(r["profile"]) == profile]
        out: Dict[str, Any] = {
            "source": "sim_profile_summary",
            "profile": profile,
            "num_samples": int(len(subset)),
        }
        for key in metric_keys:
            out.update(_metric_block(key, [float(r.get(key, np.nan)) for r in subset]))
        summaries.append(out)
    return summaries


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    keys: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in keys:
                keys.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare real railway NR trace statistics with simulator profiles.")
    parser.add_argument("--real_xlsx", type=str, default="data/raw/real/nr_gnb_rsrp_sinr_km.xlsx")
    parser.add_argument("--env_config", type=str, default="configs/default_env_config.yaml")
    parser.add_argument("--profiles", type=str, default="configs/scenario_profiles.yaml")
    parser.add_argument("--profile_split", type=str, default="all")
    parser.add_argument("--num_seeds", type=int, default=5)
    parser.add_argument("--base_seed", type=int, default=51000)
    parser.add_argument("--a3_hys", type=float, default=3.0)
    parser.add_argument("--a3_ttt", type=float, default=150.0)
    parser.add_argument("--out_dir", type=str, default="results/real_sim_gap")
    args = parser.parse_args()

    real_path = PROJECT_ROOT / args.real_xlsx
    env_path = PROJECT_ROOT / args.env_config
    profiles_path = PROJECT_ROOT / args.profiles
    out_dir = PROJECT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    with env_path.open("r", encoding="utf-8") as f:
        base_env_config = yaml.safe_load(f)

    real_df = load_real_trace(real_path)
    real_global = summarize_real_global(real_df)
    real_segments = summarize_real_segments(real_df)

    sampler = ScenarioProfileSampler(profiles_path, split=args.profile_split)
    seeds = [args.base_seed + i for i in range(args.num_seeds)]
    sim_rows: List[Dict[str, Any]] = []
    for profile_name in sampler.profile_names():
        sim_rows.extend(
            run_sim_profile_samples(
                base_env_config=base_env_config,
                sampler=sampler,
                profile_name=profile_name,
                seeds=seeds,
                a3_hys=args.a3_hys,
                a3_ttt=args.a3_ttt,
            )
        )
    sim_summary = summarize_sim_rows(sim_rows)

    write_csv(out_dir / "sim_profile_samples.csv", sim_rows)
    write_csv(out_dir / "sim_profile_summary.csv", sim_summary)

    payload = {
        "real_xlsx": str(real_path),
        "env_config": str(env_path),
        "profiles": str(profiles_path),
        "profile_split": args.profile_split,
        "num_seeds": int(args.num_seeds),
        "a3_baseline": {"hys_db": float(args.a3_hys), "ttt_ms": float(args.a3_ttt)},
        "real_global": real_global,
        "real_segments": real_segments,
        "sim_summary": sim_summary,
        "interpretation_note": (
            "实测是长线路多PCI serving-only测量，仿真是双小区局部切换episode；"
            "差距表用于决定校准方向，不应直接作为最终性能结论。"
        ),
    }
    with (out_dir / "real_sim_gap_summary.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(f"写入: {out_dir / 'real_sim_gap_summary.json'}")
    print(f"写入: {out_dir / 'sim_profile_summary.csv'}")
    print(f"写入: {out_dir / 'sim_profile_samples.csv'}")
    print("实测全局摘要:")
    print(json.dumps(real_global, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
