"""按场景 profile 评估 RL 策略泛化能力。"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import yaml

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from envs.train_ho_env import TrainHandoverEnv
from models import ActionSpace, ObservationWindow, RainbowWithForecast
from scripts.eval.test_batch_comparison import collect_kpis
from scripts.eval.test_simple import TraditionalA3Policy, resolve_checkpoint_path, run_episode
from utils.scenario_generator import ScenarioGenerator
from utils.scenario_profiles import ScenarioProfileSampler


RL_POLICY_NAME = "RL_Rainbow"
FIXED_A3_NAME = "FixedA3_Hys3_TTT150"


def load_model(checkpoint_path: str, model_config: Dict, device: str) -> RainbowWithForecast:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    hys_set = model_config["action_space"]["hys_set"]
    ttt_set = model_config["action_space"]["ttt_set"]
    obs_dim = model_config["observation"]["obs_dim"]
    window_size = model_config["observation"]["window_size"]
    use_noisy = checkpoint.get("config", {}).get("use_noisy", False)
    if "use_cql" in checkpoint or "rainbow_offline" in checkpoint_path:
        use_noisy = False

    model = RainbowWithForecast(
        obs_dim=obs_dim,
        num_actions=len(hys_set) * len(ttt_set),
        n_steps=window_size,
        feature_hidden=model_config["network"]["shared"]["hidden_dim"],
        encoder_hidden=model_config["network"]["encoder"]["hidden_dim"],
        num_atoms=model_config["network"]["rainbow"]["num_atoms"],
        v_min=model_config["network"]["rainbow"]["v_min"],
        v_max=model_config["network"]["rainbow"]["v_max"],
        use_noisy=use_noisy,
    ).to(device)
    model.load_state_dict(checkpoint["online_net_state_dict"])
    model.eval()
    return model


def summarize_rows(rows: List[Dict[str, float]], policy: str, profile_name: str) -> Dict[str, float | str]:
    if not rows:
        return {"profile": profile_name, "policy": policy, "num_episodes": 0}

    summary: Dict[str, float | str] = {
        "profile": profile_name,
        "policy": policy,
        "num_episodes": int(len(rows)),
    }
    keys = sorted({k for row in rows for k in row.keys() if isinstance(row.get(k), (int, float, np.number))})
    for key in keys:
        values = np.array([float(row.get(key, np.nan)) for row in rows], dtype=np.float64)
        values = values[np.isfinite(values)]
        if values.size == 0:
            continue
        summary[f"{key}_mean"] = float(np.mean(values))
        summary[f"{key}_std"] = float(np.std(values))
        summary[f"{key}_min"] = float(np.min(values))
        summary[f"{key}_max"] = float(np.max(values))
        if key in {"sinr_mean_db", "sinr_p5_db", "overlap_zone_sinr_mean_db"}:
            summary[f"{key}_worst10"] = float(np.percentile(values, 10))
        elif key in {"outage_time_ratio", "interruption_total_time", "ping_pong_count", "ho_count"}:
            summary[f"{key}_worst10"] = float(np.percentile(values, 90))
    return summary


def write_csv(path: Path, rows: List[Dict]) -> None:
    if not rows:
        return
    keys: List[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def run_profile_eval(args: argparse.Namespace) -> Dict:
    base_dir = PROJECT_ROOT
    env_config_path = os.path.join(base_dir, "configs", "default_env_config.yaml")
    model_config_path = os.path.join(base_dir, "configs", "model_config.yaml")

    with open(env_config_path, "r", encoding="utf-8") as f:
        base_env_config = yaml.safe_load(f)
    with open(model_config_path, "r", encoding="utf-8") as f:
        model_config = yaml.safe_load(f)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = resolve_checkpoint_path(base_dir, args.checkpoint_path)
    model = load_model(checkpoint_path, model_config, device)

    profiles_path = args.scenario_profiles_path
    if not os.path.isabs(profiles_path):
        profiles_path = os.path.join(base_dir, profiles_path)
    sampler = ScenarioProfileSampler(profiles_path, split=args.profile_split)
    selected_profiles = args.profiles or sampler.profile_names()

    window_size = model_config["observation"]["window_size"]
    obs_dim = model_config["observation"]["obs_dim"]
    obs_window = ObservationWindow(window_size=window_size, obs_dim=obs_dim)
    action_space = ActionSpace()

    episode_rows: List[Dict] = []
    summary_rows: List[Dict] = []

    for profile_name in selected_profiles:
        profile_policy_rows = {RL_POLICY_NAME: [], FIXED_A3_NAME: []}
        for idx in range(args.num_seeds):
            env_seed = args.base_seed + idx
            profile_seed = args.profile_seed_offset + env_seed
            env_config, profile_meta = sampler.build_config(
                base_env_config,
                seed=profile_seed,
                profile_name=profile_name,
            )

            scenario_gen = ScenarioGenerator(env_config)
            scenario_data = scenario_gen.generate_scenario(seed=env_seed, position_resolution_m=1.0)

            env_rl = TrainHandoverEnv(config=env_config)
            obs_window.reset()
            traj_rl = run_episode(
                env_rl,
                None,
                RL_POLICY_NAME,
                obs_window=obs_window,
                model=model,
                device=device,
                seed=None,
                scenario_data=scenario_data,
                action_hold_steps=args.action_hold_steps,
            )
            kpis_rl = collect_kpis(
                traj_rl["trajectory"],
                traj_rl.get("last_info", {}).get("kpis", {}),
            )
            profile_policy_rows[RL_POLICY_NAME].append(kpis_rl)
            episode_rows.append({
                "profile": profile_name,
                "policy": RL_POLICY_NAME,
                "seed": int(env_seed),
                **kpis_rl,
            })

            env_a3 = TrainHandoverEnv(config=env_config)
            a3_policy = TraditionalA3Policy(args.a3_hys, args.a3_ttt, action_space)
            a3_policy.reset()
            traj_a3 = run_episode(
                env_a3,
                lambda obs, info, dt, p=a3_policy: p.decide(obs, info, dt),
                FIXED_A3_NAME,
                obs_window=None,
                model=None,
                device=device,
                seed=None,
                scenario_data=scenario_data,
                a3_policy=a3_policy,
            )
            kpis_a3 = collect_kpis(
                traj_a3["trajectory"],
                traj_a3.get("last_info", {}).get("kpis", {}),
            )
            profile_policy_rows[FIXED_A3_NAME].append(kpis_a3)
            episode_rows.append({
                "profile": profile_name,
                "policy": FIXED_A3_NAME,
                "seed": int(env_seed),
                **kpis_a3,
            })

        for policy, rows in profile_policy_rows.items():
            summary_rows.append(summarize_rows(rows, policy, profile_name))

    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        if args.output_dir:
            out_dir = Path(base_dir) / out_dir
        else:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_dir = Path(base_dir) / "experiments" / "runs" / f"{stamp}_profile_generalization"
    out_dir.mkdir(parents=True, exist_ok=True)

    write_csv(out_dir / "profile_episode_metrics.csv", episode_rows)
    write_csv(out_dir / "profile_summary.csv", summary_rows)
    payload = {
        "checkpoint_path": os.path.abspath(checkpoint_path),
        "scenario_profiles_path": os.path.abspath(profiles_path),
        "profile_split": args.profile_split,
        "profiles": selected_profiles,
        "num_seeds": int(args.num_seeds),
        "base_seed": int(args.base_seed),
        "obs_dim": int(obs_dim),
        "window_size": int(window_size),
        "a3_baseline": {"hys_db": float(args.a3_hys), "ttt_ms": float(args.a3_ttt)},
        "summary": summary_rows,
    }
    with (out_dir / "profile_generalization_summary.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(f"Wrote {out_dir / 'profile_episode_metrics.csv'}")
    print(f"Wrote {out_dir / 'profile_summary.csv'}")
    print(f"Wrote {out_dir / 'profile_generalization_summary.json'}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile-based generalization evaluation.")
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--scenario_profiles_path", type=str, default="configs/scenario_profiles.yaml")
    parser.add_argument("--profile_split", type=str, default="test")
    parser.add_argument("--profiles", nargs="*", default=None)
    parser.add_argument("--num_seeds", type=int, default=20)
    parser.add_argument("--base_seed", type=int, default=20000)
    parser.add_argument("--profile_seed_offset", type=int, default=3000000)
    parser.add_argument("--output_dir", type=str, default="")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--a3_hys", type=float, default=3.0)
    parser.add_argument("--a3_ttt", type=float, default=150.0)
    parser.add_argument("--action_hold_steps", type=int, default=0)
    args = parser.parse_args()
    run_profile_eval(args)


if __name__ == "__main__":
    main()
