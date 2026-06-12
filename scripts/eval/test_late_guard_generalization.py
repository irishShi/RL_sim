"""评估低 SINR/快速退化晚切保护层的效果。"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import Counter
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
from models import ActionSpace, ObservationWindow
from scripts.eval.test_batch_comparison import collect_kpis
from scripts.eval.test_profile_generalization import load_model, summarize_rows, write_csv
from scripts.eval.test_simple import TraditionalA3Policy, resolve_checkpoint_path, run_episode
from utils.late_handover_guard import LateHandoverGuardConfig, select_guarded_action
from utils.scenario_generator import ScenarioGenerator
from utils.scenario_profiles import ScenarioProfileSampler


RAW_POLICY_NAME = "RL_Raw_50ms"
GUARD_POLICY_NAME = "RL_LateGuard_50ms"
FIXED_A3_NAME = "FixedA3_Hys3_TTT150"


def build_guard_config(args: argparse.Namespace) -> LateHandoverGuardConfig:
    return LateHandoverGuardConfig(
        enabled=True,
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


def run_model_episode_50ms(
    env: TrainHandoverEnv,
    model,
    obs_window: ObservationWindow,
    action_space: ActionSpace,
    device: str,
    scenario_data: Dict,
    policy_name: str,
    guard_config: LateHandoverGuardConfig | None = None,
) -> Dict:
    obs_raw, info = env.reset(seed=None, options={"scenario_data": scenario_data})
    obs_window.reset()
    obs_window.update_time(0.0)
    for _ in range(obs_window.window_size):
        obs_window.build_observation(
            obs_raw,
            info,
            velocity_mps=env.velocity_mps,
            track_length_m=env.cfg["track_length_m"],
        )

    done = False
    step_count = 0
    trajectory = []
    dt = float(env.cfg["delta_t_s"])
    guard_counter: Counter[str] = Counter()
    guard_checked = 0

    while not done:
        window = obs_window.get_window()
        window_tensor = torch.as_tensor(window, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            q_values = model.get_q_values(window_tensor).squeeze(0).detach().cpu().numpy()
        raw_action = int(np.argmax(q_values))
        action = raw_action
        guard_info = {
            "raw_action_id": raw_action,
            "guard_applied": False,
            "guard_reason": "",
            "guard_hys_cap_db": None,
            "guard_ttt_cap_ms": None,
        }

        if guard_config is not None:
            guard_checked += 1
            rsrp_serv = float(info.get("rsrp_serv_dbm", -90.0))
            rsrp_neig = float(info.get("rsrp_neig_dbm", -90.0))
            delta_rsrp = float(info.get("delta_rsrp_dbm", rsrp_neig - rsrp_serv))
            sinr = float(info.get("sinr_serv_db", 0.0))
            current_time = step_count * dt
            time_since_ho = float(info.get("time_since_last_ho", current_time - env.ho_logic.last_ho_time))
            speed_kmh = float(env.velocity_mps * 3.6)
            action, guard_info = select_guarded_action(
                action_space=action_space,
                q_values=q_values,
                raw_action_id=raw_action,
                speed_kmh=speed_kmh,
                delta_rsrp_db=delta_rsrp,
                sinr_db=sinr,
                time_since_ho_s=time_since_ho,
                cfg=guard_config,
            )
            if guard_info.get("guard_applied", False):
                guard_counter[str(guard_info.get("guard_reason", ""))] += 1

        action_hys, action_ttt = action_space.action_to_hys_ttt(action)
        raw_hys, raw_ttt = action_space.action_to_hys_ttt(raw_action)

        next_obs_raw, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        step_count += 1

        current_time = env.time_step * dt
        obs_window.update_time(current_time)
        if info.get("ho_executed", False):
            obs_window.update_ho_time(current_time)
        obs_window.update_params(
            info.get("current_hys", action_hys),
            info.get("current_ttt", action_ttt),
        )
        obs_window.build_observation(
            next_obs_raw,
            info,
            velocity_mps=env.velocity_mps,
            track_length_m=env.cfg["track_length_m"],
        )

        trajectory.append({
            "step": step_count,
            "x": info.get("position_m", 0.0),
            "serving_cell": info.get("serving_cell", 0),
            "rsrp_A_dbm": info.get("rsrp_A_dbm", 0.0),
            "rsrp_B_dbm": info.get("rsrp_B_dbm", 0.0),
            "rsrp_serv_dbm": info.get("rsrp_serv_dbm", 0.0),
            "rsrp_neig_dbm": info.get("rsrp_neig_dbm", 0.0),
            "sinr_serv_db": info.get("sinr_serv_db", 0.0),
            "ho_executed": info.get("ho_executed", False),
            "ho_triggered": info.get("ho_triggered", False),
            "ho_blocked_by_guard": info.get("ho_blocked_by_guard", False),
            "current_hys": info.get("current_hys", action_hys),
            "current_ttt": info.get("current_ttt", action_ttt),
            "action_id": int(action),
            "action_hys": float(action_hys),
            "action_ttt": float(action_ttt),
            "raw_action_id": int(raw_action),
            "raw_hys": float(raw_hys),
            "raw_ttt": float(raw_ttt),
            "guard_applied": bool(guard_info.get("guard_applied", False)),
            "guard_reason": str(guard_info.get("guard_reason", "")),
        })
        obs_raw = next_obs_raw

        if step_count >= 10000:
            break

    return {
        "policy_name": policy_name,
        "trajectory": trajectory,
        "last_info": info,
        "guard_stats": {
            "guard_checked_steps": int(guard_checked),
            "guard_applied_steps": int(sum(guard_counter.values())),
            "guard_applied_ratio": float(sum(guard_counter.values()) / guard_checked) if guard_checked else 0.0,
            "guard_reason_counts": dict(guard_counter),
        },
    }


def append_guard_metrics(kpis: Dict, traj: Dict) -> Dict:
    stats = traj.get("guard_stats", {})
    kpis = dict(kpis)
    kpis["guard_checked_steps"] = float(stats.get("guard_checked_steps", 0))
    kpis["guard_applied_steps"] = float(stats.get("guard_applied_steps", 0))
    kpis["guard_applied_ratio"] = float(stats.get("guard_applied_ratio", 0.0))
    reasons = stats.get("guard_reason_counts", {})
    for reason, count in reasons.items():
        kpis[f"guard_reason_{reason}"] = float(count)
    return kpis


def run_eval(args: argparse.Namespace) -> Dict:
    base_dir = PROJECT_ROOT
    with open(os.path.join(base_dir, "configs", "default_env_config.yaml"), "r", encoding="utf-8") as f:
        base_env_config = yaml.safe_load(f)
    with open(os.path.join(base_dir, "configs", "model_config.yaml"), "r", encoding="utf-8") as f:
        model_config = yaml.safe_load(f)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = resolve_checkpoint_path(base_dir, args.checkpoint_path)
    model = load_model(checkpoint_path, model_config, device)
    action_space = ActionSpace()
    obs_window = ObservationWindow(
        window_size=model_config["observation"]["window_size"],
        obs_dim=model_config["observation"]["obs_dim"],
    )
    guard_config = build_guard_config(args)

    profiles_path = args.scenario_profiles_path
    if not os.path.isabs(profiles_path):
        profiles_path = os.path.join(base_dir, profiles_path)
    sampler = ScenarioProfileSampler(profiles_path, split=args.profile_split)
    selected_profiles = args.profiles or sampler.profile_names()

    episode_rows: List[Dict] = []
    summary_rows: List[Dict] = []

    for profile_name in selected_profiles:
        policy_rows = {RAW_POLICY_NAME: [], GUARD_POLICY_NAME: [], FIXED_A3_NAME: []}
        for idx in range(args.num_seeds):
            env_seed = args.base_seed + idx
            profile_seed = args.profile_seed_offset + env_seed
            env_config, _ = sampler.build_config(
                base_env_config,
                seed=profile_seed,
                profile_name=profile_name,
            )
            scenario_data = ScenarioGenerator(env_config).generate_scenario(
                seed=env_seed,
                position_resolution_m=1.0,
            )

            for policy_name, guard in [(RAW_POLICY_NAME, None), (GUARD_POLICY_NAME, guard_config)]:
                env = TrainHandoverEnv(config=env_config)
                obs_window.reset()
                traj = run_model_episode_50ms(
                    env=env,
                    model=model,
                    obs_window=obs_window,
                    action_space=action_space,
                    device=device,
                    scenario_data=scenario_data,
                    policy_name=policy_name,
                    guard_config=guard,
                )
                kpis = collect_kpis(traj["trajectory"], traj.get("last_info", {}).get("kpis", {}))
                kpis = append_guard_metrics(kpis, traj)
                policy_rows[policy_name].append(kpis)
                episode_rows.append({
                    "profile": profile_name,
                    "policy": policy_name,
                    "seed": int(env_seed),
                    **kpis,
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
            kpis_a3 = collect_kpis(traj_a3["trajectory"], traj_a3.get("last_info", {}).get("kpis", {}))
            policy_rows[FIXED_A3_NAME].append(kpis_a3)
            episode_rows.append({
                "profile": profile_name,
                "policy": FIXED_A3_NAME,
                "seed": int(env_seed),
                **kpis_a3,
            })

        for policy_name, rows in policy_rows.items():
            summary_rows.append(summarize_rows(rows, policy_name, profile_name))

    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        if args.output_dir:
            out_dir = Path(base_dir) / out_dir
        else:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_dir = Path(base_dir) / "experiments" / "runs" / f"{stamp}_late_guard_eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    write_csv(out_dir / "late_guard_episode_metrics.csv", episode_rows)
    write_csv(out_dir / "late_guard_summary.csv", summary_rows)
    payload = {
        "checkpoint_path": os.path.abspath(checkpoint_path),
        "scenario_profiles_path": os.path.abspath(profiles_path),
        "profile_split": args.profile_split,
        "profiles": selected_profiles,
        "num_seeds": int(args.num_seeds),
        "base_seed": int(args.base_seed),
        "guard_config": guard_config.to_dict(),
        "summary": summary_rows,
    }
    with (out_dir / "late_guard_summary.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(f"Wrote {out_dir / 'late_guard_episode_metrics.csv'}")
    print(f"Wrote {out_dir / 'late_guard_summary.csv'}")
    print(f"Wrote {out_dir / 'late_guard_summary.json'}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate late-handover guard for Rainbow DQN.")
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--scenario_profiles_path", type=str, default="configs/scenario_profiles.yaml")
    parser.add_argument("--profile_split", type=str, default="test")
    parser.add_argument("--profiles", nargs="*", default=None)
    parser.add_argument("--num_seeds", type=int, default=20)
    parser.add_argument("--base_seed", type=int, default=23000)
    parser.add_argument("--profile_seed_offset", type=int, default=4000000)
    parser.add_argument("--output_dir", type=str, default="")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--a3_hys", type=float, default=3.0)
    parser.add_argument("--a3_ttt", type=float, default=150.0)

    parser.add_argument("--guard_low_sinr_db", type=float, default=-3.0)
    parser.add_argument("--guard_critical_sinr_db", type=float, default=-6.0)
    parser.add_argument("--guard_fast_sinr_db", type=float, default=0.0)
    parser.add_argument("--guard_moderate_sinr_db", type=float, default=3.0)
    parser.add_argument("--guard_min_delta_db", type=float, default=1.5)
    parser.add_argument("--guard_strong_delta_db", type=float, default=6.0)
    parser.add_argument("--guard_fast_speed_kmh", type=float, default=350.0)
    parser.add_argument("--guard_min_time_since_ho_s", type=float, default=0.5)
    parser.add_argument("--guard_low_max_ttt_ms", type=float, default=150.0)
    parser.add_argument("--guard_critical_max_ttt_ms", type=float, default=100.0)
    parser.add_argument("--guard_fast_max_ttt_ms", type=float, default=150.0)
    parser.add_argument("--guard_hys_cap_db", type=float, default=3.0)
    parser.add_argument("--guard_critical_hys_cap_db", type=float, default=2.5)
    parser.add_argument("--guard_delta3_hys_cap_db", type=float, default=2.5)
    args = parser.parse_args()
    run_eval(args)


if __name__ == "__main__":
    main()

