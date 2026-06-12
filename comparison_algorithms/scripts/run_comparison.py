"""运行对比算法 profile 评估。

示例：
python comparison_algorithms/scripts/run_comparison.py --num_seeds 5 --include_rainbow --train_tabular_q
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
EVAL_DIR = PROJECT_ROOT / "scripts" / "eval"
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))

from envs.train_ho_env import TrainHandoverEnv
from models import ActionSpace
from utils.late_handover_guard import LateHandoverGuardConfig
from utils.scenario_generator import ScenarioGenerator
from utils.scenario_profiles import ScenarioProfileSampler

from comparison_algorithms.baseline_policies import (
    TabularQPolicy,
    build_default_comparison_policies,
)
from comparison_algorithms.evaluation import (
    add_episode_metadata,
    collect_episode_kpis,
    evaluate_fixed_oracle,
    load_rainbow_policy,
    load_yaml,
    run_policy_episode,
    summarize_rows,
    write_csv,
    write_json,
)


def _resolve_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def train_tabular_q(args, base_env_config: Dict, out_dir: Path, action_space: ActionSpace) -> Path:
    profiles_path = _resolve_path(args.scenario_profiles_path)
    sampler = ScenarioProfileSampler(profiles_path, split=args.q_train_profile_split)
    profile_names = args.q_train_profiles or sampler.profile_names()
    policy = TabularQPolicy(
        action_space=action_space,
        alpha=args.q_alpha,
        gamma=args.q_gamma,
        epsilon=args.q_epsilon_start,
        training=True,
    )
    train_rows: List[Dict] = []
    eps_decay = np.linspace(args.q_epsilon_start, args.q_epsilon_end, max(1, args.q_train_episodes))

    for ep in tqdm(range(args.q_train_episodes), desc="训练 Tabular Q"):
        profile_name = profile_names[ep % len(profile_names)]
        env_seed = args.q_train_base_seed + ep
        profile_seed = args.profile_seed_offset + env_seed
        env_config, _ = sampler.build_config(base_env_config, seed=profile_seed, profile_name=profile_name)
        scenario_data = ScenarioGenerator(env_config).generate_scenario(seed=env_seed, position_resolution_m=1.0)
        policy.epsilon = float(eps_decay[ep])
        env = TrainHandoverEnv(config=env_config)
        traj = run_policy_episode(
            env,
            policy,
            scenario_data=scenario_data,
            seed=env_seed,
            action_hold_steps=args.action_hold_steps,
        )
        kpis = collect_episode_kpis(traj["trajectory"], traj.get("last_info", {}), env_config)
        train_rows.append(add_episode_metadata(kpis, traj, env_config, profile_name, "TabularQ_train", env_seed))

    q_path = out_dir / "tables" / "tabular_q_policy.json"
    policy.save(q_path)
    write_csv(out_dir / "metrics" / "tabular_q_train_metrics.csv", train_rows)
    write_json(out_dir / "metrics" / "tabular_q_train_summary.json", {
        "q_table_path": str(q_path),
        "num_states": len(policy.q_table),
        "episodes": int(args.q_train_episodes),
        "profile_split": args.q_train_profile_split,
        "profiles": profile_names,
        "alpha": float(args.q_alpha),
        "gamma": float(args.q_gamma),
        "epsilon_start": float(args.q_epsilon_start),
        "epsilon_end": float(args.q_epsilon_end),
    })
    return q_path


def build_policies(args, out_dir: Path, base_env_config: Dict, model_config: Dict, action_space: ActionSpace):
    policies = build_default_comparison_policies(action_space)

    q_table_path = args.q_table_path
    if args.train_tabular_q:
        q_table_path = str(train_tabular_q(args, base_env_config, out_dir, action_space))
    if q_table_path:
        policies.append(TabularQPolicy.load(_resolve_path(q_table_path), action_space=action_space,
                                            training=False, epsilon=0.0))

    if args.include_rainbow:
        device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
        policies.append(load_rainbow_policy(args.checkpoint_path, model_config, device, action_space=action_space))
        if args.include_rainbow_late_guard:
            policies.append(load_rainbow_policy(
                args.checkpoint_path,
                model_config,
                device,
                action_space=action_space,
                late_guard_config=build_late_guard_config(args),
            ))
        if args.include_rainbow_physics_risk:
            policies.append(load_rainbow_policy(
                args.checkpoint_path,
                model_config,
                device,
                action_space=action_space,
                use_physics_risk=True,
                risk_penalty=args.risk_penalty,
                risk_horizon_index=args.risk_horizon_index,
            ))

    return policies


def build_late_guard_config(args) -> LateHandoverGuardConfig:
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


def write_markdown_report(out_dir: Path, args, overall_rows: List[Dict], profile_rows: List[Dict]) -> None:
    lines = [
        "# 对比算法评估报告",
        "",
        f"- 输出目录：`{out_dir}`",
        f"- profile split：`{args.profile_split}`",
        f"- seeds/profile：`{args.num_seeds}`",
        f"- base_seed：`{args.base_seed}`",
        f"- action_hold_steps：`{args.action_hold_steps}`（0 表示按 TTT 自适应保持）",
        "",
        "## Overall Summary",
        "",
    ]
    if overall_rows:
        lines.append("| policy | episodes | ho | sinr_p5 | outage | pingpong | overlap_sinr | overlap_interrupt |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
        for row in overall_rows:
            lines.append(
                "| {policy} | {n} | {ho:.3f} | {sinr:.3f} | {out:.5f} | {pp:.3f} | {oz:.3f} | {intoz:.5f} |".format(
                    policy=row.get("policy", ""),
                    n=int(row.get("num_episodes", 0)),
                    ho=float(row.get("ho_count_mean", 0.0)),
                    sinr=float(row.get("sinr_p5_db_mean", 0.0)),
                    out=float(row.get("outage_time_ratio_mean", 0.0)),
                    pp=float(row.get("ping_pong_count_mean", 0.0)),
                    oz=float(row.get("overlap_zone_sinr_mean_db_mean", 0.0)),
                    intoz=float(row.get("comm_interruption_ratio_in_overlap_zone_mean", 0.0)),
                )
            )
    lines.extend([
        "",
        "## 文件",
        "",
        "- `metrics/comparison_episode_metrics.csv`：逐 episode 明细。",
        "- `metrics/comparison_summary_by_profile.csv`：按 profile 和 policy 汇总。",
        "- `metrics/comparison_summary_overall.csv`：跨 profile 汇总。",
        "- `metrics/run_config.json`：本次命令与配置。",
        "- `tables/tabular_q_policy.json`：若启用 Q-learning 训练，则保存训练出的 Q 表。",
        "",
        "## 说明",
        "",
        "这些 baseline 是可复现实验代理，不声称完全等同原论文实现；具体近似关系见 `comparison_algorithms/docs/literature_to_baselines.md`。",
    ])
    (out_dir / "experiment_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args) -> Path:
    env_config_path = _resolve_path(args.env_config_path)
    model_config_path = _resolve_path(args.model_config_path)
    profiles_path = _resolve_path(args.scenario_profiles_path)
    base_env_config = load_yaml(env_config_path)
    model_config = load_yaml(model_config_path)

    if args.output_dir:
        out_dir = _resolve_path(args.output_dir)
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = PROJECT_ROOT / "experiments" / "runs" / f"{stamp}_comparison_algorithms"
    (out_dir / "metrics").mkdir(parents=True, exist_ok=True)
    (out_dir / "tables").mkdir(parents=True, exist_ok=True)

    action_space = ActionSpace()
    policies = build_policies(args, out_dir, base_env_config, model_config, action_space)
    sampler = ScenarioProfileSampler(profiles_path, split=args.profile_split)
    profile_names = args.profiles or sampler.profile_names()

    episode_rows: List[Dict] = []
    oracle_rows: List[Dict] = []

    for profile_name in profile_names:
        for idx in tqdm(range(args.num_seeds), desc=f"评估 {profile_name}"):
            env_seed = args.base_seed + idx
            profile_seed = args.profile_seed_offset + env_seed
            env_config, profile_meta = sampler.build_config(
                base_env_config,
                seed=profile_seed,
                profile_name=profile_name,
            )
            scenario_data = ScenarioGenerator(env_config).generate_scenario(seed=env_seed, position_resolution_m=1.0)

            for policy in policies:
                env = TrainHandoverEnv(config=env_config)
                traj = run_policy_episode(
                    env,
                    policy,
                    scenario_data=scenario_data,
                    seed=env_seed,
                    action_hold_steps=args.action_hold_steps,
                )
                kpis = collect_episode_kpis(traj["trajectory"], traj.get("last_info", {}), env_config)
                episode_rows.append(add_episode_metadata(kpis, traj, env_config, profile_name, policy.name, env_seed))

            if args.include_oracle_a3:
                best = evaluate_fixed_oracle(
                    env_config,
                    scenario_data,
                    action_space,
                    seed=env_seed,
                    ho_penalty_weight=args.oracle_ho_penalty_weight,
                    action_hold_steps=args.action_hold_steps,
                )
                if best:
                    policy_name = "OracleA3_FullGrid"
                    row = add_episode_metadata(
                        best["kpis"],
                        best["traj"],
                        env_config,
                        profile_name,
                        policy_name,
                        env_seed,
                    )
                    row["oracle_hys_db"] = float(best["hys_db"])
                    row["oracle_ttt_ms"] = float(best["ttt_ms"])
                    episode_rows.append(row)
                    oracle_rows.append(row)

    by_profile = summarize_rows(episode_rows, ["profile", "policy"])
    overall = summarize_rows(episode_rows, ["policy"])

    write_csv(out_dir / "metrics" / "comparison_episode_metrics.csv", episode_rows)
    write_csv(out_dir / "metrics" / "comparison_summary_by_profile.csv", by_profile)
    write_csv(out_dir / "metrics" / "comparison_summary_overall.csv", overall)
    write_csv(out_dir / "metrics" / "oracle_a3_selections.csv", oracle_rows)
    write_json(out_dir / "metrics" / "run_config.json", {
        "output_dir": str(out_dir),
        "env_config_path": str(env_config_path),
        "model_config_path": str(model_config_path),
        "scenario_profiles_path": str(profiles_path),
        "profile_split": args.profile_split,
        "profiles": profile_names,
        "num_seeds": int(args.num_seeds),
        "base_seed": int(args.base_seed),
        "profile_seed_offset": int(args.profile_seed_offset),
        "policies": [p.name for p in policies] + (["OracleA3_FullGrid"] if args.include_oracle_a3 else []),
        "include_rainbow": bool(args.include_rainbow),
        "checkpoint_path": args.checkpoint_path,
        "train_tabular_q": bool(args.train_tabular_q),
        "q_table_path": args.q_table_path,
        "include_oracle_a3": bool(args.include_oracle_a3),
        "oracle_ho_penalty_weight": float(args.oracle_ho_penalty_weight),
        "include_rainbow_late_guard": bool(args.include_rainbow_late_guard),
        "include_rainbow_physics_risk": bool(args.include_rainbow_physics_risk),
        "risk_penalty": float(args.risk_penalty),
        "risk_horizon_index": int(args.risk_horizon_index),
        "late_guard_config": build_late_guard_config(args).to_dict(),
    })
    write_markdown_report(out_dir, args, overall, by_profile)

    print(f"\n对比实验完成: {out_dir}")
    print(f"Overall summary: {out_dir / 'metrics' / 'comparison_summary_overall.csv'}")
    print(f"Report: {out_dir / 'experiment_report.md'}")
    return out_dir


def parse_args():
    parser = argparse.ArgumentParser(description="对比算法 profile 评估")
    parser.add_argument("--env_config_path", type=str, default="configs/default_env_config.yaml")
    parser.add_argument("--model_config_path", type=str, default="configs/model_config.yaml")
    parser.add_argument("--scenario_profiles_path", type=str, default="configs/scenario_profiles.yaml")
    parser.add_argument("--profile_split", type=str, default="test")
    parser.add_argument("--profiles", nargs="*", default=None)
    parser.add_argument("--num_seeds", type=int, default=5)
    parser.add_argument("--base_seed", type=int, default=20000)
    parser.add_argument("--profile_seed_offset", type=int, default=3000000)
    parser.add_argument("--output_dir", type=str, default="")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--action_hold_steps", type=int, default=0)
    parser.add_argument("--include_rainbow", action="store_true")
    parser.add_argument("--include_rainbow_late_guard", action="store_true")
    parser.add_argument("--include_rainbow_physics_risk", action="store_true")
    parser.add_argument("--risk_penalty", type=float, default=2.0)
    parser.add_argument("--risk_horizon_index", type=int, default=-1)
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="experiments/runs/20260604_v1_domain_random_obs7_nocql_seed20260604/checkpoints/rainbow_offline_best.pth",
    )
    parser.add_argument("--train_tabular_q", action="store_true")
    parser.add_argument("--q_table_path", type=str, default="")
    parser.add_argument("--q_train_episodes", type=int, default=80)
    parser.add_argument("--q_train_profile_split", type=str, default="train")
    parser.add_argument("--q_train_profiles", nargs="*", default=None)
    parser.add_argument("--q_train_base_seed", type=int, default=60000)
    parser.add_argument("--q_alpha", type=float, default=0.08)
    parser.add_argument("--q_gamma", type=float, default=0.98)
    parser.add_argument("--q_epsilon_start", type=float, default=0.25)
    parser.add_argument("--q_epsilon_end", type=float, default=0.02)
    parser.add_argument("--include_oracle_a3", action="store_true")
    parser.add_argument("--oracle_ho_penalty_weight", type=float, default=0.5)
    parser.add_argument("--guard_low_sinr_db", type=float, default=-5.0)
    parser.add_argument("--guard_critical_sinr_db", type=float, default=-7.0)
    parser.add_argument("--guard_fast_sinr_db", type=float, default=-3.0)
    parser.add_argument("--guard_moderate_sinr_db", type=float, default=0.0)
    parser.add_argument("--guard_min_delta_db", type=float, default=3.0)
    parser.add_argument("--guard_strong_delta_db", type=float, default=8.0)
    parser.add_argument("--guard_fast_speed_kmh", type=float, default=350.0)
    parser.add_argument("--guard_min_time_since_ho_s", type=float, default=1.0)
    parser.add_argument("--guard_low_max_ttt_ms", type=float, default=150.0)
    parser.add_argument("--guard_critical_max_ttt_ms", type=float, default=150.0)
    parser.add_argument("--guard_fast_max_ttt_ms", type=float, default=150.0)
    parser.add_argument("--guard_hys_cap_db", type=float, default=3.0)
    parser.add_argument("--guard_critical_hys_cap_db", type=float, default=2.5)
    parser.add_argument("--guard_delta3_hys_cap_db", type=float, default=2.5)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
