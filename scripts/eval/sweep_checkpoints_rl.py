"""快速筛选离线 Rainbow checkpoint 的 RL 策略表现。"""
import argparse
import glob
import json
import os
import re
import sys
from collections import Counter

import numpy as np
import torch
import yaml
from tqdm import tqdm

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
EVAL_DIR = os.path.dirname(__file__)
if EVAL_DIR not in sys.path:
    sys.path.insert(0, EVAL_DIR)

from envs.train_ho_env import TrainHandoverEnv
from models import ActionSpace, ObservationWindow, RainbowWithForecast
from scripts.eval.test_batch_comparison import collect_kpis
from scripts.eval.test_simple import run_episode
from utils.scenario_generator import ScenarioGenerator


def _checkpoint_sort_key(path: str):
    name = os.path.basename(path)
    if name == "rainbow_offline_best.pth":
        return (10_000, name)
    if name == "rainbow_offline_final.pth":
        return (10_001, name)
    match = re.search(r"epoch_(\d+)", name)
    return (int(match.group(1)) if match else 9999, name)


def _discover_checkpoints(run_dir: str, pattern: str, epochs: str) -> list[str]:
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    paths = sorted(glob.glob(os.path.join(ckpt_dir, pattern)), key=_checkpoint_sort_key)
    if not epochs:
        return paths

    wanted = {token.strip() for token in epochs.split(",") if token.strip()}
    selected = []
    for path in paths:
        name = os.path.basename(path)
        epoch_match = re.search(r"epoch_(\d+)", name)
        epoch = epoch_match.group(1) if epoch_match else None
        if name in wanted or (epoch is not None and epoch in wanted):
            selected.append(path)
    return selected


def _load_model(checkpoint_path: str, model_config: dict, device: str) -> RainbowWithForecast:
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


def _summarize_checkpoint(
    checkpoint_path: str,
    scenarios: list[dict],
    env_config_path: str,
    model_config: dict,
    device: str,
    top_k: int,
    action_hold_steps: int,
) -> dict:
    model = _load_model(checkpoint_path, model_config, device)
    obs_dim = model_config["observation"]["obs_dim"]
    window_size = model_config["observation"]["window_size"]
    action_space = ActionSpace()
    action_counter = Counter()
    all_kpis = []

    for scenario_data in tqdm(scenarios, desc=os.path.basename(checkpoint_path), leave=False):
        env = TrainHandoverEnv(config_path=env_config_path)
        obs_window = ObservationWindow(window_size=window_size, obs_dim=obs_dim)
        traj = run_episode(
            env,
            None,
            "RL策略 (Rainbow DQN)",
            obs_window=obs_window,
            model=model,
            device=device,
            seed=None,
            scenario_data=scenario_data,
            action_hold_steps=action_hold_steps,
        )
        kpis_from_env = None
        if traj["trajectory"]:
            kpis_from_env = traj.get("last_info", {}).get("kpis")
        all_kpis.append(collect_kpis(traj["trajectory"], kpis_from_env))

        for point in traj["trajectory"]:
            hys = float(point.get("current_hys", 3.0))
            ttt = float(point.get("current_ttt", 150.0))
            action = action_space.hys_ttt_to_action(hys, ttt)
            hys_c, ttt_c = action_space.action_to_hys_ttt(action)
            action_counter[(float(hys_c), float(ttt_c))] += 1

    metric_keys = all_kpis[0].keys() if all_kpis else []
    stats = {}
    for key in metric_keys:
        values = np.array([k[key] for k in all_kpis], dtype=np.float64)
        stats[f"{key}_mean"] = float(values.mean())
        stats[f"{key}_std"] = float(values.std())

    total_actions = sum(action_counter.values())
    top_actions = []
    for (hys, ttt), count in action_counter.most_common(top_k):
        top_actions.append({
            "hys_db": float(hys),
            "ttt_ms": float(ttt),
            "count": int(count),
            "ratio": float(count / max(total_actions, 1)),
        })

    return {
        "checkpoint": os.path.abspath(checkpoint_path),
        "checkpoint_name": os.path.basename(checkpoint_path),
        "stats": stats,
        "top_actions": top_actions,
    }


def main():
    parser = argparse.ArgumentParser(description="快速筛选多个离线 Rainbow checkpoint 的 RL 表现")
    parser.add_argument("--run_dir", type=str, required=True, help="包含 checkpoints/ 的实验目录")
    parser.add_argument("--pattern", type=str, default="rainbow_offline_epoch_*.pth",
                        help="checkpoint 文件匹配模式")
    parser.add_argument("--epochs", type=str, default=None,
                        help="只评估指定 epoch/文件名，逗号分隔，例如 220,240,260,best")
    parser.add_argument("--include_best", action="store_true", help="额外加入 rainbow_offline_best.pth")
    parser.add_argument("--include_final", action="store_true", help="额外加入 rainbow_offline_final.pth")
    parser.add_argument("--num_scenarios", type=int, default=20, help="评估场景数量")
    parser.add_argument("--base_seed", type=int, default=10000, help="场景随机种子起点")
    parser.add_argument("--top_k", type=int, default=5, help="输出动作分布Top-K")
    parser.add_argument("--action_hold_steps", type=int, default=0,
                        help="动作保持步数（0=根据TTT自适应，>0=固定步数）")
    parser.add_argument("--output_path", type=str, default=None, help="结果 JSON 输出路径")
    args = parser.parse_args()

    run_dir = args.run_dir if os.path.isabs(args.run_dir) else os.path.join(PROJECT_ROOT, args.run_dir)
    checkpoints = _discover_checkpoints(run_dir, args.pattern, args.epochs)
    if args.include_best:
        checkpoints.append(os.path.join(run_dir, "checkpoints", "rainbow_offline_best.pth"))
    if args.include_final:
        checkpoints.append(os.path.join(run_dir, "checkpoints", "rainbow_offline_final.pth"))
    checkpoints = sorted({os.path.abspath(p) for p in checkpoints if os.path.exists(p)}, key=_checkpoint_sort_key)
    if not checkpoints:
        raise FileNotFoundError(f"未找到 checkpoint: {run_dir}")

    env_config_path = os.path.join(PROJECT_ROOT, "configs", "default_env_config.yaml")
    model_config_path = os.path.join(PROJECT_ROOT, "configs", "model_config.yaml")
    with open(model_config_path, "r", encoding="utf-8") as f:
        model_config = yaml.safe_load(f)

    env_template = TrainHandoverEnv(config_path=env_config_path)
    scenario_gen = ScenarioGenerator(env_template.cfg)
    scenarios = [
        scenario_gen.generate_scenario(seed=args.base_seed + idx, position_resolution_m=1.0)
        for idx in range(args.num_scenarios)
    ]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")
    print(f"待评估 checkpoint 数: {len(checkpoints)} | 场景数: {args.num_scenarios}")

    results = []
    for checkpoint_path in checkpoints:
        result = _summarize_checkpoint(
            checkpoint_path,
            scenarios,
            env_config_path,
            model_config,
            device,
            top_k=max(1, int(args.top_k)),
            action_hold_steps=int(args.action_hold_steps),
        )
        results.append(result)
        s = result["stats"]
        top = result["top_actions"][0] if result["top_actions"] else {}
        print(
            f"{result['checkpoint_name']:<32} "
            f"HO={s.get('ho_count_mean', 0.0):.3f} "
            f"outage={s.get('outage_time_ratio_mean', 0.0):.4f} "
            f"overlapSINR={s.get('overlap_zone_sinr_mean_db_mean', 0.0):.3f} "
            f"top={top.get('hys_db', 0.0):.1f}/{top.get('ttt_ms', 0.0):.0f} "
            f"ratio={top.get('ratio', 0.0):.3f}"
        )

    ranked = sorted(
        results,
        key=lambda r: (
            r["stats"].get("ho_count_mean", 0.0),
            -r["stats"].get("overlap_zone_sinr_mean_db_mean", 0.0),
            r["stats"].get("outage_time_ratio_mean", 0.0),
        ),
    )
    payload = {
        "run_dir": os.path.abspath(run_dir),
        "num_scenarios": int(args.num_scenarios),
        "base_seed": int(args.base_seed),
        "ranking_rule": [
            "minimize RL ho_count_mean",
            "maximize overlap_zone_sinr_mean_db_mean",
            "minimize outage_time_ratio_mean",
        ],
        "best": ranked[0],
        "results": ranked,
    }

    output_path = args.output_path
    if output_path is None:
        output_path = os.path.join(run_dir, "metrics", "checkpoint_sweep_rl.json")
    output_path = output_path if os.path.isabs(output_path) else os.path.join(PROJECT_ROOT, output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"\n筛选结果已保存: {output_path}")


if __name__ == "__main__":
    main()
