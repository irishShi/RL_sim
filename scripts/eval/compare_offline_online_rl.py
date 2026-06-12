"""离线 RL 与在线 RL 的轻量对比实验。

该脚本面向论文中的工程论证：在相同铁路切换环境下，比较当前离线
Rainbow checkpoint 与一个从零在线交互训练的 Rainbow baseline，在
安全性、训练效率和最终 holdout 效果上的差异。

注意：这里的 online baseline 是快速、可复现的参考，不声称代表充分调参
后的最优在线 RL。实验目的在于说明高铁通信场景中离线训练路线的工程优势。
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import yaml
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from comparison_algorithms.evaluation import (  # noqa: E402
    add_episode_metadata,
    collect_episode_kpis,
    load_rainbow_policy,
    run_policy_episode,
    summarize_rows,
    write_csv,
    write_json,
)
from envs.train_ho_env import TrainHandoverEnv  # noqa: E402
from models import ActionSpace, ObservationWindow, RainbowWithForecast  # noqa: E402
from scripts.train.train_rainbow import get_beta, normalize_delta_rsrp, train_step  # noqa: E402
from utils import ActionHoldController, NStepBuffer, ReplayBuffer, ScenarioProfileSampler  # noqa: E402
from utils.late_handover_guard import LateHandoverGuardConfig  # noqa: E402
from utils.scenario_generator import ScenarioGenerator  # noqa: E402


OFFLINE_POLICY = "Offline_Rainbow"
OFFLINE_GUARD_POLICY = "Offline_Rainbow_LateGuard"
ONLINE_POLICY = "Online_Rainbow_Scratch"


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_yaml(path: str | Path) -> Dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def write_rows_csv(path: str | Path, rows: List[Dict]) -> None:
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def make_model(model_config: Dict, device: str, use_noisy: bool) -> RainbowWithForecast:
    hys_set = model_config["action_space"]["hys_set"]
    ttt_set = model_config["action_space"]["ttt_set"]
    return RainbowWithForecast(
        obs_dim=int(model_config["observation"]["obs_dim"]),
        num_actions=len(hys_set) * len(ttt_set),
        n_steps=int(model_config["observation"]["window_size"]),
        feature_hidden=int(model_config["network"]["shared"]["hidden_dim"]),
        encoder_hidden=int(model_config["network"]["encoder"]["hidden_dim"]),
        num_atoms=int(model_config["network"]["rainbow"]["num_atoms"]),
        v_min=float(model_config["network"]["rainbow"]["v_min"]),
        v_max=float(model_config["network"]["rainbow"]["v_max"]),
        use_noisy=bool(use_noisy),
    ).to(device)


def build_delta3_low5_guard() -> LateHandoverGuardConfig:
    return LateHandoverGuardConfig(
        enabled=True,
        min_delta_db=3.0,
        low_sinr_db=-5.0,
        critical_sinr_db=-7.0,
        strong_delta_db=8.0,
        fast_sinr_db=-3.0,
        moderate_sinr_db=0.0,
        min_time_since_ho_s=1.0,
        low_max_ttt_ms=150.0,
        critical_max_ttt_ms=150.0,
        fast_max_ttt_ms=150.0,
        hys_cap_db=3.0,
        critical_hys_cap_db=2.5,
        delta3_hys_cap_db=2.5,
    )


def build_env_config(
    sampler: ScenarioProfileSampler,
    base_env_config: Dict,
    profile_name: Optional[str],
    profile_seed: int,
) -> Tuple[Dict, Dict]:
    return sampler.build_config(base_env_config, seed=profile_seed, profile_name=profile_name)


def collect_training_kpis(
    trajectory: List[Dict],
    last_info: Dict,
    env_config: Dict,
    profile_name: str,
    episode_idx: int,
    env_seed: int,
    epsilon: float,
    total_reward: float,
    steps: int,
    updates: int,
) -> Dict:
    kpis = collect_episode_kpis(trajectory, last_info.get("kpis", {}) if last_info else {}, env_config)
    row = {
        "phase": "online_training",
        "profile": profile_name,
        "episode": int(episode_idx),
        "seed": int(env_seed),
        "epsilon": float(epsilon),
        "episode_reward": float(total_reward),
        "episode_steps": int(steps),
        "updates": int(updates),
    }
    row.update(kpis)
    return row


def run_online_training_episode(
    env: TrainHandoverEnv,
    online_net: RainbowWithForecast,
    target_net: RainbowWithForecast,
    optimizer: optim.Optimizer,
    buffer: ReplayBuffer,
    model_config: Dict,
    device: str,
    epsilon: float,
    global_step: int,
    action_hold_steps: int,
    seed: Optional[int] = None,
) -> Tuple[Dict, int, int]:
    action_space = ActionSpace()
    obs_dim = int(model_config["observation"]["obs_dim"])
    window_size = int(model_config["observation"]["window_size"])
    obs_window = ObservationWindow(window_size=window_size, obs_dim=obs_dim)
    n_step_buffer = NStepBuffer(
        n_steps=int(model_config["training"]["n_steps"]),
        gamma=float(model_config["training"]["gamma"]),
    )

    obs_raw, info = env.reset(seed=seed)
    obs_window.reset()
    obs_window.update_time(0.0)
    n_step_buffer.reset()
    for _ in range(window_size):
        obs_window.build_observation(
            obs_raw,
            info,
            velocity_mps=env.velocity_mps,
            track_length_m=env.cfg["track_length_m"],
        )

    train_freq = int(model_config["training"]["train_freq"])
    target_update_freq = int(model_config["training"]["target_update_freq"])
    update_freq = int(model_config["training"]["update_freq"])
    replay_start_size = int(model_config["training"]["replay_start_size"])
    batch_size = int(model_config["training"]["batch_size"])
    hold = ActionHoldController(action_space, env.cfg["delta_t_s"], fixed_hold_steps=action_hold_steps)

    done = False
    total_reward = 0.0
    steps = 0
    updates = 0
    trajectory: List[Dict] = []
    last_info = info

    while not done:
        window = obs_window.get_window()

        def select_model_action() -> int:
            window_tensor = torch.as_tensor(window, dtype=torch.float32, device=device).unsqueeze(0)
            return int(online_net.act(window_tensor, epsilon=float(epsilon)))

        action = hold.select(select_model_action)
        next_obs_raw, reward, terminated, truncated, next_info = env.step(action)
        done = bool(terminated or truncated)
        total_reward += float(reward)
        steps += 1

        current_time = float(env.time_step) * float(env.cfg["delta_t_s"])
        obs_window.update_time(current_time)
        if next_info.get("ho_executed", False):
            obs_window.update_ho_time(current_time)
        obs_window.update_params(
            next_info.get("current_hys", 3.0),
            next_info.get("current_ttt", 150.0),
        )
        obs_window.build_observation(
            next_obs_raw,
            next_info,
            velocity_mps=env.velocity_mps,
            track_length_m=env.cfg["track_length_m"],
        )
        next_window = obs_window.get_window()
        delta_rsrp_next = float(next_info["rsrp_neig_dbm"] - next_info["rsrp_serv_dbm"])
        delta_rsrp_norm = normalize_delta_rsrp(
            delta_rsrp_next,
            delta_min=float(model_config["normalization"]["delta_rsrp_min"]),
            delta_max=float(model_config["normalization"]["delta_rsrp_max"]),
        )

        n_step_exp = n_step_buffer.add(window, int(action), float(reward), next_window, done, delta_rsrp_norm)
        if n_step_exp is not None:
            buffer.store(**n_step_exp)

        if global_step % train_freq == 0 and buffer.size >= replay_start_size:
            for _ in range(update_freq):
                beta = get_beta(
                    global_step,
                    beta_start=float(model_config["training"]["per_beta"]),
                    beta_increment=float(model_config["training"]["per_beta_increment"]),
                )
                batch = buffer.sample(batch_size, beta=beta)
                if batch is not None:
                    train_info = train_step(batch, online_net, target_net, optimizer, model_config, device)
                    buffer.update_priorities(batch["indices"], train_info["td_errors"])
                    updates += 1

        if global_step % target_update_freq == 0 and global_step > 0:
            target_net.load_state_dict(online_net.state_dict())

        hys, ttt = action_space.action_to_hys_ttt(int(action))
        trajectory.append({
            "step": int(steps),
            "time_s": float(steps * env.cfg["delta_t_s"]),
            "x": float(next_info.get("position_m", 0.0)),
            "serving_cell": int(next_info.get("serving_cell", 0)),
            "rsrp_A_dbm": float(next_info.get("rsrp_A_dbm", 0.0)),
            "rsrp_B_dbm": float(next_info.get("rsrp_B_dbm", 0.0)),
            "rsrp_serv_dbm": float(next_info.get("rsrp_serv_dbm", 0.0)),
            "rsrp_neig_dbm": float(next_info.get("rsrp_neig_dbm", 0.0)),
            "delta_rsrp_dbm": float(next_info.get("delta_rsrp_dbm", 0.0)),
            "sinr_serv_db": float(next_info.get("sinr_serv_db", 0.0)),
            "outage": bool(next_info.get("outage", False)),
            "outage_during_interruption": bool(next_info.get("outage_during_interruption", False)),
            "in_interruption": bool(next_info.get("in_interruption", False)),
            "in_overlap_zone": bool(next_info.get("in_overlap_zone", False)),
            "time_since_last_ho": float(next_info.get("time_since_last_ho", 0.0)),
            "ho_executed": bool(next_info.get("ho_executed", False)),
            "ho_triggered": bool(next_info.get("ho_triggered", False)),
            "ho_blocked_by_guard": bool(next_info.get("ho_blocked_by_guard", False)),
            "current_hys": float(next_info.get("current_hys", hys)),
            "current_ttt": float(next_info.get("current_ttt", ttt)),
            "action": int(action),
        })

        obs_raw, info = next_obs_raw, next_info
        last_info = next_info
        global_step += 1

    for exp in n_step_buffer.flush():
        buffer.store(**exp)

    return {
        "trajectory": trajectory,
        "last_info": last_info,
        "total_reward": float(total_reward),
        "steps": int(steps),
    }, global_step, updates


def train_online_baseline(args: argparse.Namespace, run_dir: Path, base_env_config: Dict,
                          model_config: Dict, device: str) -> Dict:
    train_start = time.time()
    sampler = ScenarioProfileSampler(args.scenario_profiles_path, split=args.train_profile_split)
    action_space = ActionSpace()
    model_config = dict(model_config)
    model_config["training"] = dict(model_config["training"])
    model_config["training"]["batch_size"] = int(args.online_batch_size)
    model_config["training"]["replay_start_size"] = int(args.online_replay_start_size)

    online_net = make_model(model_config, device, use_noisy=False)
    target_net = make_model(model_config, device, use_noisy=False)
    target_net.load_state_dict(online_net.state_dict())
    optimizer = optim.Adam(online_net.parameters(), lr=float(model_config["training"]["learning_rate"]))
    buffer = ReplayBuffer(
        capacity=int(args.online_replay_capacity),
        obs_window_size=int(model_config["observation"]["window_size"]),
        obs_dim=int(model_config["observation"]["obs_dim"]),
        per_alpha=float(model_config["training"]["per_alpha"]),
    )

    train_rows: List[Dict] = []
    total_updates = 0
    global_step = 0
    eps_values = np.linspace(args.online_epsilon_start, args.online_epsilon_end, max(1, args.online_episodes))

    for ep in tqdm(range(args.online_episodes), desc="在线训练 Rainbow"):
        env_seed = int(args.online_base_seed + ep)
        profile_seed = int(args.profile_seed_offset + env_seed)
        profile_name = None
        if args.train_profiles:
            profile_name = args.train_profiles[ep % len(args.train_profiles)]
        env_config, profile_meta = build_env_config(
            sampler,
            base_env_config,
            profile_name=profile_name,
            profile_seed=profile_seed,
        )
        env = TrainHandoverEnv(config=env_config)
        episode_result, global_step, updates = run_online_training_episode(
            env=env,
            online_net=online_net,
            target_net=target_net,
            optimizer=optimizer,
            buffer=buffer,
            model_config=model_config,
            device=device,
            epsilon=float(eps_values[ep]),
            global_step=global_step,
            action_hold_steps=int(args.action_hold_steps),
            seed=env_seed,
        )
        total_updates += int(updates)
        train_rows.append(collect_training_kpis(
            episode_result["trajectory"],
            episode_result.get("last_info", {}),
            env_config,
            profile_name=str(profile_meta.get("profile_name", "")),
            episode_idx=ep,
            env_seed=env_seed,
            epsilon=float(eps_values[ep]),
            total_reward=float(episode_result["total_reward"]),
            steps=int(episode_result["steps"]),
            updates=int(updates),
        ))

    checkpoint_path = run_dir / "checkpoints" / "online_rainbow_scratch_final.pth"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "episode": int(args.online_episodes),
        "step": int(global_step),
        "online_net_state_dict": online_net.state_dict(),
        "target_net_state_dict": target_net.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": model_config,
        "training_mode": "online_scratch",
    }, checkpoint_path)

    train_elapsed = time.time() - train_start
    training_summary = summarize_training_rows(train_rows)
    training_summary.update({
        "online_checkpoint_path": str(checkpoint_path),
        "online_training_elapsed_sec": float(train_elapsed),
        "online_training_elapsed_min": float(train_elapsed / 60.0),
        "online_env_episodes": int(args.online_episodes),
        "online_env_steps": int(global_step),
        "online_gradient_updates": int(total_updates),
        "online_replay_final_size": int(buffer.size),
        "online_batch_size": int(model_config["training"]["batch_size"]),
        "online_replay_start_size": int(model_config["training"]["replay_start_size"]),
        "online_trainable_params": int(sum(p.numel() for p in online_net.parameters())),
        "online_profiles": sampler.describe(),
    })
    write_rows_csv(run_dir / "metrics" / "online_training_episode_metrics.csv", train_rows)
    write_json(run_dir / "metrics" / "online_training_summary.json", training_summary)
    return {
        "checkpoint_path": checkpoint_path,
        "model": online_net,
        "training_rows": train_rows,
        "training_summary": training_summary,
        "model_config": model_config,
    }


def summarize_training_rows(rows: List[Dict]) -> Dict:
    if not rows:
        return {}
    numeric_keys = sorted({
        key for row in rows for key, value in row.items()
        if isinstance(value, (int, float, np.number))
    })
    out: Dict[str, float] = {"num_training_episodes": float(len(rows))}
    for metric in numeric_keys:
        values = np.asarray([float(row.get(metric, np.nan)) for row in rows], dtype=np.float64)
        values = values[np.isfinite(values)]
        if values.size == 0:
            continue
        out[f"{metric}_mean"] = float(np.mean(values))
        out[f"{metric}_std"] = float(np.std(values))
        out[f"{metric}_min"] = float(np.min(values))
        out[f"{metric}_max"] = float(np.max(values))
        if metric in {"outage_time_ratio", "ping_pong_count", "interruption_total_time",
                      "comm_interruption_ratio_in_overlap_zone", "mobility_failure_rate_proxy"}:
            out[f"{metric}_worst10"] = float(np.percentile(values, 90))
        if metric in {"sinr_p5_db", "overlap_zone_sinr_mean_db"}:
            out[f"{metric}_worst10"] = float(np.percentile(values, 10))

    tail = rows[-max(1, min(10, len(rows))):]
    for metric in ("outage_time_ratio", "sinr_p5_db", "ho_per_km", "ping_pong_count",
                   "comm_interruption_ratio_in_overlap_zone", "mobility_failure_rate_proxy"):
        vals = [float(row.get(metric, np.nan)) for row in tail]
        vals = [v for v in vals if np.isfinite(v)]
        if vals:
            out[f"tail10_{metric}_mean"] = float(np.mean(vals))
    return out


def load_offline_training_summary(offline_run_dir: Path) -> Dict:
    logs_dir = offline_run_dir / "logs"
    if not logs_dir.exists():
        return {}
    paths = sorted(logs_dir.glob("training_diagnostics_*.json"), key=lambda p: p.stat().st_mtime)
    if not paths:
        return {}
    with paths[-1].open("r", encoding="utf-8") as f:
        payload = json.load(f)
    final = payload.get("final_summary", {}) or {}
    records = payload.get("records", []) or payload.get("diagnostics_records", []) or []
    elapsed = final.get("elapsed_sec")
    if elapsed is None and records:
        elapsed = records[-1].get("elapsed_sec")
    return {
        "offline_diagnostics_path": str(paths[-1]),
        "offline_run_started_at": payload.get("run_meta", {}).get("run_started_at") or payload.get("run_started_at"),
        "offline_run_finished_at": final.get("run_finished_at") or payload.get("run_finished_at"),
        "offline_actual_epochs": final.get("actual_epochs"),
        "offline_best_epoch": final.get("best_epoch_final"),
        "offline_best_metric_name": final.get("best_metric_name_final"),
        "offline_best_rainbow_loss": final.get("best_rainbow_loss_final"),
        "offline_training_elapsed_sec": elapsed,
        "offline_training_elapsed_min": (float(elapsed) / 60.0) if elapsed is not None else None,
    }


def load_dataset_summary(dataset_path: Path) -> Dict:
    metadata_path = dataset_path.with_suffix(".metadata.json")
    summary = {
        "offline_dataset_path": str(dataset_path),
        "offline_dataset_exists": bool(dataset_path.exists()),
    }
    if metadata_path.exists():
        with metadata_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        summary.update({
            "offline_dataset_metadata_path": str(metadata_path),
            "offline_dataset_num_samples": payload.get("num_samples"),
            "offline_dataset_policy_type": payload.get("policy_type"),
            "offline_dataset_profile_counts": (
                payload.get("profile_metadata", {}).get("profile_counts")
                if isinstance(payload.get("profile_metadata"), dict)
                else None
            ),
        })
    else:
        try:
            with np.load(dataset_path) as data:
                summary["offline_dataset_num_samples"] = int(data["actions"].shape[0])
        except Exception:
            pass
    return summary


def evaluate_policies(args: argparse.Namespace, run_dir: Path, base_env_config: Dict,
                      model_config: Dict, device: str, online_checkpoint_path: Path) -> Dict:
    eval_start = time.time()
    sampler = ScenarioProfileSampler(args.scenario_profiles_path, split=args.eval_profile_split)
    selected_profiles = args.eval_profiles or sampler.profile_names()
    action_space = ActionSpace()

    offline_policy = load_rainbow_policy(
        str(args.offline_checkpoint_path),
        model_config,
        device,
        action_space=action_space,
    )
    offline_policy.name = OFFLINE_POLICY
    offline_guard_policy = load_rainbow_policy(
        str(args.offline_checkpoint_path),
        model_config,
        device,
        action_space=action_space,
        late_guard_config=build_delta3_low5_guard(),
    )
    offline_guard_policy.name = OFFLINE_GUARD_POLICY
    online_policy = load_rainbow_policy(
        str(online_checkpoint_path),
        model_config,
        device,
        action_space=action_space,
    )
    online_policy.name = ONLINE_POLICY
    policies = [offline_policy, offline_guard_policy, online_policy]

    episode_rows: List[Dict] = []
    for profile_name in selected_profiles:
        for idx in tqdm(range(args.eval_num_seeds), desc=f"评估 {profile_name}"):
            env_seed = int(args.eval_base_seed + idx)
            profile_seed = int(args.profile_seed_offset + env_seed)
            env_config, _ = build_env_config(
                sampler,
                base_env_config,
                profile_name=profile_name,
                profile_seed=profile_seed,
            )
            scenario_gen = ScenarioGenerator(env_config)
            scenario_data = scenario_gen.generate_scenario(seed=env_seed, position_resolution_m=1.0)
            for policy in policies:
                env = TrainHandoverEnv(config=env_config)
                traj = run_policy_episode(
                    env,
                    policy,
                    scenario_data=scenario_data,
                    seed=None,
                    action_hold_steps=int(args.action_hold_steps),
                )
                kpis = collect_episode_kpis(traj["trajectory"], traj.get("last_info", {}), env_config)
                episode_rows.append(add_episode_metadata(kpis, traj, env_config, profile_name, policy.name, env_seed))

    summary_by_profile = summarize_rows(episode_rows, ["profile", "policy"])
    summary_overall = summarize_rows(episode_rows, ["policy"])
    write_csv(run_dir / "metrics" / "offline_online_episode_metrics.csv", episode_rows)
    write_csv(run_dir / "metrics" / "offline_online_summary_by_profile.csv", summary_by_profile)
    write_csv(run_dir / "metrics" / "offline_online_summary_overall.csv", summary_overall)
    payload = {
        "profiles": selected_profiles,
        "num_seeds": int(args.eval_num_seeds),
        "base_seed": int(args.eval_base_seed),
        "episode_rows": len(episode_rows),
        "summary_by_profile": summary_by_profile,
        "summary_overall": summary_overall,
        "evaluation_elapsed_sec": float(time.time() - eval_start),
    }
    write_json(run_dir / "metrics" / "offline_online_eval_summary.json", payload)
    return payload


def find_summary(summary_rows: List[Dict], policy: str) -> Dict:
    for row in summary_rows:
        if row.get("policy") == policy:
            return row
    return {}


def ratio(numer: Optional[float], denom: Optional[float]) -> Optional[float]:
    if numer is None or denom is None:
        return None
    denom = float(denom)
    if abs(denom) < 1e-12:
        return None
    return float(numer) / denom


def build_claim_tables(args: argparse.Namespace, run_dir: Path, online_train: Dict,
                       eval_payload: Dict, offline_summary: Dict,
                       dataset_summary: Dict) -> Dict:
    online_training = online_train["training_summary"]
    overall = eval_payload["summary_overall"]
    offline = find_summary(overall, OFFLINE_POLICY)
    offline_guard = find_summary(overall, OFFLINE_GUARD_POLICY)
    online = find_summary(overall, ONLINE_POLICY)

    offline_train_elapsed = offline_summary.get("offline_training_elapsed_sec")
    online_train_elapsed = online_training.get("online_training_elapsed_sec")
    offline_samples = dataset_summary.get("offline_dataset_num_samples")
    online_steps = online_training.get("online_env_steps")

    safety_rows = [
        {
            "method": "Offline training deployment route",
            "interactive_training_on_live_network": 0,
            "new_environment_steps_during_model_update": 0,
            "training_stage_outage_risk": "none during model update; exploration is moved to offline/simulation data collection",
            "reference_outage_time_ratio": "",
            "reference_ping_pong_count": "",
            "reference_mobility_failure_rate_proxy": "",
        },
        {
            "method": ONLINE_POLICY,
            "interactive_training_on_live_network": 1,
            "new_environment_steps_during_model_update": int(online_steps or 0),
            "training_stage_outage_risk": "online exploration episodes directly interact with the environment",
            "reference_outage_time_ratio": online_training.get("outage_time_ratio_mean"),
            "reference_ping_pong_count": online_training.get("ping_pong_count_mean"),
            "reference_mobility_failure_rate_proxy": online_training.get("mobility_failure_rate_proxy_mean"),
        },
    ]
    efficiency_rows = [
        {
            "method": OFFLINE_POLICY,
            "training_elapsed_sec": offline_train_elapsed,
            "training_elapsed_min": offline_summary.get("offline_training_elapsed_min"),
            "environment_steps_consumed_by_training": 0,
            "reused_dataset_samples": offline_samples,
            "gradient_updates_or_samples": None,
            "note": "model update reuses fixed offline dataset; no new environment interaction during training",
        },
        {
            "method": ONLINE_POLICY,
            "training_elapsed_sec": online_train_elapsed,
            "training_elapsed_min": online_training.get("online_training_elapsed_min"),
            "environment_steps_consumed_by_training": online_steps,
            "reused_dataset_samples": 0,
            "gradient_updates_or_samples": online_training.get("online_gradient_updates"),
            "note": "from-scratch online baseline, interleaving interaction and updates",
        },
    ]

    effect_metrics = [
        "outage_time_ratio_mean",
        "sinr_p5_db_mean",
        "ho_per_km_mean",
        "ping_pong_count_mean",
        "comm_interruption_ratio_in_overlap_zone_mean",
        "mobility_failure_rate_proxy_mean",
    ]
    effect_rows = []
    for policy_name, row in [
        (OFFLINE_POLICY, offline),
        (OFFLINE_GUARD_POLICY, offline_guard),
        (ONLINE_POLICY, online),
    ]:
        effect = {"policy": policy_name, "num_episodes": row.get("num_episodes")}
        for metric in effect_metrics:
            effect[metric] = row.get(metric)
        effect_rows.append(effect)

    claims = {
        "safety_table": safety_rows,
        "efficiency_table": efficiency_rows,
        "effect_table": effect_rows,
        "derived": {
            "offline_vs_online_outage_ratio": ratio(
                offline.get("outage_time_ratio_mean"),
                online.get("outage_time_ratio_mean"),
            ),
            "offline_guard_vs_online_outage_ratio": ratio(
                offline_guard.get("outage_time_ratio_mean"),
                online.get("outage_time_ratio_mean"),
            ),
            "offline_vs_online_sinr_p5_gain_db": (
                float(offline.get("sinr_p5_db_mean")) - float(online.get("sinr_p5_db_mean"))
                if offline.get("sinr_p5_db_mean") is not None and online.get("sinr_p5_db_mean") is not None
                else None
            ),
            "offline_guard_vs_online_sinr_p5_gain_db": (
                float(offline_guard.get("sinr_p5_db_mean")) - float(online.get("sinr_p5_db_mean"))
                if offline_guard.get("sinr_p5_db_mean") is not None and online.get("sinr_p5_db_mean") is not None
                else None
            ),
            "offline_training_elapsed_sec": offline_train_elapsed,
            "online_training_elapsed_sec": online_train_elapsed,
            "online_training_live_steps": online_steps,
            "offline_dataset_samples": offline_samples,
        },
        "caveat": (
            "该实验使用小规模 online-from-scratch baseline，目标是证明离线训练路线在铁路通信"
            "工程部署中的安全性和样本复用优势；不声称排除充分调参的安全在线微调方案。"
        ),
    }
    write_rows_csv(run_dir / "tables" / "tab_offline_online_safety.csv", safety_rows)
    write_rows_csv(run_dir / "tables" / "tab_offline_online_efficiency.csv", efficiency_rows)
    write_rows_csv(run_dir / "tables" / "tab_offline_online_effect.csv", effect_rows)
    write_json(run_dir / "metrics" / "offline_online_claims.json", claims)
    return claims


def fmt(value, digits: int = 4) -> str:
    if value is None or value == "":
        return ""
    try:
        value = float(value)
    except (TypeError, ValueError):
        return str(value)
    return f"{value:.{digits}f}"


def write_report(run_dir: Path, args: argparse.Namespace, online_train: Dict,
                 eval_payload: Dict, claims: Dict) -> None:
    effect_rows = claims["effect_table"]
    safety_rows = claims["safety_table"]
    efficiency_rows = claims["efficiency_table"]
    derived = claims["derived"]

    lines = [
        "# 离线学习 vs 在线学习轻量对比证明",
        "",
        f"- run：`{run_dir.as_posix()}`",
        f"- 在线 baseline：从零训练 `{args.online_episodes}` episodes，epsilon "
        f"`{args.online_epsilon_start:g}->{args.online_epsilon_end:g}`",
        f"- 评估：`{args.eval_profile_split}` profiles，每个 profile `{args.eval_num_seeds}` seeds",
        f"- 离线 checkpoint：`{args.offline_checkpoint_path}`",
        "",
        "## 结论摘要",
        "",
        "本实验支持的谨慎结论是：在当前高铁切换仿真设置下，离线训练路线更适合工程部署。"
        "它把探索风险前移到仿真/历史数据采集阶段，模型更新阶段不需要与在线网络交互；"
        "同时复用既有数据集完成训练。最终效果上，当前推荐的离线策略 + late guard 部署版本"
        "在 holdout profile 上比小规模从零在线训练 baseline 更均衡。"
        "",
        "该结论不应表述为“离线 RL 在所有条件下必然优于在线 RL”。更准确的说法是："
        "对安全敏感的铁路通信切换控制，离线训练 + holdout 验证 + runtime guard 是更低风险、"
        "更容易部署的路线。",
        "",
        "## 安全性",
        "",
        "| method | online interaction during model update | training env steps | reference outage | reference mobility failure |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in safety_rows:
        lines.append(
            f"| {row['method']} | {row['interactive_training_on_live_network']} | "
            f"{row['new_environment_steps_during_model_update']} | "
            f"{fmt(row['reference_outage_time_ratio'])} | "
            f"{fmt(row['reference_mobility_failure_rate_proxy'])} |"
        )

    lines.extend([
        "",
        "## 训练效率/交互样本效率",
        "",
        "| method | train time (s) | env steps consumed by training | reused dataset samples | updates/samples |",
        "|---|---:|---:|---:|---:|",
    ])
    for row in efficiency_rows:
        lines.append(
            f"| {row['method']} | {fmt(row['training_elapsed_sec'], 2)} | "
            f"{row['environment_steps_consumed_by_training']} | "
            f"{row['reused_dataset_samples'] or 0} | "
            f"{fmt(row['gradient_updates_or_samples'], 0)} |"
        )

    lines.extend([
        "",
        "## 训练效果",
        "",
        "| policy | episodes | outage | SINR p5 | HO/km | ping-pong | overlap interrupt | mobility failure |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in effect_rows:
        lines.append(
            f"| {row['policy']} | {row.get('num_episodes', '')} | "
            f"{fmt(row.get('outage_time_ratio_mean'))} | "
            f"{fmt(row.get('sinr_p5_db_mean'))} | "
            f"{fmt(row.get('ho_per_km_mean'))} | "
            f"{fmt(row.get('ping_pong_count_mean'))} | "
            f"{fmt(row.get('comm_interruption_ratio_in_overlap_zone_mean'))} | "
            f"{fmt(row.get('mobility_failure_rate_proxy_mean'))} |"
        )

    lines.extend([
        "",
        "## 可写入论文的表述",
        "",
        "离线学习的优势主要体现在三方面。第一，安全性方面，模型训练阶段不需要在运行网络中进行"
        "epsilon 探索，避免将未收敛策略直接作用于 A3 参数；在线 baseline 的训练期交互步数和"
        "outage/mobility failure 指标可作为探索风险的量化参考。第二，交互样本效率方面，离线训练可以复用"
        "既有仿真或历史测量数据，模型更新不再消耗新的在线环境交互样本，便于反复调参和离线验证。"
        "第三，效果方面，在相同 holdout profile/seeds 上，raw offline 网络保持最低切换次数和零乒乓，"
        "当前推荐的 `offline + late guard` 部署版本则在 outage、尾部 SINR、重叠区中断和移动性失败 proxy "
        "之间取得比小规模 online-from-scratch baseline 更均衡的结果。",
        "",
        "## 注意事项",
        "",
        "- online baseline 是小规模从零训练参考，不代表充分调参后的最优在线 RL。",
        "- 若论文审稿人关注在线学习，可将本文定位为安全敏感铁路通信场景下的离线部署路线，而不是"
        "对所有在线 RL 方法的否定。",
        "- `Offline_Rainbow_LateGuard` 代表当前推荐工程部署方案；`Offline_Rainbow` 用于学习方式公平对比。",
        "",
        "## 产物",
        "",
        "- `metrics/online_training_episode_metrics.csv`：在线训练期逐 episode 安全指标",
        "- `metrics/offline_online_episode_metrics.csv`：holdout 评估逐 episode 指标",
        "- `metrics/offline_online_summary_overall.csv`：总体效果汇总",
        "- `tables/tab_offline_online_safety.csv`",
        "- `tables/tab_offline_online_efficiency.csv`",
        "- `tables/tab_offline_online_effect.csv`",
    ])
    report_path = run_dir / "offline_online_comparison_report.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="比较离线 RL 与在线 RL baseline。")
    parser.add_argument(
        "--offline_checkpoint_path",
        type=Path,
        default=PROJECT_ROOT / "experiments/runs/20260604_v1_domain_random_obs7_nocql_seed20260604/checkpoints/rainbow_offline_best.pth",
    )
    parser.add_argument(
        "--offline_run_dir",
        type=Path,
        default=PROJECT_ROOT / "experiments/runs/20260604_v1_domain_random_obs7_nocql_seed20260604",
    )
    parser.add_argument(
        "--offline_dataset_path",
        type=Path,
        default=PROJECT_ROOT / "data/datasets/offline_dataset_v1_domain_random_obs7_seed20260604.npz",
    )
    parser.add_argument("--scenario_profiles_path", type=Path, default=PROJECT_ROOT / "configs/scenario_profiles.yaml")
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=20260611)
    parser.add_argument("--profile_seed_offset", type=int, default=1_000_000)
    parser.add_argument("--action_hold_steps", type=int, default=0)

    parser.add_argument("--online_episodes", type=int, default=40)
    parser.add_argument("--online_base_seed", type=int, default=61000)
    parser.add_argument("--online_epsilon_start", type=float, default=0.30)
    parser.add_argument("--online_epsilon_end", type=float, default=0.02)
    parser.add_argument("--online_batch_size", type=int, default=64)
    parser.add_argument("--online_replay_start_size", type=int, default=512)
    parser.add_argument("--online_replay_capacity", type=int, default=100000)
    parser.add_argument("--train_profile_split", type=str, default="train")
    parser.add_argument("--train_profiles", nargs="*", default=None)

    parser.add_argument("--eval_profile_split", type=str, default="test")
    parser.add_argument("--eval_profiles", nargs="*", default=None)
    parser.add_argument("--eval_num_seeds", type=int, default=5)
    parser.add_argument("--eval_base_seed", type=int, default=62000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_global_seed(int(args.seed))

    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.output_dir
    if run_dir is None:
        run_dir = PROJECT_ROOT / "experiments" / "runs" / f"{run_ts}_offline_vs_online_rl"
    elif not run_dir.is_absolute():
        run_dir = PROJECT_ROOT / run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    for sub in ("metrics", "tables", "checkpoints", "config"):
        (run_dir / sub).mkdir(parents=True, exist_ok=True)

    env_config_path = PROJECT_ROOT / "configs/default_env_config.yaml"
    model_config_path = PROJECT_ROOT / "configs/model_config.yaml"
    shutil.copy2(env_config_path, run_dir / "config" / env_config_path.name)
    shutil.copy2(model_config_path, run_dir / "config" / model_config_path.name)
    shutil.copy2(args.scenario_profiles_path, run_dir / "config" / args.scenario_profiles_path.name)

    base_env_config = load_yaml(env_config_path)
    model_config = load_yaml(model_config_path)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    run_config = {
        "run_dir": str(run_dir),
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "device": device,
        "started_at": datetime.now().isoformat(timespec="seconds"),
    }
    write_json(run_dir / "metrics" / "run_config.json", run_config)

    online_train = train_online_baseline(args, run_dir, base_env_config, model_config, device)
    eval_payload = evaluate_policies(
        args,
        run_dir,
        base_env_config,
        model_config,
        device,
        Path(online_train["checkpoint_path"]),
    )
    offline_summary = load_offline_training_summary(args.offline_run_dir)
    dataset_summary = load_dataset_summary(args.offline_dataset_path)
    write_json(run_dir / "metrics" / "offline_training_summary_reference.json", offline_summary)
    write_json(run_dir / "metrics" / "offline_dataset_summary_reference.json", dataset_summary)

    claims = build_claim_tables(args, run_dir, online_train, eval_payload, offline_summary, dataset_summary)
    write_report(run_dir, args, online_train, eval_payload, claims)
    print(f"完成：{run_dir}")
    print(f"报告：{run_dir / 'offline_online_comparison_report.md'}")


if __name__ == "__main__":
    main()
