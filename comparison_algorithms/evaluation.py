"""对比算法评估与训练共用工具。"""

from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
EVAL_DIR = PROJECT_ROOT / "scripts" / "eval"
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))

from envs.train_ho_env import TrainHandoverEnv
from models import ActionSpace, RainbowWithForecast, RainbowWithPhysicsRisk
from scripts.eval.test_batch_comparison import collect_kpis
from scripts.eval.test_simple import resolve_checkpoint_path
from utils.action_hold import ActionHoldController
from utils.scenario_generator import ScenarioGenerator
from utils.scenario_profiles import ScenarioProfileSampler

from comparison_algorithms.baseline_policies import (
    ComparisonPolicy,
    RainbowLateGuardPolicy,
    RainbowPhysicsRiskPolicy,
    RainbowPolicy,
)


def load_yaml(path: str | Path) -> Dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_rainbow_policy(
    checkpoint_path: str,
    model_config: Dict,
    device: str,
    action_space: Optional[ActionSpace] = None,
    late_guard_config=None,
    use_physics_risk: bool = False,
    risk_penalty: float = 2.0,
    risk_horizon_index: int = -1,
) -> RainbowPolicy:
    checkpoint_path = resolve_checkpoint_path(str(PROJECT_ROOT), checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    hys_set = model_config["action_space"]["hys_set"]
    ttt_set = model_config["action_space"]["ttt_set"]
    obs_dim = int(model_config["observation"]["obs_dim"])
    window_size = int(model_config["observation"]["window_size"])
    use_noisy = checkpoint.get("config", {}).get("use_noisy", False)
    if "use_cql" in checkpoint or "rainbow_offline" in str(checkpoint_path):
        use_noisy = False

    checkpoint_config = checkpoint.get("config", {}) or {}
    checkpoint_has_risk = bool(checkpoint.get("physics_risk_enabled", False))
    if use_physics_risk and not checkpoint_has_risk:
        raise ValueError(
            "请求使用 PhysicsRisk 策略，但 checkpoint 不包含 physics risk head。"
            f" checkpoint={checkpoint_path}"
        )
    model_has_risk = checkpoint_has_risk
    risk_cfg = checkpoint_config.get("physics_risk", {}) or model_config.get("physics_risk", {}) or {}
    num_risk_horizons = int(risk_cfg.get("num_horizons", len(risk_cfg.get("horizons_ms", [100, 200, 300]))))
    model_kwargs = dict(
        obs_dim=obs_dim,
        num_actions=len(hys_set) * len(ttt_set),
        n_steps=window_size,
        feature_hidden=model_config["network"]["shared"]["hidden_dim"],
        encoder_hidden=model_config["network"]["encoder"]["hidden_dim"],
        num_atoms=model_config["network"]["rainbow"]["num_atoms"],
        v_min=model_config["network"]["rainbow"]["v_min"],
        v_max=model_config["network"]["rainbow"]["v_max"],
        use_noisy=use_noisy,
    )
    if model_has_risk:
        model = RainbowWithPhysicsRisk(
            **model_kwargs,
            num_risk_horizons=num_risk_horizons,
            risk_hidden_dim=int(risk_cfg.get("risk_hidden_dim", 256)),
            physics_short_window_steps=int(risk_cfg.get("short_window_steps", 5)),
            physics_delta_t_s=float(risk_cfg.get("delta_t_s", 0.05)),
            physics_l3_alpha=float(risk_cfg.get("l3_alpha", 0.7)),
        ).to(device)
    else:
        model = RainbowWithForecast(**model_kwargs).to(device)
    model.load_state_dict(checkpoint["online_net_state_dict"])
    model.eval()
    if use_physics_risk and late_guard_config is None:
        return RainbowPhysicsRiskPolicy(
            model,
            device,
            window_size,
            obs_dim,
            action_space=action_space,
            risk_penalty=risk_penalty,
            horizon_index=risk_horizon_index,
        )
    policy_cls = RainbowLateGuardPolicy if late_guard_config is not None else RainbowPolicy
    if late_guard_config is not None:
        return policy_cls(
            model,
            device,
            window_size,
            obs_dim,
            action_space=action_space,
            guard_config=late_guard_config,
        )
    return policy_cls(model, device, window_size, obs_dim, action_space=action_space)


def run_policy_episode(
    env: TrainHandoverEnv,
    policy: ComparisonPolicy,
    scenario_data: Optional[Dict] = None,
    seed: Optional[int] = None,
    action_hold_steps: int = 0,
    max_steps: int = 10000,
) -> Dict:
    if scenario_data is not None:
        obs, info = env.reset(seed=None, options={"scenario_data": scenario_data})
    else:
        obs, info = env.reset(seed=seed)

    policy.reset()
    policy.on_episode_start(obs, info, env)
    dt = float(env.cfg["delta_t_s"])
    action_space = ActionSpace()
    hold = ActionHoldController(action_space, dt, fixed_hold_steps=action_hold_steps)

    done = False
    step_count = 0
    trajectory = []
    total_reward = 0.0

    while not done and step_count < int(max_steps):
        action = hold.select(lambda: policy.select_action(obs, info, dt, env))
        action, action_meta = policy.postprocess_action(obs, info, dt, env, action)
        next_obs, reward, terminated, truncated, next_info = env.step(action)
        done = bool(terminated or truncated)
        total_reward += float(reward)
        step_count += 1
        policy.observe_step(obs, info, int(action), float(reward), next_obs, next_info, done, env)

        hys, ttt = action_space.action_to_hys_ttt(int(action))
        trajectory.append({
            "step": int(step_count),
            "time_s": float(step_count * dt),
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
            "raw_action": int(action_meta.get("raw_action_id", action)),
            "guard_applied": bool(action_meta.get("guard_applied", False)),
            "guard_reason": str(action_meta.get("guard_reason", "")),
        })
        obs, info = next_obs, next_info

    return {
        "policy_name": policy.name,
        "trajectory": trajectory,
        "last_info": info,
        "total_reward": float(total_reward),
        "steps": int(step_count),
        "policy_metrics": policy.get_episode_metrics(),
    }


def add_episode_metadata(kpis: Dict, traj: Dict, env_config: Dict, profile: str,
                         policy_name: str, seed: int) -> Dict:
    row = {
        "profile": profile,
        "policy": policy_name,
        "seed": int(seed),
        "track_length_m": float(env_config.get("track_length_m", 3000.0)),
        "v_default_kmh": float(env_config.get("v_default_kmh", 0.0)),
        "episode_steps": int(traj.get("steps", 0)),
        "episode_reward": float(traj.get("total_reward", 0.0)),
    }
    kpis = dict(kpis)
    kpis.update(traj.get("policy_metrics", {}) or {})
    row.update({k: _json_safe(v) for k, v in kpis.items()})
    return row


def collect_episode_kpis(trajectory: List[Dict], last_info: Dict, env_config: Dict) -> Dict:
    """收集 KPI，并按当前 profile 的真实线路长度修正 ho_per_km。"""
    kpis = collect_kpis(trajectory, last_info.get("kpis", {}) if last_info else {})
    track_length_km = float(env_config.get("track_length_m", 3000.0)) / 1000.0
    if track_length_km > 0:
        kpis["ho_per_km"] = float(kpis.get("ho_count", 0.0)) / track_length_km
    kpis.update(compute_handover_reliability_proxy(trajectory, env_config))
    return kpis


def compute_handover_reliability_proxy(trajectory: List[Dict], env_config: Dict) -> Dict:
    """计算论文可解释的切换成功率与移动性失败 proxy。

    口径：
    - HO attempt：A3 条件满足且 TTT 到达，即环境中的 ``ho_triggered``。
    - HO success：attempt 实际执行后，在验证窗口内没有非切换中断导致的 outage，
      且没有很快回切到原小区。
    - late-HO failure proxy：未及时切换时，服务小区连续 outage 且邻区明显更强。

    该指标不改变环境动力学，只作为评估后处理，避免“执行即成功”的乐观口径。
    """
    if not trajectory:
        return {
            "ho_attempt_count": 0,
            "ho_success_count": 0,
            "ho_failure_count": 0,
            "ho_attempt_success_rate": 0.0,
            "ho_attempt_failure_rate": 0.0,
            "late_ho_failure_count": 0,
            "mobility_event_count_proxy": 0,
            "mobility_success_rate_proxy": 0.0,
            "mobility_failure_rate_proxy": 0.0,
        }

    dt = float(env_config.get("delta_t_s", 0.05))
    outage_db = float(env_config.get("sinr_outage_db", -6.0))
    t_out_s = float(env_config.get("T_out_s", 0.2))
    validation_window_s = float(env_config.get("ho_success_validation_window_s", 0.5))
    grace_default_s = max(float(env_config.get("ho_interruption_slots", 1)) * dt, t_out_s)
    post_ho_grace_s = float(env_config.get("ho_success_grace_s", grace_default_s))
    late_delta_db = float(env_config.get("ho_late_failure_delta_db", 3.0))

    attempt_indices = [idx for idx, row in enumerate(trajectory) if bool(row.get("ho_triggered", False))]

    success_count = 0
    blocked_failure_count = 0
    post_ho_rlf_failure_count = 0
    post_ho_pingpong_failure_count = 0
    validation_incomplete_count = 0

    for idx in attempt_indices:
        row = trajectory[idx]
        if not bool(row.get("ho_executed", False)):
            blocked_failure_count += 1
            continue

        t0 = float(row.get("time_s", (idx + 1) * dt))
        target_cell = int(row.get("serving_cell", 0))
        window_rows = [
            future
            for future in trajectory[idx + 1:]
            if float(future.get("time_s", 0.0)) <= t0 + validation_window_s + 1e-9
        ]
        useful_rows = [
            future
            for future in window_rows
            if (
                not bool(future.get("in_interruption", False))
                and float(future.get("time_s", 0.0)) >= t0 + post_ho_grace_s - 1e-9
            )
        ]
        if not useful_rows:
            validation_incomplete_count += 1

        post_ho_rlf = _has_continuous_outage_segment(
            useful_rows,
            outage_db=outage_db,
            min_duration_s=t_out_s,
            dt=dt,
        )
        post_ho_pingpong = any(
            int(future.get("serving_cell", target_cell)) != target_cell
            for future in useful_rows
        )

        if post_ho_rlf:
            post_ho_rlf_failure_count += 1
        elif post_ho_pingpong:
            post_ho_pingpong_failure_count += 1
        else:
            success_count += 1

    attempt_count = len(attempt_indices)
    ho_failure_count = blocked_failure_count + post_ho_rlf_failure_count + post_ho_pingpong_failure_count

    late_failure_count = _count_late_handover_failure_segments(
        trajectory=trajectory,
        outage_db=outage_db,
        late_delta_db=late_delta_db,
        min_duration_s=t_out_s,
        dt=dt,
    )
    mobility_event_count = attempt_count + late_failure_count
    mobility_failure_count = ho_failure_count + late_failure_count

    return {
        "ho_attempt_count": int(attempt_count),
        "ho_success_count": int(success_count),
        "ho_failure_count": int(ho_failure_count),
        "ho_blocked_failure_count": int(blocked_failure_count),
        "post_ho_rlf_failure_count": int(post_ho_rlf_failure_count),
        "post_ho_pingpong_failure_count": int(post_ho_pingpong_failure_count),
        "ho_validation_incomplete_count": int(validation_incomplete_count),
        "ho_attempt_success_rate": float(success_count / attempt_count) if attempt_count > 0 else 0.0,
        "ho_attempt_failure_rate": float(ho_failure_count / attempt_count) if attempt_count > 0 else 0.0,
        "late_ho_failure_count": int(late_failure_count),
        "mobility_event_count_proxy": int(mobility_event_count),
        "mobility_failure_count_proxy": int(mobility_failure_count),
        "mobility_success_rate_proxy": float(success_count / mobility_event_count) if mobility_event_count > 0 else 0.0,
        "mobility_failure_rate_proxy": float(mobility_failure_count / mobility_event_count) if mobility_event_count > 0 else 0.0,
        "ho_success_validation_window_s": float(validation_window_s),
        "ho_success_grace_s": float(post_ho_grace_s),
        "ho_late_failure_delta_db": float(late_delta_db),
    }


def _has_continuous_outage_segment(
    rows: List[Dict],
    outage_db: float,
    min_duration_s: float,
    dt: float,
) -> bool:
    """判断窗口内是否存在连续非中断 outage 片段。"""
    run_duration = 0.0
    threshold = max(float(min_duration_s), float(dt))
    for row in rows:
        non_interruption_outage = (
            not bool(row.get("in_interruption", False))
            and (
                (bool(row.get("outage", False)) and not bool(row.get("outage_during_interruption", False)))
                or float(row.get("sinr_serv_db", 0.0)) < outage_db
            )
        )
        if non_interruption_outage:
            run_duration += float(dt)
            if run_duration + 1e-12 >= threshold:
                return True
        else:
            run_duration = 0.0
    return False


def _count_late_handover_failure_segments(
    trajectory: List[Dict],
    outage_db: float,
    late_delta_db: float,
    min_duration_s: float,
    dt: float,
) -> int:
    """统计连续 outage 且邻区明显更强的 late-HO failure 片段数。"""
    count = 0
    run_duration = 0.0
    run_counted = False
    min_steps_duration = max(float(min_duration_s), float(dt))

    for row in trajectory:
        non_interruption_outage = (
            not bool(row.get("in_interruption", False))
            and (
                (bool(row.get("outage", False)) and not bool(row.get("outage_during_interruption", False)))
                or float(row.get("sinr_serv_db", 0.0)) < outage_db
            )
        )
        neighbor_clearly_better = float(row.get("delta_rsrp_dbm", 0.0)) >= late_delta_db
        risk = non_interruption_outage and neighbor_clearly_better

        if risk:
            run_duration += float(dt)
            if not run_counted and run_duration + 1e-12 >= min_steps_duration:
                count += 1
                run_counted = True
        else:
            run_duration = 0.0
            run_counted = False

    return int(count)


def _json_safe(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def summarize_rows(rows: List[Dict], group_keys: Iterable[str]) -> List[Dict]:
    group_keys = list(group_keys)
    groups: Dict[tuple, List[Dict]] = {}
    for row in rows:
        key = tuple(row.get(k, "") for k in group_keys)
        groups.setdefault(key, []).append(row)

    summaries: List[Dict] = []
    for key, items in sorted(groups.items(), key=lambda kv: tuple(str(x) for x in kv[0])):
        out = {k: v for k, v in zip(group_keys, key)}
        out["num_episodes"] = int(len(items))
        numeric_keys = sorted({
            k for item in items for k, v in item.items()
            if k not in group_keys and isinstance(v, (int, float, np.number))
        })
        for metric in numeric_keys:
            values = np.asarray([float(item.get(metric, np.nan)) for item in items], dtype=np.float64)
            values = values[np.isfinite(values)]
            if values.size == 0:
                continue
            out[f"{metric}_mean"] = float(np.mean(values))
            out[f"{metric}_std"] = float(np.std(values))
            out[f"{metric}_min"] = float(np.min(values))
            out[f"{metric}_max"] = float(np.max(values))
            if metric in {"sinr_mean_db", "sinr_p5_db", "overlap_zone_sinr_mean_db"}:
                out[f"{metric}_worst10"] = float(np.percentile(values, 10))
            elif metric in {"outage_time_ratio", "interruption_total_time", "ping_pong_count", "ho_count"}:
                out[f"{metric}_worst10"] = float(np.percentile(values, 90))
        summaries.append(out)
    return summaries


def write_csv(path: str | Path, rows: List[Dict]) -> None:
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


def write_json(path: str | Path, payload: Dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def select_oracle_kpis(candidates: List[Dict], ho_penalty_weight: float = 0.5) -> Dict:
    if not candidates:
        return {}

    def score(item: Dict) -> float:
        k = item["kpis"]
        return (
            float(k.get("overlap_zone_sinr_mean_db", 0.0))
            - float(ho_penalty_weight) * float(k.get("ho_count", 0.0))
            - 10.0 * float(k.get("outage_time_ratio", 0.0))
        )

    return max(
        candidates,
        key=lambda item: (
            score(item),
            -float(item["kpis"].get("ho_count", 0.0)),
            -float(item["kpis"].get("outage_time_ratio", 0.0)),
        ),
    )


def evaluate_fixed_oracle(
    env_config: Dict,
    scenario_data: Dict,
    action_space: ActionSpace,
    seed: int,
    ho_penalty_weight: float,
    action_hold_steps: int,
) -> Dict:
    from comparison_algorithms.baseline_policies import FixedA3Policy

    candidates = []
    for hys, ttt in action_space.get_all_actions():
        policy = FixedA3Policy(hys, ttt, action_space, name=f"OracleCandidate_Hys{hys:g}_TTT{int(ttt)}")
        env = TrainHandoverEnv(config=env_config)
        traj = run_policy_episode(env, policy, scenario_data=scenario_data, seed=seed,
                                  action_hold_steps=action_hold_steps)
        kpis = collect_episode_kpis(traj["trajectory"], traj.get("last_info", {}), env_config)
        candidates.append({"hys_db": float(hys), "ttt_ms": float(ttt), "kpis": kpis, "traj": traj})
    best = select_oracle_kpis(candidates, ho_penalty_weight=ho_penalty_weight)
    return best
