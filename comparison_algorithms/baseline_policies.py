"""高速铁路 A3/Hys/TTT 对比策略。

这些策略面向本仓库已有的 48 动作空间：
8 个 Hys × 6 个 TTT。策略只输出动作索引，真正的 A3 条件、TTT 计时、
切换保护和 KPI 仍由 envs.train_ho_env.TrainHandoverEnv 执行。
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from models import ActionSpace, ObservationWindow, RainbowWithForecast, RainbowWithPhysicsRisk
from utils.late_handover_guard import LateHandoverGuardConfig, select_guarded_action


def _nearest_action(action_space: ActionSpace, hys_db: float, ttt_ms: float) -> int:
    return int(action_space.hys_ttt_to_action(float(hys_db), float(ttt_ms)))


def _delta_rsrp(info: Dict) -> float:
    if "delta_rsrp_dbm" in info:
        return float(info.get("delta_rsrp_dbm", 0.0))
    return float(info.get("rsrp_neig_dbm", -90.0)) - float(info.get("rsrp_serv_dbm", -90.0))


class ComparisonPolicy:
    """评估脚本使用的最小策略接口。"""

    name: str

    def reset(self) -> None:
        pass

    def on_episode_start(self, obs, info: Dict, env) -> None:
        pass

    def select_action(self, obs, info: Dict, dt: float, env) -> int:
        raise NotImplementedError

    def postprocess_action(self, obs, info: Dict, dt: float, env, action: int) -> Tuple[int, Dict]:
        return int(action), {}

    def observe_step(self, obs, info: Dict, action: int, reward: float, next_obs,
                     next_info: Dict, done: bool, env) -> None:
        pass

    def get_episode_metrics(self) -> Dict:
        return {}


@dataclass
class FixedA3Policy(ComparisonPolicy):
    """固定 A3 参数基线。"""

    hys_db: float = 3.0
    ttt_ms: float = 150.0
    action_space: ActionSpace = None
    name: str = ""

    def __post_init__(self) -> None:
        if self.action_space is None:
            self.action_space = ActionSpace()
        if not self.name:
            self.name = f"FixedA3_Hys{self.hys_db:g}_TTT{int(round(self.ttt_ms))}"

    def select_action(self, obs, info: Dict, dt: float, env) -> int:
        return _nearest_action(self.action_space, self.hys_db, self.ttt_ms)


class SpeedAdaptiveA3Policy(ComparisonPolicy):
    """速度自适应 A3 基线。

    复现意图：对应高铁 LTE-R/5G-R 文献中的 speed/doppler-aware handover
    parameter adaptation。这里用分段函数近似：速度越高，TTT 越短；Hys 略增
    以抑制快衰落导致的乒乓。
    """

    name = "SpeedAdaptiveA3"

    def __init__(self, action_space: Optional[ActionSpace] = None):
        self.action_space = action_space or ActionSpace()
        self.current_hys = 3.0
        self.current_ttt = 150.0

    def reset(self) -> None:
        self.current_hys = 3.0
        self.current_ttt = 150.0

    def select_action(self, obs, info: Dict, dt: float, env) -> int:
        v_kmh = float(getattr(env, "velocity_mps", 0.0) or 0.0) * 3.6
        if v_kmh < 160.0:
            self.current_hys, self.current_ttt = 2.0, 150.0
        elif v_kmh < 250.0:
            self.current_hys, self.current_ttt = 2.5, 150.0
        elif v_kmh < 350.0:
            self.current_hys, self.current_ttt = 3.0, 100.0
        elif v_kmh < 420.0:
            self.current_hys, self.current_ttt = 3.5, 50.0
        else:
            self.current_hys, self.current_ttt = 4.0, 50.0
        return _nearest_action(self.action_space, self.current_hys, self.current_ttt)


class PositionPriorA3Policy(ComparisonPolicy):
    """轨道几何先验基线。

    复现意图：对应基于边界/切换点预测的铁路场景算法。当前 Python 环境是一维
    双小区轨道，因此用位置归一化作为 cell boundary prediction 的轻量代理。
    """

    name = "PositionPriorA3"

    def __init__(self, action_space: Optional[ActionSpace] = None):
        self.action_space = action_space or ActionSpace()
        self.current_hys = 3.0
        self.current_ttt = 150.0

    def reset(self) -> None:
        self.current_hys = 3.0
        self.current_ttt = 150.0

    def select_action(self, obs, info: Dict, dt: float, env) -> int:
        track_length = float(env.cfg.get("track_length_m", 3000.0))
        pos_norm = float(info.get("position_m", 0.0)) / max(track_length, 1e-6)
        serving = int(info.get("serving_cell", 0))

        if serving == 0:
            if pos_norm < 0.40:
                self.current_hys, self.current_ttt = 4.0, 300.0
            elif pos_norm < 0.50:
                self.current_hys, self.current_ttt = 3.0, 100.0
            else:
                self.current_hys, self.current_ttt = 2.0, 50.0
        else:
            # 已经切到 B 后，用保守参数减少 B->A 回切。
            self.current_hys, self.current_ttt = 4.5, 300.0

        return _nearest_action(self.action_space, self.current_hys, self.current_ttt)


class SignalTrendGuardA3Policy(ComparisonPolicy):
    """信号趋势保护基线。

    复现意图：近似 LIM2 / Kalman+SARSA 一类“预测 + 参数自适应”算法，但不训练
    SARSA，仅用指数平滑预测 Delta RSRP 与低 SINR 风险触发更激进的 Hys/TTT。
    """

    name = "SignalTrendGuardA3"

    def __init__(self, action_space: Optional[ActionSpace] = None, alpha: float = 0.65):
        self.action_space = action_space or ActionSpace()
        self.alpha = float(alpha)
        self.last_delta: Optional[float] = None
        self.ema_delta = 0.0
        self.ema_slope = 0.0
        self.current_hys = 3.0
        self.current_ttt = 150.0

    def reset(self) -> None:
        self.last_delta = None
        self.ema_delta = 0.0
        self.ema_slope = 0.0
        self.current_hys = 3.0
        self.current_ttt = 150.0

    def select_action(self, obs, info: Dict, dt: float, env) -> int:
        delta = _delta_rsrp(info)
        if self.last_delta is None:
            slope = 0.0
            self.ema_delta = delta
        else:
            slope = (delta - self.last_delta) / max(float(dt), 1e-6)
            self.ema_delta = self.alpha * delta + (1.0 - self.alpha) * self.ema_delta
        self.ema_slope = self.alpha * slope + (1.0 - self.alpha) * self.ema_slope
        self.last_delta = delta

        sinr = float(info.get("sinr_serv_db", 0.0))
        v_kmh = float(getattr(env, "velocity_mps", 0.0) or 0.0) * 3.6
        horizon_s = 0.15 if v_kmh < 350.0 else 0.10
        pred_delta = self.ema_delta + self.ema_slope * horizon_s

        if sinr < -5.0 or pred_delta > 5.0:
            self.current_hys, self.current_ttt = 1.5, 50.0
        elif sinr < -3.0 or pred_delta > 3.0:
            self.current_hys, self.current_ttt = 2.0, 50.0
        elif pred_delta > 1.0:
            self.current_hys, self.current_ttt = 2.5, 100.0
        elif pred_delta < -4.0 and sinr > 2.0:
            self.current_hys, self.current_ttt = 4.5, 300.0
        else:
            self.current_hys, self.current_ttt = 3.0, 150.0

        return _nearest_action(self.action_space, self.current_hys, self.current_ttt)


class RainbowPolicy(ComparisonPolicy):
    """当前 Rainbow+GRU 策略包装器，用于同一评估脚本横向比较。"""

    name = "RL_Rainbow_GRU"

    def __init__(self, model: RainbowWithForecast, device: str, window_size: int,
                 obs_dim: int, action_space: Optional[ActionSpace] = None):
        self.model = model
        self.device = device
        self.obs_window = ObservationWindow(window_size=window_size, obs_dim=obs_dim)
        self.action_space = action_space or ActionSpace()

    def reset(self) -> None:
        self.obs_window.reset()

    def on_episode_start(self, obs, info: Dict, env) -> None:
        self.obs_window.reset()
        self.obs_window.update_time(0.0)
        for _ in range(self.obs_window.window_size):
            self.obs_window.build_observation(
                obs,
                info,
                velocity_mps=env.velocity_mps,
                track_length_m=env.cfg["track_length_m"],
            )

    def select_action(self, obs, info: Dict, dt: float, env) -> int:
        q_values = self.get_q_values()
        return int(np.argmax(q_values))

    def get_q_values(self) -> np.ndarray:
        import torch

        window = self.obs_window.get_window()
        window_tensor = torch.as_tensor(window, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            return self.model.get_q_values(window_tensor).squeeze(0).detach().cpu().numpy()

    def observe_step(self, obs, info: Dict, action: int, reward: float, next_obs,
                     next_info: Dict, done: bool, env) -> None:
        dt = float(env.cfg["delta_t_s"])
        current_time = float(env.time_step) * dt
        self.obs_window.update_time(current_time)
        if next_info.get("ho_executed", False):
            self.obs_window.update_ho_time(current_time)
        self.obs_window.update_params(
            next_info.get("current_hys", 3.0),
            next_info.get("current_ttt", 150.0),
        )
        self.obs_window.build_observation(
            next_obs,
            next_info,
            velocity_mps=env.velocity_mps,
            track_length_m=env.cfg["track_length_m"],
        )


class RainbowLateGuardPolicy(RainbowPolicy):
    """Rainbow 策略叠加低 SINR/晚切风险保护层。

    保护层不改变网络结构；它把物理可解释的风险条件作为动作后处理约束：
    当服务 SINR 低、邻区明显更强或高速退化风险出现时，只允许较小 Hys/TTT，
    并在受限动作集合内选择当前 Q 值最高的动作。
    """

    name = "RL_Rainbow_GRU_LateGuard"

    def __init__(
        self,
        model: RainbowWithForecast,
        device: str,
        window_size: int,
        obs_dim: int,
        action_space: Optional[ActionSpace] = None,
        guard_config: Optional[LateHandoverGuardConfig] = None,
    ):
        super().__init__(model, device, window_size, obs_dim, action_space=action_space)
        self.guard_config = guard_config or LateHandoverGuardConfig()
        self.guard_checked_steps = 0
        self.guard_applied_steps = 0
        self.guard_reason_counts: Dict[str, int] = {}

    def reset(self) -> None:
        super().reset()
        self.guard_checked_steps = 0
        self.guard_applied_steps = 0
        self.guard_reason_counts = {}

    def postprocess_action(self, obs, info: Dict, dt: float, env, action: int) -> Tuple[int, Dict]:
        self.guard_checked_steps += 1
        rsrp_serv = float(info.get("rsrp_serv_dbm", -90.0))
        rsrp_neig = float(info.get("rsrp_neig_dbm", -90.0))
        delta_rsrp = float(info.get("delta_rsrp_dbm", rsrp_neig - rsrp_serv))
        sinr = float(info.get("sinr_serv_db", 0.0))
        current_time = float(getattr(env, "time_step", 0) or 0) * float(dt)
        time_since_ho = float(info.get("time_since_last_ho", current_time - env.ho_logic.last_ho_time))
        speed_kmh = float(getattr(env, "velocity_mps", 0.0) or 0.0) * 3.6

        q_values = self.get_q_values()
        guarded_action, guard_info = select_guarded_action(
            action_space=self.action_space,
            q_values=q_values,
            raw_action_id=int(action),
            speed_kmh=speed_kmh,
            delta_rsrp_db=delta_rsrp,
            sinr_db=sinr,
            time_since_ho_s=time_since_ho,
            cfg=self.guard_config,
        )
        if bool(guard_info.get("guard_applied", False)):
            self.guard_applied_steps += 1
            reason = str(guard_info.get("guard_reason", ""))
            self.guard_reason_counts[reason] = self.guard_reason_counts.get(reason, 0) + 1

        return int(guarded_action), guard_info

    def get_episode_metrics(self) -> Dict:
        checked = int(self.guard_checked_steps)
        metrics: Dict[str, float] = {
            "guard_checked_steps": float(checked),
            "guard_applied_steps": float(self.guard_applied_steps),
            "guard_applied_ratio": float(self.guard_applied_steps / checked) if checked else 0.0,
        }
        for reason, count in sorted(self.guard_reason_counts.items()):
            metrics[f"guard_reason_{reason}"] = float(count)
        return metrics


class RainbowPhysicsRiskPolicy(RainbowPolicy):
    """Rainbow 策略叠加神经网络未来风险预测。

    动作选择使用 `Q(s,a) - risk_penalty * Risk(s,a,H)`，其中 Risk 来自
    `RainbowWithPhysicsRisk` 的风险头。late guard 后续可继续作为独立兜底策略叠加。
    """

    name = "RL_Rainbow_GRU_PhysicsRisk"

    def __init__(
        self,
        model: RainbowWithPhysicsRisk,
        device: str,
        window_size: int,
        obs_dim: int,
        action_space: Optional[ActionSpace] = None,
        risk_penalty: float = 2.0,
        horizon_index: int = -1,
    ):
        super().__init__(model, device, window_size, obs_dim, action_space=action_space)
        self.risk_penalty = float(risk_penalty)
        self.horizon_index = int(horizon_index)
        self.selected_risks: List[float] = []
        self.mean_risks: List[float] = []

    def reset(self) -> None:
        super().reset()
        self.selected_risks = []
        self.mean_risks = []

    def select_action(self, obs, info: Dict, dt: float, env) -> int:
        import torch

        window = self.obs_window.get_window()
        window_tensor = torch.as_tensor(window, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_values_t, risk_prob_t = self.model.get_q_and_risk(window_tensor)
        q_values = q_values_t.squeeze(0).detach().cpu().numpy()
        risk_prob = risk_prob_t.squeeze(0).detach().cpu().numpy()
        h_idx = self.horizon_index
        if h_idx < 0:
            h_idx = risk_prob.shape[1] + h_idx
        h_idx = int(np.clip(h_idx, 0, risk_prob.shape[1] - 1))
        risk = risk_prob[:, h_idx]
        score = q_values - self.risk_penalty * risk
        action = int(np.argmax(score))
        self.selected_risks.append(float(risk[action]))
        self.mean_risks.append(float(np.mean(risk)))
        return action

    def get_episode_metrics(self) -> Dict:
        return {
            "risk_selected_mean": float(np.mean(self.selected_risks)) if self.selected_risks else 0.0,
            "risk_all_action_mean": float(np.mean(self.mean_risks)) if self.mean_risks else 0.0,
            "risk_penalty": float(self.risk_penalty),
        }


class TabularQPolicy(ComparisonPolicy):
    """离散状态 tabular Q-learning/SARSA 风格对比策略。"""

    name = "TabularQ_A3"

    def __init__(
        self,
        action_space: Optional[ActionSpace] = None,
        alpha: float = 0.08,
        gamma: float = 0.98,
        epsilon: float = 0.05,
        training: bool = False,
    ):
        self.action_space = action_space or ActionSpace()
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.epsilon = float(epsilon)
        self.training = bool(training)
        self.q_table: Dict[Tuple[int, ...], np.ndarray] = {}
        self.fallback_action = self.action_space.hys_ttt_to_action(3.0, 150.0)

    def reset(self) -> None:
        pass

    @staticmethod
    def state_from_info(info: Dict, env) -> Tuple[int, ...]:
        delta = _delta_rsrp(info)
        sinr = float(info.get("sinr_serv_db", 0.0))
        track_length = float(env.cfg.get("track_length_m", 3000.0))
        pos_norm = float(info.get("position_m", 0.0)) / max(track_length, 1e-6)
        v_kmh = float(getattr(env, "velocity_mps", 0.0) or 0.0) * 3.6
        time_since_ho = float(info.get("time_since_last_ho", 10.0))
        serving = int(info.get("serving_cell", 0))
        in_overlap = int(bool(info.get("in_overlap_zone", False)))

        return (
            int(np.digitize(delta, [-10, -6, -3, -1, 0, 1, 3, 6, 10])),
            int(np.digitize(sinr, [-8, -5, -3, 0, 3, 6, 10])),
            int(np.digitize(pos_norm, [0.35, 0.43, 0.48, 0.52, 0.57, 0.65])),
            int(np.digitize(v_kmh, [120, 200, 300, 350, 420])),
            int(np.digitize(time_since_ho, [0.2, 0.5, 1.0, 2.0, 5.0])),
            serving,
            in_overlap,
        )

    def _values(self, state: Tuple[int, ...]) -> np.ndarray:
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.action_space.num_actions, dtype=np.float32)
        return self.q_table[state]

    def select_action(self, obs, info: Dict, dt: float, env) -> int:
        state = self.state_from_info(info, env)
        if self.training and np.random.random() < self.epsilon:
            return int(np.random.randint(0, self.action_space.num_actions))
        values = self._values(state)
        if not np.any(np.isfinite(values)) or float(np.max(np.abs(values))) == 0.0:
            return int(self.fallback_action)
        return int(np.argmax(values))

    def observe_step(self, obs, info: Dict, action: int, reward: float, next_obs,
                     next_info: Dict, done: bool, env) -> None:
        if not self.training:
            return
        state = self.state_from_info(info, env)
        next_state = self.state_from_info(next_info, env)
        values = self._values(state)
        next_values = self._values(next_state)
        target = float(reward)
        if not done:
            target += self.gamma * float(np.max(next_values))
        values[int(action)] += self.alpha * (target - float(values[int(action)]))

    def to_json_dict(self) -> Dict:
        return {
            "name": self.name,
            "alpha": self.alpha,
            "gamma": self.gamma,
            "epsilon": self.epsilon,
            "num_states": len(self.q_table),
            "fallback_action": int(self.fallback_action),
            "q_table": {
                ",".join(str(x) for x in state): values.astype(float).tolist()
                for state, values in self.q_table.items()
            },
        }

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(self.to_json_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: str | Path, action_space: Optional[ActionSpace] = None,
             training: bool = False, epsilon: float = 0.0) -> "TabularQPolicy":
        path = Path(path)
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        policy = cls(
            action_space=action_space,
            alpha=float(payload.get("alpha", 0.08)),
            gamma=float(payload.get("gamma", 0.98)),
            epsilon=float(epsilon),
            training=training,
        )
        policy.fallback_action = int(payload.get("fallback_action", policy.fallback_action))
        for key, values in payload.get("q_table", {}).items():
            state = tuple(int(x) for x in key.split(",") if x != "")
            arr = np.asarray(values, dtype=np.float32)
            if arr.size == policy.action_space.num_actions:
                policy.q_table[state] = arr
        return policy


def build_default_comparison_policies(action_space: Optional[ActionSpace] = None) -> List[ComparisonPolicy]:
    action_space = action_space or ActionSpace()
    return [
        FixedA3Policy(3.0, 150.0, action_space, name="FixedA3_Hys3_TTT150"),
        FixedA3Policy(2.5, 100.0, action_space, name="FixedA3_Hys2p5_TTT100"),
        SpeedAdaptiveA3Policy(action_space),
        PositionPriorA3Policy(action_space),
        SignalTrendGuardA3Policy(action_space),
    ]
