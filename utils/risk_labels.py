"""从离线轨迹生成未来切换风险监督标签。"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Sequence

import numpy as np

from models.physics_features import horizon_steps, parse_horizons_ms


@dataclass
class FutureRiskLabelConfig:
    horizons_ms: tuple[int, ...] = (100, 200, 300)
    delta_t_s: float = 0.05
    sinr_min_db: float = -10.0
    sinr_max_db: float = 20.0
    delta_min_db: float = -30.0
    delta_max_db: float = 30.0
    low_sinr_db: float = -3.0
    outage_sinr_db: float = -6.0
    delta_advantage_db: float = 1.5
    strong_delta_db: float = 6.0
    high_ttt_ms: float = 300.0
    high_hys_db: float = 4.0

    def to_dict(self) -> Dict:
        return asdict(self)


def _denorm_sinr(sinr_norm: np.ndarray, cfg: FutureRiskLabelConfig) -> np.ndarray:
    return sinr_norm * (cfg.sinr_max_db - cfg.sinr_min_db) + cfg.sinr_min_db


def _denorm_delta(delta_norm: np.ndarray, cfg: FutureRiskLabelConfig) -> np.ndarray:
    return delta_norm * (cfg.delta_max_db - cfg.delta_min_db) + cfg.delta_min_db


def _episode_slices(dones: np.ndarray) -> list[tuple[int, int]]:
    slices: list[tuple[int, int]] = []
    start = 0
    for i, done in enumerate(np.asarray(dones, dtype=bool)):
        if bool(done):
            slices.append((start, i + 1))
            start = i + 1
    if start < len(dones):
        slices.append((start, len(dones)))
    return slices


def build_future_risk_labels(
    dataset: Dict,
    action_hys_ttt: np.ndarray,
    cfg: FutureRiskLabelConfig | None = None,
) -> Dict[str, np.ndarray]:
    """构造未来风险标签。

    第一版只为实际执行动作生成监督标签，形状为 `[N, H]`。未执行动作的
    风险由训练时的物理一致性损失约束。
    """
    cfg = cfg or FutureRiskLabelConfig()
    horizons = parse_horizons_ms(cfg.horizons_ms)
    h_steps = horizon_steps(horizons, cfg.delta_t_s)

    obs = np.asarray(dataset["obs"], dtype=np.float32)
    actions = np.asarray(dataset["actions"], dtype=np.int64)
    dones = np.asarray(dataset["dones"], dtype=bool)
    if obs.ndim != 3 or obs.shape[-1] < 7:
        raise ValueError(f"dataset['obs'] 期望形状 [N, T, >=7]，实际为 {obs.shape}")

    n = obs.shape[0]
    sinr = _denorm_sinr(obs[:, -1, 3], cfg)
    delta = _denorm_delta(obs[:, -1, 2], cfg)
    hys = action_hys_ttt[actions, 0].astype(np.float32)
    ttt = action_hys_ttt[actions, 1].astype(np.float32)

    labels = np.zeros((n, len(horizons)), dtype=np.float32)
    weights = np.zeros((n, len(horizons)), dtype=np.float32)
    future_min_sinr = np.full((n, len(horizons)), np.nan, dtype=np.float32)
    future_delta_adv_ratio = np.zeros((n, len(horizons)), dtype=np.float32)

    for start, end in _episode_slices(dones):
        for idx in range(start, end):
            for h_idx, step_count in enumerate(h_steps):
                j_end = min(end, idx + step_count + 1)
                if j_end <= idx + 1:
                    continue
                future_slice = slice(idx + 1, j_end)
                f_sinr = sinr[future_slice]
                f_delta = delta[future_slice]
                min_sinr = float(np.min(f_sinr))
                adv_ratio = float(np.mean(f_delta >= cfg.delta_advantage_db))
                future_min_sinr[idx, h_idx] = min_sinr
                future_delta_adv_ratio[idx, h_idx] = adv_ratio

                low_or_outage = min_sinr <= cfg.low_sinr_db
                outage_like = min_sinr <= cfg.outage_sinr_db
                neighbor_persistent = adv_ratio >= 0.5
                late_sensitive_action = (
                    float(ttt[idx]) >= cfg.high_ttt_ms or float(hys[idx]) >= cfg.high_hys_db
                )
                current_late_context = (
                    sinr[idx] <= cfg.low_sinr_db and delta[idx] >= cfg.delta_advantage_db
                ) or (
                    sinr[idx] <= 0.0 and delta[idx] >= cfg.strong_delta_db
                )

                labels[idx, h_idx] = float(
                    outage_like
                    or (low_or_outage and neighbor_persistent)
                    or (current_late_context and late_sensitive_action)
                )
                weights[idx, h_idx] = 1.0

    return {
        "future_risk_labels": labels,
        "future_risk_weights": weights,
        "future_min_sinr_db": future_min_sinr,
        "future_delta_advantage_ratio": future_delta_adv_ratio,
        "risk_horizons_ms": np.asarray(horizons, dtype=np.int64),
        "risk_label_config": cfg.to_dict(),
    }


def attach_future_risk_labels(
    dataset: Dict,
    action_hys_ttt: np.ndarray,
    cfg: FutureRiskLabelConfig | None = None,
) -> Dict:
    """返回带风险标签的新 dataset 字典，不原地修改输入。"""
    enriched = dict(dataset)
    enriched.update(build_future_risk_labels(enriched, action_hys_ttt, cfg))
    return enriched
