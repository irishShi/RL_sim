"""低 SINR/快速退化场景下的晚切安全约束。"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Optional, Sequence, Tuple

import numpy as np


@dataclass
class LateHandoverGuardConfig:
    enabled: bool = True
    low_sinr_db: float = -3.0
    critical_sinr_db: float = -6.0
    fast_sinr_db: float = 0.0
    moderate_sinr_db: float = 3.0
    min_delta_db: float = 1.5
    strong_delta_db: float = 6.0
    fast_speed_kmh: float = 350.0
    min_time_since_ho_s: float = 0.5
    low_max_ttt_ms: float = 150.0
    critical_max_ttt_ms: float = 100.0
    fast_max_ttt_ms: float = 150.0
    hys_cap_db: float = 3.0
    critical_hys_cap_db: float = 2.5
    delta3_hys_cap_db: float = 2.5

    def to_dict(self) -> Dict[str, float | bool]:
        return asdict(self)


def _hys_cap_for_delta(delta_rsrp_db: float, requested_cap_db: float, delta3_cap_db: float) -> float:
    if float(delta_rsrp_db) < 4.0:
        return min(float(requested_cap_db), float(delta3_cap_db))
    return float(requested_cap_db)


def late_ho_guard_limits(
    speed_kmh: float,
    delta_rsrp_db: float,
    sinr_db: float,
    time_since_ho_s: float,
    cfg: LateHandoverGuardConfig,
) -> Tuple[Optional[float], Optional[float], str]:
    """返回当前状态下的 Hys/TTT 上限；空 reason 表示不启用保护。"""
    if not cfg.enabled:
        return None, None, ""
    if time_since_ho_s < cfg.min_time_since_ho_s:
        return None, None, ""
    if delta_rsrp_db < cfg.min_delta_db:
        return None, None, ""

    if sinr_db <= cfg.critical_sinr_db:
        hys_cap = _hys_cap_for_delta(delta_rsrp_db, cfg.critical_hys_cap_db, cfg.delta3_hys_cap_db)
        return hys_cap, cfg.critical_max_ttt_ms, "critical_sinr"

    if sinr_db <= cfg.low_sinr_db:
        hys_cap = _hys_cap_for_delta(delta_rsrp_db, cfg.hys_cap_db, cfg.delta3_hys_cap_db)
        return hys_cap, cfg.low_max_ttt_ms, "low_sinr"

    if speed_kmh >= cfg.fast_speed_kmh and sinr_db <= cfg.fast_sinr_db:
        hys_cap = _hys_cap_for_delta(delta_rsrp_db, cfg.hys_cap_db, cfg.delta3_hys_cap_db)
        return hys_cap, cfg.fast_max_ttt_ms, "fast_degradation_risk"

    if delta_rsrp_db >= cfg.strong_delta_db and sinr_db <= cfg.moderate_sinr_db:
        hys_cap = _hys_cap_for_delta(delta_rsrp_db, cfg.hys_cap_db, cfg.delta3_hys_cap_db)
        return hys_cap, cfg.low_max_ttt_ms, "strong_neighbor_low_margin"

    return None, None, ""


def select_guarded_action(
    action_space,
    q_values: Sequence[float],
    raw_action_id: int,
    speed_kmh: float,
    delta_rsrp_db: float,
    sinr_db: float,
    time_since_ho_s: float,
    cfg: LateHandoverGuardConfig,
) -> Tuple[int, Dict[str, float | int | str | bool | None]]:
    """在晚切风险状态下，选择满足 Hys/TTT 上限的最高 Q 动作。"""
    hys_cap, ttt_cap, reason = late_ho_guard_limits(
        speed_kmh=speed_kmh,
        delta_rsrp_db=delta_rsrp_db,
        sinr_db=sinr_db,
        time_since_ho_s=time_since_ho_s,
        cfg=cfg,
    )
    raw_action_id = int(raw_action_id)
    raw_hys, raw_ttt = action_space.action_to_hys_ttt(raw_action_id)
    info: Dict[str, float | int | str | bool | None] = {
        "raw_action_id": raw_action_id,
        "raw_hys_db": float(raw_hys),
        "raw_ttt_ms": float(raw_ttt),
        "guard_reason": reason,
        "guard_hys_cap_db": hys_cap,
        "guard_ttt_cap_ms": ttt_cap,
        "guard_applied": False,
    }
    if reason == "":
        return raw_action_id, info
    if raw_hys <= hys_cap and raw_ttt <= ttt_cap:
        return raw_action_id, info

    q_arr = np.asarray(q_values, dtype=np.float64)
    best_action = raw_action_id
    best_q = -np.inf
    for action_id, q_value in enumerate(q_arr):
        hys_db, ttt_ms = action_space.action_to_hys_ttt(action_id)
        if hys_db <= hys_cap and ttt_ms <= ttt_cap and q_value > best_q:
            best_q = float(q_value)
            best_action = int(action_id)

    info["guard_applied"] = best_action != raw_action_id
    return best_action, info

