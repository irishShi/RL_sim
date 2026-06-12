"""从 Obs7 历史窗口派生物理可解释特征。"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import torch


PHYSICS_FEATURE_NAMES = [
    "sinr_current_centered",
    "sinr_std_scaled",
    "sinr_min_centered",
    "sinr_p10_centered",
    "sinr_l3_residual_scaled",
    "effective_fading_margin_scaled",
    "delta_current_scaled",
    "delta_slope_scaled",
    "delta_positive_ratio",
    "delta_advantage_ratio",
    "speed_norm",
    "time_since_ho_norm",
]

PHYSICS_FEATURE_DIM = len(PHYSICS_FEATURE_NAMES)


def _as_horizon_list(horizons_ms: Sequence[int] | str | None) -> list[int]:
    if horizons_ms is None:
        return [100, 200, 300]
    if isinstance(horizons_ms, str):
        return [int(x.strip()) for x in horizons_ms.split(",") if x.strip()]
    return [int(x) for x in horizons_ms]


def parse_horizons_ms(horizons_ms: Sequence[int] | str | None) -> list[int]:
    """解析风险预测窗口配置。"""
    parsed = _as_horizon_list(horizons_ms)
    if not parsed:
        raise ValueError("风险预测窗口不能为空")
    return parsed


def _center_clip_np(x: np.ndarray, x_min: float, x_max: float) -> np.ndarray:
    y = (x - x_min) / (x_max - x_min + 1e-8)
    return np.clip(2.0 * y - 1.0, -1.0, 1.0)


def _center_clip_torch(x: torch.Tensor, x_min: float, x_max: float) -> torch.Tensor:
    y = (x - x_min) / (x_max - x_min + 1e-8)
    return torch.clamp(2.0 * y - 1.0, -1.0, 1.0)


def _denorm_sinr_np(sinr_norm: np.ndarray, sinr_min_db: float, sinr_max_db: float) -> np.ndarray:
    return sinr_norm * (sinr_max_db - sinr_min_db) + sinr_min_db


def _denorm_delta_np(delta_norm: np.ndarray, delta_min_db: float, delta_max_db: float) -> np.ndarray:
    return delta_norm * (delta_max_db - delta_min_db) + delta_min_db


def _denorm_sinr_torch(sinr_norm: torch.Tensor, sinr_min_db: float, sinr_max_db: float) -> torch.Tensor:
    return sinr_norm * (sinr_max_db - sinr_min_db) + sinr_min_db


def _denorm_delta_torch(delta_norm: torch.Tensor, delta_min_db: float, delta_max_db: float) -> torch.Tensor:
    return delta_norm * (delta_max_db - delta_min_db) + delta_min_db


def compute_physics_features_np(
    obs_windows: np.ndarray,
    *,
    short_window_steps: int = 5,
    delta_t_s: float = 0.05,
    l3_alpha: float = 0.7,
    sinr_min_db: float = -10.0,
    sinr_max_db: float = 20.0,
    delta_min_db: float = -30.0,
    delta_max_db: float = 30.0,
    delta_advantage_db: float = 1.5,
) -> np.ndarray:
    """由 `[B, N, 7]` Obs7 窗口计算物理派生特征。

    返回特征已做尺度压缩，便于直接拼接到神经网络共享特征后。
    """
    obs = np.asarray(obs_windows, dtype=np.float32)
    if obs.ndim != 3 or obs.shape[-1] < 7:
        raise ValueError(f"obs_windows 期望形状 [B, N, >=7]，实际为 {obs.shape}")

    k = max(2, min(int(short_window_steps), obs.shape[1]))
    recent = obs[:, -k:, :]
    sinr = _denorm_sinr_np(recent[:, :, 3], sinr_min_db, sinr_max_db)
    delta = _denorm_delta_np(recent[:, :, 2], delta_min_db, delta_max_db)

    sinr_current = sinr[:, -1]
    sinr_std = sinr.std(axis=1)
    sinr_min = sinr.min(axis=1)
    sinr_p10 = np.percentile(sinr, 10, axis=1)

    ema = sinr[:, 0].copy()
    for i in range(1, k):
        ema = float(l3_alpha) * ema + (1.0 - float(l3_alpha)) * sinr[:, i]
    sinr_l3_residual = sinr_current - ema
    fading_margin = ema - sinr_p10

    delta_current = delta[:, -1]
    t = (np.arange(k, dtype=np.float32) - float(k - 1) / 2.0) * float(delta_t_s)
    denom = float(np.sum(t ** 2)) + 1e-8
    delta_centered = delta - delta.mean(axis=1, keepdims=True)
    delta_slope = np.sum(delta_centered * t.reshape(1, -1), axis=1) / denom
    delta_positive_ratio = (delta > 0.0).mean(axis=1)
    delta_advantage_ratio = (delta >= float(delta_advantage_db)).mean(axis=1)

    speed_norm = obs[:, -1, 4]
    time_since_ho_norm = obs[:, -1, 6]

    features = np.stack(
        [
            _center_clip_np(sinr_current, sinr_min_db, sinr_max_db),
            np.clip(sinr_std / 10.0, 0.0, 1.0),
            _center_clip_np(sinr_min, sinr_min_db, sinr_max_db),
            _center_clip_np(sinr_p10, sinr_min_db, sinr_max_db),
            np.clip(sinr_l3_residual / 10.0, -1.0, 1.0),
            np.clip(fading_margin / 10.0, 0.0, 1.0),
            np.clip(delta_current / 15.0, -1.0, 1.0),
            np.clip(delta_slope / 30.0, -1.0, 1.0),
            np.clip(delta_positive_ratio, 0.0, 1.0),
            np.clip(delta_advantage_ratio, 0.0, 1.0),
            np.clip(speed_norm, 0.0, 1.0),
            np.clip(time_since_ho_norm, 0.0, 1.0),
        ],
        axis=1,
    )
    return features.astype(np.float32)


def compute_physics_features_torch(
    obs_windows: torch.Tensor,
    *,
    short_window_steps: int = 5,
    delta_t_s: float = 0.05,
    l3_alpha: float = 0.7,
    sinr_min_db: float = -10.0,
    sinr_max_db: float = 20.0,
    delta_min_db: float = -30.0,
    delta_max_db: float = 30.0,
    delta_advantage_db: float = 1.5,
) -> torch.Tensor:
    """Torch 版本物理派生特征，用于模型前向传播。"""
    if obs_windows.dim() != 3 or obs_windows.size(-1) < 7:
        raise ValueError(f"obs_windows 期望形状 [B, N, >=7]，实际为 {tuple(obs_windows.shape)}")

    k = max(2, min(int(short_window_steps), obs_windows.size(1)))
    recent = obs_windows[:, -k:, :]
    sinr = _denorm_sinr_torch(recent[:, :, 3], sinr_min_db, sinr_max_db)
    delta = _denorm_delta_torch(recent[:, :, 2], delta_min_db, delta_max_db)

    sinr_current = sinr[:, -1]
    sinr_std = sinr.std(dim=1, unbiased=False)
    sinr_min = sinr.min(dim=1).values
    sinr_sorted = sinr.sort(dim=1).values
    p10_idx = max(0, min(k - 1, int(np.ceil(0.1 * k)) - 1))
    sinr_p10 = sinr_sorted[:, p10_idx]

    ema = sinr[:, 0]
    for i in range(1, k):
        ema = float(l3_alpha) * ema + (1.0 - float(l3_alpha)) * sinr[:, i]
    sinr_l3_residual = sinr_current - ema
    fading_margin = ema - sinr_p10

    delta_current = delta[:, -1]
    t = (torch.arange(k, device=obs_windows.device, dtype=obs_windows.dtype) - float(k - 1) / 2.0)
    t = t * float(delta_t_s)
    denom = torch.sum(t ** 2) + 1e-8
    delta_centered = delta - delta.mean(dim=1, keepdim=True)
    delta_slope = torch.sum(delta_centered * t.view(1, -1), dim=1) / denom
    delta_positive_ratio = (delta > 0.0).to(obs_windows.dtype).mean(dim=1)
    delta_advantage_ratio = (delta >= float(delta_advantage_db)).to(obs_windows.dtype).mean(dim=1)

    speed_norm = torch.clamp(obs_windows[:, -1, 4], 0.0, 1.0)
    time_since_ho_norm = torch.clamp(obs_windows[:, -1, 6], 0.0, 1.0)

    return torch.stack(
        [
            _center_clip_torch(sinr_current, sinr_min_db, sinr_max_db),
            torch.clamp(sinr_std / 10.0, 0.0, 1.0),
            _center_clip_torch(sinr_min, sinr_min_db, sinr_max_db),
            _center_clip_torch(sinr_p10, sinr_min_db, sinr_max_db),
            torch.clamp(sinr_l3_residual / 10.0, -1.0, 1.0),
            torch.clamp(fading_margin / 10.0, 0.0, 1.0),
            torch.clamp(delta_current / 15.0, -1.0, 1.0),
            torch.clamp(delta_slope / 30.0, -1.0, 1.0),
            torch.clamp(delta_positive_ratio, 0.0, 1.0),
            torch.clamp(delta_advantage_ratio, 0.0, 1.0),
            speed_norm,
            time_since_ho_norm,
        ],
        dim=1,
    )


def horizon_steps(horizons_ms: Iterable[int], delta_t_s: float = 0.05) -> list[int]:
    """将毫秒预测窗口转换为样本步数。"""
    dt_ms = max(float(delta_t_s) * 1000.0, 1e-6)
    return [max(1, int(np.ceil(float(h) / dt_ms))) for h in horizons_ms]
