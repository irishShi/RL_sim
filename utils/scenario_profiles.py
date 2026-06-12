"""场景 profile 采样工具。

该模块只负责根据 YAML 配置生成环境参数覆盖项，不改变观测维度和模型结构。
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import yaml


IMPORTANT_ENV_KEYS = [
    "track_length_m",
    "random_speed",
    "v_default_kmh",
    "v_min_kmh",
    "v_max_kmh",
    "Ptx_A_dbm",
    "Ptx_B_dbm",
    "pathloss_exp_A",
    "pathloss_exp_B",
    "shadow_sigma_A",
    "shadow_sigma_B",
    "shadow_corr_distance_m",
    "noise_dbm",
    "use_dynamic_interference",
    "enable_extra_interference",
    "extra_interference_dbm",
    "dynamic_interference_attenuation_db",
    "min_link_distance_m",
    "trackside_offset_m",
    "bs_height_m",
    "ue_height_m",
    "max_rsrp_dbm",
    "enable_fast_fading",
    "rician_k_factor_db",
]


def _as_split_set(profile: Dict[str, Any]) -> set[str]:
    if "splits" in profile:
        raw = profile.get("splits") or []
    else:
        raw = [profile.get("split", "train")]
    if isinstance(raw, str):
        raw = [raw]
    return {str(x) for x in raw}


def _is_distribution_spec(value: Any) -> bool:
    return isinstance(value, dict) and any(k in value for k in ("uniform", "choice", "int_choice"))


def _resolve_value(value: Any, rng: np.random.Generator) -> Any:
    """解析 YAML 中的采样表达式，返回一次具体取值。"""
    if _is_distribution_spec(value):
        if "uniform" in value:
            low, high = value["uniform"]
            return float(rng.uniform(float(low), float(high)))
        if "choice" in value:
            choices = value["choice"]
            idx = int(rng.integers(0, len(choices)))
            return copy.deepcopy(choices[idx])
        if "int_choice" in value:
            choices = value["int_choice"]
            idx = int(rng.integers(0, len(choices)))
            return int(choices[idx])

    if isinstance(value, dict):
        return {k: _resolve_value(v, rng) for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve_value(v, rng) for v in value]
    return copy.deepcopy(value)


def _deep_update(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict) and not _is_distribution_spec(value):
            base[key] = _deep_update(copy.deepcopy(base[key]), value)
        else:
            base[key] = copy.deepcopy(value)
    return base


class ScenarioProfileSampler:
    """按 split 和 seed 可复现采样场景 profile。"""

    def __init__(self, profiles_path: str | Path, split: str = "train"):
        self.profiles_path = Path(profiles_path)
        self.split = split
        with self.profiles_path.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        self.version = raw.get("version", 1)
        self.all_profiles = raw.get("profiles", [])
        self.profiles = [
            p for p in self.all_profiles
            if split == "all" or split in _as_split_set(p)
        ]
        if not self.profiles:
            raise ValueError(f"No scenario profiles found for split={split!r} in {self.profiles_path}")

        weights = np.array([float(p.get("weight", 1.0)) for p in self.profiles], dtype=np.float64)
        if np.any(weights < 0) or weights.sum() <= 0:
            raise ValueError(f"Invalid profile weights for split={split!r}")
        self.weights = weights / weights.sum()

    def profile_names(self) -> List[str]:
        return [str(p["name"]) for p in self.profiles]

    def get_profile(self, name: str) -> Dict[str, Any]:
        for profile in self.profiles:
            if profile.get("name") == name:
                return profile
        all_names = ", ".join(self.profile_names())
        raise KeyError(f"Unknown profile {name!r} for split={self.split!r}; available: {all_names}")

    def sample_profile(self, seed: Optional[int] = None) -> Dict[str, Any]:
        rng = np.random.default_rng(seed)
        idx = int(rng.choice(len(self.profiles), p=self.weights))
        return self.profiles[idx]

    def build_config(
        self,
        base_config: Dict[str, Any],
        seed: Optional[int] = None,
        profile_name: Optional[str] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """返回合并后的环境配置和本次采样 metadata。"""
        rng = np.random.default_rng(seed)
        profile = self.get_profile(profile_name) if profile_name else self.sample_profile(seed)
        resolved_overrides = _resolve_value(profile.get("overrides", {}), rng)
        env_config = _deep_update(copy.deepcopy(base_config), resolved_overrides)
        important_config = {k: env_config.get(k) for k in IMPORTANT_ENV_KEYS if k in env_config}

        metadata = {
            "profile_name": str(profile.get("name")),
            "profile_split": self.split,
            "profile_seed": None if seed is None else int(seed),
            "profile_description": str(profile.get("description", "")),
            "profiles_path": str(self.profiles_path),
            "resolved_overrides": resolved_overrides,
            "important_config": important_config,
        }
        return env_config, metadata

    def describe(self) -> Dict[str, Any]:
        return {
            "profiles_path": str(self.profiles_path),
            "split": self.split,
            "profile_names": self.profile_names(),
            "weights": {str(p["name"]): float(w) for p, w in zip(self.profiles, self.weights)},
        }
