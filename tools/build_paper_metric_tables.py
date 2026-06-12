"""从对比实验结果生成论文用指标表。

默认读取 2026-06-11 的 handover success proxy 对比实验结果，输出：
- paper/assets/tables/final/table_main_python_comparison_metrics.*
- paper/assets/tables/final/table_profile_python_comparison_metrics.*
- paper/assets/tables/source/table_weighted_handover_success_proxy.*
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]


POLICY_ORDER = [
    "FixedA3_Hys3_TTT150",
    "PositionPriorA3",
    "SpeedAdaptiveA3",
    "FixedA3_Hys2p5_TTT100",
    "RL_Rainbow_GRU_LateGuard",
    "TabularQ_A3",
    "SignalTrendGuardA3",
    "OracleA3_FullGrid",
    "RL_Rainbow_GRU",
]

POLICY_LABELS = {
    "FixedA3_Hys3_TTT150": "Fixed A3 (Hys=3, TTT=150)",
    "FixedA3_Hys2p5_TTT100": "Aggressive A3 (Hys=2.5, TTT=100)",
    "SpeedAdaptiveA3": "Speed-adaptive A3",
    "PositionPriorA3": "Position-prior A3",
    "SignalTrendGuardA3": "Signal-trend guard A3",
    "TabularQ_A3": "Tabular Q-learning",
    "RL_Rainbow_GRU": "Rainbow-GRU",
    "RL_Rainbow_GRU_LateGuard": "Rainbow-GRU + late guard",
    "OracleA3_FullGrid": "Oracle A3 grid",
}


def resolve_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def markdown_table(df: pd.DataFrame, float_digits: int = 3) -> str:
    """生成 GitHub 风格 Markdown 表，避免依赖 pandas 的 tabulate 可选包。"""
    formatted = df.copy()
    for col in formatted.columns:
        if pd.api.types.is_integer_dtype(formatted[col]):
            formatted[col] = formatted[col].map(lambda x: "" if pd.isna(x) else f"{int(x)}")
        elif pd.api.types.is_numeric_dtype(formatted[col]):
            formatted[col] = formatted[col].map(
                lambda x: "" if pd.isna(x) else f"{float(x):.{float_digits}f}"
            )
        else:
            formatted[col] = formatted[col].fillna("").astype(str)

    headers = [str(c) for c in formatted.columns]
    rows = formatted.astype(str).values.tolist()
    widths = [
        max(len(headers[i]), *(len(row[i]) for row in rows)) if rows else len(headers[i])
        for i in range(len(headers))
    ]

    def fmt_row(values: Iterable[str]) -> str:
        return "| " + " | ".join(str(v).ljust(widths[i]) for i, v in enumerate(values)) + " |"

    align = [
        ("-" * max(width - 1, 3) + ":") if pd.api.types.is_numeric_dtype(df[col]) else ("-" * max(width, 3))
        for width, col in zip(widths, df.columns)
    ]
    lines = [fmt_row(headers), fmt_row(align)]
    lines.extend(fmt_row(row) for row in rows)
    return "\n".join(lines) + "\n"


def sort_by_policy_order(df: pd.DataFrame, policy_col: str = "policy") -> pd.DataFrame:
    order = {policy: idx for idx, policy in enumerate(POLICY_ORDER)}
    out = df.copy()
    out["_policy_order"] = out[policy_col].map(order).fillna(9999).astype(int)
    return out.sort_values("_policy_order").drop(columns=["_policy_order"])


def build_tables(run_dir: Path, final_dir: Path, source_dir: Path) -> None:
    metrics_dir = run_dir / "metrics"
    overall = pd.read_csv(metrics_dir / "comparison_summary_overall.csv")
    by_profile = pd.read_csv(metrics_dir / "comparison_summary_by_profile.csv")
    episodes = pd.read_csv(metrics_dir / "comparison_episode_metrics.csv")

    final_dir.mkdir(parents=True, exist_ok=True)
    source_dir.mkdir(parents=True, exist_ok=True)

    main_cols = [
        "policy",
        "mobility_success_rate_proxy_mean",
        "ho_attempt_success_rate_mean",
        "outage_time_ratio_mean",
        "sinr_p5_db_mean",
        "ho_per_km_mean",
        "ping_pong_count_mean",
        "comm_interruption_ratio_in_overlap_zone_mean",
        "overlap_zone_sinr_mean_db_mean",
    ]
    main = sort_by_policy_order(overall[main_cols])
    main.insert(0, "Policy", main["policy"].map(POLICY_LABELS).fillna(main["policy"]))
    main = main.drop(columns=["policy"]).rename(columns={
        "mobility_success_rate_proxy_mean": "Mobility success",
        "ho_attempt_success_rate_mean": "HO attempt success",
        "outage_time_ratio_mean": "Outage ratio",
        "sinr_p5_db_mean": "SINR p5 (dB)",
        "ho_per_km_mean": "HO/km",
        "ping_pong_count_mean": "Ping-pong",
        "comm_interruption_ratio_in_overlap_zone_mean": "Overlap interruption",
        "overlap_zone_sinr_mean_db_mean": "Overlap SINR (dB)",
    })
    main.to_csv(final_dir / "table_main_python_comparison_metrics.csv", index=False, encoding="utf-8-sig")
    (final_dir / "table_main_python_comparison_metrics.md").write_text(
        markdown_table(main, float_digits=3),
        encoding="utf-8",
    )

    profile_cols = [
        "profile",
        "policy",
        "mobility_success_rate_proxy_mean",
        "ho_attempt_success_rate_mean",
        "outage_time_ratio_mean",
        "sinr_p5_db_mean",
        "ho_per_km_mean",
        "ping_pong_count_mean",
        "comm_interruption_ratio_in_overlap_zone_mean",
    ]
    profile = by_profile[profile_cols].copy()
    profile["Policy"] = profile["policy"].map(POLICY_LABELS).fillna(profile["policy"])
    profile["_policy_order"] = profile["policy"].map({p: i for i, p in enumerate(POLICY_ORDER)}).fillna(9999).astype(int)
    profile = profile.sort_values(["profile", "_policy_order"])
    profile = profile.drop(columns=["policy", "_policy_order"]).rename(columns={
        "profile": "Profile",
        "mobility_success_rate_proxy_mean": "Mobility success",
        "ho_attempt_success_rate_mean": "HO attempt success",
        "outage_time_ratio_mean": "Outage ratio",
        "sinr_p5_db_mean": "SINR p5 (dB)",
        "ho_per_km_mean": "HO/km",
        "ping_pong_count_mean": "Ping-pong",
        "comm_interruption_ratio_in_overlap_zone_mean": "Overlap interruption",
    })
    profile = profile[[
        "Profile",
        "Policy",
        "Mobility success",
        "HO attempt success",
        "Outage ratio",
        "SINR p5 (dB)",
        "HO/km",
        "Ping-pong",
        "Overlap interruption",
    ]]
    profile.to_csv(final_dir / "table_profile_python_comparison_metrics.csv", index=False, encoding="utf-8-sig")
    (final_dir / "table_profile_python_comparison_metrics.md").write_text(
        markdown_table(profile, float_digits=3),
        encoding="utf-8",
    )

    sum_cols = [
        "ho_attempt_count",
        "ho_success_count",
        "ho_failure_count",
        "late_ho_failure_count",
        "mobility_event_count_proxy",
        "mobility_failure_count_proxy",
    ]
    weighted = episodes.groupby("policy")[sum_cols].sum()
    weighted["weighted_ho_attempt_success_rate"] = (
        weighted["ho_success_count"] / weighted["ho_attempt_count"].replace(0, pd.NA)
    )
    weighted["weighted_mobility_success_rate_proxy"] = (
        weighted["ho_success_count"] / weighted["mobility_event_count_proxy"].replace(0, pd.NA)
    )
    weighted["weighted_mobility_failure_rate_proxy"] = (
        weighted["mobility_failure_count_proxy"] / weighted["mobility_event_count_proxy"].replace(0, pd.NA)
    )
    weighted = weighted.reset_index()
    weighted = sort_by_policy_order(weighted)
    weighted.insert(0, "Policy", weighted["policy"].map(POLICY_LABELS).fillna(weighted["policy"]))
    weighted = weighted.drop(columns=["policy"]).rename(columns={
        "ho_attempt_count": "HO attempts",
        "ho_success_count": "HO success",
        "ho_failure_count": "HO failures",
        "late_ho_failure_count": "Late-HO failures",
        "mobility_event_count_proxy": "Mobility events",
        "mobility_failure_count_proxy": "Mobility failures",
        "weighted_ho_attempt_success_rate": "Weighted HO attempt success",
        "weighted_mobility_success_rate_proxy": "Weighted mobility success",
        "weighted_mobility_failure_rate_proxy": "Weighted mobility failure",
    })
    count_cols = [
        "HO attempts",
        "HO success",
        "HO failures",
        "Late-HO failures",
        "Mobility events",
        "Mobility failures",
    ]
    weighted[count_cols] = weighted[count_cols].round().astype(int)
    weighted.to_csv(source_dir / "table_weighted_handover_success_proxy.csv", index=False, encoding="utf-8-sig")
    (source_dir / "table_weighted_handover_success_proxy.md").write_text(
        markdown_table(weighted, float_digits=3),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成论文用核心指标表")
    parser.add_argument(
        "--run_dir",
        type=str,
        default="experiments/runs/20260611_handover_success_proxy_comparison_test20",
    )
    parser.add_argument("--final_dir", type=str, default="paper/assets/tables/final")
    parser.add_argument("--source_dir", type=str, default="paper/assets/tables/source")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = resolve_path(args.run_dir)
    final_dir = resolve_path(args.final_dir)
    source_dir = resolve_path(args.source_dir)
    build_tables(run_dir, final_dir, source_dir)
    print(f"Wrote final tables to {final_dir}")
    print(f"Wrote source tables to {source_dir}")


if __name__ == "__main__":
    main()
