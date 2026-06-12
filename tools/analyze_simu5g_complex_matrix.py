"""批量分析 Simu5G 复杂场景矩阵结果。"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.eval.analyze_simu5g_results import (  # noqa: E402
    build_summary,
    detect_handover_events,
    find_result_files,
    parse_sca,
    parse_vec,
    plot_timeseries,
    write_summary_csv,
    write_timeseries_csv,
)


KEY_METRICS = [
    "handover_count",
    "first_handover_x_m",
    "app_packet_loss_ratio",
    "app_frame_delay_mean_s",
    "app_frame_delay_s_max",
    "rlc_delay_dl_mean_s",
    "mac_delay_dl_mean_s",
    "cqi_dl_mean",
    "cqi_dl_p5",
    "measured_sinr_dl_mean_db",
    "measured_sinr_dl_db_p5",
    "serving_sinr_proxy_mean_db",
    "railway_policy_hys_mean_db",
    "railway_policy_ttt_mean_s",
    "railway_policy_q_mean",
]


def to_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def case_parts(config: str) -> tuple[str, str]:
    if config.endswith("-FixedA3"):
        return config.removesuffix("-FixedA3"), "FixedA3"
    if config.endswith("-RLTable"):
        return config.removesuffix("-RLTable"), "RLTable"
    return config, "Unknown"


def read_summary_csv(path: Path) -> Dict[str, str]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return next(reader)


def write_rows(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    keys: List[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def iter_case_dirs(raw_root: Path) -> Iterable[Path]:
    for path in sorted(raw_root.iterdir()):
        if path.is_dir() and list(path.glob("*.sca")) and list(path.glob("*.vec")):
            yield path


def analyze_case(case_dir: Path, out_dir: Path, window_s: float) -> Dict[str, object]:
    sca_path, vec_path = find_result_files(case_dir)
    configs, scalars, _ = parse_sca(sca_path)
    series, _ = parse_vec(vec_path)
    summary = build_summary(configs, scalars, series, case_dir.name)
    handovers = detect_handover_events(series.get("serving_cell", []))

    out_dir.mkdir(parents=True, exist_ok=True)
    write_summary_csv(out_dir / "summary.csv", [summary])
    write_timeseries_csv(out_dir / "key_timeseries.csv", series)
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump({"summary": summary, "handovers": handovers}, f, indent=2, ensure_ascii=False)

    title = f"{case_dir.name} railway handover KPIs"
    plot_timeseries(out_dir / "timeseries_full.png", series, handovers, title)
    if handovers:
        t0 = handovers[0]["time_s"]
        plot_timeseries(
            out_dir / "handover_window.png",
            series,
            handovers,
            f"{case_dir.name} around first handover",
            window=(t0 - window_s, t0 + window_s),
        )

    scenario, strategy = case_parts(case_dir.name)
    summary["scenario"] = scenario
    summary["strategy"] = strategy
    return summary


def build_delta_rows(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    by_scenario: Dict[str, Dict[str, Dict[str, object]]] = {}
    for row in rows:
        by_scenario.setdefault(str(row["scenario"]), {})[str(row["strategy"])] = row

    delta_rows: List[Dict[str, object]] = []
    for scenario, strategies in sorted(by_scenario.items()):
        fixed = strategies.get("FixedA3")
        rl = strategies.get("RLTable")
        if not fixed or not rl:
            continue
        delta: Dict[str, object] = {"scenario": scenario}
        for metric in KEY_METRICS:
            fixed_value = to_float(fixed.get(metric))
            rl_value = to_float(rl.get(metric))
            delta[f"{metric}_fixed"] = fixed_value
            delta[f"{metric}_rl"] = rl_value
            delta[f"{metric}_delta_rl_minus_fixed"] = rl_value - fixed_value
        delta_rows.append(delta)
    return delta_rows


def plot_matrix(rows: List[Dict[str, object]], out_png: Path) -> None:
    if not rows:
        return

    metrics = [
        ("app_packet_loss_ratio", "Packet loss ratio"),
        ("measured_sinr_dl_db_p5", "SINR p5 / dB"),
        ("cqi_dl_mean", "CQI mean"),
        ("app_frame_delay_s_max", "App delay max / s"),
        ("first_handover_x_m", "First HO x / m"),
        ("railway_policy_ttt_mean_s", "RL TTT mean / s"),
    ]

    scenarios = sorted({str(row["scenario"]) for row in rows})
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    axes = axes.ravel()
    width = 0.38
    x = list(range(len(scenarios)))

    for ax, (metric, title) in zip(axes, metrics):
        fixed_values = []
        rl_values = []
        for scenario in scenarios:
            fixed = next((r for r in rows if r["scenario"] == scenario and r["strategy"] == "FixedA3"), {})
            rl = next((r for r in rows if r["scenario"] == scenario and r["strategy"] == "RLTable"), {})
            fixed_values.append(to_float(fixed.get(metric)))
            rl_values.append(to_float(rl.get(metric)))
        ax.bar([i - width / 2 for i in x], fixed_values, width=width, label="FixedA3", color="tab:blue")
        ax.bar([i + width / 2 for i in x], rl_values, width=width, label="RLTable", color="tab:orange")
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels([s.replace("Railway-300-DL-", "") for s in scenarios], rotation=35, ha="right")
        ax.grid(axis="y", alpha=0.25)
        ax.legend(fontsize=8)

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze a Simu5G complex railway matrix.")
    parser.add_argument("--raw_root", required=True, help="包含各配置 .sca/.vec 的 raw 根目录。")
    parser.add_argument("--parsed_root", required=True, help="分析输出目录。")
    parser.add_argument("--window_s", type=float, default=2.0, help="首次切换窗口半宽，单位秒。")
    args = parser.parse_args()

    raw_root = Path(args.raw_root)
    parsed_root = Path(args.parsed_root)
    parsed_root.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, object]] = []
    for case_dir in iter_case_dirs(raw_root):
        print(f"Analyze {case_dir.name}")
        rows.append(analyze_case(case_dir, parsed_root / case_dir.name, args.window_s))

    if not rows:
        raise FileNotFoundError(f"No Simu5G result cases found under {raw_root}")

    write_rows(parsed_root / "comparison_summary.csv", rows)
    delta_rows = build_delta_rows(rows)
    write_rows(parsed_root / "comparison_delta_rl_minus_fixed.csv", delta_rows)
    with (parsed_root / "comparison_delta_rl_minus_fixed.json").open("w", encoding="utf-8") as f:
        json.dump(delta_rows, f, indent=2, ensure_ascii=False)
    plot_matrix(rows, parsed_root / "comparison_overview.png")

    print(f"Wrote {parsed_root / 'comparison_summary.csv'}")
    print(f"Wrote {parsed_root / 'comparison_delta_rl_minus_fixed.csv'}")
    print(f"Wrote {parsed_root / 'comparison_overview.png'}")


if __name__ == "__main__":
    main()
