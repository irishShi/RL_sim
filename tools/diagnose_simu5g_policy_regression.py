"""Diagnose Simu5G policy regressions on selected railway scenarios.

该工具用于对比两个已解析的 Simu5G complex matrix 结果目录，重点查看
Stress / HighInterference 中切换窗口、无线质量、业务时延和策略动作差异。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List


DEFAULT_SCENARIOS = [
    "Railway-300-DL-Stress",
    "Railway-300-DL-HighInterference",
]

SUMMARY_METRICS = [
    "handover_count",
    "first_handover_time_s",
    "first_handover_x_m",
    "app_packet_loss_ratio",
    "app_frame_delay_mean_s",
    "app_frame_delay_s_max",
    "rlc_delay_dl_mean_s",
    "mac_delay_dl_mean_s",
    "harq_error_rate_dl",
    "cqi_dl_mean",
    "cqi_dl_p5",
    "measured_sinr_dl_mean_db",
    "measured_sinr_dl_db_p5",
    "serving_sinr_proxy_mean_db",
    "railway_policy_hys_mean_db",
    "railway_policy_ttt_mean_s",
]

WINDOW_METRICS = [
    "delta_rsrp_db",
    "serving_sinr_proxy_db",
    "measured_sinr_dl_db",
    "received_sinr_dl_db",
    "cqi_dl",
    "app_frame_delay_s",
    "rlc_delay_dl_s",
    "mac_delay_dl_s",
    "railway_policy_hys_db",
    "railway_policy_ttt_s",
    "railway_policy_action",
]


def _to_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _read_single_row_csv(path: Path) -> Dict[str, str]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return next(reader)


def _read_timeseries(path: Path) -> Dict[str, List[tuple[float, float]]]:
    series: Dict[str, List[tuple[float, float]]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                metric = str(row["metric"])
                time_s = float(row["time_s"])
                value = float(row["value"])
            except (KeyError, ValueError):
                continue
            series.setdefault(metric, []).append((time_s, value))
    for values in series.values():
        values.sort(key=lambda item: item[0])
    return series


def _percentile(values: List[float], q: float) -> float:
    if not values:
        return math.nan
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return ordered[lo]
    frac = pos - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def _stats(values: List[float]) -> Dict[str, float]:
    if not values:
        return {
            "count": 0,
            "mean": math.nan,
            "min": math.nan,
            "p5": math.nan,
            "p50": math.nan,
            "p95": math.nan,
            "max": math.nan,
        }
    return {
        "count": len(values),
        "mean": sum(values) / len(values),
        "min": min(values),
        "p5": _percentile(values, 0.05),
        "p50": _percentile(values, 0.50),
        "p95": _percentile(values, 0.95),
        "max": max(values),
    }


def _window_values(
    series: Dict[str, List[tuple[float, float]]],
    metric: str,
    start_s: float,
    end_s: float,
) -> List[float]:
    return [value for time_s, value in series.get(metric, []) if start_s <= time_s <= end_s]


def _nearest(values: List[tuple[float, float]], time_s: float, max_gap_s: float = 0.08) -> float | None:
    if not values:
        return None
    nearest_time, nearest_value = min(values, key=lambda item: abs(item[0] - time_s))
    if abs(nearest_time - time_s) > max_gap_s:
        return None
    return nearest_value


def _first_handover_from_series(series: Dict[str, List[tuple[float, float]]]) -> float | None:
    last_cell: float | None = None
    for time_s, cell in series.get("serving_cell", []):
        if last_cell is None:
            last_cell = cell
            continue
        if time_s > 0.05 and cell != last_cell:
            return time_s
        last_cell = cell
    return None


def _action_counts(series: Dict[str, List[tuple[float, float]]], top_k: int = 8) -> Dict[str, object]:
    actions = [int(round(value)) for _, value in series.get("railway_policy_action", [])]
    counts = Counter(actions)
    total = sum(counts.values())
    top = [
        {"action": action, "count": count, "ratio": count / total if total else 0.0}
        for action, count in counts.most_common(top_k)
    ]
    return {"total": total, "top_actions": top}


def _risk_points(
    series: Dict[str, List[tuple[float, float]]],
    *,
    sinr_db: float,
    delta_db: float,
) -> Dict[str, object]:
    points: List[Dict[str, float]] = []
    sinr_values = series.get("serving_sinr_proxy_db", [])
    hys_values = series.get("railway_policy_hys_db", [])
    ttt_values = series.get("railway_policy_ttt_s", [])
    action_values = series.get("railway_policy_action", [])

    for time_s, delta in series.get("delta_rsrp_db", []):
        sinr = _nearest(sinr_values, time_s)
        if sinr is None or delta < delta_db or sinr > sinr_db:
            continue
        points.append(
            {
                "time_s": time_s,
                "delta_rsrp_db": delta,
                "serving_sinr_proxy_db": sinr,
                "hys_db": _nearest(hys_values, time_s) or math.nan,
                "ttt_s": _nearest(ttt_values, time_s) or math.nan,
                "action": _nearest(action_values, time_s) or math.nan,
            }
        )

    hys = [p["hys_db"] for p in points if math.isfinite(p["hys_db"])]
    ttt = [p["ttt_s"] for p in points if math.isfinite(p["ttt_s"])]
    return {
        "count": len(points),
        "hys_mean_db": sum(hys) / len(hys) if hys else math.nan,
        "ttt_mean_s": sum(ttt) / len(ttt) if ttt else math.nan,
        "first_points": points[:20],
    }


def _write_rows(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _case_dir(parsed_root: Path, scenario: str, strategy: str) -> Path:
    return parsed_root / f"{scenario}-{strategy}"


def _iter_policy_specs(args: argparse.Namespace) -> Iterable[tuple[str, Path]]:
    yield args.candidate_label, args.candidate_parsed_root
    yield args.baseline_label, args.baseline_parsed_root


def diagnose(args: argparse.Namespace) -> Dict[str, object]:
    summary_rows: List[Dict[str, object]] = []
    window_rows: List[Dict[str, object]] = []
    risk_rows: List[Dict[str, object]] = []
    action_payload: Dict[str, object] = {}

    for label, parsed_root in _iter_policy_specs(args):
        for scenario in args.scenarios:
            case = _case_dir(parsed_root, scenario, args.strategy)
            summary_path = case / "summary.csv"
            timeseries_path = case / "key_timeseries.csv"
            if not summary_path.exists() or not timeseries_path.exists():
                raise FileNotFoundError(f"缺少 Simu5G parsed case: {case}")

            summary = _read_single_row_csv(summary_path)
            series = _read_timeseries(timeseries_path)
            first_ho_s = _to_float(summary.get("first_handover_time_s"))
            if not math.isfinite(first_ho_s):
                first_ho_s = _first_handover_from_series(series) or math.nan

            summary_row: Dict[str, object] = {"policy": label, "scenario": scenario}
            for metric in SUMMARY_METRICS:
                summary_row[metric] = _to_float(summary.get(metric))
            summary_rows.append(summary_row)

            windows = {
                "pre_2s": (first_ho_s - 2.0, first_ho_s),
                "post_2s": (first_ho_s, first_ho_s + 2.0),
                "around_4s": (first_ho_s - 2.0, first_ho_s + 2.0),
                "full": (-math.inf, math.inf),
            }
            for window_name, (start_s, end_s) in windows.items():
                for metric in WINDOW_METRICS:
                    stats = _stats(_window_values(series, metric, start_s, end_s))
                    window_rows.append(
                        {
                            "policy": label,
                            "scenario": scenario,
                            "window": window_name,
                            "metric": metric,
                            **stats,
                        }
                    )

            risk = _risk_points(
                series,
                sinr_db=float(args.risk_sinr_db),
                delta_db=float(args.risk_delta_db),
            )
            risk_rows.append(
                {
                    "policy": label,
                    "scenario": scenario,
                    "risk_sinr_db": float(args.risk_sinr_db),
                    "risk_delta_db": float(args.risk_delta_db),
                    "risk_point_count": risk["count"],
                    "risk_hys_mean_db": risk["hys_mean_db"],
                    "risk_ttt_mean_s": risk["ttt_mean_s"],
                }
            )
            action_payload[f"{label}:{scenario}"] = _action_counts(series)
            action_payload[f"{label}:{scenario}:risk_points"] = risk["first_points"]

    delta_rows: List[Dict[str, object]] = []
    by_key = {(row["policy"], row["scenario"]): row for row in summary_rows}
    for scenario in args.scenarios:
        cand = by_key[(args.candidate_label, scenario)]
        base = by_key[(args.baseline_label, scenario)]
        row: Dict[str, object] = {"scenario": scenario}
        for metric in SUMMARY_METRICS:
            c = _to_float(cand.get(metric))
            b = _to_float(base.get(metric))
            row[f"{metric}_{args.candidate_label}"] = c
            row[f"{metric}_{args.baseline_label}"] = b
            row[f"{metric}_delta_candidate_minus_baseline"] = c - b
        delta_rows.append(row)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_rows(args.output_dir / "summary_metrics.csv", summary_rows)
    _write_rows(args.output_dir / "summary_delta_candidate_minus_baseline.csv", delta_rows)
    _write_rows(args.output_dir / "window_metrics.csv", window_rows)
    _write_rows(args.output_dir / "risk_points.csv", risk_rows)
    with (args.output_dir / "action_distributions.json").open("w", encoding="utf-8") as handle:
        json.dump(action_payload, handle, indent=2, ensure_ascii=False)

    return {
        "summary_metrics": str(args.output_dir / "summary_metrics.csv"),
        "summary_delta": str(args.output_dir / "summary_delta_candidate_minus_baseline.csv"),
        "window_metrics": str(args.output_dir / "window_metrics.csv"),
        "risk_points": str(args.output_dir / "risk_points.csv"),
        "action_distributions": str(args.output_dir / "action_distributions.json"),
        "delta_rows": delta_rows,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate_parsed_root", required=True, type=Path)
    parser.add_argument("--baseline_parsed_root", required=True, type=Path)
    parser.add_argument("--candidate_label", default="candidate")
    parser.add_argument("--baseline_label", default="baseline")
    parser.add_argument("--strategy", default="RLTable")
    parser.add_argument("--scenarios", nargs="*", default=DEFAULT_SCENARIOS)
    parser.add_argument("--risk_sinr_db", type=float, default=-5.0)
    parser.add_argument("--risk_delta_db", type=float, default=2.0)
    parser.add_argument("--output_dir", required=True, type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = diagnose(args)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
