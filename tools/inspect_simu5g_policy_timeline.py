"""Inspect Simu5G policy/signal timeline around the first handover.

用法示例：
    python tools/inspect_simu5g_policy_timeline.py \
        --timeseries results/simu5g/.../key_timeseries.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


METRICS = [
    "serving_cell",
    "ue_position_x_m",
    "delta_rsrp_db",
    "serving_sinr_proxy_db",
    "measured_sinr_dl_db",
    "railway_policy_hys_db",
    "railway_policy_ttt_s",
    "railway_policy_action",
]


def _load_series(path: Path) -> dict[str, dict[float, float]]:
    series = {metric: {} for metric in METRICS}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            metric = row.get("metric", "")
            if metric not in series:
                continue
            try:
                time_s = round(float(row["time_s"]), 3)
                value = float(row["value"])
            except (KeyError, ValueError):
                continue
            series[metric][time_s] = value
    return series


def _nearest(values: dict[float, float], time_s: float, max_gap_s: float) -> float | None:
    if not values:
        return None
    nearest_t = min(values, key=lambda t: abs(t - time_s))
    if abs(nearest_t - time_s) > max_gap_s:
        return None
    return values[nearest_t]


def _first_handover_time(serving_cell: dict[float, float]) -> float | None:
    last_cell: float | None = None
    for time_s, cell in sorted(serving_cell.items()):
        if last_cell is None:
            last_cell = cell
            continue
        if time_s > 0.05 and cell != last_cell:
            return time_s
        last_cell = cell
    return None


def _fmt(value: float | None) -> str:
    return "" if value is None else f"{value:.3f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeseries", required=True, type=Path)
    parser.add_argument("--window_s", default=2.4, type=float)
    parser.add_argument("--step_s", default=0.2, type=float)
    parser.add_argument("--risk_delta_db", default=2.0, type=float)
    parser.add_argument("--risk_sinr_db", default=-5.0, type=float)
    args = parser.parse_args()

    series = _load_series(args.timeseries)
    first_ho_s = _first_handover_time(series["serving_cell"])
    center_s = first_ho_s if first_ho_s is not None else 20.0

    print(f"file,{args.timeseries}")
    print(f"first_handover_time_s,{'' if first_ho_s is None else first_ho_s:.3f}")
    print("time_s,x_m,delta_rsrp_db,serving_sinr_proxy_db,measured_sinr_dl_db,hys_db,ttt_s,action")

    steps = int(round(args.window_s / args.step_s))
    for offset in range(-steps, steps + 1):
        time_s = round(center_s + offset * args.step_s, 3)
        row = [
            time_s,
            _nearest(series["ue_position_x_m"], time_s, 0.08),
            _nearest(series["delta_rsrp_db"], time_s, 0.08),
            _nearest(series["serving_sinr_proxy_db"], time_s, 0.08),
            _nearest(series["measured_sinr_dl_db"], time_s, 0.08),
            _nearest(series["railway_policy_hys_db"], time_s, 0.08),
            _nearest(series["railway_policy_ttt_s"], time_s, 0.08),
            _nearest(series["railway_policy_action"], time_s, 0.08),
        ]
        print(",".join([f"{row[0]:.3f}"] + [_fmt(value) for value in row[1:]]))

    risk_points: list[tuple[float, float, float, float | None, float | None]] = []
    for time_s, delta in sorted(series["delta_rsrp_db"].items()):
        sinr = _nearest(series["serving_sinr_proxy_db"], time_s, 0.08)
        if sinr is None or delta < args.risk_delta_db or sinr > args.risk_sinr_db:
            continue
        hys = _nearest(series["railway_policy_hys_db"], time_s, 0.08)
        ttt = _nearest(series["railway_policy_ttt_s"], time_s, 0.08)
        risk_points.append((time_s, delta, sinr, hys, ttt))

    print(f"risk_point_count,{len(risk_points)}")
    print("risk_time_s,delta_rsrp_db,serving_sinr_proxy_db,hys_db,ttt_s")
    for time_s, delta, sinr, hys, ttt in risk_points[:20]:
        print(f"{time_s:.3f},{delta:.3f},{sinr:.3f},{_fmt(hys)},{_fmt(ttt)}")


if __name__ == "__main__":
    main()
