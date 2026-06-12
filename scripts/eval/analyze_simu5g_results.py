"""Analyze Simu5G railway handover result files.

This script reads OMNeT++/Simu5G .sca and .vec files, extracts the KPIs that
matter for railway handover experiments, and writes summary tables plus plots.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


Series = List[Tuple[float, float]]


METRIC_PATTERNS = {
    "serving_cell": ("ue[0].cellularNic.nrPhy", "servingCell"),
    "best_neighbor_cell": ("ue[0].cellularNic.nrPhy", "bestNeighborCell"),
    "distance_m": ("ue[0].cellularNic.nrPhy", "distance"),
    "serving_rsrp_db": ("ue[0].cellularNic.nrPhy", "servingRsrp"),
    "neighbor_rsrp_db": ("ue[0].cellularNic.nrPhy", "neighborRsrp"),
    "delta_rsrp_db": ("ue[0].cellularNic.nrPhy", "deltaRsrp"),
    "serving_sinr_proxy_db": ("ue[0].cellularNic.nrPhy", "servingSinr"),
    "ue_position_x_m": ("ue[0].cellularNic.nrPhy", "uePositionX"),
    "ue_speed_mps": ("ue[0].cellularNic.nrPhy", "ueSpeed"),
    "time_since_last_handover_s": ("ue[0].cellularNic.nrPhy", "timeSinceLastHandover"),
    "railway_policy_action": ("ue[0].cellularNic.nrPhy", "railwayPolicyAction"),
    "railway_policy_hys_db": ("ue[0].cellularNic.nrPhy", "railwayPolicyHys"),
    "railway_policy_ttt_s": ("ue[0].cellularNic.nrPhy", "railwayPolicyTtt"),
    "railway_policy_q": ("ue[0].cellularNic.nrPhy", "railwayPolicyQ"),
    "cqi_dl": ("ue[0].cellularNic.nrPhy", "averageCqiDl"),
    "measured_sinr_dl_db": ("ue[0].cellularNic.nrChannelModel[0]", "measuredSinrDl"),
    "received_sinr_dl_db": ("ue[0].cellularNic.nrChannelModel[0]", "rcvdSinrDl"),
    "rlc_delay_dl_s": ("ue[0].cellularNic.nrRlc.um", "rlcDelayDl"),
    "rlc_throughput_dl_bps": ("ue[0].cellularNic.nrRlc.um", "rlcThroughputDl"),
    "mac_delay_dl_s": ("ue[0].cellularNic.nrMac", "macDelayDl"),
    "app_frame_delay_s": ("ue[0].app[0]", "cbrFrameDelay"),
    "app_received_bytes": ("ue[0].app[0]", "cbrReceivedBytes"),
    "app_received_pkt_index": ("ue[0].app[0]", "cbrRcvdPkt"),
}


SCALAR_PATTERNS = {
    "server_udp_sent_count": ("MultiCell_Standalone.server.udp", "packetSent:count"),
    "ue_udp_received_count": ("MultiCell_Standalone.ue[0].udp", "packetReceived:count"),
    "app_received_bytes_sum": ("MultiCell_Standalone.ue[0].app[0]", "cbrReceivedBytes:sum"),
    "app_received_throughput_bps": (
        "MultiCell_Standalone.ue[0].app[0]",
        "cbrReceivedThroughput:simu5g_rateavg",
    ),
    "app_frame_delay_mean_s": ("MultiCell_Standalone.ue[0].app[0]", "cbrFrameDelay:mean"),
    "rlc_delay_dl_mean_s": ("MultiCell_Standalone.ue[0].cellularNic.nrRlc.um", "rlcDelayDl:mean"),
    "rlc_throughput_dl_bps": (
        "MultiCell_Standalone.ue[0].cellularNic.nrRlc.um",
        "rlcThroughputDl:simu5g_rateavg",
    ),
    "mac_delay_dl_mean_s": ("MultiCell_Standalone.ue[0].cellularNic.nrMac", "macDelayDl:mean"),
    "harq_error_rate_dl": ("MultiCell_Standalone.ue[0].cellularNic.nrMac", "harqErrorRateDl:mean"),
    "cqi_dl_mean": ("MultiCell_Standalone.ue[0].cellularNic.nrPhy", "averageCqiDl:mean"),
    "serving_sinr_proxy_mean_db": ("MultiCell_Standalone.ue[0].cellularNic.nrPhy", "servingSinr:mean"),
    "railway_policy_action_mean": (
        "MultiCell_Standalone.ue[0].cellularNic.nrPhy",
        "railwayPolicyAction:mean",
    ),
    "railway_policy_hys_mean_db": (
        "MultiCell_Standalone.ue[0].cellularNic.nrPhy",
        "railwayPolicyHys:mean",
    ),
    "railway_policy_ttt_mean_s": (
        "MultiCell_Standalone.ue[0].cellularNic.nrPhy",
        "railwayPolicyTtt:mean",
    ),
    "railway_policy_q_mean": (
        "MultiCell_Standalone.ue[0].cellularNic.nrPhy",
        "railwayPolicyQ:mean",
    ),
    "measured_sinr_dl_mean_db": (
        "MultiCell_Standalone.ue[0].cellularNic.nrChannelModel[0]",
        "measuredSinrDl:mean",
    ),
    "received_sinr_dl_mean_db": (
        "MultiCell_Standalone.ue[0].cellularNic.nrChannelModel[0]",
        "rcvdSinrDl:mean",
    ),
}


def _to_float(text: str) -> float:
    try:
        return float(text)
    except ValueError:
        return math.nan


def _clean_signal(signal: str) -> str:
    return signal.split(":", 1)[0]


def parse_sca(path: Path) -> Tuple[Dict[str, str], Dict[str, float], Dict[str, str]]:
    configs: Dict[str, str] = {}
    scalars: Dict[str, float] = {}
    scalar_raw: Dict[str, str] = {}

    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("config "):
                parts = line.split(maxsplit=2)
                if len(parts) == 3:
                    configs[parts[1]] = parts[2].strip('"')
                continue
            if not line.startswith("scalar "):
                continue
            parts = line.split(maxsplit=3)
            if len(parts) != 4:
                continue
            _, module, stat, value = parts
            key = f"{module} {stat}"
            scalar_raw[key] = value
            scalars[key] = _to_float(value)
    return configs, scalars, scalar_raw


def parse_vec(path: Path) -> Tuple[Dict[str, Series], Dict[int, Tuple[str, str]]]:
    vector_defs: Dict[int, Tuple[str, str]] = {}
    selected_ids: Dict[int, str] = {}
    data: Dict[str, Series] = {name: [] for name in METRIC_PATTERNS}

    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith("vector "):
                parts = line.split()
                if len(parts) >= 4:
                    vec_id = int(parts[1])
                    module = parts[2]
                    signal = _clean_signal(parts[3])
                    vector_defs[vec_id] = (module, signal)
                    for metric, (module_part, signal_name) in METRIC_PATTERNS.items():
                        if module_part in module and signal == signal_name:
                            selected_ids[vec_id] = metric
                continue

            parts = line.split()
            if len(parts) < 4 or not parts[0].isdigit():
                continue
            vec_id = int(parts[0])
            metric = selected_ids.get(vec_id)
            if metric is None:
                continue
            time_s = _to_float(parts[2])
            value = _to_float(parts[3])
            if not math.isnan(time_s) and not math.isnan(value):
                data[metric].append((time_s, value))
    return data, vector_defs


def series_stats(series: Series) -> Dict[str, float]:
    values = np.array([v for _, v in series if np.isfinite(v) and v > -900.0], dtype=np.float64)
    if values.size == 0:
        return {
            "count": 0,
            "min": math.nan,
            "mean": math.nan,
            "max": math.nan,
            "p5": math.nan,
            "p95": math.nan,
        }
    return {
        "count": int(values.size),
        "min": float(np.min(values)),
        "mean": float(np.mean(values)),
        "max": float(np.max(values)),
        "p5": float(np.percentile(values, 5)),
        "p95": float(np.percentile(values, 95)),
    }


def detect_handover_events(serving_cell: Series) -> List[Dict[str, float]]:
    events: List[Dict[str, float]] = []
    previous: Optional[Tuple[float, float]] = None
    for time_s, cell in serving_cell:
        if previous is not None:
            prev_t, prev_cell = previous
            if cell != prev_cell and prev_cell > 0 and cell > 0:
                events.append(
                    {
                        "from_cell": prev_cell,
                        "to_cell": cell,
                        "time_before_s": prev_t,
                        "time_s": time_s,
                    }
                )
        previous = (time_s, cell)
    return events


def parse_speed_mps(configs: Dict[str, str]) -> Optional[float]:
    raw = configs.get("*.ue[0].mobility.speed", "")
    match = re.search(r"([-+]?\d+(?:\.\d+)?)", raw)
    if not match:
        return None
    return float(match.group(1))


def lookup_scalar(scalars: Dict[str, float], module: str, stat: str) -> float:
    return scalars.get(f"{module} {stat}", math.nan)


def build_summary(
    configs: Dict[str, str],
    scalars: Dict[str, float],
    series: Dict[str, Series],
    config_name: str,
) -> Dict[str, float | str]:
    speed_mps = parse_speed_mps(configs)
    handovers = detect_handover_events(series.get("serving_cell", []))
    first_ho_time = handovers[0]["time_s"] if handovers else math.nan
    first_ho_x = first_ho_time * speed_mps if handovers and speed_mps is not None else math.nan

    sent = lookup_scalar(scalars, *SCALAR_PATTERNS["server_udp_sent_count"])
    received = lookup_scalar(scalars, *SCALAR_PATTERNS["ue_udp_received_count"])
    loss_ratio = (sent - received) / sent if sent and np.isfinite(sent) else math.nan

    summary: Dict[str, float | str] = {
        "config": config_name,
        "sim_time_limit": configs.get("sim-time-limit", ""),
        "speed_mps": speed_mps if speed_mps is not None else math.nan,
        "speed_kmh": speed_mps * 3.6 if speed_mps is not None else math.nan,
        "handover_count": len(handovers),
        "first_handover_time_s": first_ho_time,
        "first_handover_x_m": first_ho_x,
        "server_udp_sent_count": sent,
        "ue_udp_received_count": received,
        "app_packet_loss_ratio": loss_ratio,
    }

    for metric, (module, stat) in SCALAR_PATTERNS.items():
        summary[metric] = lookup_scalar(scalars, module, stat)

    for metric in [
        "serving_rsrp_db",
        "neighbor_rsrp_db",
        "delta_rsrp_db",
        "serving_sinr_proxy_db",
        "ue_position_x_m",
        "ue_speed_mps",
        "time_since_last_handover_s",
        "railway_policy_action",
        "railway_policy_hys_db",
        "railway_policy_ttt_s",
        "railway_policy_q",
        "measured_sinr_dl_db",
        "received_sinr_dl_db",
        "cqi_dl",
        "rlc_delay_dl_s",
        "rlc_throughput_dl_bps",
        "mac_delay_dl_s",
        "app_frame_delay_s",
    ]:
        stats = series_stats(series.get(metric, []))
        for stat_name, value in stats.items():
            summary[f"{metric}_{stat_name}"] = value

    return summary


def write_summary_csv(path: Path, rows: List[Dict[str, float | str]]) -> None:
    if not rows:
        return
    keys: List[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def write_timeseries_csv(path: Path, series: Dict[str, Series]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "time_s", "value"])
        for metric, values in series.items():
            for time_s, value in values:
                writer.writerow([metric, f"{time_s:.9g}", f"{value:.9g}"])


def _series_to_arrays(series: Series) -> Tuple[np.ndarray, np.ndarray]:
    if not series:
        return np.array([]), np.array([])
    arr = np.array(series, dtype=np.float64)
    return arr[:, 0], arr[:, 1]


def plot_timeseries(
    output_path: Path,
    series: Dict[str, Series],
    handovers: List[Dict[str, float]],
    title: str,
    window: Optional[Tuple[float, float]] = None,
) -> None:
    fig, axes = plt.subplots(5, 1, figsize=(12, 12), sharex=True)

    def maybe_window(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if window is None or x.size == 0:
            return x, y
        mask = (x >= window[0]) & (x <= window[1])
        return x[mask], y[mask]

    x, y = maybe_window(*_series_to_arrays(series.get("serving_cell", [])))
    if x.size:
        axes[0].step(x, y, where="post", color="tab:blue")
    axes[0].set_ylabel("Serving cell")
    axes[0].grid(True, alpha=0.25)

    x, y = maybe_window(*_series_to_arrays(series.get("serving_rsrp_db", [])))
    if x.size:
        mask = y > -900.0
        axes[1].plot(x[mask], y[mask], color="tab:blue", label="Serving RSRP")
    x, y = maybe_window(*_series_to_arrays(series.get("neighbor_rsrp_db", [])))
    if x.size:
        mask = y > -900.0
        axes[1].plot(x[mask], y[mask], color="tab:orange", label="Neighbor RSRP")
    ax_delta = axes[1].twinx()
    x, y = maybe_window(*_series_to_arrays(series.get("delta_rsrp_db", [])))
    if x.size:
        mask = y > -900.0
        ax_delta.plot(x[mask], y[mask], color="tab:purple", alpha=0.55, label="Delta RSRP")
    axes[1].axhline(0, color="gray", linewidth=0.8, alpha=0.5)
    axes[1].set_ylabel("RSRP / dB")
    ax_delta.set_ylabel("Delta / dB")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(loc="lower right", fontsize=8)

    x, y = maybe_window(*_series_to_arrays(series.get("measured_sinr_dl_db", [])))
    if x.size:
        axes[2].plot(x, y, color="tab:green", label="SINR DL")
    x, y = maybe_window(*_series_to_arrays(series.get("serving_sinr_proxy_db", [])))
    if x.size:
        axes[2].plot(x, y, color="tab:blue", alpha=0.55, label="Policy SINR proxy")
    ax2 = axes[2].twinx()
    x2, y2 = maybe_window(*_series_to_arrays(series.get("cqi_dl", [])))
    if x2.size:
        ax2.plot(x2, y2, color="tab:orange", alpha=0.7, label="CQI DL")
    axes[2].set_ylabel("SINR / dB")
    ax2.set_ylabel("CQI")
    axes[2].grid(True, alpha=0.25)

    for metric, label, color in [
        ("app_frame_delay_s", "App frame delay", "tab:red"),
        ("rlc_delay_dl_s", "RLC DL delay", "tab:purple"),
        ("mac_delay_dl_s", "MAC DL delay", "tab:brown"),
    ]:
        x, y = maybe_window(*_series_to_arrays(series.get(metric, [])))
        if x.size:
            axes[3].plot(x, y * 1000.0, label=label, alpha=0.85, color=color)
    axes[3].set_ylabel("Delay / ms")
    axes[3].grid(True, alpha=0.25)
    axes[3].legend(loc="upper right", fontsize=8)

    x, y = maybe_window(*_series_to_arrays(series.get("rlc_throughput_dl_bps", [])))
    if x.size:
        axes[4].plot(x, y / 1000.0, color="tab:cyan")
    axes[4].set_ylabel("RLC DL / kbps")
    axes[4].set_xlabel("Time / s")
    axes[4].grid(True, alpha=0.25)

    for ax in axes:
        for event in handovers:
            t = event["time_s"]
            if window is None or window[0] <= t <= window[1]:
                ax.axvline(t, color="black", linestyle="--", linewidth=1.0, alpha=0.7)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def find_result_files(result_dir: Path) -> Tuple[Path, Path]:
    sca_files = sorted(result_dir.glob("*.sca"))
    vec_files = sorted(result_dir.glob("*.vec"))
    if not sca_files:
        raise FileNotFoundError(f"No .sca file found in {result_dir}")
    if not vec_files:
        raise FileNotFoundError(f"No .vec file found in {result_dir}")
    return sca_files[0], vec_files[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze Simu5G railway handover results.")
    parser.add_argument("--result_dir", required=True, help="Directory containing Simu5G .sca/.vec files.")
    parser.add_argument("--out_dir", required=True, help="Output directory for tables and figures.")
    parser.add_argument("--window_s", type=float, default=2.0, help="Half window around the first handover.")
    args = parser.parse_args()

    result_dir = Path(args.result_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sca_path, vec_path = find_result_files(result_dir)
    configs, scalars, _ = parse_sca(sca_path)
    series, _ = parse_vec(vec_path)
    config_name = configs.get("configname", result_dir.name)
    summary = build_summary(configs, scalars, series, result_dir.name)
    handovers = detect_handover_events(series.get("serving_cell", []))

    write_summary_csv(out_dir / "summary.csv", [summary])
    write_timeseries_csv(out_dir / "key_timeseries.csv", series)
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump({"summary": summary, "handovers": handovers}, f, indent=2, ensure_ascii=False)

    title = f"{result_dir.name} railway handover KPIs"
    plot_timeseries(out_dir / "timeseries_full.png", series, handovers, title)
    if handovers:
        t0 = handovers[0]["time_s"]
        plot_timeseries(
            out_dir / "handover_window.png",
            series,
            handovers,
            f"{result_dir.name} around first handover",
            window=(t0 - args.window_s, t0 + args.window_s),
        )

    print(f"Wrote {out_dir / 'summary.csv'}")
    print(f"Wrote {out_dir / 'summary.json'}")
    print(f"Wrote {out_dir / 'key_timeseries.csv'}")
    print(f"Wrote {out_dir / 'timeseries_full.png'}")
    if handovers:
        print(f"Wrote {out_dir / 'handover_window.png'}")


if __name__ == "__main__":
    main()
