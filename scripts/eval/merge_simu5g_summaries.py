"""Merge Simu5G per-run summary.csv files and plot sensitivity KPIs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PLOT_METRICS = [
    ("app_packet_loss_ratio", "Packet loss ratio"),
    ("measured_sinr_dl_db_p5", "SINR DL p5 / dB"),
    ("cqi_dl_mean", "CQI DL mean"),
    ("app_frame_delay_s_mean", "App delay mean / s"),
    ("app_frame_delay_s_max", "App delay max / s"),
    ("first_handover_x_m", "First HO position / m"),
]


def read_summary(path: Path) -> Dict[str, str]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return next(reader)


def to_float(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge Simu5G summary CSV files.")
    parser.add_argument("--root", default="results/simu5g", help="Root directory containing per-config outputs.")
    parser.add_argument("--out_csv", default="results/simu5g/summary_all.csv")
    parser.add_argument("--out_png", default="results/simu5g/summary_comparison.png")
    args = parser.parse_args()

    root = Path(args.root)
    rows: List[Dict[str, str]] = []
    for path in sorted(root.glob("Railway-*/summary.csv")):
        rows.append(read_summary(path))

    if not rows:
        raise FileNotFoundError(f"No summary.csv files found under {root}")

    keys: List[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)

    labels = [row.get("config", "") for row in rows]
    short_labels = [label.replace("Railway-300-DL-", "").replace("Railway-300-DL", "Base") for label in labels]

    fig, axes = plt.subplots(3, 2, figsize=(14, 11))
    axes = axes.ravel()
    for ax, (metric, title) in zip(axes, PLOT_METRICS):
        values = [to_float(row.get(metric, "")) for row in rows]
        ax.bar(short_labels, values, color="tab:blue", alpha=0.8)
        ax.set_title(title)
        ax.tick_params(axis="x", labelrotation=35)
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()

    out_png = Path(args.out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=220)
    plt.close(fig)

    print(f"Wrote {out_csv}")
    print(f"Wrote {out_png}")


if __name__ == "__main__":
    main()
