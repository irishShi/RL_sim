from __future__ import annotations

import csv
from pathlib import Path


def to_float(row: dict[str, str], key: str, scale: float = 1.0) -> float:
    try:
        return float(row.get(key, "nan")) * scale
    except ValueError:
        return float("nan")


def main() -> None:
    path = Path("results/simu5g/CodexA3Sweep-20260603-rlbridge/summary_all.csv")
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    print("case,ho,x_m,loss_pct,sinr_p5,cqi_mean,delay_max_ms,delta_max")
    for row in sorted(rows, key=lambda item: item["config"]):
        print(
            f"{row['config']},"
            f"{to_float(row, 'handover_count'):.0f},"
            f"{to_float(row, 'first_handover_x_m'):.1f},"
            f"{to_float(row, 'app_packet_loss_ratio', 100):.2f},"
            f"{to_float(row, 'measured_sinr_dl_db_p5'):.2f},"
            f"{to_float(row, 'cqi_dl_mean'):.2f},"
            f"{to_float(row, 'app_frame_delay_s_max', 1000):.1f},"
            f"{to_float(row, 'delta_rsrp_db_max'):.2f}"
        )


if __name__ == "__main__":
    main()
