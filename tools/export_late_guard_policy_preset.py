"""Export late-handover guarded policy tables with reproducible presets.

这个脚本用于绕开 Windows venv launcher 在超长命令行下偶发的启动问题，
同时把重要的 guard 调参方案固化成可复现实验预设。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.export.export_policy_table import main as export_policy_table_main


CHECKPOINT = (
    "experiments/runs/20260604_v1_domain_random_obs7_nocql_seed20260604/"
    "checkpoints/rainbow_offline_best.pth"
)


PRESETS: dict[str, dict[str, str]] = {
    "delta3_low5": {
        "output_path": "results/policy_tables/"
        "policy_table_v1_domain_random_obs7_late_guard_delta3_low5_seed20260604.csv",
        "guard_min_delta_db": "3",
        "guard_low_sinr_db": "-5",
        "guard_critical_sinr_db": "-7",
        "guard_strong_delta_db": "8",
        "guard_fast_sinr_db": "-3",
        "guard_moderate_sinr_db": "0",
        "guard_min_time_since_ho_s": "1.0",
        "guard_low_max_ttt_ms": "150",
        "guard_critical_max_ttt_ms": "150",
        "guard_fast_max_ttt_ms": "150",
        "guard_hys_cap_db": "3.0",
        "guard_critical_hys_cap_db": "2.5",
        "guard_delta3_hys_cap_db": "2.5",
    },
    "delta4_low6": {
        "output_path": "results/policy_tables/"
        "policy_table_v1_domain_random_obs7_late_guard_tuned_seed20260604.csv",
        "guard_min_delta_db": "4",
        "guard_low_sinr_db": "-6",
        "guard_critical_sinr_db": "-8",
        "guard_strong_delta_db": "8",
        "guard_fast_sinr_db": "-4",
        "guard_moderate_sinr_db": "-1",
        "guard_min_time_since_ho_s": "1.0",
        "guard_low_max_ttt_ms": "150",
        "guard_critical_max_ttt_ms": "150",
        "guard_fast_max_ttt_ms": "150",
        "guard_hys_cap_db": "3.0",
        "guard_critical_hys_cap_db": "2.5",
        "guard_delta3_hys_cap_db": "2.5",
    },
}


GRID_ARGS = {
    "device": "cpu",
    "positions_m": "0:3000:50",
    "speed_bins_kmh": "200,300,350,400",
    "rsrp_serv_bins_dbm": "-120,-110,-100,-90,-80,-70,-60",
    "delta_rsrp_bins_db": "-15,-10,-6,-3,0,3,6,10,15",
    "sinr_bins_db": "-10,-6,-3,0,3,6,10,15,20",
    "time_since_ho_bins_s": "0,0.5,1,2,5,10",
    "top_k": "3",
}


def _build_export_argv(preset: dict[str, str]) -> list[str]:
    argv = [
        "export_policy_table.py",
        "--checkpoint_path",
        CHECKPOINT,
        "--late_ho_guard",
    ]
    merged = {**GRID_ARGS, **preset}
    for key, value in merged.items():
        argv.extend([f"--{key}", value])
    return argv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("preset", choices=sorted(PRESETS))
    parser.add_argument(
        "--copy_to_default",
        action="store_true",
        help="导出后复制为 results/policy_tables/policy_table.csv。",
    )
    args = parser.parse_args()

    output_path = Path(PRESETS[args.preset]["output_path"])
    sys.argv = _build_export_argv(PRESETS[args.preset])
    export_policy_table_main()

    if args.copy_to_default:
        import shutil

        default_path = Path("results/policy_tables/policy_table.csv")
        default_meta_path = Path("results/policy_tables/policy_table.metadata.json")
        shutil.copy2(output_path, default_path)
        shutil.copy2(output_path.with_suffix(".metadata.json"), default_meta_path)
        print(f"[OK] copied to {default_path}")


if __name__ == "__main__":
    main()
