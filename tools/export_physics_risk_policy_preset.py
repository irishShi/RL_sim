"""Export PhysicsRisk policy tables with reproducible dense grids.

第二轮 PhysicsRisk 评估必须使用与 late guard 基线一致的密集查表网格，
否则 Simu5G 侧会混入 grid/domain gap，难以判断模型本身是否改进。
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.export.export_policy_table import main as export_policy_table_main


DEFAULT_CHECKPOINT = (
    "experiments/runs/20260606_physics_risk_obs7_seed20260606/"
    "checkpoints/rainbow_offline_best.pth"
)


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


PRESETS: dict[str, dict[str, str]] = {
    "p2_dense": {
        "risk_penalty": "2.0",
        "risk_horizon_index": "-1",
        "output_path": "results/policy_tables/policy_table_physics_risk_p2_dense_seed20260606.csv",
    },
    "p4_dense": {
        "risk_penalty": "4.0",
        "risk_horizon_index": "-1",
        "output_path": "results/policy_tables/policy_table_physics_risk_p4_dense_seed20260606.csv",
    },
}


def _build_export_argv(checkpoint_path: str, preset: dict[str, str]) -> list[str]:
    argv = [
        "export_policy_table.py",
        "--checkpoint_path",
        checkpoint_path,
        "--use_physics_risk",
    ]
    merged = {**GRID_ARGS, **preset}
    for key, value in merged.items():
        argv.extend([f"--{key}", value])
    return argv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("preset", choices=sorted(PRESETS))
    parser.add_argument("--checkpoint_path", default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--output_path",
        default=None,
        help="覆盖 preset 的输出路径。",
    )
    parser.add_argument(
        "--copy_to_default",
        action="store_true",
        help="导出后复制为 results/policy_tables/policy_table.csv。",
    )
    args = parser.parse_args()

    preset = dict(PRESETS[args.preset])
    if args.output_path:
        preset["output_path"] = args.output_path

    output_path = Path(preset["output_path"])
    sys.argv = _build_export_argv(args.checkpoint_path, preset)
    export_policy_table_main()

    if args.copy_to_default:
        default_path = Path("results/policy_tables/policy_table.csv")
        default_meta_path = Path("results/policy_tables/policy_table.metadata.json")
        shutil.copy2(output_path, default_path)
        shutil.copy2(output_path.with_suffix(".metadata.json"), default_meta_path)
        print(f"[OK] copied to {default_path}")


if __name__ == "__main__":
    main()
