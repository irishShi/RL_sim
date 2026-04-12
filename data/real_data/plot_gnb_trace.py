"""
绘制 gnb_trace_processed.csv 的信号强度随距离变化图

输出：
- data/real_data/gnb_trace_rsrp_vs_distance.png

图内容（默认）：
- serving RSRP（rsrp_dbm）
- estimated neighbor RSRP（rsrp_neig_dbm_est）
- 真实切换点（ho_true=True，PCI变化）竖线标注
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        type=str,
        default="data/real_data/gnb_trace_processed.csv",
        help="预处理后的轨迹CSV路径",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="输出图片路径（默认与CSV同目录）",
    )
    parser.add_argument(
        "--use_km",
        action="store_true",
        help="横轴使用 km（rel_km0），否则使用 m（pos_m）",
    )
    parser.add_argument(
        "--max_ho_markers",
        type=int,
        default=80,
        help="最多标注多少个切换点（避免过密）",
    )
    args = parser.parse_args()

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    csv_path = os.path.join(project_root, args.csv) if not os.path.isabs(args.csv) else args.csv
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV不存在: {csv_path}")

    df = pd.read_csv(csv_path)
    required = {"rsrp_dbm", "ho_true"}
    if not required.issubset(df.columns):
        raise ValueError(f"CSV缺少必要列: {sorted(required - set(df.columns))}")

    # 横轴
    if args.use_km:
        if "rel_km0" not in df.columns:
            raise ValueError("CSV缺少 rel_km0 列，无法用 km 作为横轴")
        x = df["rel_km0"].to_numpy(dtype=float)
        x_label = "相对里程 (km)"
    else:
        if "pos_m" not in df.columns:
            raise ValueError("CSV缺少 pos_m 列，无法用 m 作为横轴")
        x = df["pos_m"].to_numpy(dtype=float)
        x_label = "相对距离 (m)"

    rsrp_serv = df["rsrp_dbm"].to_numpy(dtype=float)
    rsrp_neig = df["rsrp_neig_dbm_est"].to_numpy(dtype=float) if "rsrp_neig_dbm_est" in df.columns else None

    # 真实切换点（PCI变化）
    ho_mask = df["ho_true"].astype(bool).to_numpy()
    ho_x = x[ho_mask]

    # 画图
    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(x, rsrp_serv, label="Serving RSRP (dBm)", color="tab:blue", linewidth=1.6, alpha=0.9)

    if rsrp_neig is not None:
        ax.plot(x, rsrp_neig, label="Estimated Neighbor RSRP (dBm)", color="tab:orange", linewidth=1.4, alpha=0.8)

    # 标注切换点（竖线）；过多时做均匀抽样
    if len(ho_x) > 0:
        if len(ho_x) > args.max_ho_markers:
            idx = np.linspace(0, len(ho_x) - 1, args.max_ho_markers).astype(int)
            ho_x_plot = ho_x[idx]
        else:
            ho_x_plot = ho_x

        for xv in ho_x_plot:
            ax.axvline(x=xv, color="red", alpha=0.15, linewidth=1)

        ax.text(
            0.99,
            0.02,
            f"HO events (PCI change): {int(ho_mask.sum())}",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.7, edgecolor="gray", linewidth=0.5),
        )

    ax.set_xlabel(x_label)
    ax.set_ylabel("RSRP (dBm)")
    ax.set_title("5G-R 轨迹：信号强度随距离变化（含真实切换点）")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")

    # 输出路径
    if args.out is None:
        out_path = os.path.join(os.path.dirname(csv_path), "gnb_trace_rsrp_vs_distance.png")
    else:
        out_path = os.path.join(project_root, args.out) if not os.path.isabs(args.out) else args.out

    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    print(f"已保存图表: {out_path}")


if __name__ == "__main__":
    main()

