import os
import sys
import numpy as np
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from envs.train_ho_env import TrainHandoverEnv
from envs.channel_model import ChannelModel

# 配置 matplotlib 支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']  # 使用黑体或微软雅黑
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


def main():
    # 为了和 run_env_test 中的场景保持一致，这里也从 YAML 读取配置
    base_dir = PROJECT_ROOT
    cfg_path = os.path.join(base_dir, "configs", "default_env_config.yaml")
    env = TrainHandoverEnv(config_path=cfg_path)
    cfg = env.cfg  # 使用环境最终生效的配置

    # 关掉快衰落，只看大尺度+阴影，避免曲线太“毛”
    cfg["enable_fast_fading"] = False

    channel = ChannelModel(cfg)
    channel.set_rng(np.random.default_rng(0))

    D = cfg["track_length_m"]
    xs = np.linspace(0.0, D, 600)
    rsrp_A, rsrp_B = [], []

    for x in xs:
        rA, rB, _, _ = channel.compute_link_metrics(x)
        rsrp_A.append(rA)
        rsrp_B.append(rB)

    rsrp_A = np.array(rsrp_A)
    rsrp_B = np.array(rsrp_B)

    plt.figure(figsize=(8, 5))
    plt.plot(xs, rsrp_A, label="RSRP A小区", color="tab:blue")
    plt.plot(xs, rsrp_B, label="RSRP B小区", color="tab:orange")
    plt.xlabel("距离 x / m")
    plt.ylabel("RSRP / dBm")
    plt.title("A/B 基站沿轨道的 RSRP 曲线（与环境配置一致）")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
