# Simu5G policy_table 对比实验记录

日期：2026-06-04

## 目标

验证 `results/policy_tables/policy_table.csv` 能否被 Simu5G 读取，并在铁路 3 km 双 gNB 场景中动态控制 A3 切换参数 `Hys/TTT`，再与固定 A3 基线对比。

## Simu5G 改造点

WSL Simu5G 路径：

```text
/home/qcshi/simu5g-workspace/simu5g-1.4.4
```

改动文件：

```text
src/simu5g/stack/phy/LtePhyUe.h
src/simu5g/stack/phy/LtePhyUe.cc
src/simu5g/stack/phy/LtePhyUe.ned
simulations/nr/railway_handover/omnetpp.ini
```

新增能力：

- `railwayPolicyTableEnabled`
- `railwayPolicyTablePath`
- `railwayPolicyUpdateInterval`
- `railwayPolicyNoise`
- 仿真启动时加载 `policy_table.csv`
- UE 移动/观测更新时按最近邻查表选择 `Hys/TTT`
- 记录 `railwayPolicyAction / railwayPolicyHys / railwayPolicyTtt / railwayPolicyQ`
- 记录 `servingSinr`，当前为基于服务 RSRP、邻区 RSRP、噪声的查表用 SINR proxy

## 运行命令

编译：

```bash
cd ~/simu5g-workspace
opp_env run simu5g-1.4.4 -c 'cd /home/qcshi/simu5g-workspace/simu5g-1.4.4 && make MODE=release -j2'
```

RL 表策略：

```bash
cd ~/simu5g-workspace
opp_env run simu5g-1.4.4 -c 'cd /home/qcshi/simu5g-workspace/simu5g-1.4.4/simulations/nr/railway_handover && ./run -u Cmdenv -c Railway-300-DL-RLTable'
```

固定 A3 基线：

```bash
cd ~/simu5g-workspace
opp_env run simu5g-1.4.4 -c 'cd /home/qcshi/simu5g-workspace/simu5g-1.4.4/simulations/nr/railway_handover && ./run -u Cmdenv -c Railway-300-DL-FixedA3'
```

## 结果位置

Simu5G 原始结果：

```text
/home/qcshi/simu5g-workspace/simu5g-1.4.4/simulations/nr/railway_handover/results/RailwayPolicyCompare
```

项目内复制结果：

```text
results/simu5g/raw/RailwayPolicyCompare
```

分析结果：

```text
results/simu5g/RailwayPolicyCompare/comparison_summary.csv
results/simu5g/RailwayPolicyCompare/comparison_delta.json
results/simu5g/RailwayPolicyCompare/rl_policy_action_distribution.csv
results/simu5g/RailwayPolicyCompare/FixedA3/summary.csv
results/simu5g/RailwayPolicyCompare/RLTable/summary.csv
```

## 本次关键结果

| 指标 | FixedA3 | RLTable | RL - Fixed |
| --- | ---: | ---: | ---: |
| 切换次数 | 1 | 1 | 0 |
| 首次切换位置 | 1767.49 m | 1808.33 m | +40.83 m |
| 应用层丢包率 | 0 | 0 | 0 |
| CQI 均值 | 12.57 | 12.33 | -0.24 |
| DL SINR p5 | 8.90 dB | 8.79 dB | -0.11 dB |
| 应用层平均时延 | 6.394 ms | 6.321 ms | -0.073 ms |
| 应用层最大时延 | 93 ms | 67 ms | -26 ms |
| RLC DL 平均时延 | 6.393 ms | 6.320 ms | -0.073 ms |

## 初步判断

这次实验已经验证了 CSV 表可以被 Simu5G 成功读取并实际控制 A3 参数。RLTable 不是固定参数策略，动作分布中最常见的是：

```text
action 27: Hys=3.5 dB, TTT=150 ms
action 47: Hys=5.0 dB, TTT=650 ms
action 38: Hys=4.5 dB, TTT=100 ms
```

从 KPI 看，当前 RL 表策略相对固定 A3 更偏晚切换和保守 TTT：它没有减少切换次数，但把最大时延峰值从 93 ms 降到 67 ms；代价是 CQI/SINR 均值略低。这个结果符合当前 `policy_table.csv` 来自 `repeat_current` 近似蒸馏、且离线策略整体偏保守的预期。

下一阶段建议用 Simu5G 真实 trace 生成 15 步历史窗口，再重新导出 `trace_window` 版 policy table，以减少 GRU 历史信息丢失造成的策略偏差。
