# Simu5G 复杂场景矩阵首轮结果

本文档记录 `2026-06-04` 执行的 Simu5G 复杂铁路切换矩阵实验。该实验基于 3 km 双 gNB 铁路切换场景，比较固定 A3 参数与 RL `policy_table.csv` 查表策略在多种压力条件下的表现。

## 1. 实验配置

- 仿真工程：`/home/qcshi/simu5g-workspace/simu5g-1.4.4/simulations/nr/railway_handover`
- 本项目结果目录：`results/simu5g/RailwayComplexMatrix-20260604-123813`
- 策略表：`results/policy_tables/policy_table.csv`
- 场景长度：3 km
- 速度：300 km/h
- 基站布局：两端 gNB，站间距约 3 km
- 业务：下行 CBR
- 对比策略：
  - `FixedA3`：`Hys=3 dB`，`TTT=160 ms`
  - `RLTable`：每 50 ms 查表更新 `Hys/TTT`

本轮共运行 14 个配置，即 7 个场景乘以 2 种策略。

## 2. 场景矩阵

| 场景 | 含义 |
| --- | --- |
| `TrackOffset` | 基础铁路偏置场景，gNB 横向离轨道约 150 m |
| `WeakCoverage` | 降低发射功率、站高和天线增益 |
| `HighNoise` | 提高接收噪声与 UE 噪声系数 |
| `NlosFading` | 开启 shadowing、NLOS 和 Rayleigh fading |
| `HighInterference` | 开启上下行、背景和外部干扰，并降低资源块 |
| `HeavyTraffic` | 提高 CBR 发包频率和包大小 |
| `Stress` | 叠加 NLOS、弱覆盖、高噪声、低资源和重业务 |

其中 `Stress` 就是之前提到的综合压力场景。它不是一个新名字，而是已经写入 `omnetpp.ini` 的配置组，本轮把它纳入了 FixedA3 vs RLTable 的正式对比矩阵。

## 3. 关键结果

| 场景 | HO 固定/RL | 首次切换位置 固定/RL (m) | 丢包率 固定/RL | SINR p5 固定/RL (dB) | CQI 均值 固定/RL | 最大 App 时延 固定/RL (s) | RL 平均 Hys/TTT |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `HeavyTraffic` | 1 / 1 | 1600.8 / 1641.7 | 0.0007 / 0.0008 | 9.03 / 9.38 | 12.52 / 12.52 | 0.093 / 0.083 | 3.56 dB / 280 ms |
| `HighInterference` | 1 / 1 | 1684.2 / 1891.7 | 0 / 0 | 12.10 / 11.88 | 12.91 / 12.94 | 0.087 / 0.067 | 3.62 dB / 332 ms |
| `HighNoise` | 1 / 1 | 1767.5 / 1808.3 | 0.0056 / 0.0056 | -1.74 / -1.61 | 9.75 / 9.78 | 0.311 / 0.311 | 3.80 dB / 307 ms |
| `NlosFading` | 1 / 1 | 1600.8 / 1641.7 | 0 / 0 | 7.54 / 7.75 | 12.26 / 12.61 | 0.087 / 0.067 | 3.56 dB / 312 ms |
| `Stress` | 1 / 1 | 1684.2 / 1808.3 | 0.3007 / 0.3360 | -16.60 / -16.76 | 5.86 / 6.19 | 1.678 / 2.719 | 3.94 dB / 418 ms |
| `TrackOffset` | 1 / 1 | 1767.5 / 1808.3 | 0 / 0 | 8.90 / 8.79 | 12.57 / 12.33 | 0.093 / 0.067 | 3.88 dB / 334 ms |
| `WeakCoverage` | 1 / 1 | 1767.5 / 1808.3 | 0.0134 / 0.0117 | -3.67 / -3.04 | 9.52 / 9.43 | 0.311 / 0.305 | 3.65 dB / 400 ms |

## 4. 初步判断

本轮所有场景下 `FixedA3` 和 `RLTable` 都只发生 1 次切换。因此当前 RL 查表策略的主要影响不是减少切换次数，而是改变切换位置和切换触发参数。

比较明显的现象：

- 在 `NlosFading`、`HighNoise`、`WeakCoverage` 中，RLTable 对 SINR p5 或业务层指标有小幅改善或接近持平。
- 在 `HighInterference` 中，RLTable 把切换点明显后移，丢包仍为 0，最大 App 时延更低，但 SINR p5 略低。
- 在 `Stress` 中，RLTable 的 CQI 均值略高，但丢包率和最大 App 时延更差。这说明当前离线策略表在综合压力分布下还不稳定，不能直接声称优于固定 A3。
- RLTable 的平均 TTT 多数落在 280 ms 到 418 ms，整体比固定基线 160 ms 更保守；这解释了为什么 RL 往往把首次切换位置后移。

## 5. 输出文件

- 原始 `.sca/.vec/.vci`：`results/simu5g/RailwayComplexMatrix-20260604-123813/raw`
- 单场景解析结果：`results/simu5g/RailwayComplexMatrix-20260604-123813/parsed/<config>/`
- 总览表：`results/simu5g/RailwayComplexMatrix-20260604-123813/parsed/comparison_summary.csv`
- RL 相对固定 A3 的差值表：`results/simu5g/RailwayComplexMatrix-20260604-123813/parsed/comparison_delta_rl_minus_fixed.csv`
- 总览图：`results/simu5g/RailwayComplexMatrix-20260604-123813/parsed/comparison_overview.png`

每个配置目录中还包含：

- `summary.csv`
- `summary.json`
- `key_timeseries.csv`
- `timeseries_full.png`
- `handover_window.png`

## 6. 下一步建议

本轮结果适合作为“RL 策略接入 Simu5G 后的首轮复杂场景迁移验证”。如果要支撑论文结论，建议继续做三步：

1. 多随机种子重复：每个场景至少 20 到 50 个 seed，统计均值、方差和置信区间。
2. 重新生成 stress-aware policy table：在 Python 环境中加入弱覆盖、高噪声、NLOS、干扰、重业务对应的观测分布，再导出策略表。
3. 扩展速度矩阵：优先跑 `TrackOffset/NlosFading/Stress x 200/300/350/400 km/h`，观察 RL 是否只是对 300 km/h 单点过拟合。

当前最重要的结论是：`Stress` 场景确实已经存在，并且首轮结果显示它是最能暴露 RL 策略迁移问题的场景，后续应把它作为鲁棒性评估的核心场景之一。
