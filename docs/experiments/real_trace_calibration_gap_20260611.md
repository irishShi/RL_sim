# 实测轨迹与仿真 profile 差距审计及首版校准记录（2026-06-11）

## 目标

本记录用于说明 `data/raw/real/` 下实测 NR 轨迹如何服务于本文仿真可信度校准。

当前阶段不直接重训模型，也不把实测数据用于扩散生成。先完成两件事：

- 对比实测轨迹与现有 `scenario_profiles.yaml` 的统计差距。
- 新增一个不进入默认训练/测试的校准 profile：`realistic_trace_calib_v0`。

## 数据与工具

实测轻量表：

```text
data/raw/real/nr_gnb_rsrp_sinr_km.xlsx
```

对比脚本：

```text
tools/compare_real_sim_gap.py
```

脚本输出：

```text
results/real_sim_gap/baseline_20_seed52000/
results/real_sim_gap/calib_v0_tuned_20_seed52000/
```

运行命令：

```powershell
.\.venv\Scripts\python.exe tools\compare_real_sim_gap.py `
  --num_seeds 20 `
  --base_seed 52000 `
  --out_dir results/real_sim_gap/baseline_20_seed52000

.\.venv\Scripts\python.exe tools\compare_real_sim_gap.py `
  --profile_split calib `
  --num_seeds 20 `
  --base_seed 52000 `
  --out_dir results/real_sim_gap/calib_v0_tuned_20_seed52000
```

## 实测轨迹摘要

| 指标 | 数值 |
| --- | ---: |
| rows | 2560 |
| track span | 197.23 km |
| PCI 数量 | 63 |
| gNodeB 数量 | 16 |
| PCI change | 71 |
| PCI change per km | 0.360 |
| dt median | 1.030 s |
| speed mean / p50 | 245.27 / 293.28 km/h |
| RSRP mean / p5 / p50 | -70.74 / -88.55 / -71.03 dBm |
| SINR mean / p5 / p50 | 15.52 / 3.02 / 15.11 dB |
| SINR < 5 dB | 9.96% |
| SINR < 10 dB | 28.09% |

按真实 PCI change 划分的局部片段中，片段长度均值约 `3.02 km`，中位数约 `2.89 km`。这说明当前 3 km 双小区 episode 的空间尺度与实测相邻小区片段基本一致。

## Baseline gap

20 seeds 对比显示，现有 profile 中 `weak_coverage` 与实测最接近，但仍存在明显 SINR 尾部偏差：

| profile | RSRP mean | RSRP p5 | SINR mean | SINR p5 | SINR<5 | SINR<10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| real | -70.74 | -88.55 | 15.52 | 3.02 | 0.100 | 0.281 |
| weak_coverage | -71.22 | -86.28 | 16.91 | -0.19 | 0.194 | 0.358 |
| mixed_mild | -68.54 | -85.38 | 16.20 | -1.91 | 0.211 | 0.377 |
| stress_radio_holdout | -80.05 | -100.30 | 8.70 | -10.07 | 0.476 | 0.622 |

判断：

- `weak_coverage` 的 RSRP 均值接近实测，但 SINR 低分位过低，低 SINR 比例约为实测 2 倍。
- `normal/high_noise/high_interference` 的 RSRP 明显偏强。
- `stress_radio_holdout` 明显比该实测线路更重，不适合作为实测校准目标。

## 校准 profile

新增 profile：

```text
realistic_trace_calib_v0
```

split：

```text
calib
```

该 profile 不进入默认 `train` 或 `test` split，只用于实测-仿真差距审计和后续外部压力测试。

同时在信道模型中新增一组默认关闭的校准参数：

- `min_link_distance_m`：避免列车被建模为距离基站 1 m，造成不现实的 RSRP/SINR 峰值。
- `dynamic_interference_attenuation_db`：表示邻区同频干扰受到资源隔离、波束方向、测量口径等影响，并不总是满功率互扰。
- `trackside_offset_m`、`bs_height_m`、`ue_height_m`：使用轨旁三维几何计算有效链路距离。
- `max_rsrp_dbm`：限制近站点过高的接收功率峰值，等效表示天线方向图、接收机测量上限和未显式建模的近站损耗。

默认值保持旧行为，因此现有 profile 不受影响。

当前 `realistic_trace_calib_v0` 采用吴卫等《高速铁路宽带无线通信系统切换方案研究》中系统模型的几何参照：RAU 到轨道距离约 `10 m`、RAU 高度约 `15 m`、车顶 MR 高度约 `3 m`。因此链路距离由原来的一维距离改为：

```text
d_link = sqrt(d_along^2 + trackside_offset_m^2 + (bs_height_m - ue_height_m)^2)
```

该论文的 RAU 覆盖半径和 RAU 间距属于 DAS/RAU 架构，不能直接搬到本文 km 级双小区 A3 场景；这里仅采用轨旁距离和高度作为几何参照。

## 校准结果

调优后的 `realistic_trace_calib_v0`，即三维轨旁几何 + RSRP 峰值限制版本：

| profile | RSRP mean | RSRP p5 | RSRP p50 | RSRP p95 | SINR mean | SINR p5 | SINR p50 | SINR p95 | SINR<5 | SINR<10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| real | -70.74 | -88.55 | -71.03 | -51.45 | 15.52 | 3.02 | 15.11 | 28.47 | 0.100 | 0.281 |
| realistic_trace_calib_v0 | -70.80 | -85.66 | -72.57 | -47.42 | 18.27 | 2.05 | 16.02 | 43.27 | 0.126 | 0.276 |

综合差距评分（越低越接近，基于 RSRP mean/p5、SINR mean/p5、SINR<5、SINR<10）：

| profile | score |
| --- | ---: |
| realistic_trace_calib_v0 | 0.557 |
| weak_coverage | 0.812 |
| mixed_mild | 1.116 |

判断：

- 首版校准 profile 已明显优于现有 `weak_coverage`。
- RSRP 均值、SINR p50、SINR<10 比例已较接近实测。
- RSRP p5 仍略偏强，SINR mean/p95 偏高，说明近站峰值和全程高 SINR 片段仍需进一步校正。
- 三维几何版本的综合低尾部分数略弱于纯经验 `min_link_distance_m=60 m` 版本，但物理解释更清晰；因此后续优先沿三维几何模型继续微调，而不是回退到经验距离下限。
- `ho_per_km` 不应在当前阶段强行对齐，因为实测是多 PCI serving 变化，仿真是双小区 A3 执行次数，口径不同。

## 校准 profile 上的策略验证

使用当前推荐 checkpoint：

```text
experiments/runs/20260604_v1_domain_random_obs7_nocql_seed20260604/checkpoints/rainbow_offline_best.pth
```

在 `realistic_trace_calib_v0` 上运行 profile 泛化评估：

```text
experiments/runs/20260611_realistic_trace_calib_v0_3dgeom_capped/profile_generalization_seed53000
```

| policy | HO mean | HO/km | outage | SINR mean | SINR p5 | ping-pong | interruption | overlap SINR | overlap interruption |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| RL_Rainbow | 1.80 | 0.600 | 0.01325 | 17.66 | -0.04 | 0.00 | 0.090 | 10.25 | 0.00033 |
| FixedA3_Hys3_TTT150 | 3.90 | 1.300 | 0.01325 | 18.03 | 2.37 | 0.15 | 0.195 | 9.95 | 0.01191 |

判断：

- 在更贴近实测的无线分布下，RL 仍表现为“少切换、低乒乓、低中断”的保守风格。
- 代价是 SINR p5 低于 Fixed A3，说明低尾部晚切风险仍存在。
- 这与实测轨迹模型推理中“预测切换少、匹配精度高但召回低”的现象一致。

进一步运行 `delta3_low5` late guard 诊断：

```text
experiments/runs/20260611_realistic_trace_calib_v0_3dgeom_capped/late_guard_delta3_low5_seed54000
```

| policy | HO mean | HO/km | outage | SINR mean | SINR p5 | guard ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| RL_Raw_50ms | 1.70 | 0.567 | 0.01907 | 18.24 | -0.97 | 0.0000 |
| RL_LateGuard_50ms | 1.60 | 0.533 | 0.01019 | 18.51 | 0.03 | 0.0091 |
| FixedA3_Hys3_TTT150 | 4.90 | 1.633 | 0.01651 | 18.71 | 1.24 | - |

判断：

- 在 50 ms 诊断口径下，`delta3_low5` guard 只低频介入，触发比例约 `0.91%`。
- guard 将 RL outage 从 `0.0191` 降至 `0.0102`，SINR p5 从 `-0.97 dB` 提升到 `0.03 dB`。
- 切换次数维持在约 `1.6`，说明 guard 主要处理尾部晚切风险，并未显著破坏少切换风格。

注意：late guard 诊断脚本为每 50 ms 检查策略，profile 泛化评估使用 action hold 逻辑，两者口径不同；应分别作为“主线策略评估”和“guard 触发方向诊断”解读。

## 下一步

建议按以下顺序继续：

1. 补跑 Oracle A3 / A3 参数 sweep，确认 `realistic_trace_calib_v0` 下可达到的切换质量上界。
2. 继续微调校准 profile，优先降低近站高 SINR 峰值，使 SINR mean/p95 更接近实测，同时不要破坏 RSRP 均值和低 SINR 比例。
3. 若 raw RL 与 Oracle 的差距主要来自晚切，再考虑把校准 profile 作为小权重 domain randomization 场景加入训练。
4. 若主要损失来自两个候选小区都差或链路/资源瓶颈，则保留为算法边界，不应盲目扩大模型。
5. 扩散模型暂作为后续支线，优先尝试生成实测残差，而不是直接生成完整长线路轨迹。
