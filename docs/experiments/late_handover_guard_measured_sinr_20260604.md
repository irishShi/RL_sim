# 低 SINR/快速退化晚切优化验证记录（2026-06-04）

## 目标

针对 V1 `Obs7 + GRU + Rainbow DQN` 策略在低 SINR、快速退化场景下偏保守、TTT 偏长、可能晚切的问题，加入小型安全层：

- 不扩展模型入参，保持 `obs_dim=7`。
- 不重训模型，先在 policy table 导出阶段做 guard。
- 在 Simu5G 桥接侧改进查表用的 SINR 观测口径。
- 通过 Python holdout profile 和 Simu5G 复杂矩阵验证。

## 代码与产物

新增/修改：

- `utils/late_handover_guard.py`
- `scripts/export/export_policy_table.py`
- `scripts/eval/test_late_guard_generalization.py`
- `tools/export_late_guard_policy_preset.py`
- `tools/inspect_simu5g_policy_timeline.py`
- Simu5G `LtePhyUe.cc/.h`：查表 `sinr_db` 优先使用服务小区 `getSINR()` 均值，初始化阶段回退到 RSRP proxy。

最终推荐策略表：

- `results/policy_tables/policy_table_v1_domain_random_obs7_late_guard_delta3_low5_seed20260604.csv`
- `results/policy_tables/policy_table.csv`

最终 Simu5G 验证目录：

- `results/simu5g/RailwayComplexMatrix-V1LateGuardMeasuredSinrDelta3-20260604/parsed`

## Guard 规则

最终推荐预设：`delta3_low5`

```text
min_delta_db = 3
low_sinr_db = -5
critical_sinr_db = -7
fast_sinr_db = -3
moderate_sinr_db = 0
strong_delta_db = 8
min_time_since_ho_s = 1.0
max_ttt_ms = 150
hys_cap_db = 3.0
critical_hys_cap_db = 2.5
delta3_hys_cap_db = 2.5
```

导出表统计：

- rows: `830088`
- guard applied rows: `88680`
- guard applied ratio: `10.68%`

## 关键发现

第一版只在导出表中用 RSRP 推算的 `serving_sinr_proxy_db` 触发 guard。Simu5G Stress 中实际测得 SINR 很低，但 proxy 的低分位没有充分反映 NLOS/fading、资源减少、重业务导致的真实退化，导致 guard 触发偏晚。

因此在 Simu5G 桥接侧改成：

- 优先用当前服务小区 `primaryChannelModel_->getSINR()` 均值作为查表 `sinr_db`。
- 如果初始化阶段或计算失败，则回退到原 `computeRailwayPolicySinrProxy()`。

这使 policy table 的 `sinr_db` 维度更接近最终评价用的 `measured_sinr_dl_db`。

## Python 侧验证

使用 `test_late_guard_generalization.py`，20 seeds，test split：

| profile | Raw outage | Guard outage | Raw SINR p5 | Guard SINR p5 | Raw HO | Guard HO |
|---|---:|---:|---:|---:|---:|---:|
| stress_radio_holdout | 0.1957 | 0.1615 | -13.38 | -11.10 | 1.2 | 3.6 |
| speed_400 | 0.0306 | 0.0145 | -3.37 | -1.94 | 1.1 | 1.7 |
| distance_2km | 0.0251 | 0.0173 | -2.77 | -2.21 | 1.1 | 1.2 |
| distance_4km | 0.0263 | 0.0163 | -3.17 | -2.40 | 1.7 | 2.3 |

结论：Python 环境中，late guard 明确降低 holdout profile 的 outage，并改善 SINR 低分位；代价是 HO/interruption 略增，但仍小于固定 A3 的切换频率。

## Simu5G 最终验证

最终版本为 `measured-SINR + delta3_low5`，与原 V1 RLTable 对比：

| scenario | first HO 原始/最终 m | loss 原始/最终 | SINR p5 原始/最终 dB | CQI 原始/最终 | max delay 原始/最终 s |
|---|---:|---:|---:|---:|---:|
| Stress | 1724.99 / 1683.33 | 0.3008 / 0.2829 | -16.60 / -16.40 | 5.98 / 6.08 | 1.804 / 2.946 |
| WeakCoverage | 1808.33 / 1808.33 | 0.0117 / 0.00223 | -3.04 / -1.04 | 9.43 / 9.88 | 0.305 / 0.311 |
| HighNoise | 1808.33 / 1766.66 | 0.00558 / 0.00223 | -1.61 / -3.63 | 9.78 / 9.62 | 0.311 / 0.305 |
| HighInterference | 1891.66 / 1849.99 | 0.0000 / 0.00056 | 11.88 / 6.98 | 12.94 / 12.55 | 0.067 / 0.311 |
| TrackOffset | 1808.33 / 1779.16 | 0.0000 / 0.0000 | 8.79 / 7.62 | 12.33 / 12.38 | 0.067 / 0.067 |
| NlosFading | 1641.66 / 1612.49 | 0.0000 / 0.0000 | 7.75 / 8.96 | 12.61 / 12.56 | 0.067 / 0.067 |
| HeavyTraffic | 1641.66 / 1808.33 | 0.00084 / 0.0000 | 9.38 / 9.05 | 12.52 / 12.59 | 0.083 / 0.083 |

## 结论

本轮优化达到了“缓解低 SINR/快速退化时晚切”的目标：

- Stress 切换点提前约 `41.7 m`，丢包率下降约 `5.9%` 相对值，CQI 均值提升。
- WeakCoverage 丢包显著下降，SINR p5 明显改善。
- NlosFading、HeavyTraffic 等场景保持稳定或改善。
- 代价是 Stress 最大时延仍偏高，HighInterference 的 SINR p5 与 max delay 变差。

Stress 的剩余问题不完全是切换时机导致，而是低 SINR、NLOS/fading、资源减少和重业务共同造成的排队/丢包。对于仍存在可用邻区质量优势的退化片段，Hys/TTT 和 late guard 应继续发挥缓解作用；但若片段中相邻候选基站都质量很差或主要受资源排队限制，则单靠 Hys/TTT 安全层无法完全修复。后续若继续优化，应先区分“切换可控损失”和“覆盖/资源极限损失”，避免为少数极端样本盲目增加模型复杂度。

## 推荐使用

当前推荐默认表：

```text
results/policy_tables/policy_table.csv
```

对应预设：

```powershell
.\.venv\Scripts\python.exe tools\export_late_guard_policy_preset.py delta3_low5 --copy_to_default
```

Simu5G 验证命令：

```powershell
wsl -d Ubuntu -- bash -lc "cd /mnt/c/Users/qichengshi/Desktop/RL_sim && bash tools/run_simu5g_complex_matrix.sh RailwayComplexMatrix-V1LateGuardMeasuredSinrDelta3-20260604"
.\.venv\Scripts\python.exe tools\analyze_simu5g_complex_matrix.py --raw_root results\simu5g\RailwayComplexMatrix-V1LateGuardMeasuredSinrDelta3-20260604\raw --parsed_root results\simu5g\RailwayComplexMatrix-V1LateGuardMeasuredSinrDelta3-20260604\parsed --window_s 2.0
```

## 下一步建议

如果继续优化 Stress，建议按从轻到重的顺序推进：

1. 先做时间线归因，判断低 SINR、丢包和 delay 是否由晚切直接造成，还是相邻候选基站都不可用或资源本身不足。
2. 保留小模型，优先评估 late guard 阈值、action hold 和轻量特征消融。
3. 只有多 seed 证据显示核心损失仍由切换时机造成时，再考虑 R6 robust reward 或业务/队列观测；新增项必须证明核心 KPI 收益大于复杂度和推理成本。
