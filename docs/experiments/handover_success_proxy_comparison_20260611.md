# 切换成功率与移动性失败 proxy 对比实验（2026-06-11）

## 目标

很多切换算法论文使用 `handover success rate` 作为核心指标。本文当前环境中，A3+TTT 一旦触发通常会直接完成小区切换，如果简单使用“执行切换次数 / 触发次数”，会高估保守策略的可靠性。因此本实验新增两个口径：

- `ho_attempt_success_rate`：已触发切换尝试后的成功率。
- `mobility_success_rate_proxy`：把“未及时切换导致的 late-HO outage/RLF proxy”也计入移动性失败后的成功率。

第二个口径更适合本文，因为它能识别“策略很少切换，所以每次切换看似成功，但切换前已经持续低 SINR/outage”的情况。

## 指标定义

切换尝试：

```text
HO attempt = A3 条件满足且 TTT 计时达到阈值，即环境中的 ho_triggered=True。
```

切换尝试成功：

```text
HO success = HO attempt 实际执行后，在 post-HO 验证窗口内：
             1) 经过切换中断/滤波恢复宽限期；
             2) 没有连续非中断 outage 达到 T_out；
             3) 没有很快回切到原小区。
```

移动性成功率 proxy：

```text
mobility_success_rate_proxy
  = HO success count / (HO attempt count + late-HO failure count)
```

其中：

```text
late-HO failure = 未发生及时切换时，服务小区连续 outage 且邻区明显更强。
```

当前参数：

| 参数 | 数值 |
|---|---:|
| post-HO validation window | 0.5 s |
| grace window | max(切换中断时长, T_out) |
| T_out | 0.2 s |
| late-HO delta RSRP threshold | 3 dB |

实现位置：

```text
comparison_algorithms/evaluation.py
```

## 实验设置

输出目录：

```text
experiments/runs/20260611_handover_success_proxy_comparison_test20
```

命令：

```powershell
.\.venv\Scripts\python.exe comparison_algorithms\scripts\run_comparison.py `
  --include_rainbow `
  --include_rainbow_late_guard `
  --q_table_path experiments\runs\20260604_comparison_algorithms_test20_v2\tables\tabular_q_policy.json `
  --profile_split test `
  --num_seeds 20 `
  --base_seed 25000 `
  --include_oracle_a3 `
  --output_dir experiments\runs\20260611_handover_success_proxy_comparison_test20
```

评估 profile：

- `stress_radio_holdout`
- `speed_400`
- `distance_2km`
- `distance_4km`

总计：

```text
4 profiles x 20 seeds x 9 policies = 720 episodes
```

## Overall 结果（逐 episode 比率平均）

| policy | HO/km | HO attempt success | late-HO failure | mobility success proxy | outage | SINR p5 | ping-pong | overlap interruption |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| FixedA3_Hys3_TTT150 | 1.954 | 0.848 | 0.725 | 0.803 | 0.05235 | -3.989 | 0.238 | 0.01398 |
| PositionPriorA3 | 1.538 | 0.824 | 1.500 | 0.730 | 0.05569 | -4.438 | 0.200 | 0.00723 |
| SpeedAdaptiveA3 | 2.931 | 0.754 | 0.688 | 0.730 | 0.05789 | -4.620 | 0.525 | 0.02261 |
| FixedA3_Hys2p5_TTT100 | 3.142 | 0.747 | 0.788 | 0.725 | 0.06057 | -4.815 | 0.625 | 0.02475 |
| RL_Rainbow_GRU_LateGuard | 1.106 | 0.934 | 1.725 | 0.713 | 0.05660 | -4.919 | 0.000 | 0.00208 |
| TabularQ_A3 | 2.831 | 0.739 | 0.800 | 0.698 | 0.06183 | -4.826 | 0.875 | 0.02683 |
| SignalTrendGuardA3 | 6.685 | 0.592 | 0.188 | 0.591 | 0.07874 | -7.106 | 2.025 | 0.04388 |
| OracleA3_FullGrid | 0.515 | 0.959 | 2.313 | 0.587 | 0.05833 | -4.752 | 0.000 | 0.00047 |
| RL_Rainbow_GRU | 0.565 | 0.958 | 3.063 | 0.434 | 0.07558 | -6.344 | 0.000 | 0.00165 |

## Overall 结果（总量加权口径）

| policy | attempts | HO success | late-HO failure | weighted HO success | weighted mobility success | weighted mobility failure |
|---|---:|---:|---:|---:|---:|---:|
| FixedA3_Hys3_TTT150 | 460 | 368 | 58 | 0.800 | 0.710 | 0.290 |
| TabularQ_A3 | 670 | 469 | 64 | 0.700 | 0.639 | 0.361 |
| SpeedAdaptiveA3 | 698 | 467 | 55 | 0.669 | 0.620 | 0.380 |
| RL_Rainbow_GRU_LateGuard | 262 | 237 | 138 | 0.905 | 0.593 | 0.407 |
| FixedA3_Hys2p5_TTT100 | 752 | 475 | 63 | 0.632 | 0.583 | 0.417 |
| PositionPriorA3 | 364 | 240 | 120 | 0.659 | 0.496 | 0.504 |
| SignalTrendGuardA3 | 1596 | 648 | 15 | 0.406 | 0.402 | 0.598 |
| OracleA3_FullGrid | 120 | 109 | 185 | 0.908 | 0.357 | 0.643 |
| RL_Rainbow_GRU | 132 | 128 | 245 | 0.970 | 0.340 | 0.660 |

## 关键观察

1. `RL_Rainbow_GRU` 的已尝试切换成功率最高，逐 episode 平均为 `0.958`，总量加权为 `0.970`。但这不是好消息：它切换尝试太少，late-HO failure 平均达到 `3.063`，加权移动性成功率只有 `0.340`。
2. `RL_Rainbow_GRU_LateGuard` 将 late-HO failure 从 `3.063` 降到 `1.725`，移动性成功率 proxy 从 `0.434` 提升到 `0.713`（逐 episode 平均）。代价是 HO/km 从 `0.565` 增加到 `1.106`，但仍明显低于固定 A3 和启发式基线。
3. `FixedA3_Hys3_TTT150` 在该移动性成功率 proxy 下整体最高，说明固定中等参数在尾部可靠性上仍是很强的基线。不过它的 ping-pong 和 overlap interruption 明显高于 `RL + LateGuard`。
4. `OracleA3_FullGrid` 的 HO attempt success 很高，但 mobility success proxy 不高。原因是当前 Oracle 选择目标仍按旧评分函数选择，偏向少切换/低中断，并没有把 late-HO failure 纳入 oracle 目标，因此不能作为该新指标下的真正上界。

## 分 profile 观察

`RL + LateGuard` 相对原始 RL 的移动性成功率 proxy：

| profile | RL raw | RL + guard | 提升 |
|---|---:|---:|---:|
| distance_2km | 0.603 | 0.814 | +0.211 |
| distance_4km | 0.578 | 0.827 | +0.249 |
| speed_400 | 0.411 | 0.794 | +0.382 |
| stress_radio_holdout | 0.142 | 0.418 | +0.275 |

Stress 中仍然最低，说明剩余失败不只是切换触发问题，也包含弱覆盖、NLOS/fading、强干扰和资源/链路边界。

## 论文使用建议

论文中可以新增一组指标：

```text
HO attempt success rate
Late-HO failure count
Mobility success rate proxy
```

但不要只汇报 `HO attempt success rate`。原始 RL 正是反例：它的 attempt success rate 很高，却因为过保守导致 late-HO failure 多。更稳妥的表述是：

> 仅统计已触发切换的成功率会高估少切换策略的可靠性。本文进一步引入 late-HO failure proxy，将低 SINR 且邻区明显更强但未及时切换的片段计为移动性失败。该指标显示 late guard 能显著缓解原始 RL 的保守晚切问题，同时保持低 ping-pong 和低重叠区中断。

后续如果要把该指标作为主表之一，建议同步修改 Oracle 评分函数，将 `late_ho_failure_count` 或 `mobility_failure_rate_proxy` 纳入 Oracle 目标，得到更合理的可靠性上界。

## 论文表格产物

已根据本次 run 生成论文主表和辅助表：

```text
paper/assets/tables/final/table_main_python_comparison_metrics.csv
paper/assets/tables/final/table_main_python_comparison_metrics.md
paper/assets/tables/final/table_profile_python_comparison_metrics.csv
paper/assets/tables/final/table_profile_python_comparison_metrics.md
paper/assets/tables/source/table_weighted_handover_success_proxy.csv
paper/assets/tables/source/table_weighted_handover_success_proxy.md
```

生成脚本：

```text
tools/build_paper_metric_tables.py
```

运行命令：

```powershell
.\.venv\Scripts\python.exe tools\build_paper_metric_tables.py
```

其中：

- `table_main_python_comparison_metrics`：推荐作为 Python 多 profile 主结果表。
- `table_profile_python_comparison_metrics`：推荐用于附表或按 profile 分析。
- `table_weighted_handover_success_proxy`：用于解释逐 episode 平均口径与总量加权口径的差异，不建议直接替代主表。
