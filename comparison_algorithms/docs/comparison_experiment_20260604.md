# 对比算法 Python 评估记录（2026-06-04）

## 实验入口

代码与文档子目录：

- `comparison_algorithms/baseline_policies.py`
- `comparison_algorithms/evaluation.py`
- `comparison_algorithms/scripts/run_comparison.py`
- `comparison_algorithms/docs/literature_to_baselines.md`

正式小批量评估 run：

- `experiments/runs/20260604_comparison_algorithms_test20_v2`
- profile split：`test`
- profiles：`stress_radio_holdout`、`speed_400`、`distance_2km`、`distance_4km`
- seeds/profile：`20`
- base seed：`25000`
- 当前 RL checkpoint：`experiments/runs/20260604_v1_domain_random_obs7_nocql_seed20260604/checkpoints/rainbow_offline_best.pth`
- Tabular Q 训练：train split 120 episodes，`alpha=0.08`，`gamma=0.98`，epsilon `0.25 -> 0.02`
- Q 表状态数：`6410`

运行命令：

```powershell
.\.venv\Scripts\python.exe comparison_algorithms\scripts\run_comparison.py `
  --include_rainbow `
  --train_tabular_q `
  --q_train_episodes 120 `
  --profile_split test `
  --num_seeds 20 `
  --base_seed 25000 `
  --q_train_base_seed 62000 `
  --include_oracle_a3 `
  --output_dir experiments\runs\20260604_comparison_algorithms_test20_v2
```

## 策略

| 策略 | 含义 |
|---|---|
| `FixedA3_Hys3_TTT150` | 标准固定 A3 参数 |
| `FixedA3_Hys2p5_TTT100` | 偏激进固定 A3 参数 |
| `SpeedAdaptiveA3` | 速度/多普勒自适应启发式 |
| `PositionPriorA3` | 一维轨道切换点先验启发式 |
| `SignalTrendGuardA3` | Delta RSRP 趋势 + 低 SINR 风险保护启发式 |
| `TabularQ_A3` | 离散状态 Q-learning 表策略 |
| `RL_Rainbow_GRU` | 当前 Obs7 + GRU + Rainbow DQN checkpoint |
| `OracleA3_FullGrid` | 每个 episode 枚举 48 组固定 A3 的上帝视角上界 |

## Overall 结果

80 episodes 汇总。`ho_per_km` 已按各 profile 实际线路长度（2/3/4 km）修正。

| policy | ho | ho/km | SINR p5 | outage | pingpong | overlap SINR | overlap interruption |
|---|---:|---:|---:|---:|---:|---:|---:|
| `OracleA3_FullGrid` | 1.35 | 0.465 | -4.911 | 0.05945 | 0.00 | 6.424 | 0.00042 |
| `RL_Rainbow_GRU` | 1.75 | 0.598 | -6.538 | 0.07946 | 0.00 | 5.744 | 0.00165 |
| `PositionPriorA3` | 4.55 | 1.538 | -4.315 | 0.05425 | 0.10 | 5.430 | 0.00668 |
| `FixedA3_Hys3_TTT150` | 5.70 | 1.938 | -3.903 | 0.04983 | 0.20 | 5.254 | 0.01378 |
| `TabularQ_A3` | 8.43 | 2.848 | -4.861 | 0.06325 | 0.93 | 4.311 | 0.02963 |
| `SpeedAdaptiveA3` | 8.58 | 2.881 | -4.576 | 0.06062 | 0.49 | 4.685 | 0.02375 |
| `FixedA3_Hys2p5_TTT100` | 9.23 | 3.083 | -4.767 | 0.06018 | 0.64 | 4.640 | 0.02601 |
| `SignalTrendGuardA3` | 17.90 | 6.002 | -7.077 | 0.07354 | 1.64 | 4.599 | 0.03350 |

## Profile 观察

- `distance_2km`：`RL_Rainbow_GRU` 切换最少（1.6 次）且无乒乓，但 `PositionPriorA3` 的 outage 更低（0.01917 vs 0.03563）。
- `distance_4km`：`RL_Rainbow_GRU` 切换较少（2.5 次）且重叠区 SINR 接近 `PositionPriorA3`，但 SINR p5 低于固定 A3。
- `speed_400`：`RL_Rainbow_GRU` 切换次数接近 Oracle（1.2 vs 1.1），重叠区中断为 0，但 SINR p5 和 outage 不如 Oracle / 固定 A3。
- `stress_radio_holdout`：所有非 Oracle 策略都暴露尾部问题。`RL_Rainbow_GRU` 极大压低切换和乒乓，但 SINR p5/outage 明显弱于固定 A3 和 Tabular Q。这与现有结论一致：Stress 剩余问题不只是切换时机，还包含无线覆盖、NLOS/fading、干扰和业务负载耦合。

## 产物

- `metrics/comparison_episode_metrics.csv`：逐 episode 明细。
- `metrics/comparison_summary_by_profile.csv`：按 profile 和策略汇总。
- `metrics/comparison_summary_overall.csv`：跨 profile 汇总。
- `metrics/oracle_a3_selections.csv`：逐 episode Oracle 参数选择。
- `metrics/tabular_q_train_summary.json`：Q 表训练摘要。
- `tables/tabular_q_policy.json`：训练出的 Tabular Q 表。
- `experiment_report.md`：脚本自动生成的简表报告。

## 限制

- `TabularQ_A3` 只训练 120 episodes，是快速可跑的对比基线，不代表充分调参后的最优表格 RL。
- `SpeedAdaptiveA3`、`PositionPriorA3`、`SignalTrendGuardA3` 是文献思想的 Python 场景代理实现，不等同原论文完整系统模型。
- `OracleA3_FullGrid` 使用每个 episode 的完整枚举结果选优，是上界参考，不是可部署算法。
- 本轮没有叠加 late handover guard；若论文主方法使用 guard，应另跑 `Rainbow+guard` 与这些 baseline 的同场景对比。

