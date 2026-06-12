# 对比算法复现目录

本目录单独存放用于论文/实验对比的 baseline 文档与代码实现，避免污染主训练链路。

## 对比实验目标与边界

本目录的对比实验只评价“切换参数自适应”是否有效，即不同策略如何选择 A3 的 `(Hys, TTT)`，以及它们对切换次数、乒乓、outage、SINR 低分位和切换区中断的影响。

对比结果不应被解读为算法能够修复所有极端弱覆盖或资源拥塞场景。弱覆盖、强干扰、NLOS/fading 的中等退化应纳入对比，因为更合理的切换可以缓解晚切/早切造成的损失；但若某些 profile 中两个候选小区都很差、目标小区质量不足，或退化主要来自业务负载和资源瓶颈，则应作为算法边界分析。相关结论应和 [docs/design/problem_scope_and_boundaries.md](../docs/design/problem_scope_and_boundaries.md) 保持一致。

## 已实现策略

- `FixedA3_Hys3_TTT150`：传统固定 A3 参数。
- `FixedA3_Hys2p5_TTT100`：偏激进固定 A3 参数。
- `SpeedAdaptiveA3`：速度/多普勒自适应参数代理，速度越高 TTT 越短。
- `PositionPriorA3`：一维铁路几何切换点预测代理。
- `SignalTrendGuardA3`：基于 Delta RSRP 趋势和低 SINR 风险的预测式保护代理。
- `TabularQ_A3`：离散状态 tabular Q-learning 策略，可在线训练并保存 Q 表。
- `RL_Rainbow_GRU`：当前仓库主线 Rainbow+GRU checkpoint 包装器。
- `OracleA3_FullGrid`：可选上帝视角逐 episode 固定 A3 全网格枚举，只用于上界参考。

## 快速运行

```powershell
.\.venv\Scripts\python.exe comparison_algorithms\scripts\run_comparison.py `
  --include_rainbow `
  --train_tabular_q `
  --q_train_episodes 80 `
  --profile_split test `
  --num_seeds 5
```

小规模冒烟：

```powershell
.\.venv\Scripts\python.exe comparison_algorithms\scripts\run_comparison.py `
  --include_rainbow `
  --train_tabular_q `
  --q_train_episodes 8 `
  --profile_split test `
  --profiles speed_400 distance_2km `
  --num_seeds 2 `
  --output_dir experiments\runs\smoke_comparison_algorithms
```

输出写入 `experiments/runs/<run_id>/`：

- `metrics/comparison_episode_metrics.csv`
- `metrics/comparison_summary_by_profile.csv`
- `metrics/comparison_summary_overall.csv`
- `metrics/run_config.json`
- `experiment_report.md`
- `tables/tabular_q_policy.json`（启用 Q-learning 训练时）

## 已完成的一轮评估

当前已跑通一轮 test holdout 对比：

- run：`experiments/runs/20260604_comparison_algorithms_test20_v2`
- 记录：`comparison_algorithms/docs/comparison_experiment_20260604.md`
- 范围：4 个 test profile × 20 seeds/profile，包含 Rainbow、Tabular Q、启发式 baseline 和 Oracle A3 全网格上界。

## 注意

这些 baseline 是为当前 Python 场景构建的可复现实验代理。它们保持 `[15, 7]` 观测和 48 动作空间不变，不修改 Simu5G 链路；策略只选择 `(Hys, TTT)`，A3 事件、TTT 计时、保护窗口、中断、outage 和 KPI 仍由 `TrainHandoverEnv` 统一执行。
