# 风险感知 Late Guard 对比验证（2026-06-05）

本文记录第一阶段模型改进：在当前 `Obs7 + GRU + Rainbow DQN` checkpoint 上叠加物理风险感知 late handover guard，形成 `RL_Rainbow_GRU_LateGuard`。该方案不改变观测维度、动作空间、checkpoint 或 policy table 导出链路，只在每个 `50 ms` 测量点根据低 SINR、邻区优势、速度和距上次切换时间对过保守 `Hys/TTT` 动作做安全约束。

## 改进目标

本阶段不追求预测瞬时快衰落，而是用物理规则识别晚切风险：

```text
低 SINR + 邻区明显更强 + 已过最小切换间隔
    -> 限制过大 Hys / 过长 TTT
    -> 在满足约束的动作中选择 Rainbow Q 值最高动作
```

验收重点：

- 相对 `RL_Rainbow_GRU` 降低 `outage_time_ratio`。
- 提升 `sinr_p5_db`，降低 `sinr_below_minus3db_ratio`。
- 保持 `ping_pong_count` 接近 0。
- `comm_interruption_ratio_in_overlap_zone` 保持显著低于固定 A3、速度自适应和 TabularQ 等基线。
- `ho_per_km` 可适度上升，但应显著低于激进启发式/表格策略。

完整计划见 `docs/design/risk_aware_model_improvement_plan.md`。

## 代码入口

- 策略实现：`comparison_algorithms/baseline_policies.py`
- 评估入口：`comparison_algorithms/scripts/run_comparison.py`
- 新增策略名：`RL_Rainbow_GRU_LateGuard`
- guard 逻辑：`utils/late_handover_guard.py`

`RL_Rainbow_GRU_LateGuard` 通过 `ComparisonPolicy.postprocess_action()` 作为每步动作后处理，不改变 `ActionHoldController` 的全局语义。其他基线和训练流程不受影响。

## 实验设置

输出目录：

```text
experiments/runs/20260605_rainbow_late_guard_comparison_test20
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
  --output_dir experiments\runs\20260605_rainbow_late_guard_comparison_test20
```

评估 profile：

- `stress_radio_holdout`
- `speed_400`
- `distance_2km`
- `distance_4km`

每个 profile 20 seeds，总计 80 episodes。为保证与 `20260604_comparison_algorithms_test20_v2` 可比，本轮沿用相同 `base_seed=25000`，TabularQ 使用前一轮已训练 Q 表。Oracle A3 不在本轮重新枚举，作为不可部署上界继续参考 `20260604_comparison_algorithms_test20_v2`。

Guard 参数采用 `delta3_low5` 风格：

```text
guard_min_delta_db = 3
guard_low_sinr_db = -5
guard_critical_sinr_db = -7
guard_fast_sinr_db = -3
guard_moderate_sinr_db = 0
guard_min_time_since_ho_s = 1.0
TTT caps = 150 ms
Hys caps = 3.0 / 2.5 dB
```

## Overall 结果

| policy | ho/km | SINR p5 | outage | pingpong | overlap SINR | overlap interrupt |
|---|---:|---:|---:|---:|---:|---:|
| `FixedA3_Hys3_TTT150` | 1.904 | -3.934 | 0.04897 | 0.225 | 5.194 | 0.01413 |
| `PositionPriorA3` | 1.663 | -4.454 | 0.05529 | 0.200 | 5.037 | 0.00765 |
| `RL_Rainbow_GRU` | 0.590 | -6.670 | 0.07955 | 0.000 | 5.173 | 0.00165 |
| `RL_Rainbow_GRU_LateGuard` | 1.106 | -4.948 | 0.05663 | 0.000 | 5.570 | 0.00219 |
| `FixedA3_Hys2p5_TTT100` | 3.125 | -4.858 | 0.06011 | 0.588 | 4.754 | 0.02459 |
| `TabularQ_A3` | 2.898 | -4.853 | 0.06129 | 1.038 | 4.554 | 0.02834 |
| `SpeedAdaptiveA3` | 3.098 | -4.877 | 0.06240 | 0.675 | 4.525 | 0.02472 |
| `SignalTrendGuardA3` | 6.360 | -7.122 | 0.07512 | 1.675 | 4.649 | 0.03403 |

相对原始 `RL_Rainbow_GRU`：

- `outage_time_ratio_mean`: `0.07955 -> 0.05663`，下降约 `28.8%`。
- `sinr_p5_db_mean`: `-6.670 -> -4.948`，提升约 `1.72 dB`。
- `sinr_below_minus3db_ratio_mean`: 下降约 `0.0277`。
- `ho_per_km_mean`: `0.590 -> 1.106`，切换增加但仍明显低于固定 A3、SpeedAdaptiveA3、TabularQ。
- `ping_pong_count_mean`: 保持 `0.000`。
- `comm_interruption_ratio_in_overlap_zone_mean`: `0.00165 -> 0.00219`，略升但仍显著低于 `FixedA3_Hys3_TTT150` 的 `0.01413` 和多数启发式/表格策略。

按六个核心指标做简单平均排名：

```text
outage_time_ratio 越低越好
sinr_p5_db 越高越好
comm_interruption_ratio_in_overlap_zone 越低越好
ping_pong_count 越低越好
ho_per_km 越低越好
overlap_zone_sinr_mean_db 越高越好
```

`RL_Rainbow_GRU_LateGuard` 平均排名为第 1，说明它在当前可部署/可复现基线中处于相对较优位置。它不是单项最优策略：固定 A3 的整体 outage 和 SINR p5 更好，但固定 A3 的切换、中断和乒乓代价明显更高；原始 RL 的切换和中断最低，但尾部 SINR/outage 明显不足。LateGuard 方案处于这两者之间，取得更均衡的折中。

## Profile 观察

- `stress_radio_holdout`：LateGuard 将原始 RL 的 `outage` 从 `0.21637` 降到 `0.15839`，`SINR p5` 从 `-15.020 dB` 提升到 `-11.547 dB`，同时 `pingpong=0`，重叠区中断仅 `0.00062`。
- `speed_400`：LateGuard 将原始 RL 的 `outage` 从 `0.03972` 降到 `0.02926`，`SINR p5` 从 `-4.782 dB` 提升到 `-3.422 dB`，仍无乒乓。
- `distance_2km`：LateGuard 将原始 RL 的 `outage` 从 `0.03563` 降到 `0.02125`，`SINR p5` 从 `-3.506 dB` 提升到 `-2.354 dB`。
- `distance_4km`：LateGuard 将原始 RL 的 `outage` 从 `0.02648` 降到 `0.01764`，`SINR p5` 从 `-3.372 dB` 提升到 `-2.469 dB`。

## 结论

第一阶段目标达成：`RL_Rainbow_GRU_LateGuard` 明显修复原始 RL 的保守晚切尾部问题，并保持低乒乓、低重叠区中断和较低切换次数。当前最稳妥的表述是：

> 风险感知 late guard 不是全面优于固定 A3，而是在低切换/低乒乓/低重叠区中断的 RL 策略基础上，显著改善尾部 SINR 与 outage，使整体多 KPI 表现进入当前可部署对比策略中的相对较优位置。

后续如果继续推进，应先做 Doppler-correlated fading profile 和风险预测辅助头的消融，而不是直接扩大主网络或预测瞬时快衰落波形。

