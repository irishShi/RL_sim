# 实验记录索引

本目录记录项目从 `Obs7 + GRU + Rainbow DQN` 主线、domain randomization、late handover guard、Simu5G 查表验证到论文表格准备的阶段性实验。结论应以具体记录文件为准，不使用“最新结果”这类不带路径和日期的说法。

## 当前主线证据

| 日期 | 文件 | 主要内容 | 论文使用建议 |
| --- | --- | --- | --- |
| 2026-06-04 | [v1_domain_random_obs7_full_pipeline_20260604.md](v1_domain_random_obs7_full_pipeline_20260604.md) | V1 domain random Obs7 数据、离线训练、Python 评估、policy table 导出、Simu5G 复杂矩阵 | 方法与主实验背景，可引用数据集/checkpoint/run |
| 2026-06-04 | [late_handover_guard_measured_sinr_20260604.md](late_handover_guard_measured_sinr_20260604.md) | `delta3_low5` late guard、measured SINR 查表口径、Python holdout 和 Simu5G 验证 | 主方法安全层证据 |
| 2026-06-11 | [handover_success_proxy_comparison_20260611.md](handover_success_proxy_comparison_20260611.md) | `RL + LateGuard` 与 baseline 同批对比，新增移动性成功率 proxy | 推荐作为 Python 主结果表来源 |
| 2026-06-11 | [offline_vs_online_rl_basic_20260611.md](offline_vs_online_rl_basic_20260611.md) | 离线训练路线与小规模 online-from-scratch baseline 对比 | 支撑工程部署合理性，注意不要夸大为离线 RL 全面优越 |
| 2026-06-11 | [real_trace_calibration_gap_20260611.md](real_trace_calibration_gap_20260611.md) | 实测 NR 轨迹与仿真 profile 差距审计、`realistic_trace_calib_v0` | 支撑仿真可信度和局限性讨论 |

## 支撑与规划记录

| 日期 | 文件 | 主要内容 |
| --- | --- | --- |
| 2026-06-04 | [robust_generalization_v1_plan.md](robust_generalization_v1_plan.md) | V1 鲁棒泛化训练框架设计 |
| 2026-06-04 | [simu5g_complex_scenario_plan_20260604.md](simu5g_complex_scenario_plan_20260604.md) | Simu5G 复杂铁路场景扩展方案 |
| 2026-06-04 | [simu5g_policy_table_compare_20260604.md](simu5g_policy_table_compare_20260604.md) | policy table 初次接入 Simu5G 对比 |
| 2026-06-04 | [simu5g_complex_matrix_20260604.md](simu5g_complex_matrix_20260604.md) | Simu5G 复杂场景矩阵首轮结果 |
| 2026-06-05 | [stage_progress_and_publication_plan_20260605.md](stage_progress_and_publication_plan_20260605.md) | 阶段进展、短板、投稿路线评估 |
| 2026-06-05 | [risk_aware_late_guard_comparison_20260605.md](risk_aware_late_guard_comparison_20260605.md) | 风险感知 late guard 早期对比 |

## 探索支线

以下记录用于保留探索过程和负结果，不建议替代当前默认方案：

| 日期 | 文件 | 结论定位 |
| --- | --- | --- |
| 2026-06-06 | [physics_informed_risk_head_smoke_20260606.md](physics_informed_risk_head_smoke_20260606.md) | risk head 能跑通 smoke，但不是默认策略 |
| 2026-06-06 | [physics_informed_risk_model_evaluation_20260606.md](physics_informed_risk_model_evaluation_20260606.md) | physics risk 方案未稳定优于 `late guard` |
| 2026-06-06 | [physics_informed_risk_v2_stress_highinterference_20260606.md](physics_informed_risk_v2_stress_highinterference_20260606.md) | Stress/HighInterference 专项探索，仍需归因 |

## 当前推荐结论口径

论文和项目说明中建议采用以下表述：

- `Obs7 + GRU + Rainbow DQN` 学到的是低切换、低乒乓、低中断的保守策略。
- 原始 RL 在低 SINR 或邻区明显更强时可能晚切，因此需要 `late guard`。
- `RL + LateGuard` 显著减少原始 RL 的 late-HO failure，并保持低 ping-pong 和低 overlap interruption。
- 固定 A3 在部分 outage / SINR 尾部指标上仍是强基线，不能声称 RL 全面优于固定 A3。
- Stress / HighInterference 的剩余问题要区分切换可控损失和覆盖、干扰、资源或队列边界。

## 新实验记录模板

新增实验记录建议包含以下小节：

```text
# 实验标题（日期）

## 目标
## 代码与产物
## 数据、checkpoint 与命令
## 指标结果
## 关键观察
## 论文使用建议
## 后续限制与下一步
```
