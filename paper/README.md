# 论文材料索引

本目录保存论文写作相关材料，包括最终候选表格、文献调研、后续 PaperSpine 产物和投稿材料。实验完整输出仍应保存在 `experiments/runs/` 或 `results/`，这里仅放论文写作需要直接引用或加工的材料。

## 当前可用材料

| 文件 | 来源 | 用途 |
| --- | --- | --- |
| [references/literature_review_handover_rl_5gr_20260611.md](references/literature_review_handover_rl_5gr_20260611.md) | 2026-06-11 文献调研 | 相关工作、baseline 引文映射、5G-R 背景 |
| [assets/tables/final/table_main_python_comparison_metrics.md](assets/tables/final/table_main_python_comparison_metrics.md) | `20260611_handover_success_proxy_comparison_test20` | Python 多 profile 主结果表 |
| [assets/tables/final/table_profile_python_comparison_metrics.md](assets/tables/final/table_profile_python_comparison_metrics.md) | `20260611_handover_success_proxy_comparison_test20` | 分 profile 结果附表或分析表 |
| [assets/tables/source/table_weighted_handover_success_proxy.md](assets/tables/source/table_weighted_handover_success_proxy.md) | `20260611_handover_success_proxy_comparison_test20` | 解释逐 episode 平均与总量加权口径差异 |

## 建议论文主线

当前材料适合支撑一篇以工程可部署性为重点的论文：

1. 高速铁路 5G-R 相邻小区重叠区内的 A3 `Hys/TTT` 自适应控制。
2. `Obs7 + GRU + Rainbow DQN` 离线训练策略，结合多 profile domain randomization。
3. 针对原始 RL 偏保守、晚切风险高的问题，引入轻量 `late handover guard`。
4. 用 Python 多 profile baseline 对比验证切换次数、乒乓、outage、SINR p5、重叠区中断和移动性成功率 proxy。
5. 用 Simu5G policy table 查表链路验证部署可行性。
6. 用实测轨迹校准 profile 说明仿真分布与真实线路统计的接近程度和限制。

注意：不要把当前结论写成“全面优于固定 A3”。更稳妥的贡献表达是：`RL + LateGuard` 在低切换、低乒乓和低重叠区中断方面有优势，并显著缓解原始 RL 的晚切失败；固定 A3 在部分尾部 SINR/outage 指标上仍是强基线。

## PaperSpine 准备状态

当前已经具备启动 PaperSpine `build_from_materials` 的条件，但还没有生成 `paper_rewriting_output/paper_spine_config.json`。建议启动后优先把以下材料纳入 `source_map.md`：

- [../docs/design/problem_scope_and_boundaries.md](../docs/design/problem_scope_and_boundaries.md)
- [../docs/design/README_MODEL.md](../docs/design/README_MODEL.md)
- [../docs/experiments/v1_domain_random_obs7_full_pipeline_20260604.md](../docs/experiments/v1_domain_random_obs7_full_pipeline_20260604.md)
- [../docs/experiments/late_handover_guard_measured_sinr_20260604.md](../docs/experiments/late_handover_guard_measured_sinr_20260604.md)
- [../docs/experiments/handover_success_proxy_comparison_20260611.md](../docs/experiments/handover_success_proxy_comparison_20260611.md)
- [../docs/experiments/offline_vs_online_rl_basic_20260611.md](../docs/experiments/offline_vs_online_rl_basic_20260611.md)
- [../docs/experiments/real_trace_calibration_gap_20260611.md](../docs/experiments/real_trace_calibration_gap_20260611.md)
- [references/literature_review_handover_rl_5gr_20260611.md](references/literature_review_handover_rl_5gr_20260611.md)

PaperSpine 第一轮目标应是生成动机选项、证据库、章节蓝图和写作矩阵，而不是立即定稿。投稿前仍建议补充或确认：

- Simu5G 多 seed 或多扰动统计。
- 关键消融：raw RL vs `RL + LateGuard`，domain randomization，reward/action hold/guard 阈值。
- 复杂度与部署成本：参数量、policy table 大小、查表耗时或推理耗时。
- Oracle 评分函数在移动性成功率 proxy 下的解释，避免把旧评分 Oracle 写成严格上界。

## 目录约定

```text
paper/
├─ manuscript/          # 论文正文草稿或 PaperSpine 中间成稿
├─ assets/
│  ├─ figures/source/   # 原始或可编辑图
│  ├─ figures/final/    # 最终候选图
│  ├─ tables/source/    # 口径解释、加权表、加工前表格
│  └─ tables/final/     # 论文主表和附表候选
├─ references/          # 文献调研、BibTeX/RIS 等
├─ notes/               # 写作备忘
└─ submission/          # 投稿模板、cover letter、审稿回复等
```
