# 阶段性进展、下一阶段规划与投稿评估（2026-06-05）

本文档整理当前 `RL_sim` 项目的阶段性进展、下一阶段工作规划，以及作为期刊论文投稿国内铁路核心 / CSCD 核心和国际 JCR 期刊的可行性评估。

## 1. 当前工作进展

### 1.1 问题边界已明确

项目当前定位为高速铁路蜂窝/5G-R 场景下的 A3 `Hys/TTT` 自适应切换优化：

- 主要解决晚切、过早切、乒乓切换、切换中断和 outage 之间的折中。
- 弱覆盖、强干扰、NLOS/fading 的中等退化仍属于鲁棒切换需要覆盖的对象。
- 当两个候选基站都质量很差、目标小区不可用，或主要瓶颈来自资源不足 / 业务排队时，不应为了极端少数样本盲目扩大模型。

相关说明已经写入：

- `docs/design/problem_scope_and_boundaries.md`
- `README.md`
- `AGENTS.md`
- `docs/design/README_MODEL.md`

### 1.2 主模型与训练链路已成型

当前主线模型保持为 `Obs7 + GRU + Rainbow DQN`：

- 输入：15 步历史窗口，每步 7 维观测。
- 动作：48 个离散 `(Hys, TTT)`。
- 奖励：默认 `R5`，聚焦切换重叠区。
- 数据：`data/datasets/offline_dataset_v1_domain_random_obs7_seed20260604.npz`
- checkpoint：`experiments/runs/20260604_v1_domain_random_obs7_nocql_seed20260604/checkpoints/rainbow_offline_best.pth`

已经完成多 profile domain randomization 训练，覆盖 `normal / weak_coverage / high_noise / nlos_fading / high_interference / mixed_mild` 等训练 profile，并用 holdout profile 做泛化评估。

### 1.3 Python 与 Simu5G 评估已打通

已经完成：

- Python profile 泛化评估。
- 100 场景 batch comparison。
- Simu5G `policy_table.csv` 查表控制 A3 `Hys/TTT`。
- Simu5G 复杂场景矩阵验证。
- late handover guard 导出与 measured-SINR 触发改进。

当前结论是：

- RL 明显降低激进 A3 的切换次数和乒乓。
- 原始 V1 策略偏长 TTT / 少切换，存在晚切风险。
- late guard 在 Python holdout profile 中降低 outage、改善 SINR p5；在 Simu5G 中对 WeakCoverage、NlosFading、HeavyTraffic 等有改善或持平。
- Stress 场景仍不能简单声称全面优于固定 A3，需要做归因。

### 1.4 对比算法复现框架已建立

已新增 `comparison_algorithms/`，包含：

- 固定 A3。
- 速度自适应 A3。
- 位置先验 A3。
- 信号趋势保护 A3。
- Tabular Q-learning。
- 当前 Rainbow GRU。
- Oracle A3 全网格上界。

已完成 4 个 holdout profile × 20 seeds 的对比实验：

- run：`experiments/runs/20260604_comparison_algorithms_test20_v2`
- 结果文档：`comparison_algorithms/docs/comparison_experiment_20260604.md`

当前对比表明：`RL_Rainbow_GRU` 在切换次数、乒乓和重叠区中断方面较强，但 SINR p5 / outage 在部分 profile 上仍不如固定 A3 或 Oracle。

## 2. 当前短板

如果按期刊论文要求，目前主要短板是：

1. `Rainbow + late guard` 尚未和所有 baseline 在同一 comparison runner 中正式重跑。
2. Simu5G 复杂矩阵目前更像工程验证，随机种子 / 重复次数和置信区间还不足。
3. 缺少系统消融：无 domain randomization、无 GRU、无 R5、无 guard、不同 action hold、不同 guard 阈值等。
4. 当前 Simu5G 查表默认 `repeat_current`，对 GRU 历史信息表达不完整；若论文强调时序模型，需要解释或补充 `trace_window` / 在线推理。
5. 需要增加复杂度和效率分析：参数量、推理耗时、policy table 大小、部署延迟。
6. 需要将文献指标映射为正式论文 KPI 表，并统一主指标、辅助指标和边界指标。

## 3. 下一阶段工作规划

### 阶段 A：补齐论文核心实验

优先级最高，建议先做：

1. 在 `comparison_algorithms` 中加入 `Rainbow + late guard` 策略。
2. 同一批 test profile / seeds 重跑所有 baseline。
3. 每个策略输出均值、标准差、95% CI。
4. 增加核心 KPI：
   - `ho_per_km`
   - `ping_pong_count`
   - `outage_time_ratio`
   - `sinr_p5_db`
   - `overlap_zone_sinr_mean_db`
   - `comm_interruption_ratio_in_overlap_zone`
   - 可选 `effective_handover_rate`

目标：形成一张可直接放论文的主结果表。

### 阶段 B：做小规模消融

建议控制复杂度，只做最能支撑结论的消融：

1. `RL raw` vs `RL + late guard`
2. `single scenario training` vs `domain randomization`
3. `R5` vs 历史 `R3/R4` 或已有旧 checkpoint
4. `GRU` vs 非时序 MLP / repeat-current 近似
5. guard 阈值：`delta3_low5` vs 较保守/较激进预设

目标：证明改进不是只靠一个 heuristic，而是训练分布、奖励设计和安全层共同产生效果。

### 阶段 C：Simu5G 论文级验证

建议不要继续无限扩大场景矩阵，而是聚焦：

1. Normal / WeakCoverage / NlosFading / HighInterference / Stress。
2. FixedA3、RL raw、RL + guard 三类策略。
3. 每个场景至少多 seed 或多初始扰动重复，给出均值和区间。
4. 对 Stress 做时间线归因：低 SINR 是晚切导致，还是两个候选基站都不可用 / 业务排队导致。

目标：把 Simu5G 从“工程打通证明”提升为“论文验证证据”。

### 阶段 D：成文与投稿准备

论文建议结构：

1. 高速铁路 5G-R 切换问题与极端工况边界。
2. A3 `Hys/TTT` 动态决策 MDP 建模。
3. `Obs7 + GRU + Rainbow DQN` 策略与 R5 切换区奖励。
4. domain randomization 与 holdout profile。
5. late guard 安全约束。
6. Python baseline 对比。
7. Simu5G 迁移验证。
8. 算法边界与工程部署讨论。

## 4. 工作量与创新性评估

### 4.1 对国内 CSCD / 铁路核心

当前工作量已经接近国内铁路核心期刊的基本要求，但还需要补齐正式对比、消融和统计分析。

创新性可以概括为：

1. 面向高速铁路 / 5G-R 切换区的 A3 参数自适应，而不是一般蜂窝场景。
2. 使用 `Obs7 + GRU + Rainbow DQN` 处理高铁移动中的时序无线观测。
3. 提出切换区聚焦型 R5 奖励，减少区外无关 SINR 噪声对训练的干扰。
4. 通过多 profile domain randomization 覆盖弱覆盖、干扰、噪声、NLOS/fading 等退化。
5. 将 RL 策略导出为 Simu5G 可部署 policy table，并加入轻量 late guard。
6. 构建文献对应 baseline 与 Oracle 上界对比。

判断：

- 投 `铁道科学与工程学报`：较可行，建议作为国内优先目标之一。
- 投 `中国铁道科学`：可行但需要更突出铁路工程背景、5G-R 实用价值和 Simu5G 验证。
- 投 `铁道学报`：有机会但门槛更高，需要更强的工程验证、严谨统计和更清楚的铁路通信贡献。
- 也可考虑 `西南交通大学学报`、`交通运输工程学报` 等交通工程类核心，但要看栏目对通信/智能算法的接受度。

### 4.2 对 IEEE / 国际 JCR

当前版本若直接投 IEEE Transactions 级别期刊，创新性和理论深度还不够稳。问题不在工作量，而在“Rainbow DQN + guard”的算法理论新意有限，国际高水平通信期刊会要求更强的泛化验证、理论分析或更接近真实网络的系统建模。

较现实的国际路线：

- `IEEE Access`：应用导向、跨领域，适合“高铁 5G-R + RL handover + Simu5G 验证”的工程型论文。
- `Journal on Wireless Communications and Networking`：无线通信与网络应用方向匹配，可考虑。
- `IET Communications`：通信方向合适，难度中等。
- `Wireless Networks` / `Mobile Networks and Applications`：可考虑，但需要把贡献写成移动网络 / mobility management，而不是纯铁路仿真。

较高风险路线：

- `IEEE Transactions on Vehicular Technology`：范围匹配，甚至包含 railway communications and networking，但竞争强，需要更强理论、更多真实/Simu5G验证和更充分 SOTA 对比。
- `IEEE Transactions on Intelligent Transportation Systems`：如果只做切换参数优化，交通系统层面贡献可能偏弱；除非扩展到列车运行、业务 QoS、移动性管理与智能交通耦合。

## 5. 推荐投稿路线

建议采用“两级目标”：

### 主路线：国内铁路核心 / CSCD

先按国内核心标准打磨一篇完整中文论文：

1. 第一目标：`铁道科学与工程学报`
2. 第二目标：`中国铁道科学`
3. 冲刺目标：`铁道学报`

国内版本重点强调铁路场景、5G-R 切换区、工程可部署性、Simu5G 验证和算法边界。

### 拓展路线：国际 JCR

如果后续补齐消融、统计和 Simu5G 多 seed 验证，可以转写英文版：

1. 稳妥：`IEEE Access`
2. 稳妥/中等：`Journal on Wireless Communications and Networking`
3. 中等：`IET Communications`
4. 冲刺：`IEEE Transactions on Vehicular Technology`

英文版需要更重视：

- SOTA 对比。
- 统计显著性。
- 复杂度分析。
- 可复现实验设置。
- 清晰区分 `RL raw`、`RL + guard`、Oracle 与固定 A3。

## 6. 参考信息

- CSCD 2025-2026 来源期刊列表显示：`铁道科学与工程学报`、`铁道学报`、`中国铁道科学` 均为核心库。
- `铁道学报` 官网介绍其被 EI Compendex、Scopus、CSCD 等数据库收录。
- 万方期刊信息显示 `铁道科学与工程学报` 为 CSCD核心、EI、北大核心等。
- IEEE Access 官方页面显示其覆盖 IEEE 全领域，2024 JCR Journal Impact Factor 为 3.6。
- IEEE TVT 官方 scope 明确包含 mobility management、machine learning for wireless communications，以及 railway communications and networking。
- Springer 的 Journal on Wireless Communications and Networking aims and scope 强调无线通信与网络技术的理论和应用。

