# 高速铁路 5G-R/LTE-R 切换优化文献调研

生成日期：2026-06-11

## 1. 调研范围与筛选原则

本文献表围绕本项目的核心问题筛选：高速铁路相邻小区重叠区内，基于 A3 事件的 `Hys/TTT` 自适应控制，以及强化学习策略在降低晚切、乒乓、outage 和切换中断中的作用。

筛选优先级如下：

1. IEEE 期刊/会议论文，特别是 IEEE T-ITS、IEEE TMC、IEEE Access、IEEE Communications Magazine、VTC/IWCMC 等。
2. 国内铁路、通信领域核心期刊或高水平高校学报论文，特别是高速铁路通信、5G-R/LTE-R、越区切换、信道/场景建模相关工作。
3. 与本文对比实验方法直接对应的论文，包括固定 A3 参数优化、速度/多普勒自适应、位置/边界预测、信号预测 + RL、Tabular Q-learning、Rainbow DQN、离线 RL/CQL。

说明：CNKI 检索入口在本次调研中触发安全验证，因此国内文献只正式列入已通过期刊官网或公开题录核验到作者、刊名、年份、页码或 DOI 的条目。若只在百度学术、万方摘要或搜索片段中出现，放入“待复核候选”而不作为正式参考文献。

## 2. 推荐正式纳入的核心文献

### 2.1 与本文对比方法直接相关

| 编号 | 文献 | 类型/质量 | 关键词 | 与本文关系 |
|---|---|---|---|---|
| R1 | Cheng Wu, Xingqiang Cai, Jie Sheng, Ziwen Tang, Bo Ai, Yiming Wang. "Parameter Adaptation and Situation Awareness of LTE-R Handover for High-Speed Railway Communication." IEEE Transactions on Intelligent Transportation Systems, 23(3):1767-1781, 2022. DOI: 10.1109/TITS.2020.3026195. | IEEE T-ITS 期刊 | LTE-R, 高铁切换, 参数自适应, situation awareness, TD value cube | 最贴近本文的铁路切换参数自适应论文，可支撑 `TabularQ_A3`、状态离散化、`Hys/TTT` 在线调参和高铁专用通信场景。 |
| R2 | Xingqiang Cai, Cheng Wu, Jie Sheng, Jin Zhang, Yiming Wang. "A Parameter Optimization Method for LTE-R Handover Based on Reinforcement Learning." 2020 International Wireless Communications and Mobile Computing (IWCMC), IEEE, 2020. IEEE Xplore document 9148194. | IEEE 会议 | LTE-R, Q-learning, handover parameter optimization | 可直接对应本文 `TabularQ_A3`。本文实现是工程代理版本：状态/动作/奖励按本仓库环境重写，不声称逐项复现原文。 |
| R3 | Yong Chen, Kaiyu Niu, Zhen Wang. "Adaptive Handover Algorithm for LTE-R System in High-Speed Railway Scenario." IEEE Access, 2021. IEEE Xplore document 9406799. | IEEE Access 期刊 | LTE-R, adaptive handover, high-speed railway, random suppression | 可支撑 `SpeedAdaptiveA3` 一类速度/场景自适应切换基线。公开检索能确认题名、作者、IEEE Access 与文档号；正式投稿前建议从 IEEE Xplore 导出 BibTeX 补齐卷页/DOI。 |
| R4 | Raja Karmakar, Georges Kaddoum, Samiran Chattopadhyay. "Mobility Management in 5G and Beyond: A Novel Smart Handover with Adaptive Time-to-Trigger and Hysteresis Margin." IEEE Transactions on Mobile Computing, 2022/2023. arXiv:2207.01706. DOI: 10.1109/TMC.2022.3188212. | IEEE TMC 期刊 | 5G, smart handover, adaptive TTT, hysteresis, Kalman filter, SARSA | 对应 `SignalTrendGuardA3` 的“信号质量预测 + 自适应 TTT/Hys”思想，也可作为本文 late guard 的方法背景。 |
| R5 | Thomas Jansen, Irina Balan, John Turk, Ingrid Moerman, Thomas Kürner. "Handover Parameter Optimization in LTE Self-Organizing Networks." IEEE Vehicular Technology Conference, 2010. IEEE Xplore document 5594245. | IEEE VTC 会议 | LTE SON, hysteresis, TTT, handover optimization | 固定 A3、激进 A3、Oracle A3 网格的基础对照文献，说明 `Hys/TTT` 是经典可优化 HO 控制参数。 |
| R6 | Volodymyr Mnih et al. "Human-level control through deep reinforcement learning." Nature, 518:529-533, 2015. | Nature | DQN, deep RL, experience replay, target network | 本文 Rainbow DQN 的基础源流，可在方法部分简述 DQN 价值函数学习。 |
| R7 | Hado van Hasselt, Arthur Guez, David Silver. "Deep Reinforcement Learning with Double Q-learning." AAAI, 2016. arXiv:1509.06461. | AAAI | Double DQN, overestimation bias | Rainbow 中 Double Q-learning 组件的来源。 |
| R8 | Marc G. Bellemare, Will Dabney, Rémi Munos. "A Distributional Perspective on Reinforcement Learning." ICML, 2017. | ICML | distributional RL, C51 | Rainbow 中 C51 distributional head 的来源。 |
| R9 | Matteo Hessel et al. "Rainbow: Combining Improvements in Deep Reinforcement Learning." AAAI, 2018. arXiv:1710.02298. | AAAI | Rainbow DQN, dueling, distributional, noisy, prioritized replay | 本项目 `RainbowWithForecast` 的核心算法出处。 |
| R10 | Aviral Kumar, Aurick Zhou, George Tucker, Sergey Levine. "Conservative Q-Learning for Offline Reinforcement Learning." NeurIPS, 2020. arXiv:2006.04779. | NeurIPS | offline RL, CQL, Q-value conservatism | 对应本项目早期 `--use_cql`/`--no_cql` 消融和离线训练风险讨论。当前推荐基线为 `--no_cql`，但 CQL 仍可作为离线 RL 背景。 |

### 2.2 高速铁路/5G-R 场景、信道与切换背景

| 编号 | 文献 | 类型/质量 | 关键词 | 与本文关系 |
|---|---|---|---|---|
| R11 | Ruisi He, Bo Ai, Zhangdui Zhong, Mi Yang, Ruifeng Chen, Jianwen Ding, Zhangfeng Ma, Guiqi Sun, Changzhu Liu. "5G for Railways: the Next Generation Railway Dedicated Communications." IEEE Communications Magazine, 2022. IEEE Xplore document 9895381; arXiv:2207.03127. | IEEE Communications Magazine | 5G-R, railway dedicated communications, intelligent railway | 可作为 5G-R 背景总述，解释为何铁路专用通信从 GSM-R/LTE-R 走向 5G-R，以及高可靠低时延需求。 |
| R12 | Ruisi He, Bo Ai, Zhangdui Zhong, Mi Yang, Chen Huang, Ruifeng Chen, Jianwen Ding, Hang Mi, Zhangfeng Ma, Guiqi Sun, Changzhu Liu. "Radio Communication Scenarios in 5G-Railways." IEEE Transactions on Intelligent Transportation Systems, 2023. IEEE Xplore document 10122814; arXiv:2105.01511. | IEEE T-ITS 期刊 | 5G-R scenarios, railway propagation, scenario classification | 支撑本文 `scenario_profiles.yaml` 的多 profile/domain randomization 思路：铁路场景差异会显著影响信道与系统评估。 |
| R13 | Shengfeng Xu, Gang Zhu, Bo Ai, Zhangdui Zhong. "A Survey on High-Speed Railway Communications: A Radio Resource Management Perspective." Computer Communications, 2016; arXiv:1603.05368. | 综述 | HSR communications, RRM, mobility management, resource allocation | 用于相关工作开头，说明高铁无线通信中的移动性管理、资源分配与 QoS 约束。 |
| R14 | Yang Lu, Ke Xiong, Zhuyan Zhao, Pingyi Fan, Zhangdui Zhong. "Remote Antenna Unit Selection Assisted Seamless Handover for High-Speed Railway Communications with Distributed Antennas." IEEE VTC-Spring, 2016. DOI: 10.1109/VTCSpring.2016.7504445; arXiv:1603.06461. | IEEE VTC 会议 | HSR handover, RAU selection, seamless handover, interruption probability | 支撑高速铁路场景下切换失败概率、通信中断概率是关键 KPI；与本文 `comm_interruption_ratio_in_overlap_zone` 指标相呼应。 |
| R15 | 刘云毅, 赵军辉, 王传云. "高速铁路宽带无线通信系统越区切换技术." 电信科学, 33(11):37-46, 2017. DOI: 10.11959/j.issn.1000-0801.2017278. | 国内通信核心期刊 | 高速铁路, 越区切换, 无线通信, LTE-R/5G-R | 国内中文背景综述。可引用其对高铁越区切换困难、切换失败、频繁切换、群切换和 QoS 的总结。 |
| R16 | 陈永, 康婕, 陶瑄. "改进5G-R自适应高速铁路越区切换算法." 北京航空航天大学学报, 51(3):724-731, 2025. DOI: 10.13700/j.bh.1001-5965.2023.0148. | 国内高校学报/核心候选 | 5G-R, 多普勒频移, 自适应切换, Hys/TTT 动态函数 | 可支撑 `SpeedAdaptiveA3` 与多普勒/速度相关自适应思想；注意该文主要是解析/函数型自适应，不是强化学习。 |
| R17 | 申瑜, 欧盼, 陈付坤, 单馨漪, 何丹萍, 钟章队. "智能超表面辅助的5G高铁场景信道特性." 北京交通大学学报, 47(2):23-35, 2023. DOI: 10.11860/j.issn.1673-0291.20220098. | 国内高校学报 | 5G-R, 高铁场景, 信道特性, 射线跟踪, RIS | 用于说明 5G-R 高铁信道具有场景依赖和复杂传播特性。本文不研究 RIS，但可作为场景/信道复杂性的国内支撑。 |

## 3. 本文对比实验方法与文献对应关系

| 本项目策略/方法 | 推荐引用 | 对应关系 | 写作注意 |
|---|---|---|---|
| `FixedA3_Hys3_TTT150`、`FixedA3_Hys2p5_TTT100` | R5；3GPP A3 事件规范可另补 TS 36.331/38.331 | 固定 `Hys/TTT` 是 LTE/NR 切换参数优化研究的基本对象。 | 固定参数不是某一篇论文独有算法，可作为工程基线。 |
| `OracleA3_FullGrid` | R5 | 枚举固定 A3 参数网格，作为离线上界参考。 | 不是可部署算法，不要表述为真实 oracle mobility manager。当前评分函数未纳入 `late_ho_failure_count`，在移动性成功率 proxy 下不是严格上界。 |
| `SpeedAdaptiveA3` | R3, R16 | 用速度/多普勒或高铁场景特性调整切换参数。 | 本仓库实现是分段函数代理，不复现原文随机抑制或动态函数全部细节。 |
| `PositionPriorA3` | R14，以及高铁固定轨道/可预测移动性的相关工作 | 高铁一维轨道和小区边界可预测，位置先验可用于提前切换。 | 现有实现用归一化位置代理切换点预测；建议表述为 geometry/position-prior baseline。 |
| `SignalTrendGuardA3` | R4 | 用信号预测或滤波估计服务/邻区质量趋势，并动态调整 TTT/Hys。 | 本文实现为指数平滑趋势保护，不训练 SARSA；可写成 light-weight predictor baseline。 |
| `TabularQ_A3` | R1, R2 | 离散状态 + 强化学习表格价值函数，输出 A3 参数。 | 本文使用环境 R5 reward 和 48 个动作空间，不逐项复现原文。 |
| `RL_Rainbow_GRU` | R6-R9 | 深度强化学习价值函数，结合 GRU 历史窗口、C51/Rainbow 组件选择 A3 参数。 | Rainbow 是通用 DRL 算法，本文贡献在铁路切换观测、动作、reward 与 domain randomization。 |
| `Rainbow-GRU + late guard` | R4, R15, R16 | 在 RL 动作后加低 SINR/邻区优势/高速退化风险保护，限制过大 Hys 和过长 TTT。 | guard 是本文工程安全层，不应说成原论文算法；可作为物理约束或风险兜底。 |
| 离线训练、CQL 消融 | R10 | 离线 RL 中保守 Q 函数可降低分布外动作过估计风险。 | 当前推荐实验是 `--no_cql`，因此 CQL 只作为背景和消融说明。 |

## 4. 可直接服务于论文写作的相关工作结构

建议相关工作按如下逻辑组织：

1. 高速铁路/5G-R 场景特性：引用 R11-R13、R15、R17，说明高铁高速移动、场景差异、时变信道、弱覆盖/NLOS/强干扰使切换优化具有必要性。
2. 传统与自适应切换参数优化：引用 R5、R15、R16，说明 `Hys/TTT`、切换成功率、乒乓和中断是经典指标；固定参数难以适应速度、地形与无线退化。
3. 铁路场景强化学习切换：引用 R1、R2，说明已有 LTE-R 研究已尝试用 RL/TD/Q-learning 进行参数自适应，但多为表格/低维状态或特定场景验证。
4. 预测式智能切换与 5G mobility management：引用 R4，说明基于信号预测和 RL 的 adaptive TTT/Hysteresis 是 5G/B5G mobility management 的重要方向。
5. 本文差异：强调本文聚焦 5G-R/高速铁路相邻小区重叠区，使用 `Obs7 + GRU + Rainbow DQN`，通过多 profile domain randomization 与 holdout 泛化评估，并加入 late handover guard；不要把方法表述为覆盖增强、干扰抑制或资源调度方案。

## 5. 文献质量与使用建议

- 第一优先引用：R1、R4、R9、R11、R12、R15、R16。这些最能支撑本文问题、方法和实验对比。
- 对比实验必须引用：`TabularQ_A3` 用 R1/R2，`SignalTrendGuardA3` 用 R4，`SpeedAdaptiveA3` 用 R3/R16，固定/Oracle A3 用 R5。
- 国内文献建议至少引用 R15 与 R16。R17 可用于说明 5G 高铁信道/场景复杂性，但不要把 RIS 作为本文算法相关技术。
- R2、R3 的正式 BibTeX 建议后续从 IEEE Xplore 导出，以补齐 DOI、卷期页码。当前已核验 IEEE 文档号和公开题录，但本文件不补写未确认的 DOI。

## 6. 待复核候选，不建议直接写入正式参考文献

以下条目在公开检索中出现，但本次没有通过期刊官网/CNKI/万方完整题录核验。若后续能登录 CNKI 或 IEEE Xplore，可继续补齐。

| 候选题名 | 检索线索 | 可能用途 |
|---|---|---|
| 高速铁路 LTE-R 改进切换算法的研究 | 搜索片段显示基于 3GPP A3 事件、测量周期、切换时延分析 | 可能支撑传统 A3 改进算法，但需核验作者、刊名、年份。 |
| 基于速度触发的提前切换算法在 LTE-R 中的应用研究 | 搜索片段显示来自《电子与信息学报》PDF，疑似 DOI: 10.11999/JEIT150577 | 若核验成功，可作为中文通信核心期刊中速度触发/提前切换支撑。 |
| 铁路 5G-R 网络特点及网络规划技术研究 | 搜索片段显示《中国铁路》2022 年第 09 期 | 可用于 5G-R 网络规划背景，但需核验作者和页码。 |
| 铁路 5G-R 系统关键技术创新 | 万方搜索片段显示 5G-R 系统核心网/无线网组网关键技术 | 可用于国内 5G-R 系统背景，但需核验刊名、作者、年份。 |

## 7. 检索记录

已核验来源包括：

- IEEE Xplore 搜索结果与文档号：9215041、9148194、9406799、9895381、10122814、5594245、7504445、9813555。
- DBLP 条目：R1 的作者、出处、页码和 DOI。
- arXiv 条目：R4、R11、R12、R13、R14 的作者、摘要、arXiv 编号和部分期刊/会议信息。
- 国内期刊官网：电信科学、北京航空航天大学学报、北京交通大学学报。

CNKI 入口在本次访问时出现安全验证，因此未直接抓取 CNKI 题录。正式投稿前，建议用机构网络从 CNKI/万方导出国内文献 RIS/BibTeX，并补全所有中文参考文献格式。
