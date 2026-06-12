# 文献到对比算法映射

本文档记录本目录 baseline 与相关研究的对应关系。当前实现优先服务于“可在本仓库 Python 场景复现”的横向比较，因此部分算法是工程代理版本，不声称完全复刻原论文全部系统模型。

## 实验指标边界

相关文献常用 `handover count`、`handover success/failure`、`ping-pong rate`、`outage/RLF`、`throughput`、`delay`、`SINR/RSRP` 等指标证明算法性能。映射到本项目时，应优先使用能反映切换决策质量的指标：`ho_per_km`、`ping_pong_count`、`outage_time_ratio`、`sinr_p5_db`、`overlap_zone_sinr_mean_db`、`comm_interruption_ratio_in_overlap_zone`。

如果某个场景属于弱覆盖、强干扰或 NLOS/fading 的中等退化，且邻区仍存在可利用质量优势，这些指标可以用于证明切换策略的缓解作用。如果极端场景的主要矛盾是两个候选小区都不可用、资源不足或业务排队，则这些指标只能说明切换策略在该场景下的有限作用，不能证明单靠 A3 参数优化可以解决系统侧问题。论文表述中应把这类极端场景作为边界分析或未来跨层优化方向。

## 1. TD/Q-learning 参数表

代表文献：

- *Parameter Adaptation and Situation Awareness of LTE-R Handover for High-Speed Railway Communication*, IEEE T-ITS 2022。
- *A Parameter Optimization Method for LTE-R Handover Based on Reinforcement Learning*, IWCMC 2020。

本目录实现：

- `TabularQ_A3`

近似方式：

- 状态离散化为 `Delta RSRP`、服务小区 SINR、位置、速度、距上次切换时间、服务小区、是否在重叠区。
- 动作沿用当前项目 48 个 `(Hys, TTT)`。
- 奖励直接使用 `TrainHandoverEnv` 的 R5 reward，不额外重写 KPI 目标。
- 使用 action hold，避免每步改参数导致 TTT timer 被反复清零。

## 2. 速度/多普勒自适应切换参数

代表文献：

- *Adaptive Handover Algorithm for LTE-R System in High-Speed Railway Scenario*, IEEE Access 2021。
- 《改进5G-R自适应高速铁路越区切换算法》，北京航空航天大学学报 2025。

本目录实现：

- `SpeedAdaptiveA3`

近似方式：

- 用速度分段函数代理速度/多普勒自适应。
- 速度越高，TTT 越短；Hys 略增以降低快衰落下的乒乓。
- 保持离散动作集合不变，输出会映射到最近的 48 动作。

## 3. 切换点/边界预测

代表文献：

- LTE-R Bayesian regression handover decision。
- 高铁 5G LSTM RSRP prediction handover enhancement。

本目录实现：

- `PositionPriorA3`

近似方式：

- 当前环境是一维双小区轨道，因此用归一化位置作为 cell boundary crossing 的可观测代理。
- 在接近轨道中点时逐步降低 Hys/TTT，已切到 B 后转为保守参数抑制回切。

## 4. 预测 + 自适应 Hys/TTT

代表文献：

- *Mobility Management in 5G and Beyond: A Novel Smart Handover with Adaptive Time-to-Trigger and Hysteresis Margin*。

本目录实现：

- `SignalTrendGuardA3`

近似方式：

- 用指数平滑估计 `Delta RSRP` 与趋势，作为轻量 Kalman-like predictor。
- 当预测邻区明显更强或服务小区 SINR 偏低时，选择更低 Hys 和更短 TTT。
- 当服务小区仍明显占优且 SINR 良好时，选择较保守参数减少无效切换。

## 5. 当前主线策略与上界

本目录实现：

- `RL_Rainbow_GRU`：加载当前 checkpoint，在同一批 profile/seed 上评估。
- `OracleA3_FullGrid`：每个 episode 枚举 48 组固定 A3 并按同一评分选优，只作为上界参考，不是可部署算法。

## 推荐论文实验层次

建议报告中按如下层次组织：

1. 固定 A3 网格/Oracle 上界。
2. 速度或位置规则自适应。
3. 信号预测式启发策略。
4. Tabular Q-learning。
5. 当前 `Obs7 + GRU + Rainbow DQN` 及 late guard 版本。
