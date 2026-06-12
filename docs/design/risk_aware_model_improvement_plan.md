# 风险感知模型改进目标与验证计划

本文档固化当前主线之后的模型改进目标。核心方向不是预测瞬时快衰落波形，而是在现有 `Obs7 + GRU + Rainbow DQN` 基础上，引入物理可解释的未来切换风险约束，使策略在弱覆盖、NLOS/fading、强干扰和高速场景下减少晚切风险，同时保持较低切换次数和乒乓。

## 改进定位

当前传播模型已经包含路径损耗、距离相关阴影衰落、动态同频干扰，以及可选 Rayleigh/Rician 快衰落。高铁场景中小尺度快衰落相干时间远小于当前 `50 ms` 决策周期和 `50~650 ms` TTT 尺度，因此主线不追求对瞬时 `h(t)` 的精确预测。

改进目标应转为预测和抑制切换风险：

```text
物理传播先验 + 测量滤波 + 数据驱动策略 + 风险保护层 -> A3 Hys/TTT 自适应控制
```

其中，物理规则负责识别低 SINR、邻区明显更强、高速退化等晚切风险；Rainbow 策略负责在正常状态下选择低切换、低乒乓的动作。

## 第一阶段目标

第一阶段采用轻量、可部署的 `RL_Rainbow_GRU_LateGuard`：

- 保持当前 checkpoint、观测维度、动作空间和 policy table 导出链路不变。
- 在每个 `50 ms` 测量点检查服务 SINR、`Delta RSRP`、速度、距上次切换时间。
- 当低 SINR 或快速退化风险出现时，限制过大 `Hys` 和过长 `TTT`。
- 在满足上限约束的动作集合内选择 Rainbow Q 值最高的动作。

这等价于一个 physics-informed action shield：

```text
低 SINR + 邻区优势明显 -> 限制 Hys/TTT -> 避免过保守晚切
```

## 验收指标

第一阶段不要求全面优于所有固定 A3 或 Oracle。进入主线的最低要求是：相对原始 `RL_Rainbow_GRU`，在 test split 上至少满足以下条件：

- `outage_time_ratio_mean` 明显下降。
- `sinr_p5_db_mean` 明显提升。
- `sinr_below_minus3db_ratio_mean` 下降或不恶化。
- `ping_pong_count_mean` 保持接近 0，不出现启发式策略式的频繁乒乓。
- `comm_interruption_ratio_in_overlap_zone_mean` 保持在固定 A3 和多数启发式基线以下。
- `ho_per_km_mean` 可适度上升，但应显著低于激进固定 A3、SpeedAdaptiveA3、TabularQ_A3 等策略。

相对较优的判据采用多 KPI 排序，而不是单一 SINR 均值。推荐排序优先级：

```text
outage_time_ratio, sinr_p5_db, comm_interruption_ratio_in_overlap_zone,
ping_pong_count, ho_per_km, overlap_zone_sinr_mean_db
```

## 对比对象

正式验证应至少包含：

- `FixedA3_Hys3_TTT150`
- `FixedA3_Hys2p5_TTT100`
- `SpeedAdaptiveA3`
- `PositionPriorA3`
- `SignalTrendGuardA3`
- `TabularQ_A3`
- `RL_Rainbow_GRU`
- `RL_Rainbow_GRU_LateGuard`

Oracle A3 只作为不可部署上界参考，不作为必须超过的目标。

## 后续阶段

若第一阶段验证通过，后续再进入更高成本改进：

1. 新增 Doppler-correlated fading profile，只作 holdout/消融，不替代默认 baseline。
2. 将快衰落先经过子步采样、测量平均和 L3 滤波，再进入 `50 ms` 观测。
3. 增加风险预测辅助头，预测未来 `100~300 ms` outage 概率、`SINR_p5` 或邻区持续优势概率。
4. 设计 R6 reward，把未来风险用于切换区奖励，而不是奖励不可控的瞬时快衰落。
5. 若 policy table 的 `repeat_current` 近似限制明显，再考虑 `trace_window` 或轻量在线风险服务。

## 当前第一阶段实现入口

- 策略类：`comparison_algorithms/baseline_policies.py` 中的 `RL_Rainbow_GRU_LateGuard`
- 评估入口：`comparison_algorithms/scripts/run_comparison.py --include_rainbow --include_rainbow_late_guard`
- guard 参数：默认采用 `delta3_low5` 风格预设，即低 SINR 和邻区优势足够明显时限制 `TTT <= 150 ms`，并压低 `Hys` 上限。

