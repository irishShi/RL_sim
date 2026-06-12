# 物理规则约束的未来切换风险预测：任务要求与目标

本文档将下一阶段模型改进固化为可执行任务。目标不是继续扩大主干网络，也不是预测瞬时快衰落波形，而是在现有 `Obs7 + GRU + Rainbow DQN` 基础上，引入基于物理规则约束的神经网络未来风险预测，使策略能提前识别晚切、低 SINR 和 outage 风险。

## 总体目标

当前主线是：

```text
Rainbow Q 网络选动作 + late guard 规则后处理
```

下一阶段目标是：

```text
Obs7 历史窗口
-> 物理派生特征
-> GRU/Rainbow 共享表征
-> Q 值预测 + 未来风险预测
-> Q-risk 联合动作选择
-> late guard 仅保留为极端状态兜底
```

第一阶段必须保持外部观测维度 `[15, 7]` 不变，不新增 Simu5G 测量输入。物理特征从历史窗口内部派生。

## 第一阶段功能要求

1. 从 `Obs7` 历史窗口派生物理特征：
   - `SINR` 短窗标准差、低分位、最小值。
   - `Delta RSRP` 短窗斜率。
   - `Delta RSRP` 持续为正或持续超过门限的比例。
   - 短窗 effective fading margin。
   - pseudo L3 residual，即当前测量与 EMA/L3-like 平滑值的差值。

2. 新增神经网络风险头：
   - 复用 GRU 和 shared feature。
   - 输入为 `concat(shared_feature, physics_features)`。
   - 输出为 `risk_logits[s, action, horizon]`。
   - 第一版预测窗口为 `100 ms / 200 ms / 300 ms`。

3. 构造未来风险监督标签：
   - 从离线轨迹顺序、`done` 边界和现有归一化观测中生成。
   - 标签聚焦 `future_late_outage_risk`。
   - 第一版只监督实际执行动作对应的风险，未执行动作由物理约束损失约束。

4. 加入物理一致性损失：
   - `300 ms` 风险不应低于 `200 ms`，`200 ms` 风险不应低于 `100 ms`。
   - 在低 SINR 且邻区明显更强时，长 `TTT` / 大 `Hys` 动作风险不应低于短 `TTT` / 小 `Hys` 动作。
   - 在 critical SINR 且邻区有优势时，高延迟动作风险不应过低。

5. 推理策略后续目标：
   - 新增 `RL_Rainbow_GRU_PhysicsRisk`。
   - 动作选择从 `argmax Q(s,a)` 扩展为 `argmax [Q(s,a) - lambda * Risk(s,a)]`。
   - 在风险头验证稳定前，late guard 不删除，只降级为安全兜底。

## 验收目标

第一阶段完成后，至少应能证明：

- 离线训练脚本可开启 physics risk head 并正常完成 smoke train。
- 风险头输出维度为 `[batch, 48, 3]`。
- 风险损失和物理约束损失进入训练日志。
- 不开启 risk head 时，旧 `RainbowWithForecast` 训练链路保持兼容。
- 在 test split 对比中，`PhysicsRisk` 或 `PhysicsRisk+Guard` 至少不弱于当前 `RL_Rainbow_GRU_LateGuard` 的多 KPI 均衡表现，重点关注 `outage_time_ratio`、`sinr_p5_db`、`ping_pong_count`、`ho_per_km` 和重叠区中断。

## 当前实现顺序

1. 增加物理派生特征模块。
2. 增加未来风险标签生成模块。
3. 增加 `PhysicsRiskHead` 和 `RainbowWithPhysicsRisk`。
4. 改造离线训练入口，支持 `--enable_physics_risk`。
5. 完成编译检查和小批量 smoke 验证。
6. 再进入正式数据重训与 profile 对比。
