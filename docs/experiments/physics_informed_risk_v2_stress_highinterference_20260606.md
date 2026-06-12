# Physics-informed Risk v2 Stress/HighInterference 专项记录（2026-06-06）

本文记录第二轮 PhysicsRisk 优化的专项归因、训练、dense policy table 导出和 Simu5G 验证结果。目标是解释第一轮 PhysicsRisk 在 `Stress` / `HighInterference` 场景中的尾部副作用，并验证“物理约束下的神经网络风险预测”是否能进一步内化 late handover guard。

## 目标状态

本轮目标：

```text
Stress / HighInterference 场景专项归因 + 第二轮 PhysicsRisk 模型优化
```

当前状态：

- 已完成第一轮 PhysicsRisk 与 late guard 的 Simu5G 差异诊断。
- 已定位一个主要工程因素：第一轮 PhysicsRisk sparse policy table 与 late guard 默认 dense table 的网格不一致。
- 已实现第二轮 action physics loss，用物理风险条件直接约束风险窗口中的动作偏好。
- 已完成 v2 训练、Python Stress 小样本评估、p2/p4 dense policy table 导出、Simu5G complex matrix 实跑和诊断。
- 已确认默认 `results/policy_tables/policy_table.csv` 恢复为 2026-06-04 `delta3_low5` late guard 表。

## 专项归因

第一轮 PhysicsRisk 的主推 p2 表为 sparse grid：

```text
results/policy_tables/policy_table_physics_risk_p2_seed20260606.csv
```

该表共 `337590` 行，而 late guard 默认表为 `830088` 行。差异主要来自导出网格：

- sparse 表只覆盖较窄速度/位置网格，典型为 `speed=300`、位置步长 `100 m`。
- late guard 默认表采用 dense grid，位置步长为 `50 m`，并覆盖 `200 / 300 / 350 / 400 km/h` 等速度。

这导致 Simu5G 查表时存在额外离散化/domain gap。专项诊断显示，sparse p2 在 `Stress` 中将首次切换推迟约 `1.0 s / 83.333 m`，packet loss 从 late guard 的 `0.282921` 上升到 `0.317553`，最大帧时延从 `2.946 s` 上升到 `3.848 s`。

将第一轮 checkpoint 重新导出为 dense p2 后，副作用明显收敛：

```text
results/policy_tables/policy_table_physics_risk_p2_dense_seed20260606.csv
results/simu5g/RailwayComplexMatrix-PhysicsRiskP2Dense-20260606/parsed
```

相对 late guard，dense p2 在 `Stress` 中仍晚切 `0.5 s / 41.6665 m`，packet loss 增加 `0.015361`，最大帧时延增加 `0.902 s`。这说明 sparse grid 是重要原因，但不是全部原因：网络策略在低 SINR 且邻区明显更强的风险窗口中仍可能保留偏长 TTT。

## 第二轮训练改动

训练入口：

```text
scripts/train/train_rainbow_offline.py
```

新增参数：

```text
--lambda_action_phys
```

第二轮在原有 Rainbow loss、C51 distributional loss、物理风险辅助头之外，加入 action physics loss。核心风险上下文为：

```text
serving SINR <= -5 dB 且 Delta RSRP >= 2 dB
```

在该上下文中，对所有动作施加物理偏好：

- `TTT >= 300 ms` 或 `Hys >= 4 dB` 的保守动作视为高晚切风险动作。
- 高风险动作目标风险提高到 `0.95`。
- 非风险动作目标风险压到 `0.25`。
- 用 `lambda_action_phys=0.1` 加入总 loss。

正式训练 run：

```text
experiments/runs/20260606_physics_risk_v2_actionphys_seed2026060612
```

训练命令：

```powershell
.\.venv\Scripts\python.exe scripts\train\train_rainbow_offline.py `
  --dataset_path data\datasets\offline_dataset_v1_domain_random_obs7_seed20260604.npz `
  --enable_physics_risk `
  --resume experiments\runs\20260606_physics_risk_obs7_seed20260606\checkpoints\rainbow_offline_best.pth `
  --no_cql `
  --num_epochs 30 `
  --samples_per_epoch 12000 `
  --batch_size 64 `
  --amp `
  --val_ratio 0.05 `
  --early_stop_on val `
  --early_stop_patience 8 `
  --seed 2026060612 `
  --train_log_interval 5 `
  --lambda_risk 0.2 `
  --lambda_phys 0.05 `
  --lambda_action_phys 0.1 `
  --run_dir experiments\runs\20260606_physics_risk_v2_actionphys_seed2026060612
```

产物：

```text
experiments/runs/20260606_physics_risk_v2_actionphys_seed2026060612/checkpoints/rainbow_offline_final.pth
```

本轮从第一轮 PhysicsRisk checkpoint 继续训练；由于 validation Rainbow loss 没有超过旧 best，只保存了 final checkpoint。训练日志中 `avg_action_physics_loss` 约为 `0.25 ~ 0.27`，说明物理动作约束确实参与了优化。

## Python Stress 小样本评估

评估 run：

```text
experiments/runs/20260606_physics_risk_v2_actionphys_stress_python10
```

只针对 `stress_radio_holdout`，`num_seeds=10`，PhysicsRisk 使用 `risk_penalty=2.0`。关键结果：

| policy | HO/km | outage | SINR p5 | overlap SINR | overlap interruption |
|---|---:|---:|---:|---:|---:|
| RL raw | 0.333 | 0.17670 | -12.997 | 6.188 | 0.00000 |
| RL + LateGuard | 1.333 | 0.13981 | -10.550 | 4.978 | 0.00000 |
| RL + PhysicsRisk v2 p2 | 0.533 | 0.15506 | -11.638 | 5.203 | 0.00000 |

结论：v2 p2 相对 raw RL 有改善，但在 Stress Python 小样本上仍没有超过 late guard。因此仅靠 `risk_penalty=2.0` 不足以替代 guard。

## Dense Policy Table 导出

新增导出工具：

```text
tools/export_physics_risk_policy_preset.py
```

该工具使用与 late guard 默认表一致的 dense grid：

```text
position: 0:3000:50
speed: 200, 300, 350, 400
RSRP: -120, -110, -100, -90, -80, -70, -60
Delta RSRP: -15, -10, -6, -3, 0, 3, 6, 10, 15
SINR: -10, -6, -3, 0, 3, 6, 10, 15, 20
time_since_ho: 0, 0.5, 1, 2, 5, 10
rows: 830088
```

本轮导出的 v2 表：

```text
results/policy_tables/policy_table_physics_risk_v2_actionphys_p2_dense_seed2026060612.csv
results/policy_tables/policy_table_physics_risk_v2_actionphys_p4_dense_seed2026060612.csv
```

在 `Stress` 风险窗口中，v2 p4 会把动作从第一轮 dense p2 常见的 `Hys=3.5 / TTT=650 ms` 压到 `Hys=3.0 / TTT=150 ms`，说明更高 `risk_penalty` 能把物理风险头的输出转化为实际动作变化。

## Simu5G 结果

### v2 p2 dense

Simu5G 输出：

```text
results/simu5g/RailwayComplexMatrix-PhysicsRiskV2ActionPhysP2Dense-20260606/parsed
experiments/runs/20260606_physics_risk_v2_actionphys_seed2026060612/diagnostics/stress_highinterference_v2_vs_lateguard
```

相对 late guard：

| scenario | 首次切换差异 | packet loss 差异 | SINR p5 差异 | max delay 差异 | 结论 |
|---|---:|---:|---:|---:|---|
| Stress | +0.5 s / +41.6665 m | +0.015361 | +0.112 dB | +0.902 s | 仍有晚切副作用 |
| HighInterference | -1.5 s / -124.9995 m | 0.000000 | +0.418 dB | 0.000 s | 与 late guard 主业务 KPI 持平 |

v2 p2 的 action physics loss 已经改变训练分布，但在 Simu5G 轨迹上仍不足以改变 Stress 的关键动作序列。

### v2 p4 dense

Simu5G 输出：

```text
results/simu5g/RailwayComplexMatrix-PhysicsRiskV2ActionPhysP4Dense-20260606/parsed
experiments/runs/20260606_physics_risk_v2_actionphys_seed2026060612/diagnostics/stress_highinterference_v2_p4_vs_lateguard
```

相对 late guard：

| scenario | 首次切换差异 | packet loss 差异 | SINR p5 差异 | max delay 差异 | 结论 |
|---|---:|---:|---:|---:|---|
| Stress | 0.0 s / 0.0 m | 0.000000 | 0.000 dB | 0.000 s | 主 KPI 与 late guard 对齐 |
| HighInterference | -1.5 s / -124.9995 m | 0.000000 | +0.418 dB | 0.000 s | 与 late guard 主业务 KPI 持平 |

`Stress` 时间线中，v2 p4 的首次切换发生在 `20.200 s / 1683.327 m`，风险点共 `12` 个。风险窗口内动作稳定为：

```text
Hys = 3.0 dB
TTT = 150 ms
action = 21
```

这与 late guard 的主 KPI 对齐，说明第二轮“物理约束风险头 + 更高风险惩罚”的组合确实可以把外部 guard 的核心行为部分内化到神经网络策略表中。

## 默认策略结论

当前默认策略表仍应保持 2026-06-04 `delta3_low5` late guard：

```text
results/policy_tables/policy_table.csv
results/policy_tables/policy_table.metadata.json
```

默认表恢复校验：

```text
SHA256 C7C79049314299D2DE3D01D976EBF98F665191CE62383A9E9CE15EF39F062584 results/policy_tables/policy_table.before_physics_risk_20260606.csv
SHA256 C7C79049314299D2DE3D01D976EBF98F665191CE62383A9E9CE15EF39F062584 results/policy_tables/policy_table.csv
```

不替换默认表的原因：

- v2 p4 已经能在 `Stress` 中对齐 late guard，但目前只是对齐，不是稳定超越。
- `HighInterference` 中固定 A3 仍可能优于 RL/late guard/p4，这更像干扰、链路质量或资源边界问题，不能仅靠切换参数解决。
- Python Stress 小样本中，v2 p2 仍弱于 late guard，说明风险头对不同惩罚系数较敏感。
- 当前 v2 只验证了 dense p2/p4 单次 Simu5G complex matrix，还需要更多 seed/profile 消融。

因此，本轮结论应表述为：

> 第二轮 PhysicsRisk 已经把 late guard 的核心物理风险逻辑更深地内化到神经网络动作选择中。特别是 v2 p4 dense 在 Simu5G Stress 场景中将关键风险窗口动作压到 `TTT=150 ms`，并使主 KPI 与 late guard 对齐。但现阶段证据仍不足以替换默认 late guard；推荐将 v2 p4 dense 保留为候选策略和论文消融结果。

## 下一步任务

1. 保持 `delta3_low5` late guard 为默认 Simu5G policy table。
2. 将 v2 p4 dense 纳入后续 profile/seed 扩展评估，重点检查是否引入额外切换、中断或尾部时延副作用。
3. 若继续推进模型内化 guard，下一轮应优先增强风险标签对业务尾时延、RLF proxy 和持续邻区优势的约束，而不是只提高 `risk_penalty`。
4. 在 Simu5G 侧优先尝试 `trace_window` 或在线历史窗口推理，减少 `repeat_current` 导出对 GRU 趋势判断的压缩。
