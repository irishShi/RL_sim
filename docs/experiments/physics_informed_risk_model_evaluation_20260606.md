# Physics-informed Risk 模型训练与评估记录（2026-06-06）

本文记录 `Obs7 + GRU + Rainbow + PhysicsRiskHead` 的第一轮正式训练、Python profile 对比、policy table 导出与 Simu5G 测试准备情况。

后续第二轮 Stress / HighInterference 专项归因、action physics loss、dense p2/p4 导出和 Simu5G 复测记录见：

```text
docs/experiments/physics_informed_risk_v2_stress_highinterference_20260606.md
```

## 训练

训练命令：

```powershell
.\.venv\Scripts\python.exe scripts\train\train_rainbow_offline.py `
  --dataset_path data/datasets/offline_dataset_v1_domain_random_obs7_seed20260604.npz `
  --enable_physics_risk `
  --resume experiments/runs/20260604_v1_domain_random_obs7_nocql_seed20260604/checkpoints/rainbow_offline_best.pth `
  --no_cql `
  --num_epochs 40 `
  --samples_per_epoch 12000 `
  --batch_size 64 `
  --amp `
  --val_ratio 0.05 `
  --early_stop_on val `
  --early_stop_patience 10 `
  --seed 20260606 `
  --train_log_interval 5 `
  --run_dir experiments/runs/20260606_physics_risk_obs7_seed20260606
```

训练结果：

- run：`experiments/runs/20260606_physics_risk_obs7_seed20260606`
- checkpoint：`experiments/runs/20260606_physics_risk_obs7_seed20260606/checkpoints/rainbow_offline_best.pth`
- 数据集：`data/datasets/offline_dataset_v1_domain_random_obs7_seed20260604.npz`
- 训练样本：`464127`
- 验证样本：`24428`
- 模型参数：`1,131,476`
- warm start：从 `20260604_v1_domain_random_obs7_nocql_seed20260604` 的 best checkpoint 加载主干，risk head 随机初始化
- 最佳 epoch：`4`
- best `val_rainbow_loss`：`3.865301`
- 早停：epoch `14/40`

风险头输出为 `[B, 48, 3]`，三个 horizon 为 `100 / 200 / 300 ms`。

## Python Test Split 对比

基础命令：

```powershell
.\.venv\Scripts\python.exe comparison_algorithms\scripts\run_comparison.py `
  --include_rainbow `
  --include_rainbow_late_guard `
  --include_rainbow_physics_risk `
  --checkpoint_path experiments/runs/20260606_physics_risk_obs7_seed20260606/checkpoints/rainbow_offline_best.pth `
  --q_table_path experiments/runs/20260604_comparison_algorithms_test20_v2/tables/tabular_q_policy.json `
  --profile_split test `
  --num_seeds 20 `
  --base_seed 28000
```

已完成三组 `risk_penalty` 扫描：

- `risk_penalty=1.0`：`experiments/runs/20260606_physics_risk_penalty1_comparison_test20`
- `risk_penalty=2.0`：`experiments/runs/20260606_physics_risk_comparison_test20`
- `risk_penalty=4.0`：`experiments/runs/20260606_physics_risk_penalty4_comparison_test20`

核心 KPI 汇总保存于：

```text
experiments/runs/20260606_physics_risk_obs7_seed20260606/metrics_key_python_comparison.csv
```

总体对比：

| run | policy | HO/km | pingpong | outage | SINR p5 | below -3 dB | overlap SINR | overlap interruption |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| p1 | RL raw | 0.571 | 0.000 | 0.06848 | -5.932 | 0.11272 | 6.269 | 0.00000 |
| p1 | RL + LateGuard | 0.973 | 0.013 | 0.05235 | -4.539 | 0.09419 | 6.110 | 0.00047 |
| p1 | RL + PhysicsRisk | 0.688 | 0.000 | 0.06206 | -5.368 | 0.10668 | 6.200 | 0.00000 |
| p2 | RL raw | 0.521 | 0.000 | 0.06535 | -5.727 | 0.11055 | 6.262 | 0.00000 |
| p2 | RL + LateGuard | 0.981 | 0.025 | 0.05315 | -4.549 | 0.09320 | 6.312 | 0.00084 |
| p2 | RL + PhysicsRisk | 0.983 | 0.000 | 0.05314 | -4.742 | 0.09527 | 6.150 | 0.00000 |
| p4 | RL raw | 0.571 | 0.000 | 0.06741 | -5.839 | 0.11298 | 6.346 | 0.00000 |
| p4 | RL + LateGuard | 0.940 | 0.013 | 0.05230 | -4.546 | 0.09140 | 6.261 | 0.00039 |
| p4 | RL + PhysicsRisk | 1.431 | 0.000 | 0.05008 | -4.179 | 0.08622 | 5.880 | 0.00161 |

解释：

- `risk_penalty=1.0` 仍偏保守，风险头介入不足。
- `risk_penalty=2.0` 与 `LateGuard` 的 outage 基本持平，且保持 `pingpong=0` 和重叠区中断为 `0`，是当前最均衡的 PhysicsRisk 配置。
- `risk_penalty=4.0` 更激进，outage 和 SINR p5 明显改善，接近固定 A3，但 HO/km 与重叠区中断上升。它适合作为“更重视尾部 SINR/outage”的备选，而不是默认策略。

当前 Python 侧推荐：

```text
主推：risk_penalty=2.0
备选：risk_penalty=4.0
```

## Policy Table 导出

已备份原默认表：

```text
results/policy_tables/policy_table.before_physics_risk_20260606.csv
results/policy_tables/policy_table.before_physics_risk_20260606.metadata.json
```

导出主推 p2：

```powershell
.\.venv\Scripts\python.exe scripts\export\export_policy_table.py `
  --checkpoint_path experiments/runs/20260606_physics_risk_obs7_seed20260606/checkpoints/rainbow_offline_best.pth `
  --use_physics_risk `
  --risk_penalty 2.0 `
  --output_path results/policy_tables/policy_table_physics_risk_p2_seed20260606.csv `
  --batch_size 4096
```

导出备选 p4：

```powershell
.\.venv\Scripts\python.exe scripts\export\export_policy_table.py `
  --checkpoint_path experiments/runs/20260606_physics_risk_obs7_seed20260606/checkpoints/rainbow_offline_best.pth `
  --use_physics_risk `
  --risk_penalty 4.0 `
  --output_path results/policy_tables/policy_table_physics_risk_p4_seed20260606.csv `
  --batch_size 4096
```

Simu5G 测试过程中曾分别将默认表切到 p2 与 p4；评估结束后已恢复为 2026-06-04 的 `delta3_low5` late guard 默认表：

```text
results/policy_tables/policy_table.csv
results/policy_tables/policy_table.metadata.json
```

PhysicsRisk 候选表行数均为 `337590`。原 late guard 默认表备份与当前默认表 hash 一致：

```text
results/policy_tables/policy_table.before_physics_risk_20260606.csv
results/policy_tables/policy_table.csv
```

## Simu5G 测试状态

已完成 p2 测试：

```powershell
wsl -d Ubuntu -- bash -lc "cd /mnt/c/Users/qichengshi/Desktop/RL_sim && bash tools/run_simu5g_complex_matrix.sh RailwayComplexMatrix-PhysicsRiskP2-20260606"

.\.venv\Scripts\python.exe tools\analyze_simu5g_complex_matrix.py `
  --raw_root results/simu5g/RailwayComplexMatrix-PhysicsRiskP2-20260606/raw `
  --parsed_root results/simu5g/RailwayComplexMatrix-PhysicsRiskP2-20260606/parsed `
  --window_s 2.0
```

已完成 p4 测试：

```powershell
wsl -d Ubuntu -- bash -lc "cd /mnt/c/Users/qichengshi/Desktop/RL_sim && bash tools/run_simu5g_complex_matrix.sh RailwayComplexMatrix-PhysicsRiskP4-20260606"

.\.venv\Scripts\python.exe tools\analyze_simu5g_complex_matrix.py `
  --raw_root results/simu5g/RailwayComplexMatrix-PhysicsRiskP4-20260606/raw `
  --parsed_root results/simu5g/RailwayComplexMatrix-PhysicsRiskP4-20260606/parsed `
  --window_s 2.0
```

输出路径：

```text
results/simu5g/RailwayComplexMatrix-PhysicsRiskP2-20260606/parsed/comparison_delta_rl_minus_fixed.csv
results/simu5g/RailwayComplexMatrix-PhysicsRiskP4-20260606/parsed/comparison_delta_rl_minus_fixed.csv
experiments/runs/20260606_physics_risk_obs7_seed20260606/simu5g_key_comparison_physics_risk_p2_p4_vs_lateguard.csv
```

### Simu5G p2 结果

| scenario | packet loss delta | SINR p5 delta | max delay delta | 结论 |
|---|---:|---:|---:|---|
| HeavyTraffic | -0.000698 | +0.021 dB | -0.010 s | 小幅改善 |
| HighInterference | +0.000558 | -4.711 dB | +0.224 s | 明显副作用 |
| HighNoise | -0.003350 | -1.891 dB | -0.006 s | 丢包改善但低分位 SINR 下降 |
| NlosFading | 0.000000 | -0.115 dB | -0.020 s | 基本持平 |
| Stress | +0.016897 | +0.172 dB | +2.170 s | 丢包和尾部时延变差 |
| TrackOffset | 0.000000 | +0.677 dB | -0.020 s | 改善 |
| WeakCoverage | -0.011167 | +2.633 dB | 0.000 s | 明显改善 |

p2 相比固定 A3 在 `WeakCoverage`、`HighNoise`、`HeavyTraffic`、`TrackOffset` 上有改善或部分改善，但在 `Stress` 和 `HighInterference` 上出现不可忽略的尾部副作用。尤其 `Stress` 的 app packet loss 从 fixed 的 `0.300656` 上升到 `0.317553`，最大帧时延从 `1.678 s` 上升到 `3.848 s`。

### Simu5G p4 结果

p4 与 p2 的 policy table 文件不同，但在本次 Simu5G 轨迹主 KPI 上基本相同：handover count、first handover、packet loss、SINR p5、max delay 等关键结果没有改善。p4 主要改变了运行中记录的平均 `Hys/TTT`，但没有改变这组场景里的关键切换结果，因此不能作为替代默认策略的证据。

### 与历史 late guard 对比

历史对照基线：

```text
experiments/runs/20260606_physics_risk_obs7_seed20260606/simu5g_baseline_lateguard_measured_sinr_delta3_20260604.csv
```

该文件来自：

```text
results/simu5g/RailwayComplexMatrix-V1LateGuardMeasuredSinrDelta3-20260604/parsed/comparison_delta_rl_minus_fixed.csv
```

与历史 late guard 相比，PhysicsRisk p2/p4 的 `WeakCoverage`、`HighNoise`、`HeavyTraffic` 等场景表现接近或一致；但历史 late guard 在 `Stress` 上更稳，packet loss 为 `0.282921`，而 PhysicsRisk p2/p4 为 `0.317553`。因此当前不能用 PhysicsRisk 表替换默认 late guard 表。

## 当前结论

第一轮 PhysicsRisk 模型已经完成训练、Python test split 评估、policy table 导出和 Simu5G complex matrix 实跑。相对原始 `RL_Rainbow_GRU`，PhysicsRisk 在 Python 侧明显降低 outage、改善 SINR p5，同时保持 `pingpong=0`。与 `LateGuard` 相比：

- `risk_penalty=2.0` 是最均衡方案：outage 基本持平、乒乓更低、重叠区中断为 0，但 SINR p5 略低于 LateGuard。
- `risk_penalty=4.0` 更偏风险规避：outage 和 SINR p5 更好，但切换次数和重叠区中断代价更高。

Simu5G 侧结论更保守：

- PhysicsRisk p2/p4 可证明“物理约束风险头已经能参与动作选择并导出可运行策略表”。
- 当前 PhysicsRisk 表没有在 Simu5G 主 KPI 上稳定超过 2026-06-04 的 `delta3_low5` late guard 表。
- `Stress` 与 `HighInterference` 的尾部副作用说明，现阶段不应移除 late guard，也不应把 PhysicsRisk 表设为默认线上策略。

当前不应表述为“PhysicsRisk 全面优于 LateGuard 或固定 A3”。更准确的表述是：

> PhysicsRisk 已经把原先外部 late guard 的风险判断部分内化进神经网络动作选择，在 Python holdout test split 上达到与 LateGuard 接近的多 KPI 均衡表现，并提供了可调风险偏好的 `risk_penalty` 控制旋钮。但第一轮 Simu5G 实跑显示其跨平台落地仍不够稳，尤其在 Stress / HighInterference 场景存在尾部副作用。因此当前定位应是“已打通并可继续优化的物理规则 + 数据联合驱动模型”，而不是替代默认 late guard 的最终策略。

下一步建议：

1. 保留 2026-06-04 `delta3_low5` late guard 作为默认 Simu5G policy table。
2. 对 PhysicsRisk 做 Stress / HighInterference 专项误差归因，重点检查风险标签是否只学到低 SINR 风险、而未充分约束业务尾时延和干扰场景下的切换副作用。
3. 下一轮训练考虑加入 Simu5G 风格的 domain gap 消融：更密 policy table grid、trace window 导出、以及 Stress/HighInterference 加权风险标签。
