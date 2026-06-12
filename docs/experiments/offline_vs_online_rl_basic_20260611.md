# 离线学习 vs 在线学习基本对比证明（2026-06-11）

本文记录一个面向论文写作的轻量对比实验，用于说明当前项目采用离线强化学习路线的工程合理性。实验聚焦三点：安全性、训练效率/交互样本效率、训练效果。

注意：本实验不是为了证明离线 RL 在理论上全面优于所有在线 RL，而是证明在高铁通信切换这种安全敏感场景中，`offline training + holdout validation + late guard` 是更稳妥的工程部署路线。

## 实验入口

新增脚本：

```text
scripts/eval/compare_offline_online_rl.py
```

正式 run：

```text
experiments/runs/20260611_offline_vs_online_rl_basic
```

运行命令：

```powershell
.\.venv\Scripts\python.exe scripts\eval\compare_offline_online_rl.py `
  --online_episodes 40 `
  --eval_num_seeds 5 `
  --output_dir experiments\runs\20260611_offline_vs_online_rl_basic
```

## 对比对象

| 策略 | 含义 |
|---|---|
| `Offline_Rainbow` | 当前主线离线 checkpoint：`20260604_v1_domain_random_obs7_nocql_seed20260604` |
| `Offline_Rainbow_LateGuard` | 同一离线 checkpoint 叠加当前推荐 `delta3_low5` late guard |
| `Online_Rainbow_Scratch` | 从零开始在线交互训练 40 episodes 的 Rainbow baseline |

离线 checkpoint：

```text
experiments/runs/20260604_v1_domain_random_obs7_nocql_seed20260604/checkpoints/rainbow_offline_best.pth
```

离线数据集：

```text
data/datasets/offline_dataset_v1_domain_random_obs7_seed20260604.npz
```

数据集规模为 `488555` 条 transition，profile 覆盖 `normal / weak_coverage / high_noise / nlos_fading / high_interference / mixed_mild`。

在线 baseline 设置：

- 训练 profile split：`train`
- episodes：`40`
- epsilon：`0.30 -> 0.02`
- action hold：`0`，即按 TTT 自适应保持
- 训练期间环境交互步数：`38205`
- 梯度更新次数：`9423`
- 输出 checkpoint：

```text
experiments/runs/20260611_offline_vs_online_rl_basic/checkpoints/online_rainbow_scratch_final.pth
```

## Holdout 评估设置

评估 profile split：`test`

profile：

- `stress_radio_holdout`
- `speed_400`
- `distance_2km`
- `distance_4km`

每个 profile `5` seeds，总计 `20` episodes/policy。

输出：

```text
experiments/runs/20260611_offline_vs_online_rl_basic/metrics/offline_online_episode_metrics.csv
experiments/runs/20260611_offline_vs_online_rl_basic/metrics/offline_online_summary_by_profile.csv
experiments/runs/20260611_offline_vs_online_rl_basic/metrics/offline_online_summary_overall.csv
experiments/runs/20260611_offline_vs_online_rl_basic/offline_online_comparison_report.md
```

## 总体结果

| policy | episodes | outage | SINR p5 | HO/km | ping-pong | overlap interrupt | mobility failure proxy |
|---|---:|---:|---:|---:|---:|---:|---:|
| `Offline_Rainbow` | 20 | 0.07049 | -6.056 | 0.4625 | 0.000 | 0.00063 | 0.591 |
| `Offline_Rainbow_LateGuard` | 20 | 0.05116 | -4.101 | 0.7542 | 0.000 | 0.00063 | 0.315 |
| `Online_Rainbow_Scratch` | 20 | 0.05589 | -4.967 | 0.9042 | 0.050 | 0.00270 | 0.416 |

相对 `Online_Rainbow_Scratch`：

- `Offline_Rainbow_LateGuard` 的 outage 更低：`0.05589 -> 0.05116`。
- `Offline_Rainbow_LateGuard` 的 SINR p5 更高：`-4.967 dB -> -4.101 dB`，提升约 `0.87 dB`。
- `Offline_Rainbow_LateGuard` 的 HO/km 更低：`0.9042 -> 0.7542`。
- `Offline_Rainbow_LateGuard` 的 ping-pong 为 `0`，在线 baseline 为 `0.05`。
- `Offline_Rainbow_LateGuard` 的重叠区中断比例更低：`0.00270 -> 0.00063`。
- `Offline_Rainbow_LateGuard` 的 mobility failure proxy 更低：`0.416 -> 0.315`。

Raw offline 网络并非所有指标都优于 online baseline：`Offline_Rainbow` 的 outage 和 SINR p5 较差，但它保持最低切换次数、零乒乓和最低重叠区中断。这说明当前工程方案的关键是“离线策略 + 安全 guard”的组合，而不是单独声称 raw offline 网络全面优于在线学习。

## Profile 观察

| profile | policy | outage | SINR p5 | HO/km | ping-pong | mobility failure proxy |
|---|---|---:|---:|---:|---:|---:|
| `distance_2km` | `Offline_Rainbow_LateGuard` | 0.00792 | -1.162 | 0.500 | 0.000 | 0.100 |
| `distance_2km` | `Online_Rainbow_Scratch` | 0.01375 | -1.060 | 0.700 | 0.200 | 0.050 |
| `distance_4km` | `Offline_Rainbow_LateGuard` | 0.01602 | -2.386 | 0.650 | 0.000 | 0.250 |
| `distance_4km` | `Online_Rainbow_Scratch` | 0.02206 | -3.244 | 0.650 | 0.000 | 0.410 |
| `speed_400` | `Offline_Rainbow_LateGuard` | 0.01259 | -1.277 | 0.467 | 0.000 | 0.150 |
| `speed_400` | `Online_Rainbow_Scratch` | 0.02630 | -3.205 | 0.600 | 0.000 | 0.397 |
| `stress_radio_holdout` | `Offline_Rainbow_LateGuard` | 0.16810 | -11.578 | 1.400 | 0.000 | 0.760 |
| `stress_radio_holdout` | `Online_Rainbow_Scratch` | 0.16144 | -12.360 | 1.667 | 0.000 | 0.809 |

Stress profile 中，online baseline 的 outage 略低于 `Offline_Rainbow_LateGuard`，但 `Offline_Rainbow_LateGuard` 的 SINR p5、HO/km 和 mobility failure proxy 更好。这也再次说明 Stress 剩余问题不宜被解释为单纯学习方式问题，而是覆盖、干扰、NLOS/fading 和资源/业务耦合共同作用。

## 三方面证明

### 1. 安全性

离线训练路线在模型更新阶段不需要与在线网络交互，新增在线探索步数为 `0`。在线 baseline 为了从零学习，在训练期间产生 `38205` 个环境交互步，并直接把未收敛策略作用于 A3 参数。

在线训练期统计显示：

- mobility failure proxy mean：`0.2605`
- mobility failure proxy worst10：`0.5868`
- 最大 HO/km：`20.33`
- 最大 HO attempt 数：`61`
- 最差 SINR p5：`-14.58 dB`

这些数字可作为在线探索风险的量化参考。离线训练不是没有探索成本，而是把探索放在仿真/历史数据采集阶段，使真实部署阶段不必承受从零在线试错。

### 2. 训练效率/交互样本效率

离线训练复用固定数据集，模型更新阶段新增环境交互步数为 `0`；在线 baseline 则需要边交互边更新，本次小规模训练就消耗 `38205` 步交互和 `9423` 次梯度更新。

注意：本次 online baseline 只有 40 episodes，因此 wall-clock 时间 `47.96 s` 短于完整离线 run 的 `70.63 s`。论文中不要把该结果表述为“离线训练计算时间更短”。更稳妥的结论是：

> 离线训练具有更好的交互样本效率和部署安全性：模型更新可以复用既有数据集，不需要在运行网络中持续产生新的探索交互。

### 3. 训练效果

在 4 个 holdout profiles × 5 seeds 上，当前推荐的 `Offline_Rainbow_LateGuard` 相对 40-episode `Online_Rainbow_Scratch` 更均衡：outage、SINR p5、HO/km、ping-pong、重叠区中断和 mobility failure proxy 均有优势或接近。

但不能写成“离线学习全面优于在线学习”。更准确的是：

> 在当前小规模在线从零训练对照下，离线训练策略经过 holdout 验证和 late guard 安全约束后，形成了更适合工程部署的折中；其优势主要来自离线训练的数据复用、安全验证和部署期 guard，而不是单个 raw RL 网络在所有 KPI 上碾压在线学习。

## 论文建议表述

可在方法或实验讨论中写：

> 本文采用离线强化学习而非直接在线训练，主要出于铁路通信场景的安全约束。在线 RL 在训练早期需要探索不同 A3 参数，可能导致过多切换、乒乓、晚切和业务中断。相反，离线训练可将探索过程限制在仿真或历史数据中，模型更新阶段不再与运行网络交互，并可在部署前通过 holdout profile 和 Simu5G 验证进行筛选。本研究进一步叠加 late handover guard 作为运行时安全层。轻量对比实验表明，在 4 个 holdout profile、20 个评估 episode 上，`offline + late guard` 相比小规模 online-from-scratch baseline 取得更低 outage、更高 SINR p5、更低切换次数、零乒乓和更低移动性失败 proxy。

需要同步保留限制：

> 该对比仅作为工程合理性证明，不排除经过充分安全约束和长期训练的在线微调方法在未来取得更好性能。本文关注的是安全敏感铁路切换控制中的可部署离线训练路线。

## 后续增强

若论文审稿压力较大，可补充：

1. 将 online baseline 扩展到 `120` 或 `200` episodes，并保存中间 checkpoint，画出 KPI 随训练 episode 的变化。
2. 加入 `safe online fine-tuning from offline checkpoint`，证明从离线策略出发的安全微调优于从零在线探索。
3. 将 eval seeds 从 `5/profile` 扩到 `20/profile`，给出均值、标准差和 95% CI。
4. 若强调真实部署，再补 Simu5G 中 `Offline_Rainbow_LateGuard` 与固定 A3 的多 seed 对比；在线从零训练不建议直接放入 Simu5G 运行网络。

