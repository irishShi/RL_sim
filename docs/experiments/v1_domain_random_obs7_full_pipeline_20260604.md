# V1 Domain Random Obs7 完整实验记录

本文档记录 `2026-06-04` 执行的 V1 鲁棒泛化版 RL 自适应调参器完整流水线：数据收集、离线训练、Python 测试、policy table 导出和 Simu5G 复杂矩阵验证。

## 1. 实验目标

本轮实验保持模型结构不变：

- 输入仍为 `15 x 7`，不新增 `Delta RSRP rate`、`SINR rate`、`low SINR duration` 等特征。
- 模型仍为 `Obs7 + GRU + Rainbow DQN`。
- 奖励函数不切换到 R6，本轮只验证多 profile domain randomization 对泛化能力的影响。
- Simu5G 仍通过 `policy_table.csv` 查表控制 A3 的 `Hys/TTT`。

## 2. 数据收集

数据集：

```text
data/datasets/offline_dataset_v1_domain_random_obs7_seed20260604.npz
data/datasets/offline_dataset_v1_domain_random_obs7_seed20260604.metadata.json
```

采集命令：

```powershell
.\.venv\Scripts\python.exe scripts\data\collect_data.py `
  --num_episodes 500 `
  --output_path data/datasets/offline_dataset_v1_domain_random_obs7_seed20260604.npz `
  --policy_type stratified `
  --scenario_profiles_path configs/scenario_profiles.yaml `
  --profile_split train `
  --save_profile_metadata `
  --seed_start 2026060400 `
  --action_hold_steps 0
```

数据概况：

| 项目 | 数值 |
| --- | ---: |
| episodes | 500 |
| samples | 488555 |
| obs shape | `(488555, 15, 7)` |
| action shape | `(488555,)` |

profile 分布：

| profile | episode 数 |
| --- | ---: |
| normal | 145 |
| weak_coverage | 82 |
| high_noise | 54 |
| nlos_fading | 73 |
| high_interference | 73 |
| mixed_mild | 73 |

## 3. 离线训练

实验目录：

```text
experiments/runs/20260604_v1_domain_random_obs7_nocql_seed20260604
```

训练命令：

```powershell
.\.venv\Scripts\python.exe scripts\train\train_rainbow_offline.py `
  --dataset_path data/datasets/offline_dataset_v1_domain_random_obs7_seed20260604.npz `
  --no_cql `
  --num_epochs 120 `
  --samples_per_epoch 12000 `
  --batch_size 64 `
  --amp `
  --val_ratio 0.05 `
  --early_stop_on val `
  --early_stop_patience 18 `
  --early_stop_min_delta 0.001 `
  --seed 20260604 `
  --run_dir experiments/runs/20260604_v1_domain_random_obs7_nocql_seed20260604
```

训练结果：

| 项目 | 数值 |
| --- | ---: |
| device | CUDA |
| GPU | NVIDIA GeForce RTX 4070 SUPER |
| model params | 959300 |
| actual epochs | 34 |
| best epoch | 16 |
| best metric | val Rainbow loss |
| best val Rainbow loss | 3.893554 |
| total train time | 70.66 s |

checkpoint：

```text
experiments/runs/20260604_v1_domain_random_obs7_nocql_seed20260604/checkpoints/rainbow_offline_best.pth
experiments/runs/20260604_v1_domain_random_obs7_nocql_seed20260604/checkpoints/rainbow_offline_final.pth
```

## 4. Python 评估

### 4.1 Test Profile 泛化评估

输出目录：

```text
experiments/runs/20260604_v1_domain_random_obs7_nocql_seed20260604/eval/profile_generalization_seed2026061400
```

核心结果：

| profile | policy | HO mean | outage ratio mean | SINR mean | SINR p5 | interruption total |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| stress_radio_holdout | RL | 1.40 | 0.2021 | 7.49 | -13.69 | 0.070 |
| stress_radio_holdout | FixedA3 | 8.10 | 0.1411 | 8.45 | -11.03 | 0.405 |
| speed_400 | RL | 1.40 | 0.0379 | 15.41 | -4.28 | 0.070 |
| speed_400 | FixedA3 | 3.70 | 0.0199 | 15.98 | -1.21 | 0.185 |
| distance_2km | RL | 1.50 | 0.0370 | 15.90 | -4.07 | 0.075 |
| distance_2km | FixedA3 | 3.10 | 0.0185 | 16.52 | -0.79 | 0.155 |
| distance_4km | RL | 2.30 | 0.0262 | 16.19 | -3.20 | 0.115 |
| distance_4km | FixedA3 | 6.40 | 0.0186 | 16.51 | -1.35 | 0.320 |

判断：V1 RL 明显减少 HO 和中断时长，但 Python holdout profile 下的 outage/SINR 仍弱于固定 A3。这说明当前策略仍偏保守，少切换换来了较低中断，但在极端覆盖或高速度场景中仍可能晚切。

### 4.2 Batch Comparison

输出目录：

```text
experiments/runs/20260604_v1_domain_random_obs7_nocql_seed20260604/eval/batch_100_seed2026061500
```

100 场景摘要：

| policy | HO mean | success | SINR mean | outage ratio | ping-pong | overlap SINR | overlap interruption |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A3 Hys=1.5 TTT=150 | 5.34 | 1.0000 | 16.04 | 0.0188 | 0.1332 | 5.026 | 0.0151 |
| A3 Hys=2.5 TTT=300 | 2.70 | 1.0000 | 16.07 | 0.0094 | 0.0441 | 5.360 | 0.0041 |
| A3 Hys=3.5 TTT=300 | 2.08 | 1.0000 | 16.06 | 0.0073 | 0.0187 | 5.422 | 0.0017 |
| A3 Hys=3.5 TTT=650 | 1.44 | 1.0000 | 15.79 | 0.0167 | 0.0020 | 5.475 | 0.0000 |
| A3 Hys=4.0 TTT=300 | 1.84 | 1.0000 | 16.03 | 0.0071 | 0.0120 | 5.434 | 0.0009 |
| RL Rainbow | 1.62 | 1.0000 | 15.86 | 0.0148 | 0.0020 | 5.462 | 0.0012 |
| Oracle A3 | 1.22 | 1.0000 | 16.09 | 0.0063 | 0.0000 | 5.534 | 0.0009 |
| Fixed A3 reference | 1.22 | 1.0000 | 15.56 | 0.0253 | 0.0000 | 5.481 | 0.0000 |

判断：RL 比激进 A3 大幅减少切换和乒乓，但与 Oracle A3 仍有明显差距。与固定同参 A3 参考相比，RL 的 SINR 更好、outage 更低，但切换次数略高。

## 5. Policy Table 导出

版本化输出：

```text
results/policy_tables/policy_table_v1_domain_random_obs7_seed20260604.csv
results/policy_tables/policy_table_v1_domain_random_obs7_seed20260604.metadata.json
```

Simu5G 默认读取路径已同步覆盖：

```text
results/policy_tables/policy_table.csv
results/policy_tables/policy_table.metadata.json
```

导出表大小：

| 项目 | 数值 |
| --- | ---: |
| rows | 830088 |
| file size | 184724446 bytes |

动作分布 Top-10：

| action_id | count |
| ---: | ---: |
| 29 | 413271 |
| 23 | 195351 |
| 34 | 105992 |
| 46 | 38122 |
| 10 | 25158 |
| 3 | 22199 |
| 47 | 9352 |
| 16 | 8137 |
| 39 | 6772 |
| 11 | 2829 |

按 TTT 汇总：

| TTT ms | count |
| ---: | ---: |
| 650 | 621228 |
| 300 | 178317 |
| 150 | 30524 |
| 0 | 12 |
| 100 | 7 |

判断：策略表明显偏向长 TTT，尤其 `650 ms`。这有利于减少乒乓和不必要切换，但可能带来晚切风险。

## 6. Simu5G 复杂矩阵验证

Simu5G 输出目录：

```text
results/simu5g/RailwayComplexMatrix-V1DomainRandomObs7-20260604
```

运行脚本：

```powershell
wsl -d Ubuntu -- bash -lc "cd /mnt/c/Users/qichengshi/Desktop/RL_sim && bash tools/run_simu5g_complex_matrix.sh RailwayComplexMatrix-V1DomainRandomObs7-20260604"
```

解析命令：

```powershell
.\.venv\Scripts\python.exe tools\analyze_simu5g_complex_matrix.py `
  --raw_root results/simu5g/RailwayComplexMatrix-V1DomainRandomObs7-20260604/raw `
  --parsed_root results/simu5g/RailwayComplexMatrix-V1DomainRandomObs7-20260604/parsed `
  --window_s 2.0
```

关键结果：

| scenario | first HO Fixed/RL m | loss Fixed/RL | SINR p5 Fixed/RL dB | CQI mean Fixed/RL | max delay Fixed/RL s | RL Hys/TTT mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| HeavyTraffic | 1600.8 / 1641.7 | 0.00070 / 0.00084 | 9.03 / 9.38 | 12.52 / 12.52 | 0.093 / 0.083 | 3.20 dB / 339 ms |
| HighInterference | 1684.2 / 1891.7 | 0 / 0 | 12.10 / 11.88 | 12.91 / 12.94 | 0.087 / 0.067 | 3.20 dB / 325 ms |
| HighNoise | 1767.5 / 1808.3 | 0.00558 / 0.00558 | -1.74 / -1.61 | 9.75 / 9.78 | 0.311 / 0.311 | 3.30 dB / 337 ms |
| NlosFading | 1600.8 / 1641.7 | 0 / 0 | 7.54 / 7.75 | 12.26 / 12.61 | 0.087 / 0.067 | 3.17 dB / 366 ms |
| Stress | 1684.2 / 1725.0 | 0.30066 / 0.30080 | -16.60 / -16.60 | 5.86 / 5.98 | 1.678 / 1.804 | 3.42 dB / 449 ms |
| TrackOffset | 1767.5 / 1808.3 | 0 / 0 | 8.90 / 8.79 | 12.57 / 12.33 | 0.093 / 0.067 | 3.23 dB / 343 ms |
| WeakCoverage | 1767.5 / 1808.3 | 0.01340 / 0.01173 | -3.67 / -3.04 | 9.52 / 9.43 | 0.311 / 0.305 | 3.31 dB / 431 ms |

## 7. 初步结论

1. V1 domain randomization 对 Simu5G 复杂场景有正面作用。相比上一轮旧 policy table，`Stress` 场景的 RLTable 不再出现明显的失控式恶化；丢包率从旧表约 `0.3360` 降到 `0.3008`，最大 App delay 从旧表约 `2.719 s` 降到 `1.804 s`。
2. V1 在 `WeakCoverage`、`NlosFading`、`HighNoise` 这类无线退化场景中表现较好，常见改善包括 SINR p5 提升、最大时延下降或丢包率下降。
3. V1 仍然不是全面优于 FixedA3。在 `Stress` 中，RL 的 CQI mean 略高，但最大时延仍高于 FixedA3；在 Python holdout profile 中，RL 的 outage/SINR 也仍弱于 FixedA3。
4. 当前策略主要学到了“少切换、长 TTT、偏保守”的风格。它可以减少 ping-pong 和中断，但在极端快速退化时容易晚切。
5. 下一轮不应直接扩大模型。建议先对低 SINR/快速退化片段做归因，并优先尝试 policy table 安全约束，例如在低 SINR 或邻区明显强于服务小区时限制最大 TTT。只有多 seed 证据显示退化主要来自切换时机，才考虑 R6 robust reward 或新增观测。

## 8. 产物索引

| 类型 | 路径 |
| --- | --- |
| 数据集 | `data/datasets/offline_dataset_v1_domain_random_obs7_seed20260604.npz` |
| 数据集 metadata | `data/datasets/offline_dataset_v1_domain_random_obs7_seed20260604.metadata.json` |
| 训练 run | `experiments/runs/20260604_v1_domain_random_obs7_nocql_seed20260604` |
| best checkpoint | `experiments/runs/20260604_v1_domain_random_obs7_nocql_seed20260604/checkpoints/rainbow_offline_best.pth` |
| profile 泛化评估 | `experiments/runs/20260604_v1_domain_random_obs7_nocql_seed20260604/eval/profile_generalization_seed2026061400` |
| batch comparison | `experiments/runs/20260604_v1_domain_random_obs7_nocql_seed20260604/eval/batch_100_seed2026061500` |
| policy table | `results/policy_tables/policy_table_v1_domain_random_obs7_seed20260604.csv` |
| Simu5G raw | `results/simu5g/RailwayComplexMatrix-V1DomainRandomObs7-20260604/raw` |
| Simu5G parsed | `results/simu5g/RailwayComplexMatrix-V1DomainRandomObs7-20260604/parsed` |
