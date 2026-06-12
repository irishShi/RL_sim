# V1 鲁棒泛化训练框架说明

本文档记录第一轮“鲁棒泛化版 RL 自适应调参器”的实现方案。V1 不改变模型结构，不扩展输入维度，仍保持 `Obs7 + GRU + Rainbow DQN`，优先解决训练分布单一和测试集不隔离的问题。

## 1. 目标

V1 的目标是把当前单场景训练流程升级为多工况 domain randomization 训练流程，并建立训练/测试 profile 分离的评估入口。

本轮不做：

- 不增加 `ΔRSRP rate`、`SINR rate`、`low SINR duration` 等新入参。
- 不修改 `model_config.yaml` 中的 `obs_dim=7`。
- 不修改 Rainbow DQN 网络结构。
- 不新增 `R6 reward`。

## 2. 场景 Profile

配置文件位于：

```bash
configs/scenario_profiles.yaml
```

训练 split 包含：

- `normal`
- `weak_coverage`
- `high_noise`
- `nlos_fading`
- `high_interference`
- `mixed_mild`

测试 split 包含：

- `stress_radio_holdout`
- `speed_400`
- `distance_2km`
- `distance_4km`

训练 profile 主要覆盖单因素和轻度混合压力；测试 profile 保留更难或未见组合，用于检验泛化能力。

## 3. 数据采集

使用 profile 随机采样采集离线数据：

```bash
python scripts/data/collect_data.py \
  --num_episodes 500 \
  --policy_type stratified \
  --output_path data/datasets/offline_dataset_v1_domain_random_obs7.npz \
  --seed_start 40000 \
  --scenario_profiles_path configs/scenario_profiles.yaml \
  --profile_split train \
  --save_profile_metadata
```

输出 `.npz` 保持原字段不变，可直接用于现有离线训练脚本。额外生成：

```bash
data/datasets/offline_dataset_v1_domain_random_obs7.metadata.json
```

metadata 记录每个 episode 使用的 profile、seed 和关键环境参数，便于复现实验和统计训练分布。

## 4. 训练

训练命令沿用现有离线训练入口：

```bash
python scripts/train/train_rainbow_offline.py \
  --dataset_path data/datasets/offline_dataset_v1_domain_random_obs7.npz
```

建议 run 名称中包含：

```text
v1_domain_random_obs7
```

## 5. Python Profile 泛化评估

新增评估入口：

```bash
python scripts/eval/test_profile_generalization.py \
  --checkpoint_path <checkpoint.pth> \
  --scenario_profiles_path configs/scenario_profiles.yaml \
  --profile_split test \
  --num_seeds 20 \
  --base_seed 20000 \
  --output_dir experiments/runs/<run_name>/eval/profile_generalization
```

该脚本会对每个 test profile 和多个 seed 运行：

- `RL_Rainbow`
- `FixedA3_Hys3_TTT150`

输出：

- `profile_episode_metrics.csv`
- `profile_summary.csv`
- `profile_generalization_summary.json`

重点关注：

- `outage_time_ratio`
- `sinr_p5_db`
- `overlap_zone_sinr_mean_db`
- `interruption_total_time`
- `ho_count`
- worst 10% 表现

## 6. Simu5G 验证关系

V1 的 Python profile 测试用于快速筛选泛化趋势。最终仍需要把训练后的 checkpoint 导出为新的 `policy_table.csv`，再跑现有 Simu5G 14 组复杂矩阵。

判断目标：

- `NlosFading`、`WeakCoverage` 中保持当前 RL 的优势。
- `Stress` 中至少不再明显劣于 FixedA3。
- 正常场景不出现明显退化。

如果 V1 在 `stress_radio_holdout` 或 Simu5G `Stress` 仍明显退化，下一轮应先做误差归因和少量候选入参/guard 消融。弱覆盖、强干扰、NLOS/fading 的中等退化仍属于鲁棒切换需要覆盖的对象；只有当退化主要来自晚切、过早切或乒乓这类切换可控问题时，才进入 `R6 robust reward`。若主要来自两个候选小区都不可用、资源不足或业务排队，则应作为算法边界或跨层优化方向记录。
