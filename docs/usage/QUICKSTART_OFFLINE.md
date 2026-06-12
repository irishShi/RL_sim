# 离线学习快速开始

> 当前版本已移除天气损耗、温度观测特征以及策略输入中的当前 Hys/TTT 特征，模型输入为 `[15, 7]`。旧的 10 维/9 特征离线数据集和 legacy checkpoint 仅用于历史结果归档，继续实验前需要重新收集数据并重新训练。

> 数据采集、在线训练和评估共用同一套自适应 action hold 逻辑：每个 `(Hys, TTT)` 动作默认保持到覆盖当前 TTT 所需时间，并额外留出 2 步余量。

## 1. 收集离线数据

快速 smoke 可以使用较少 episode：

```bash
python scripts/data/collect_data.py --num_episodes 100 --output_path data/datasets/offline_dataset.npz
```

当前推荐 V1 domain random Obs7 数据集使用完整命令：

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

推荐输出数据集：

```text
data/datasets/offline_dataset_v1_domain_random_obs7_seed20260604.npz
```

## 2. 离线训练

快速 smoke 可以使用临时数据集：

```bash
python scripts/train/train_rainbow_offline.py --dataset_path data/datasets/offline_dataset.npz --no_cql
```

当前推荐基线训练使用 `--no_cql`：

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

训练输出默认进入新的实验目录：

```text
experiments/runs/YYYYMMDD_HHMMSS_rainbow_offline_seed2026/
├─ config/
├─ checkpoints/
├─ logs/
└─ metrics/
```

也可以手动指定输出目录：

```bash
python scripts/train/train_rainbow_offline.py \
  --dataset_path data/datasets/offline_dataset.npz \
  --no_cql \
  --run_dir experiments/runs/YYYYMMDD_rainbow_obs7_nocql_seed2026
```

## 3. 评估模型

单场景对比：

```powershell
.\.venv\Scripts\python.exe scripts\eval\test_simple.py `
  --checkpoint_path experiments/runs/20260604_v1_domain_random_obs7_nocql_seed20260604/checkpoints/rainbow_offline_best.pth
```

profile holdout 泛化：

```powershell
.\.venv\Scripts\python.exe scripts\eval\test_profile_generalization.py `
  --checkpoint_path experiments/runs/20260604_v1_domain_random_obs7_nocql_seed20260604/checkpoints/rainbow_offline_best.pth `
  --scenario_profiles_path configs/scenario_profiles.yaml `
  --profile_split test `
  --num_seeds 20
```

整理前的历史模型位于：

```text
experiments/runs/legacy_20260416_offline_rainbow/checkpoints/
```

## 4. 真实轨迹数据

预处理真实轨迹：

```bash
python scripts/data/validate_gnb_trace.py --preprocess_only --save_csv
```

绘制真实轨迹信号图：

```bash
python scripts/plot/plot_gnb_trace.py
```

## 5. 目录提醒

- 数据集放 `data/datasets/`
- 原始真实数据放 `data/raw/real/`
- 处理后真实数据放 `data/processed/real/`
- 实验完整输出放 `experiments/runs/<run_id>/`
- 论文最终图表放 `paper/assets/`

更多规则见 [../项目文件分类与整理规范.md](../项目文件分类与整理规范.md)。
