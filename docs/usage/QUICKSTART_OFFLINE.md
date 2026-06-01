# 离线学习快速开始

## 1. 收集离线数据

```bash
python scripts/data/collect_data.py --num_episodes 100 --output_path data/datasets/offline_dataset.npz
```

输出数据集：

```text
data/datasets/offline_dataset.npz
```

## 2. 离线训练

```bash
python scripts/train/train_rainbow_offline.py --dataset_path data/datasets/offline_dataset.npz --use_cql
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
  --use_cql \
  --run_dir experiments/runs/20260601_rainbow_cql_seed2026
```

## 3. 评估模型

单场景对比：

```bash
python scripts/eval/test_simple.py
```

批量对比：

```bash
python scripts/eval/test_batch_comparison.py
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
