# 离线学习快速开始指南

## 快速开始（3步）

### 步骤 1: 收集数据

```bash
python collect_data.py --num_episodes 100 --output_path data/offline_dataset.npz
```

这将：
- 使用随机策略收集 100 个 episode 的数据
- 保存到 `data/offline_dataset.npz`
- 自动创建 `data/` 目录（如果不存在）

### 步骤 2: 离线训练

```bash
python train_rainbow_offline.py --dataset_path data/offline_dataset.npz --use_cql
```

这将：
- 从数据集加载经验
- 使用 CQL 正则化进行离线训练
- 训练 100 个 epoch（默认）
- 保存模型到 `checkpoints/rainbow_offline_final.pth`

### 步骤 3: 测试模型

**测试离线训练模型（推荐）：**
```bash
python test_rainbow_offline.py
```

**或测试在线训练模型：**
```bash
python test_rainbow.py
```

这将：
- 加载训练好的模型
- 在环境中测试性能
- 与传统算法对比
- 生成对比图表

## 推荐配置

### 生产环境（高质量数据）

```bash
# 1. 收集更多数据（使用混合策略）
python collect_data.py \
    --num_episodes 500 \
    --policy_type uniform_mix \
    --output_path data/offline_dataset.npz

# 2. 使用 CQL 训练
python train_rainbow_offline.py \
    --dataset_path data/offline_dataset.npz \
    --use_cql \
    --cql_alpha 0.1 \
    --num_epochs 100 \
    --samples_per_epoch 20000
```

### 快速测试（小规模）

```bash
# 1. 收集少量数据
python collect_data.py --num_episodes 50 --output_path data/test_dataset.npz

# 2. 快速训练
python train_rainbow_offline.py \
    --dataset_path data/test_dataset.npz \
    --num_epochs 20 \
    --samples_per_epoch 5000
```

## 输出文件

- **数据集**: `data/offline_dataset.npz`
- **模型检查点**: `checkpoints/rainbow_offline_epoch_{N}.pth`
- **最终模型**: `checkpoints/rainbow_offline_final.pth`

## 注意事项

1. **首次运行**：确保 `data/` 和 `checkpoints/` 目录存在（脚本会自动创建）
2. **GPU 内存**：如果 GPU 内存不足，减小 `batch_size` 或 `samples_per_epoch`
3. **数据质量**：使用 `uniform_mix` 策略可以获得更好的动作覆盖

## 故障排除

### 问题：找不到数据集文件

**解决**：确保先运行 `collect_data.py` 生成数据集

### 问题：训练损失不下降

**解决**：
- 尝试使用 CQL：`--use_cql --cql_alpha 0.1`
- 增加训练轮数：`--num_epochs 200`
- 检查数据集质量（动作分布是否均匀）

### 问题：内存不足

**解决**：
- 减小 batch_size：`--batch_size 16`
- 减小 samples_per_epoch：`--samples_per_epoch 5000`

## 测试选项

### 基本测试
```bash
python test_rainbow_offline.py
```

### 完整对比测试
```bash
python test_rainbow_offline.py --num_episodes 20 --compare_online --save_plots
```

### 测试特定检查点
```bash
python test_rainbow_offline.py --checkpoint checkpoints/rainbow_offline_epoch_50.pth
```

## 更多信息

- 详细使用指南：`docs/离线学习使用指南.md`
- 测试指南：`docs/离线模型测试指南.md`
- 优化分析报告：`docs/离线学习优化分析.md`

