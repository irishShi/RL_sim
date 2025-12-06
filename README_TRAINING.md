# Rainbow DQN 训练和测试指南

## 一、训练脚本

### 1.1 文件说明

- **`train_rainbow.py`**: 完整的 Rainbow DQN 训练脚本
- **`utils/replay_buffer.py`**: 经验回放缓冲区（支持 PER 和 N-step）
- **`utils/c51_projection.py`**: C51 分布投影算法

### 1.2 使用方法

```bash
python train_rainbow.py
```

### 1.3 训练流程

1. **加载配置**：从 `configs/model_config.yaml` 读取模型和训练配置
2. **创建环境**：使用 `TrainHandoverEnv`（48 动作模式）
3. **初始化模型**：创建在线网络和目标网络
4. **训练循环**：
   - 与环境交互，收集经验
   - 使用 N-step return 和 PER
   - 定期更新目标网络
   - 每 50 个 episode 保存一次 checkpoint

### 1.4 训练输出

- **Checkpoints**: 保存在 `checkpoints/` 目录
  - `rainbow_episode_50.pth`, `rainbow_episode_100.pth`, ...
  - `rainbow_final.pth`（最终模型）

### 1.5 训练参数

主要参数在 `configs/model_config.yaml` 中配置：

```yaml
training:
  batch_size: 32
  learning_rate: 6.25e-5
  gamma: 0.99
  n_steps: 3
  lambda_aux: 0.3  # 辅助预测损失权重
  target_update_freq: 1000
  replay_buffer_size: 100000
  replay_start_size: 1000
```

---

## 二、测试脚本

### 2.1 文件说明

- **`test_rainbow.py`**: 测试脚本，包含与传统 A3 算法对比

### 2.2 使用方法

```bash
python test_rainbow.py
```

### 2.3 测试内容

1. **Rainbow DQN 模型测试**
   - 加载训练好的模型
   - 运行多个 episode 测试性能

2. **传统 A3 算法测试**
   - 测试多个固定参数组合：
     - Hys=3.0dB, TTT=160ms
     - Hys=3.0dB, TTT=320ms
     - Hys=4.0dB, TTT=160ms
     - Hys=4.0dB, TTT=320ms
     - Hys=5.0dB, TTT=160ms

3. **性能对比**
   - 平均奖励
   - 切换次数
   - 平均 SINR
   - 最小 SINR

4. **可视化**
   - RSRP 曲线与切换点对比
   - SINR 对比
   - Rainbow 参数选择动态变化
   - 性能对比柱状图

### 2.4 测试输出

- **控制台输出**：各策略的性能统计
- **可视化图片**：`test_results_comparison.png`

---

## 三、关键特性

### 3.1 Rainbow DQN 特性

✅ **Dueling 架构**：分离状态价值和动作优势  
✅ **C51 分布**：价值分布而非期望值  
✅ **NoisyNet**：参数化探索  
✅ **N-step Return**：多步奖励累积  
✅ **Double Q-learning**：减少过估计  
✅ **PER**：优先经验回放  
✅ **辅助预测**：联合训练 ΔRSRP 预测

### 3.2 训练技巧

1. **观测窗口初始化**：用第一个观测填充整个窗口
2. **延迟存储**：在 t 步存储 t-1 步的经验，此时 t 步的 ΔRSRP 已知
3. **N-step 缓冲区**：Episode 结束时清空缓冲区
4. **目标网络更新**：每 1000 步硬更新一次

---

## 四、常见问题

### Q1: 训练很慢怎么办？

**A**: 
- 减少 `replay_buffer_size`
- 减少 `batch_size`
- 减少 `max_episodes`
- 使用 GPU（如果有）

### Q2: 模型不收敛？

**A**:
- 检查学习率是否合适
- 增加 `replay_start_size`，确保有足够样本再开始训练
- 调整 `lambda_aux`（辅助预测损失权重）
- 检查奖励函数设计

### Q3: 如何调整训练参数？

**A**: 修改 `configs/model_config.yaml` 中的 `training` 部分

### Q4: 如何继续训练？

**A**: 修改训练脚本，加载 checkpoint 并继续训练：

```python
checkpoint = torch.load("checkpoints/rainbow_episode_100.pth")
online_net.load_state_dict(checkpoint['online_net_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
start_episode = checkpoint['episode']
```

---

## 五、性能指标

### 5.1 训练指标

- **Episode Reward**: 每个 episode 的累计奖励
- **Rainbow Loss**: C51 分布损失
- **Forecast Loss**: 辅助预测损失
- **TD Error**: 用于更新 PER 优先级

### 5.2 测试指标

- **平均奖励**: 多个 episode 的平均累计奖励
- **切换次数**: 平均切换次数
- **平均 SINR**: 平均信号质量
- **最小 SINR**: 最差信号质量
- **Outage 次数**: 信号中断次数

---

## 六、下一步

1. ✅ **训练模型**: 运行 `python train_rainbow.py`
2. ✅ **测试模型**: 运行 `python test_rainbow.py`
3. ⬜ **调优超参数**: 根据结果调整配置
4. ⬜ **扩展实验**: 测试不同场景和参数组合

---

## 七、文件结构

```
RL_sim/
├── train_rainbow.py          # 训练脚本
├── test_rainbow.py           # 测试脚本
├── utils/
│   ├── __init__.py
│   ├── replay_buffer.py     # 经验回放缓冲区
│   └── c51_projection.py    # C51 投影算法
├── checkpoints/              # 模型检查点（训练后生成）
│   ├── rainbow_episode_50.pth
│   └── rainbow_final.pth
└── test_results_comparison.png  # 测试结果对比图（测试后生成）
```
