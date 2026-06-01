# Rainbow DQN 模型说明

根据模型方案文档，本项目实现了完整的 Rainbow DQN + 辅助预测模型。

## 模型结构

### 1. 动作空间 (ActionSpace)

动作空间将 (Hys, TTT) 组合映射为离散动作索引：

- **Hys 集合**: {1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0} dB (8个)
- **TTT 集合**: {0, 40, 80, 160, 320, 640} ms (6个)
- **总动作数**: 8 × 6 = 48

### 2. 观测空间

#### 单步观测（10维）

每个时间步的特征包括：

0. `RSRP_serv` - 服务小区 RSRP（归一化）
1. `RSRP_neig` - 邻区 RSRP（归一化）
2. `ΔRSRP` - RSRP 差值（归一化）
3. `SINR_serv` - 服务小区 SINR（归一化）
4. `v_norm` - 速度归一化
5. `pos_norm` - 位置归一化
6. `time_since_last_ho_norm` - 距离上次切换时间归一化
7. `Hys_norm` - 当前 Hys 参数归一化
8. `TTT_norm` - 当前 TTT 参数归一化
9. `T_norm` - 温度归一化

#### 时序窗口

- **窗口长度**: N = 15 步（覆盖 1.5 秒，假设 100ms 采样）
- **输入形状**: [batch_size, N, 10]

### 3. 网络架构

```
输入 [B, N, 10]
    ↓
时序编码器 (GRU)
    ↓ [B, 128]
共享特征层 (FC)
    ↓ [B, 256]
    ├─→ Rainbow Q 头 (Dueling + C51 + Noisy)
    │       ↓ [B, 48, 51] (动作数 × 原子数)
    │
    └─→ 辅助预测头 (FC)
            ↓ [B] (预测 ΔRSRP)
```

#### 组件说明

1. **时序编码器 (TemporalEncoderGRU)**
   - 使用 GRU 处理时间序列
   - 输出维度: 128

2. **共享特征层 (SharedFeatureNet)**
   - 2层全连接网络
   - 输出维度: 256

3. **Rainbow Q 头 (RainbowHead)**
   - **Dueling 架构**: Value stream + Advantage stream
   - **C51 分布**: 51 个价值原子，范围 [-60, 10]
   - **NoisyLinear**: 用于探索，替代 ε-greedy

4. **辅助预测头 (ForecastHead)**
   - 预测下一步 ΔRSRP
   - 与主任务共享特征编码器

## 使用方法

### 基本使用

```python
from models import RainbowWithForecast, ActionSpace, ObservationWindow
import torch

# 创建动作空间
action_space = ActionSpace()

# 创建模型
model = RainbowWithForecast(
    obs_dim=10,
    num_actions=48,
    n_steps=15,
    feature_hidden=256,
    encoder_hidden=128,
    num_atoms=51,
    v_min=-60.0,
    v_max=10.0
)

# 创建观测窗口
obs_window = ObservationWindow(window_size=15, obs_dim=10)

# 构建观测
obs_raw = np.array([...])  # 7维原始观测
info = {'rsrp_serv_dbm': -90.0, 'rsrp_neig_dbm': -85.0}
obs_extended = obs_window.build_observation(
    obs_raw, info,
    velocity_mps=80.0,
    track_length_m=3000.0
)

# 获取窗口
window = obs_window.get_window()  # [15, 10]

# 模型推理
window_tensor = torch.from_numpy(window).unsqueeze(0)  # [1, 15, 10]
dist, pred_delta = model(window_tensor)
q_values = model.get_q_values(window_tensor)
action = model.act(window_tensor, epsilon=0.0)

# 获取动作对应的参数
hys, ttt = action_space.action_to_hys_ttt(action)
```

### 配置文件

模型配置位于 `configs/model_config.yaml`，包括：

- 动作空间定义
- 观测配置
- 网络结构参数
- 训练超参数
- 归一化范围

## 关键特性

### 1. C51 分布强化学习

- 使用价值分布而非期望值
- 51 个价值原子，范围 [-60, 10]
- 通过投影 Bellman 更新目标分布

### 2. Dueling 架构

- 分离状态价值和动作优势
- 提高学习效率

### 3. NoisyNet 探索

- 使用参数化噪声替代 ε-greedy
- 自适应探索策略

### 4. 辅助预测任务

- 联合训练预测 ΔRSRP
- 损失函数: `L = L_Rainbow + λ_aux * L_forecast`
- 默认 λ_aux = 0.3

## 训练流程（待实现）

完整的训练流程需要：

1. **经验回放缓冲区** (PER - Prioritized Experience Replay)
2. **N-step return** 计算
3. **C51 投影算法** 实现
4. **目标网络** 更新
5. **训练循环** 实现

这些将在后续的训练脚本中实现。

## 文件结构

```
models/
├── __init__.py              # 模块导出
├── rainbow_model.py          # 主模型实现
├── action_space.py           # 动作空间定义
└── observation_builder.py   # 观测窗口构建

configs/
└── model_config.yaml        # 模型配置文件
```

## 测试

运行测试脚本验证模型：

```bash
python test_model.py
```

## 注意事项

1. **观测归一化**: 确保所有输入特征都在 [0, 1] 范围内
2. **ΔRSRP 计算**: 从环境 info 中获取原始 RSRP 值（dBm）计算差值
3. **时间窗口**: 需要维护历史观测窗口，在环境 reset 时清空
4. **动作参数**: 模型输出动作索引，需要通过 ActionSpace 转换为 (Hys, TTT)
5. **NoisyLinear**: 训练时每次前向传播前需要调用 `reset_noise()`

## 下一步

- [ ] 实现经验回放缓冲区（PER）
- [ ] 实现 C51 投影算法
- [ ] 实现训练循环
- [ ] 实现环境包装器（适配新的动作空间）
- [ ] 实现评估和可视化工具
