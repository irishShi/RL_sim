# Rainbow DQN 模型说明

本文档说明当前主线 `Obs7 + GRU + Rainbow DQN` 模型。早期的 10 维/9 特征模型、R3 奖励和环境包装器方案已经归档；当前训练、评估和导出统一使用 `[15, 7]` 观测窗口、48 个 A3 `(Hys, TTT)` 动作、R5 奖励和 action hold。

## 模型目标与边界

当前模型的目标是在高速铁路相邻小区重叠区内自适应选择 A3 的 `Hys/TTT`。模型主要优化切换时机：避免该切不切造成晚切，也避免过早切换和频繁乒乓带来的中断。对于一定程度内的弱覆盖、强干扰、NLOS/fading，如果邻区仍存在可利用质量优势，模型应能通过更合理切换缓解退化。

如果极端工况中两个候选小区都处于低 SINR、目标小区同样不可用，或退化主要来自资源不足和业务排队，单靠扩大 Rainbow 网络或增加输入特征通常不能从根本上解决。此类现象应在实验中作为算法边界说明，必要时交给覆盖增强、资源调度或跨层可靠性机制。

因此，后续模型扩展应优先保持小模型和可部署性。只有当新增特征或模块能稳定改善 `ho_per_km`、`ping_pong_count`、`outage_time_ratio`、`sinr_p5_db`、`overlap_zone_sinr_mean_db` 或切换区中断率时，才应进入主线。完整边界说明见 [problem_scope_and_boundaries.md](problem_scope_and_boundaries.md)。

## 模型结构

### 1. 动作空间 (ActionSpace)

动作空间将 (Hys, TTT) 组合映射为离散动作索引：

- **Hys 集合**: {1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0} dB (8个)
- **TTT 集合**: {0, 50, 100, 150, 300, 650} ms (6个)
- **总动作数**: 8 × 6 = 48

### 2. 观测空间

#### 单步观测（7维）

每个时间步的特征包括：

0. `RSRP_serv` - 服务小区 RSRP（归一化）
1. `RSRP_neig` - 邻区 RSRP（归一化）
2. `ΔRSRP` - RSRP 差值（归一化）
3. `SINR_serv` - 服务小区 SINR（归一化）
4. `v_norm` - 速度归一化
5. `pos_norm` - 位置归一化
6. `time_since_last_ho_norm` - 距离上次切换时间归一化

当前 Hys/TTT 不再作为策略输入特征，以避免 Q 网络学习“当前参数是什么就继续选择什么”的动作复制捷径。

#### 时序窗口

- **窗口长度**: N = 15 步（覆盖 1.5 秒，假设 100ms 采样）
- **输入形状**: [batch_size, N, 7]

### 3. Action hold 控制频率

`Hys` 和 `TTT` 表示一段时间内持续生效的切换参数，而不是每个仿真步都应重新选择的瞬时控制量。当前主流程统一使用自适应 action hold：保持步数默认为 `max(6, ceil(TTT / delta_t) + 2)`，上限为 20 步。这样可以让较大的 TTT 有机会完成计时，同时保留模型在列车快速移动场景中的调参能力。

### 4. 网络架构

```
输入 [B, N, 7]
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

### 基本推理示例

```python
from models import RainbowWithForecast, ActionSpace, ObservationWindow
import torch
import numpy as np

# 创建动作空间
action_space = ActionSpace()

# 创建模型
model = RainbowWithForecast(
    obs_dim=7,
    num_actions=48,
    n_steps=15,
    feature_hidden=256,
    encoder_hidden=128,
    num_atoms=51,
    v_min=-60.0,
    v_max=10.0
)

# 创建观测窗口
obs_window = ObservationWindow(window_size=15, obs_dim=7)

# 构建观测
obs_raw = np.array([...])
info = {'rsrp_serv_dbm': -90.0, 'rsrp_neig_dbm': -85.0}
obs_extended = obs_window.build_observation(
    obs_raw, info,
    velocity_mps=80.0,
    track_length_m=3000.0
)

# 获取窗口
window = obs_window.get_window()  # [15, 7]

# 模型推理
window_tensor = torch.from_numpy(window).unsqueeze(0)  # [1, 15, 7]
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

## 当前训练、评估与导出入口

当前完整训练链路已经实现。推荐主线如下：

1. 使用 `scripts/data/collect_data.py` 采集 `Obs7` 离线数据，可通过 `configs/scenario_profiles.yaml` 启用多 profile domain randomization。
2. 使用 `scripts/train/train_rainbow_offline.py` 离线训练 Rainbow checkpoint。
3. 使用 `scripts/eval/test_profile_generalization.py` 和 `scripts/eval/test_late_guard_generalization.py` 做 holdout profile 泛化评估。
4. 使用 `comparison_algorithms/scripts/run_comparison.py` 做论文 baseline 对比。
5. 使用 `scripts/export/export_policy_table.py` 或 `tools/export_late_guard_policy_preset.py` 导出 Simu5G policy table。

当前推荐 checkpoint：

```text
experiments/runs/20260604_v1_domain_random_obs7_nocql_seed20260604/checkpoints/rainbow_offline_best.pth
```

当前推荐 policy table：

```text
results/policy_tables/policy_table.csv
```

## 文件结构

```
models/
├── __init__.py              # 模块导出
├── rainbow_model.py          # 主模型实现
├── physics_features.py       # 风险/物理辅助特征支线
├── action_space.py           # 动作空间定义
└── observation_builder.py   # 观测窗口构建

configs/
└── model_config.yaml        # 模型配置文件
```

## 验证命令

单场景和批量评估：

```powershell
.\.venv\Scripts\python.exe scripts\eval\test_simple.py --checkpoint_path <checkpoint.pth>
.\.venv\Scripts\python.exe scripts\eval\test_batch_comparison.py --checkpoint_path <checkpoint.pth>
```

profile 泛化与 late guard：

```powershell
.\.venv\Scripts\python.exe scripts\eval\test_profile_generalization.py `
  --checkpoint_path <checkpoint.pth> `
  --scenario_profiles_path configs/scenario_profiles.yaml `
  --profile_split test `
  --num_seeds 20

.\.venv\Scripts\python.exe scripts\eval\test_late_guard_generalization.py `
  --checkpoint_path <checkpoint.pth> `
  --scenario_profiles_path configs/scenario_profiles.yaml `
  --profile_split test `
  --num_seeds 20
```

## 注意事项

1. **观测归一化**: 确保所有输入特征都在 [0, 1] 范围内
2. **ΔRSRP 计算**: 从环境 info 中获取原始 RSRP 值（dBm）计算差值
3. **时间窗口**: 需要维护历史观测窗口，在环境 reset 时清空
4. **动作参数**: 模型输出动作索引，需要通过 ActionSpace 转换为 (Hys, TTT)
5. **NoisyLinear**: 训练时每次前向传播前需要调用 `reset_noise()`

## 当前改进方向

- 保持 `Obs7 + GRU + Rainbow DQN` 小模型作为主线。
- 优先用 `late guard` 这类轻量、安全、可解释的约束处理明确晚切风险。
- Stress / HighInterference 的剩余问题先做归因，区分切换可控损失和覆盖、干扰、资源或队列边界。
- 只有多 seed、多 profile 证据表明新增模块稳定改善核心 KPI 时，才把 physics risk head、R6 reward 或新增观测纳入主线。
