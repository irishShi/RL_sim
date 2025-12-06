# 铁路切换算法环境 (Train Handover Environment)

基于强化学习的铁路通信系统切换算法仿真环境。

## 项目结构

```
RL_sim/
├── envs/                        # 环境模块
│   ├── __init__.py
│   ├── train_ho_env.py         # 主环境：TrainHandoverEnv
│   ├── channel_model.py        # 信道模型：路径损耗、阴影衰落、RSRP/SINR
│   ├── weather_model.py        # 天气模型：采样与归一化
│   └── ho_logic.py             # 切换逻辑：HO保护时间、outage判断
├── configs/                     # 配置文件
│   └── default_env_config.yaml # 默认环境参数
├── run_env_test.py             # 测试脚本
├── requirements.txt            # 依赖包
└── README.md                   # 本文件
```

## 环境特性

### 观测空间
7维连续观测向量：
- `RSRP_serv`: 服务小区RSRP（归一化）
- `RSRP_neig`: 邻区RSRP（归一化）
- `SINR_serv`: 服务小区SINR（归一化）
- `pos_norm`: 位置归一化（0~1）
- `T_norm`: 温度归一化（0~1）
- `H_norm`: 湿度归一化（0~1）
- `PM_norm`: PM2.5归一化（0~1）

### 动作空间
离散动作空间：
- `0`: 不切换
- `1`: 切换

### 奖励函数
- **正奖励**: SINR归一化值（鼓励高信号质量）
- **负奖励**: 
  - Outage惩罚：`-C_outage`（默认-10.0）
  - 切换惩罚：`-C_ho`（默认-0.3）

### 终止条件
- 到达终点（`position >= track_length`）
- 发生Outage（`SINR < sinr_outage_db`）

## 安装依赖

```bash
pip install -r requirements.txt
```

## 快速开始

### 1. 基本使用

```python
from envs.train_ho_env import TrainHandoverEnv

# 创建环境
env = TrainHandoverEnv()

# 重置环境
obs, info = env.reset(seed=42)

# 运行一个episode
done = False
while not done:
    action = env.action_space.sample()  # 随机动作
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
```

### 2. 使用配置文件

```python
env = TrainHandoverEnv(config_path="configs/default_env_config.yaml")
```

### 3. 自定义配置

```python
custom_config = {
    "track_length_m": 5000.0,
    "v_default_kmh": 250.0,
    "C_outage": 15.0,
}
env = TrainHandoverEnv(config=custom_config)
```

### 4. 运行测试

```bash
python run_env_test.py
```

## 环境参数说明

主要配置参数（详见 `configs/default_env_config.yaml`）：

- **轨道参数**: 轨道长度、时间步长
- **速度参数**: 默认速度、速度范围
- **发射功率**: 基站A/B的发射功率
- **路径损耗**: 路径损耗指数、阴影衰落标准差
- **天气参数**: 温度、湿度、PM2.5范围及影响系数
- **L3滤波**: IIR滤波系数
- **奖励参数**: Outage惩罚、切换惩罚
- **切换保护**: 最小切换间隔时间

## 模型说明

### 路径损耗模型
```
PL(d) = PL0 + 10*n*log10(d/d0) + Shadowing + L_weather
```

### 天气影响
额外路径损耗：
```
L_weather = a_T * |T - T0| + a_H * max(0, H - 70) + a_PM * PM2.5
```

### L3滤波
使用IIR滤波器平滑RSRP/SINR测量值：
```
filtered = alpha * last_value + (1 - alpha) * current_value
```

## 与强化学习框架集成

### Stable-Baselines3

```python
from stable_baselines3 import DQN
from envs.train_ho_env import TrainHandoverEnv

env = TrainHandoverEnv()
model = DQN("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100000)
```

### Ray RLlib

```python
from ray import tune
from envs.train_ho_env import TrainHandoverEnv

tune.run(
    "PPO",
    config={
        "env": TrainHandoverEnv,
        "num_workers": 4,
    }
)
```

## 扩展功能

可以在此基础上扩展：
- 多小区场景（>2个基站）
- 更复杂的信道模型（多径、多普勒效应）
- 动态天气变化
- 更精细的奖励函数（重叠区奖励、SINR增益奖励等）
- 可视化渲染

## 许可证

本项目仅供研究使用。

