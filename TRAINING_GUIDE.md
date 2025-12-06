# Rainbow DQN 模型训练指南

本文档基于当前项目代码，提供完整的训练流程建议和实现方案。

## 一、训练架构概览

### 1.1 核心组件

训练系统需要以下组件：

1. **环境包装器**：将 48 个动作（Hys/TTT 组合）转换为环境可执行的切换逻辑
2. **经验回放缓冲区**：存储时序窗口、动作、奖励、下一窗口等
3. **N-step Return 计算器**：计算多步折扣奖励
4. **C51 投影算法**：实现分布强化学习的 Bellman 投影
5. **训练循环**：主训练流程

### 1.2 数据流

```
环境 → 观测窗口构建 → 模型推理 → 动作选择 → 环境执行
  ↑                                                      ↓
  └─────────── 经验回放缓冲区 ← 存储 (s, a, r, s', done, ΔRSRP)
```

## 二、关键实现细节

### 2.1 环境适配：动作空间转换

**问题**：当前环境只支持 2 个动作（0=不切换，1=切换），但模型输出 48 个动作（Hys/TTT 组合）。

**解决方案**：创建环境包装器，实现基于 Hys/TTT 的切换逻辑。

#### 实现思路

```python
class HandoverWrapper:
    """环境包装器：将 Hys/TTT 动作转换为切换决策"""
    
    def __init__(self, env, action_space):
        self.env = env
        self.action_space = action_space
        self.current_hys = 3.0  # 当前生效的 Hys
        self.current_ttt = 160.0  # 当前生效的 TTT
        self.ttt_timer = 0.0  # TTT 计时器
        self.a3_condition_met = False  # A3 事件条件是否满足
    
    def step(self, action_48):
        """
        执行动作：根据 Hys/TTT 参数决定是否切换
        
        Args:
            action_48: 0-47 的动作索引
            
        Returns:
            (obs, reward, terminated, truncated, info)
        """
        # 1. 将动作转换为 Hys/TTT
        hys, ttt = self.action_space.action_to_hys_ttt(action_48)
        
        # 2. 更新参数（如果参数改变，重置 TTT 计时器）
        if hys != self.current_hys or ttt != self.current_ttt:
            self.current_hys = hys
            self.current_ttt = ttt
            self.ttt_timer = 0.0
            self.a3_condition_met = False
        
        # 3. 获取当前观测（需要原始 RSRP 值）
        obs, info = self.env.get_current_obs()  # 需要扩展环境接口
        rsrp_serv = info['rsrp_serv_dbm']
        rsrp_neig = info['rsrp_neig_dbm']
        
        # 4. 检查 A3 事件：RSRP_neig - RSRP_serv > Hys
        delta_rsrp = rsrp_neig - rsrp_serv
        if delta_rsrp > self.current_hys:
            self.a3_condition_met = True
            self.ttt_timer += self.env.cfg['delta_t_s']
        else:
            self.a3_condition_met = False
            self.ttt_timer = 0.0
        
        # 5. 如果 A3 条件满足且持续超过 TTT，执行切换
        env_action = 0  # 默认不切换
        if self.a3_condition_met and self.ttt_timer >= (self.current_ttt / 1000.0):
            env_action = 1  # 执行切换
            self.ttt_timer = 0.0
            self.a3_condition_met = False
        
        # 6. 执行环境动作
        obs, reward, terminated, truncated, info = self.env.step(env_action)
        
        # 7. 在 info 中添加当前参数
        info['current_hys'] = self.current_hys
        info['current_ttt'] = self.current_ttt
        info['delta_rsrp_dbm'] = delta_rsrp
        
        return obs, reward, terminated, truncated, info
```

**注意**：需要扩展环境接口，使其能够在不执行 step 的情况下获取当前观测。

### 2.2 观测窗口管理

**问题**：模型需要时序窗口（15 步历史），但环境每次只返回单步观测。

**解决方案**：使用 `ObservationWindow` 类维护历史窗口。

#### 使用示例

```python
from models import ObservationWindow

obs_window = ObservationWindow(window_size=15, obs_dim=10)

# 在环境 reset 后
obs_window.reset()
obs_window.update_time(0.0)

# 在每一步
obs_raw, info = env.step(action)
obs_extended = obs_window.build_observation(
    obs_raw, info,
    velocity_mps=env.velocity_mps,
    track_length_m=env.cfg['track_length_m']
)
obs_window.update_time(env.time_step * env.cfg['delta_t_s'])

# 如果发生切换，更新切换时间
if info.get('ho_executed', False):
    obs_window.update_ho_time(env.time_step * env.cfg['delta_t_s'])

# 如果参数改变，更新参数
obs_window.update_params(info['current_hys'], info['current_ttt'])

# 获取完整窗口用于模型输入
window = obs_window.get_window()  # [15, 10]
```

### 2.3 下一步 ΔRSRP 真值计算

**问题**：辅助预测任务需要下一步的 ΔRSRP 真值作为监督信号。

**解决方案**：在存储经验时，预先计算或从下一步观测中获取。

#### 实现方法

**方法 1：延迟存储（推荐）**

```python
# 在 t 步时，存储 t-1 步的经验，此时 t 步的 ΔRSRP 已经知道
if len(obs_history) > 0:
    # 计算当前步的 ΔRSRP（作为上一步的"下一步"）
    delta_rsrp_next = info['rsrp_neig_dbm'] - info['rsrp_serv_dbm']
    
    # 归一化
    delta_rsrp_norm = normalize_delta_rsrp(delta_rsrp_next)
    
    # 存储上一步的经验（包含下一步的 ΔRSRP）
    buffer.store(
        obs_window_prev=obs_history[-1],
        action=action_prev,
        reward=reward_prev,
        next_obs_window=obs_window.get_window(),
        done=done_prev,
        delta_rsrp_target=delta_rsrp_norm
    )
```

**方法 2：提前一步计算**

在环境 step 之前，先"预览"下一步的 RSRP（需要扩展环境接口）。

### 2.4 经验回放缓冲区设计

#### 数据结构

```python
class ReplayBuffer:
    """经验回放缓冲区（支持 PER）"""
    
    def __init__(self, capacity, obs_window_size, obs_dim):
        self.capacity = capacity
        self.obs_window_size = obs_window_size
        self.obs_dim = obs_dim
        
        # 存储的数据
        self.obs_windows = np.zeros((capacity, obs_window_size, obs_dim))
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity)
        self.next_obs_windows = np.zeros((capacity, obs_window_size, obs_dim))
        self.dones = np.zeros(capacity, dtype=bool)
        self.delta_rsrp_targets = np.zeros(capacity)  # 辅助预测目标
        
        # PER 相关
        self.priorities = np.zeros(capacity)
        self.max_priority = 1.0
        
        self.pos = 0
        self.size = 0
    
    def store(self, obs_window, action, reward, next_obs_window, done, delta_rsrp_target):
        """存储一条经验"""
        idx = self.pos % self.capacity
        
        self.obs_windows[idx] = obs_window
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.next_obs_windows[idx] = next_obs_window
        self.dones[idx] = done
        self.delta_rsrp_targets[idx] = delta_rsrp_target
        
        # PER: 新样本优先级最高
        self.priorities[idx] = self.max_priority
        
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size, beta=0.4):
        """采样一个 batch（PER）"""
        if self.size == 0:
            return None
        
        # 计算采样概率
        priorities = self.priorities[:self.size]
        probs = priorities / priorities.sum()
        
        # 采样索引
        indices = np.random.choice(self.size, batch_size, p=probs)
        
        # 计算重要性采样权重
        weights = (self.size * probs[indices]) ** (-beta)
        weights = weights / weights.max()  # 归一化
        
        batch = {
            'obs': self.obs_windows[indices],
            'action': self.actions[indices],
            'reward': self.rewards[indices],
            'next_obs': self.next_obs_windows[indices],
            'done': self.dones[indices],
            'delta_target': self.delta_rsrp_targets[indices],
            'weights': weights,
            'indices': indices
        }
        
        return batch
    
    def update_priorities(self, indices, td_errors):
        """更新优先级（PER）"""
        priorities = np.abs(td_errors) + 1e-6
        self.priorities[indices] = priorities
        self.max_priority = max(self.max_priority, priorities.max())
```

### 2.5 N-step Return 计算

**问题**：Rainbow 使用 N-step return，需要累积未来 N 步的奖励。

**解决方案**：在存储经验时，维护一个 N-step 缓冲区。

```python
class NStepBuffer:
    """N-step return 缓冲区"""
    
    def __init__(self, n_steps, gamma):
        self.n_steps = n_steps
        self.gamma = gamma
        self.buffer = []
    
    def add(self, obs_window, action, reward, next_obs_window, done, delta_rsrp_target):
        """添加一步经验"""
        self.buffer.append({
            'obs': obs_window,
            'action': action,
            'reward': reward,
            'next_obs': next_obs_window,
            'done': done,
            'delta_target': delta_rsrp_target
        })
        
        # 如果缓冲区满了，计算 N-step return
        if len(self.buffer) >= self.n_steps:
            return self._compute_n_step_return()
        return None
    
    def _compute_n_step_return(self):
        """计算 N-step return"""
        # 累积未来 N 步的奖励
        reward_n = 0.0
        for i in range(self.n_steps):
            reward_n += (self.gamma ** i) * self.buffer[i]['reward']
        
        # 获取第一个和最后一个状态
        first = self.buffer[0]
        last = self.buffer[-1]
        
        # 返回 N-step 经验
        n_step_exp = {
            'obs': first['obs'],
            'action': first['action'],
            'reward_n': reward_n,
            'next_obs': last['next_obs'],
            'done': last['done'],
            'delta_target': first['delta_target']  # 使用第一步的目标
        }
        
        # 移除第一步
        self.buffer.pop(0)
        
        return n_step_exp
```

### 2.6 C51 投影算法实现

**关键**：将 Bellman 目标投影回 C51 分布的原子。

```python
def project_distribution(support, target_dist, target_support, v_min, v_max, gamma, reward_n, done):
    """
    C51 投影算法：将目标分布投影回 support
    
    Args:
        support: [num_atoms] 当前分布的原子值
        target_dist: [num_atoms] 目标分布（从 target network 得到）
        target_support: [num_atoms] 目标分布的原子值
        v_min, v_max: 价值范围
        gamma: 折扣因子
        reward_n: N-step 累积奖励
        done: 是否终止
        
    Returns:
        projected_dist: [num_atoms] 投影后的目标分布
    """
    num_atoms = support.size(0)
    delta_z = (v_max - v_min) / (num_atoms - 1)
    
    # 计算目标原子值：Tz = r + γ * z' (如果未终止)
    Tz = reward_n + (1.0 - done) * gamma * target_support  # [num_atoms]
    Tz = Tz.clamp(v_min, v_max)
    
    # 投影到当前 support
    b = (Tz - v_min) / delta_z  # [num_atoms]
    l = b.floor().long()
    u = b.ceil().long()
    
    # 初始化投影分布
    projected_dist = torch.zeros(num_atoms, device=support.device)
    
    # 分配概率质量
    for i in range(num_atoms):
        # 目标分布中第 i 个原子的概率
        prob = target_dist[i]
        
        # 投影到 l[i] 和 u[i]
        if l[i] == u[i]:
            projected_dist[l[i]] += prob
        else:
            # 线性插值
            projected_dist[l[i]] += prob * (u[i].float() - b[i])
            projected_dist[u[i]] += prob * (b[i] - l[i].float())
    
    # 归一化
    projected_dist = projected_dist / (projected_dist.sum() + 1e-8)
    
    return projected_dist
```

### 2.7 训练步骤实现

```python
def train_step(batch, online_net, target_net, optimizer, config):
    """执行一次训练步骤"""
    # 转换为 tensor
    obs = torch.FloatTensor(batch['obs']).to(device)
    action = torch.LongTensor(batch['action']).to(device)
    reward_n = torch.FloatTensor(batch['reward_n']).to(device)
    next_obs = torch.FloatTensor(batch['next_obs']).to(device)
    done = torch.FloatTensor(batch['done']).to(device)
    delta_target = torch.FloatTensor(batch['delta_target']).to(device)
    weights = torch.FloatTensor(batch['weights']).to(device)
    
    B = obs.size(0)
    num_atoms = config['network']['rainbow']['num_atoms']
    num_actions = config['action_space']['num_actions']
    v_min = config['network']['rainbow']['v_min']
    v_max = config['network']['rainbow']['v_max']
    gamma = config['training']['gamma']
    n_steps = config['training']['n_steps']
    lambda_aux = config['training']['lambda_aux']
    
    # 创建 support
    support = torch.linspace(v_min, v_max, num_atoms, device=device)
    
    # 重置噪声
    online_net.train()
    online_net.reset_noise()
    
    # 当前分布和预测
    dist, pred_delta = online_net(obs)
    action_idx = action.view(-1, 1, 1).expand(B, 1, num_atoms)
    dist_a = dist.gather(1, action_idx).squeeze(1)  # [B, num_atoms]
    
    # 目标分布（Double Q）
    with torch.no_grad():
        target_net.eval()
        # 用在线网络选动作
        next_dist_online, _ = online_net(next_obs)
        next_q = (next_dist_online * support.view(1, 1, -1)).sum(dim=-1)
        next_action = next_q.argmax(dim=1, keepdim=True)  # [B, 1]
        
        # 用目标网络得到分布
        next_dist_target, _ = target_net(next_obs)
        next_action_idx = next_action.view(-1, 1, 1).expand(B, 1, num_atoms)
        target_dist_a = next_dist_target.gather(1, next_action_idx).squeeze(1)  # [B, num_atoms]
        
        # C51 投影
        target_support = support.unsqueeze(0).expand(B, -1)  # [B, num_atoms]
        Tz = reward_n.unsqueeze(1) + (1.0 - done.unsqueeze(1)) * (gamma ** n_steps) * target_support
        Tz = Tz.clamp(v_min, v_max)
        
        b = (Tz - v_min) / ((v_max - v_min) / (num_atoms - 1))
        l = b.floor().long()
        u = b.ceil().long()
        
        # 初始化目标分布
        m = torch.zeros_like(dist_a)
        
        # 投影（简化版，实际需要更仔细的实现）
        for i in range(num_atoms):
            for j in range(B):
                if l[j, i] == u[j, i]:
                    m[j, l[j, i]] += target_dist_a[j, i]
                else:
                    m[j, l[j, i]] += target_dist_a[j, i] * (u[j, i].float() - b[j, i])
                    m[j, u[j, i]] += target_dist_a[j, i] * (b[j, i] - l[j, i].float())
        
        m = m / (m.sum(dim=1, keepdim=True) + 1e-8)
    
    # Rainbow 损失（KL 散度）
    log_p = torch.log(dist_a + 1e-6)
    rainbow_loss = -(m * log_p).sum(dim=1)  # [B]
    rainbow_loss = (rainbow_loss * weights).mean()
    
    # 预测损失
    forecast_loss = F.mse_loss(pred_delta, delta_target, reduction='none')
    forecast_loss = (forecast_loss * weights).mean()
    
    # 总损失
    loss = rainbow_loss + lambda_aux * forecast_loss
    
    # 反向传播
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(online_net.parameters(), config['training']['grad_clip'])
    optimizer.step()
    
    # 计算 TD error（用于更新 PER 优先级）
    td_errors = (rainbow_loss.detach() / weights).cpu().numpy()
    
    return {
        'loss': loss.item(),
        'rainbow_loss': rainbow_loss.item(),
        'forecast_loss': forecast_loss.item(),
        'td_errors': td_errors
    }
```

## 三、完整训练流程

### 3.1 训练循环伪代码

```python
def train():
    # 1. 初始化
    env = TrainHandoverEnv(config_path=env_config_path)
    wrapper = HandoverWrapper(env, action_space)
    model = RainbowWithForecast(...)
    target_model = RainbowWithForecast(...)  # 目标网络
    target_model.load_state_dict(model.state_dict())
    
    buffer = ReplayBuffer(...)
    n_step_buffer = NStepBuffer(n_steps=3, gamma=0.99)
    obs_window = ObservationWindow(...)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=6.25e-5)
    
    # 2. 训练循环
    step = 0
    for episode in range(num_episodes):
        obs_raw, info = wrapper.reset()
        obs_window.reset()
        done = False
        
        while not done:
            # 构建观测窗口
            obs_extended = obs_window.build_observation(...)
            window = obs_window.get_window()
            
            # 选择动作
            epsilon = get_epsilon(step)
            window_tensor = torch.FloatTensor(window).unsqueeze(0)
            action = model.act(window_tensor, epsilon)
            
            # 执行动作
            next_obs_raw, reward, terminated, truncated, info = wrapper.step(action)
            done = terminated or truncated
            
            # 更新窗口
            next_obs_extended = obs_window.build_observation(...)
            next_window = obs_window.get_window()
            
            # 计算下一步 ΔRSRP（用于辅助预测）
            delta_rsrp_next = info['rsrp_neig_dbm'] - info['rsrp_serv_dbm']
            delta_rsrp_norm = normalize_delta_rsrp(delta_rsrp_next)
            
            # 添加到 N-step 缓冲区
            n_step_exp = n_step_buffer.add(
                window, action, reward, next_window, done, delta_rsrp_norm
            )
            
            # 如果得到 N-step 经验，存储到回放缓冲区
            if n_step_exp is not None:
                buffer.store(**n_step_exp)
            
            # 训练
            if step % train_freq == 0 and buffer.size >= replay_start_size:
                for _ in range(update_freq):
                    batch = buffer.sample(batch_size, beta=get_beta(step))
                    train_info = train_step(batch, model, target_model, optimizer, config)
                    
                    # 更新 PER 优先级
                    buffer.update_priorities(batch['indices'], train_info['td_errors'])
            
            # 更新目标网络
            if step % target_update_freq == 0:
                target_model.load_state_dict(model.state_dict())
            
            step += 1
```

## 四、实现优先级建议

### 阶段 1：基础功能（必须）

1. ✅ **环境包装器**：实现 Hys/TTT 到切换逻辑的转换
2. ✅ **观测窗口管理**：确保时序窗口正确构建
3. ✅ **经验回放缓冲区**：基础版本（可先不用 PER）
4. ✅ **训练循环框架**：基本的训练流程

### 阶段 2：Rainbow 特性（重要）

1. ✅ **C51 投影算法**：实现分布投影
2. ✅ **N-step Return**：多步奖励累积
3. ✅ **Double Q-learning**：目标网络更新
4. ✅ **NoisyLinear**：已实现，确保正确使用

### 阶段 3：优化（可选）

1. ✅ **PER（优先经验回放）**：提高样本效率
2. ✅ **软更新**：目标网络软更新
3. ✅ **多环境并行**：加速数据收集

## 五、调试建议

1. **先测试环境包装器**：确保动作转换正确
2. **验证观测窗口**：检查窗口形状和内容
3. **小规模测试**：先用小 buffer、短 episode 测试
4. **监控损失**：观察 Rainbow 损失和预测损失的比例
5. **可视化**：绘制训练曲线、Q 值分布等

## 六、常见问题

### Q1: 如何确保时序窗口在 episode 开始时正确初始化？

**A**: 在 reset 时，用第一个观测填充整个窗口。

### Q2: 环境不支持"预览"下一步 RSRP，如何获取 ΔRSRP 真值？

**A**: 使用延迟存储：在 t 步存储 t-1 步的经验，此时 t 步的 ΔRSRP 已知。

### Q3: TTT 计时器如何与环境的 step 频率对齐？

**A**: TTT 单位是毫秒，环境 step 是 50ms，需要转换：`ttt_timer >= (ttt_ms / 1000.0)`。

### Q4: 如何处理 episode 结束时的 N-step 缓冲区？

**A**: 在 episode 结束时，将缓冲区中剩余的经验也存储（使用实际折扣）。

## 七、下一步行动

1. **创建环境包装器** (`envs/handover_wrapper.py`)
2. **实现经验回放缓冲区** (`utils/replay_buffer.py`)
3. **实现 C51 投影** (`utils/c51_projection.py`)
4. **创建训练脚本** (`train_rainbow.py`)
5. **创建评估脚本** (`eval_rainbow.py`)

---

**注意**：这是一个完整的训练方案建议。建议按阶段实现，先确保基础功能正确，再逐步添加高级特性。
