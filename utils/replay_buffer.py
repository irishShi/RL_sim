"""经验回放缓冲区（支持PER和N-step）"""
import numpy as np
from typing import Dict, Optional, Tuple
from collections import deque


class ReplayBuffer:
    """经验回放缓冲区（支持优先经验回放 PER）"""
    
    def __init__(self, capacity: int, obs_window_size: int, obs_dim: int, 
                 per_alpha: float = 0.6):
        """
        Args:
            capacity: 缓冲区容量
            obs_window_size: 观测窗口长度
            obs_dim: 单步观测维度
            per_alpha: PER 优先级指数
        """
        self.capacity = capacity
        self.obs_window_size = obs_window_size
        self.obs_dim = obs_dim
        self.per_alpha = per_alpha
        
        # 存储的数据
        self.obs_windows = np.zeros((capacity, obs_window_size, obs_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_obs_windows = np.zeros((capacity, obs_window_size, obs_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=bool)
        self.delta_rsrp_targets = np.zeros(capacity, dtype=np.float32)  # 辅助预测目标
        
        # PER 相关
        self.priorities = np.zeros(capacity)
        self.max_priority = 1.0
        
        self.pos = 0
        self.size = 0
    
    def store(self, obs_window: np.ndarray, action: int, reward: float,
              next_obs_window: np.ndarray, done: bool, delta_rsrp_target: float):
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

    def load_from_dataset(self, dataset: Dict):
        """从离线数据集批量加载（比逐条store更快）"""
        num = min(len(dataset['actions']), self.capacity)
        if num <= 0:
            return

        self.obs_windows[:num] = dataset['obs'][:num]
        self.actions[:num] = dataset['actions'][:num]
        self.rewards[:num] = dataset['rewards'][:num]
        self.next_obs_windows[:num] = dataset['next_obs'][:num]
        self.dones[:num] = dataset['dones'][:num]
        self.delta_rsrp_targets[:num] = dataset['delta_targets'][:num]

        self.priorities[:num] = self.max_priority
        self.pos = num % self.capacity
        self.size = num
    
    def sample(self, batch_size: int, beta: float = 0.4) -> Optional[Dict]:
        """
        采样一个 batch（PER）
        
        Args:
            batch_size: batch 大小
            beta: PER 重要性采样指数
            
        Returns:
            batch 字典，包含 'indices' 和 'weights'
        """
        if self.size < batch_size:
            return None
        
        # 计算采样概率
        priorities = self.priorities[:self.size]
        probs = priorities ** self.per_alpha
        probs = probs / probs.sum()
        
        # 采样索引
        indices = np.random.choice(self.size, batch_size, p=probs, replace=False)
        
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
            'weights': weights.astype(np.float32),
            'indices': indices
        }
        
        return batch
    
    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray):
        """更新优先级（PER）"""
        priorities = np.abs(td_errors) + 1e-6
        self.priorities[indices] = priorities
        self.max_priority = max(self.max_priority, priorities.max())


class NStepBuffer:
    """N-step return 缓冲区"""
    
    def __init__(self, n_steps: int, gamma: float):
        """
        Args:
            n_steps: N-step 步数
            gamma: 折扣因子
        """
        self.n_steps = n_steps
        self.gamma = gamma
        self.buffer = deque(maxlen=n_steps)
    
    def add(self, obs_window: np.ndarray, action: int, reward: float,
            next_obs_window: np.ndarray, done: bool, delta_rsrp_target: float) -> Optional[Dict]:
        """
        添加一步经验，如果缓冲区满了，返回 N-step 经验
        
        Returns:
            N-step 经验字典，或 None（如果缓冲区未满）
        """
        self.buffer.append({
            'obs_window': obs_window,
            'action': action,
            'reward': reward,
            'next_obs_window': next_obs_window,
            'done': done,
            'delta_rsrp_target': delta_rsrp_target
        })
        
        # 如果缓冲区满了，计算 N-step return
        if len(self.buffer) >= self.n_steps:
            return self._compute_n_step_return()
        return None
    
    def _compute_n_step_return(self) -> Dict:
        """计算 N-step return"""
        # 累积未来 N 步的奖励
        reward_n = 0.0
        for i in range(self.n_steps):
            reward_n += (self.gamma ** i) * self.buffer[i]['reward']
        
        # 获取第一个和最后一个状态
        first = self.buffer[0]
        last = self.buffer[-1]
        
        # 返回 N-step 经验（使用与 ReplayBuffer.store() 匹配的参数名）
        n_step_exp = {
            'obs_window': first['obs_window'],
            'action': first['action'],
            'reward': reward_n,  # N-step 累积奖励
            'next_obs_window': last['next_obs_window'],
            'done': last['done'],
            'delta_rsrp_target': last['delta_rsrp_target']  # 与 next_obs_window 对齐
        }
        
        # 移除第一步
        self.buffer.popleft()
        
        return n_step_exp
    
    def flush(self) -> list:
        """清空缓冲区，返回剩余的经验（使用实际折扣）"""
        experiences = []
        while len(self.buffer) > 0:
            # 计算剩余步数的折扣奖励
            reward_n = 0.0
            for i in range(len(self.buffer)):
                reward_n += (self.gamma ** i) * self.buffer[i]['reward']
            
            first = self.buffer[0]
            last = self.buffer[-1] if len(self.buffer) > 1 else first
            
            experiences.append({
                'obs_window': first['obs_window'],
                'action': first['action'],
                'reward': reward_n,  # 累积奖励
                'next_obs_window': last['next_obs_window'],
                'done': last['done'],
                'delta_rsrp_target': last['delta_rsrp_target']  # 与 next_obs_window 对齐
            })
            
            self.buffer.popleft()
        
        return experiences
    
    def reset(self):
        """重置缓冲区"""
        self.buffer.clear()
