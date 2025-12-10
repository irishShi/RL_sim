"""
离线数据集加载器

从预收集的数据集加载经验，用于离线学习
"""
import numpy as np
from typing import Dict, Optional
import os


class OfflineDataset:
    """离线数据集加载器"""
    
    def __init__(self, data_path: str, obs_window_size: int, obs_dim: int):
        """
        Args:
            data_path: 数据集路径（.npz文件）
            obs_window_size: 观测窗口大小
            obs_dim: 观测维度
        """
        self.data_path = data_path
        self.obs_window_size = obs_window_size
        self.obs_dim = obs_dim
        self.data = None
        
    def load(self) -> Dict:
        """
        加载数据集
        
        Returns:
            data: 数据集字典，包含：
                - obs: [N, window_size, obs_dim]
                - actions: [N]
                - rewards: [N]
                - next_obs: [N, window_size, obs_dim]
                - dones: [N]
                - delta_targets: [N]
        """
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"数据集文件不存在: {self.data_path}")
        
        print(f"加载数据集: {self.data_path}")
        data = np.load(self.data_path, allow_pickle=True)
        
        # 提取数据
        dataset = {
            'obs': data['obs'],
            'actions': data['actions'],
            'rewards': data['rewards'],
            'next_obs': data['next_obs'],
            'dones': data['dones'],
            'delta_targets': data['delta_targets'],
        }
        
        # 验证数据形状
        num_samples = dataset['obs'].shape[0]
        assert dataset['obs'].shape == (num_samples, self.obs_window_size, self.obs_dim), \
            f"观测形状不匹配: {dataset['obs'].shape} != ({num_samples}, {self.obs_window_size}, {self.obs_dim})"
        assert dataset['actions'].shape == (num_samples,), \
            f"动作形状不匹配: {dataset['actions'].shape} != ({num_samples},)"
        assert dataset['rewards'].shape == (num_samples,), \
            f"奖励形状不匹配: {dataset['rewards'].shape} != ({num_samples},)"
        assert dataset['next_obs'].shape == (num_samples, self.obs_window_size, self.obs_dim), \
            f"下一观测形状不匹配: {dataset['next_obs'].shape} != ({num_samples}, {self.obs_window_size}, {self.obs_dim})"
        assert dataset['dones'].shape == (num_samples,), \
            f"Done标志形状不匹配: {dataset['dones'].shape} != ({num_samples},)"
        assert dataset['delta_targets'].shape == (num_samples,), \
            f"Delta目标形状不匹配: {dataset['delta_targets'].shape} != ({num_samples},)"
        
        print(f"数据集加载成功！")
        print(f"  样本数: {num_samples}")
        print(f"  观测形状: {dataset['obs'].shape}")
        print(f"  动作形状: {dataset['actions'].shape}")
        
        # 统计信息
        print(f"  动作分布: {np.bincount(dataset['actions'], minlength=48)}")
        print(f"  平均奖励: {np.mean(dataset['rewards']):.4f}")
        print(f"  完成率: {np.mean(dataset['dones']):.4f}")
        
        self.data = dataset
        return dataset
    
    def get_size(self) -> int:
        """获取数据集大小"""
        if self.data is None:
            self.load()
        return self.data['obs'].shape[0]
    
    def get_statistics(self) -> Dict:
        """获取数据集统计信息"""
        if self.data is None:
            self.load()
        
        return {
            'num_samples': self.data['obs'].shape[0],
            'mean_reward': float(np.mean(self.data['rewards'])),
            'std_reward': float(np.std(self.data['rewards'])),
            'min_reward': float(np.min(self.data['rewards'])),
            'max_reward': float(np.max(self.data['rewards'])),
            'done_rate': float(np.mean(self.data['dones'])),
            'action_distribution': np.bincount(self.data['actions'], minlength=48).tolist()
        }

