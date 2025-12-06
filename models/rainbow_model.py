"""Rainbow DQN 模型：包含时序编码器、Rainbow Q头、辅助预测头"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple


class NoisyLinear(nn.Module):
    """
    NoisyNet 线性层，用于探索
    
    实现参考 Rainbow DQN 论文中的 NoisyNet
    """
    
    def __init__(self, in_features: int, out_features: int, std_init: float = 0.5):
        """
        Args:
            in_features: 输入特征维度
            out_features: 输出特征维度
            std_init: 噪声标准差初始值
        """
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init
        
        # 可学习的权重和偏置
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        
        # 噪声缓冲区（每次前向传播时重新采样）
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))
        
        self.reset_parameters()
        self.reset_noise()
    
    def reset_parameters(self):
        """初始化参数"""
        mu_range = 1.0 / np.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / np.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / np.sqrt(self.out_features))
    
    def reset_noise(self):
        """重置噪声（每次前向传播前调用）"""
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)
    
    def _scale_noise(self, size: int) -> torch.Tensor:
        """生成因子化的高斯噪声"""
        x = torch.randn(size, device=self.weight_mu.device)
        return x.sign().mul_(x.abs().sqrt_())
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        if self.training:
            # 训练时使用噪声
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            # 评估时只使用均值
            weight = self.weight_mu
            bias = self.bias_mu
        
        return F.linear(input, weight, bias)


class TemporalEncoderGRU(nn.Module):
    """
    时序编码器：使用 GRU 处理时间序列观测
    
    输入: [B, N, F] (batch_size, 时间窗口长度, 特征维度)
    输出: [B, H] (batch_size, 隐层维度)
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 128, num_layers: int = 1, bidirectional: bool = False):
        """
        Args:
            input_dim: 单步观测的特征维度
            hidden_dim: GRU 隐层维度
            num_layers: GRU 层数
            bidirectional: 是否使用双向 GRU
        """
        super(TemporalEncoderGRU, self).__init__()
        self.gru = nn.GRU(
            input_dim, hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional
        )
        self.hidden_dim = hidden_dim * (2 if bidirectional else 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, N, F] 时间序列输入
            
        Returns:
            h: [B, H] 最后时间步的隐层状态
        """
        out, h_n = self.gru(x)  # h_n: [num_layers * num_directions, B, hidden_dim]
        # 取最后一层的最后一个方向的 hidden
        h = h_n[-1]  # [B, hidden_dim]
        return h


class SharedFeatureNet(nn.Module):
    """
    共享特征层：在时序编码器输出后接全连接层
    
    输入: [B, H] (batch_size, 编码器输出维度)
    输出: [B, D] (batch_size, 特征维度)
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 256):
        """
        Args:
            input_dim: 输入维度（编码器输出维度）
            hidden_dim: 隐藏层维度
        """
        super(SharedFeatureNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.output_dim = hidden_dim
    
    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: [B, H] 编码器输出
            
        Returns:
            z: [B, D] 共享特征表示
        """
        return self.net(h)


class RainbowHead(nn.Module):
    """
    Rainbow Q 头：Dueling + C51 分布 + NoisyLinear
    
    输出每个动作的价值分布（C51）
    """
    
    def __init__(self, feature_dim: int, num_actions: int, num_atoms: int = 51, 
                 v_min: float = -60.0, v_max: float = 10.0):
        """
        Args:
            feature_dim: 共享特征维度
            num_actions: 动作数量
            num_atoms: C51 原子数量
            v_min: 价值最小值
            v_max: 价值最大值
        """
        super(RainbowHead, self).__init__()
        self.num_actions = num_actions
        self.num_atoms = num_atoms
        self.v_min = v_min
        self.v_max = v_max
        self.delta_z = (v_max - v_min) / (num_atoms - 1)
        
        # Value stream
        self.value_fc = NoisyLinear(feature_dim, 256)
        self.value_out = NoisyLinear(256, num_atoms)
        
        # Advantage stream
        self.adv_fc = NoisyLinear(feature_dim, 256)
        self.adv_out = NoisyLinear(256, num_actions * num_atoms)
    
    def reset_noise(self):
        """重置所有 NoisyLinear 的噪声"""
        self.value_fc.reset_noise()
        self.value_out.reset_noise()
        self.adv_fc.reset_noise()
        self.adv_out.reset_noise()
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: [B, feature_dim] 共享特征
            
        Returns:
            prob: [B, num_actions, num_atoms] 每个动作的价值分布（概率）
        """
        B = z.size(0)
        
        # Value stream
        v = F.relu(self.value_fc(z))  # [B, 256]
        v = self.value_out(v)  # [B, num_atoms]
        
        # Advantage stream
        a = F.relu(self.adv_fc(z))  # [B, 256]
        a = self.adv_out(a)  # [B, num_actions * num_atoms]
        a = a.view(B, self.num_actions, self.num_atoms)  # [B, num_actions, num_atoms]
        
        # Dueling combine: Q(s,a) = V(s) + A(s,a) - mean(A(s,a'))
        a_mean = a.mean(dim=1, keepdim=True)  # [B, 1, num_atoms]
        q_atoms = v.unsqueeze(1) + a - a_mean  # [B, num_actions, num_atoms]
        
        # 转为概率分布（C51 要求）
        prob = F.softmax(q_atoms, dim=-1)  # [B, num_actions, num_atoms]
        prob = prob.clamp(min=1e-6)  # 避免 log(0)
        
        return prob
    
    def get_q_values(self, dist: torch.Tensor) -> torch.Tensor:
        """
        从分布中计算期望 Q 值
        
        Args:
            dist: [B, num_actions, num_atoms] 价值分布
            
        Returns:
            q_values: [B, num_actions] 期望 Q 值
        """
        support = torch.linspace(self.v_min, self.v_max, self.num_atoms, 
                                 device=dist.device)  # [num_atoms]
        q_values = (dist * support.view(1, 1, -1)).sum(dim=-1)  # [B, num_actions]
        return q_values


class ForecastHead(nn.Module):
    """
    辅助预测头：预测下一步 ΔRSRP
    
    输入: [B, D] (batch_size, 共享特征维度)
    输出: [B] (batch_size, 预测的 ΔRSRP)
    """
    
    def __init__(self, feature_dim: int):
        """
        Args:
            feature_dim: 共享特征维度
        """
        super(ForecastHead, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)  # 预测标量 ΔRSRP_{t+1}
        )
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: [B, feature_dim] 共享特征
            
        Returns:
            pred_delta: [B] 预测的 ΔRSRP
        """
        return self.net(z).squeeze(-1)  # [B]


class RainbowWithForecast(nn.Module):
    """
    完整的 Rainbow DQN + 辅助预测模型
    
    包含：
    1. 时序编码器（GRU）
    2. 共享特征层
    3. Rainbow Q 头（Dueling + C51 + Noisy）
    4. 辅助预测头
    """
    
    def __init__(self, obs_dim: int, num_actions: int, n_steps: int = 15,
                 feature_hidden: int = 256, encoder_hidden: int = 128,
                 num_atoms: int = 51, v_min: float = -60.0, v_max: float = 10.0):
        """
        Args:
            obs_dim: 单步观测的特征维度
            num_actions: 动作数量
            n_steps: 时间窗口长度（历史步数）
            feature_hidden: 共享特征层隐藏维度
            encoder_hidden: 时序编码器隐藏维度
            num_atoms: C51 原子数量
            v_min: 价值最小值
            v_max: 价值最大值
        """
        super(RainbowWithForecast, self).__init__()
        
        self.obs_dim = obs_dim
        self.num_actions = num_actions
        self.n_steps = n_steps
        self.num_atoms = num_atoms
        self.v_min = v_min
        self.v_max = v_max
        self.delta_z = (v_max - v_min) / (num_atoms - 1)
        
        # 时序编码器
        self.encoder = TemporalEncoderGRU(
            input_dim=obs_dim,
            hidden_dim=encoder_hidden,
            num_layers=1,
            bidirectional=False
        )
        
        # 共享特征层
        self.shared = SharedFeatureNet(
            input_dim=self.encoder.hidden_dim,
            hidden_dim=feature_hidden
        )
        
        # Rainbow Q 头
        self.rainbow_head = RainbowHead(
            feature_dim=feature_hidden,
            num_actions=num_actions,
            num_atoms=num_atoms,
            v_min=v_min,
            v_max=v_max
        )
        
        # 辅助预测头
        self.forecast_head = ForecastHead(feature_dim=feature_hidden)
    
    def reset_noise(self):
        """重置所有 NoisyLinear 的噪声"""
        self.rainbow_head.reset_noise()
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            x: [B, N, F] 时间序列输入（batch_size, 时间窗口长度, 特征维度）
            
        Returns:
            dist: [B, num_actions, num_atoms] C51 分布 Q
            pred_delta: [B] 预测 ΔRSRP
        """
        h = self.encoder(x)  # [B, H]
        z = self.shared(h)  # [B, D]
        dist = self.rainbow_head(z)  # [B, A, num_atoms]
        pred_delta = self.forecast_head(z)  # [B]
        return dist, pred_delta
    
    def get_q_values(self, x: torch.Tensor) -> torch.Tensor:
        """
        获取 Q 值（用于动作选择）
        
        Args:
            x: [B, N, F] 时间序列输入
            
        Returns:
            q_values: [B, num_actions] 期望 Q 值
        """
        dist, _ = self.forward(x)
        return self.rainbow_head.get_q_values(dist)
    
    def act(self, x: torch.Tensor, epsilon: float = 0.0) -> int:
        """
        选择动作（用于推理）
        
        Args:
            x: [1, N, F] 时间序列输入（单个样本）
            epsilon: 探索率（0.0 表示贪婪策略）
            
        Returns:
            action: 动作索引
        """
        if np.random.random() < epsilon:
            return np.random.randint(0, self.num_actions)
        
        with torch.no_grad():
            q_values = self.get_q_values(x)  # [1, num_actions]
            action = q_values.argmax(dim=1).item()
        return action
