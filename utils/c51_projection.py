"""C51 分布投影算法"""
import torch
import torch.nn.functional as F


def project_distribution(support: torch.Tensor, target_dist: torch.Tensor,
                        target_support: torch.Tensor, v_min: float, v_max: float,
                        gamma: float, reward_n: torch.Tensor, done: torch.Tensor,
                        n_steps: int = 1) -> torch.Tensor:
    """
    C51 投影算法：将目标分布投影回 support
    
    Args:
        support: [num_atoms] 当前分布的原子值
        target_dist: [B, num_atoms] 目标分布（从 target network 得到）
        target_support: [B, num_atoms] 目标分布的原子值
        v_min, v_max: 价值范围
        gamma: 折扣因子
        reward_n: [B] N-step 累积奖励
        done: [B] 是否终止
        n_steps: N-step 步数
        
    Returns:
        projected_dist: [B, num_atoms] 投影后的目标分布
    """
    B = target_dist.size(0)
    num_atoms = support.size(0)
    delta_z = (v_max - v_min) / (num_atoms - 1)
    
    # 计算目标原子值：Tz = r + γ^n * z' (如果未终止)
    Tz = reward_n.unsqueeze(1) + (1.0 - done.unsqueeze(1)) * (gamma ** n_steps) * target_support
    Tz = Tz.clamp(v_min, v_max)
    
    # 投影到当前 support
    b = (Tz - v_min) / delta_z  # [B, num_atoms]
    l = b.floor().long().clamp(0, num_atoms - 1)  # [B, num_atoms]
    u = b.ceil().long().clamp(0, num_atoms - 1)   # [B, num_atoms]
    
    # 初始化投影分布
    projected_dist = torch.zeros(B, num_atoms, device=support.device)
    
    # 分配概率质量
    for i in range(B):
        for j in range(num_atoms):
            prob = target_dist[i, j]
            l_val = l[i, j].item()
            u_val = u[i, j].item()
            b_val = b[i, j].item()
            
            if l_val == u_val:
                projected_dist[i, l_val] += prob
            else:
                # 线性插值
                projected_dist[i, l_val] += prob * (u_val - b_val)
                projected_dist[i, u_val] += prob * (b_val - l_val)
    
    # 归一化
    projected_dist = projected_dist / (projected_dist.sum(dim=1, keepdim=True) + 1e-8)
    
    return projected_dist
