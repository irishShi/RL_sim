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
    
    # 向量化投影（scatter_add_ 替代 Python 双层循环）
    projected_dist = torch.zeros(B, num_atoms, device=support.device)

    # 上下界权重
    u_weight = b - l.float()          # [B, num_atoms]  (b - floor(b))
    l_weight = 1.0 - u_weight         # [B, num_atoms]  (ceil(b) - b)

    # 将概率质量按权重分配到上下界原子
    projected_dist.scatter_add_(1, l, target_dist * l_weight)
    projected_dist.scatter_add_(1, u, target_dist * u_weight)
    
    # 归一化
    projected_dist = projected_dist / (projected_dist.sum(dim=1, keepdim=True) + 1e-8)
    
    return projected_dist
