"""动作空间定义：Hys和TTT的组合"""
import numpy as np
from typing import Tuple, List


class ActionSpace:
    """
    动作空间：将 (Hys, TTT) 组合映射为离散动作索引
    
    根据模型方案：
    - Hys_set = {1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5} (8个)
    - TTT_set = {0, 40, 80, 160, 320, 640} (6个)
    - 总动作数 = 8 * 6 = 48
    """
    
    def __init__(self):
        # Hys 离散集合（dB）
        self.hys_set = np.array([1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0], dtype=np.float32)
        # TTT 离散集合（ms）
        self.ttt_set = np.array([0, 40, 80, 160, 320, 640], dtype=np.float32)
        
        self.num_hys = len(self.hys_set)
        self.num_ttt = len(self.ttt_set)
        self.num_actions = self.num_hys * self.num_ttt
        
        # 创建动作索引到 (Hys, TTT) 的映射
        self.action_to_params = []
        for hys in self.hys_set:
            for ttt in self.ttt_set:
                self.action_to_params.append((hys, ttt))
        self.action_to_params = np.array(self.action_to_params, dtype=np.float32)
    
    def action_to_hys_ttt(self, action: int) -> Tuple[float, float]:
        """
        将动作索引转换为 (Hys, TTT) 参数
        
        Args:
            action: 动作索引 (0-47)
            
        Returns:
            (Hys, TTT) 元组
        """
        assert 0 <= action < self.num_actions, f"动作索引 {action} 超出范围 [0, {self.num_actions})"
        hys, ttt = self.action_to_params[action]
        return float(hys), float(ttt)
    
    def hys_ttt_to_action(self, hys: float, ttt: float) -> int:
        """
        将 (Hys, TTT) 参数转换为动作索引
        
        Args:
            hys: 迟滞值（dB）
            ttt: 时间触发阈值（ms）
            
        Returns:
            动作索引
        """
        # 找到最接近的 Hys 和 TTT
        hys_idx = np.argmin(np.abs(self.hys_set - hys))
        ttt_idx = np.argmin(np.abs(self.ttt_set - ttt))
        action = hys_idx * self.num_ttt + ttt_idx
        return int(action)
    
    def get_all_actions(self) -> List[Tuple[float, float]]:
        """获取所有动作的 (Hys, TTT) 组合列表"""
        return [(float(hys), float(ttt)) for hys, ttt in self.action_to_params]
    
    def normalize_hys(self, hys: float, hys_min: float = 1.5, hys_max: float = 5.0) -> float:
        """归一化 Hys 到 [0, 1]"""
        return float(np.clip((hys - hys_min) / (hys_max - hys_min + 1e-8), 0.0, 1.0))
    
    def normalize_ttt(self, ttt: float, ttt_min: float = 0.0, ttt_max: float = 640.0) -> float:
        """归一化 TTT 到 [0, 1]"""
        return float(np.clip((ttt - ttt_min) / (ttt_max - ttt_min + 1e-8), 0.0, 1.0))
