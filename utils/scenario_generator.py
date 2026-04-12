"""
场景生成器：生成并保存场景数据，确保不同策略在相同场景下运行
"""
import numpy as np
import pickle
import os
from typing import Dict, Optional, Tuple


class ScenarioGenerator:
    """
    场景生成器类
    
    功能：
    1. 生成完整的场景数据（位置序列、阴影衰落序列、天气参数等）
    2. 保存场景数据到文件
    3. 从文件加载场景数据
    """
    
    def __init__(self, config: Dict):
        """
        初始化场景生成器
        
        Args:
            config: 环境配置字典
        """
        self.cfg = config
        self.track_length_m = config.get("track_length_m", 3000.0)
        self.delta_t_s = config.get("delta_t_s", 0.05)
        self.v_default_kmh = config.get("v_default_kmh", 300.0)
        self.random_speed = config.get("random_speed", False)
        self.v_min_kmh = config.get("v_min_kmh", 120.0)
        self.v_max_kmh = config.get("v_max_kmh", 350.0)
        
        # 阴影衰落参数
        self.shadow_sigma_A = config.get("shadow_sigma_A", 6.0)
        self.shadow_sigma_B = config.get("shadow_sigma_B", 6.0)
        self.shadow_corr_distance_m = config.get("shadow_corr_distance_m", 80.0)
        
        # 天气参数范围
        self.T_min = config.get("T_min", -10.0)
        self.T_max = config.get("T_max", 40.0)
        self.H_min = config.get("H_min", 20.0)
        self.H_max = config.get("H_max", 100.0)
        self.PM_min = config.get("PM_min", 0.0)
        self.PM_max = config.get("PM_max", 300.0)
    
    def generate_scenario(self, seed: Optional[int] = None, 
                         position_resolution_m: float = 1.0) -> Dict:
        """
        生成完整的场景数据
        
        Args:
            seed: 随机种子
            position_resolution_m: 位置分辨率（米），用于生成阴影衰落序列
            
        Returns:
            场景数据字典，包含：
            - positions: 位置序列（米）
            - shadow_A: 小区A的阴影衰落序列（dB）
            - shadow_B: 小区B的阴影衰落序列（dB）
            - weather: 天气参数字典（temperature, humidity, pm25）
            - velocity_mps: 速度（m/s）
            - seed: 使用的随机种子
        """
        if seed is not None:
            np.random.seed(seed)
        
        # 1. 采样速度
        if self.random_speed:
            v_kmh = np.random.uniform(self.v_min_kmh, self.v_max_kmh)
        else:
            v_kmh = self.v_default_kmh
        velocity_mps = v_kmh / 3.6
        
        # 2. 采样天气参数
        temperature = np.random.uniform(self.T_min, self.T_max)
        humidity = np.random.uniform(self.H_min, self.H_max)
        pm25 = np.random.uniform(self.PM_min, self.PM_max)
        
        # 3. 生成位置序列（基于速度和时间步长）
        # 计算总步数（估计）
        total_time_s = self.track_length_m / velocity_mps
        max_steps = int(total_time_s / self.delta_t_s) + 100  # 加一些余量
        
        # 生成位置序列（实际位置会在运行时根据速度动态计算，这里生成参考位置序列）
        # 为了生成阴影衰落序列，我们需要一个位置网格
        num_positions = int(self.track_length_m / position_resolution_m) + 1
        positions = np.linspace(0.0, self.track_length_m, num_positions)
        
        # 4. 生成阴影衰落序列（使用AR(1)模型）
        shadow_A = self._generate_shadow_sequence(positions, self.shadow_sigma_A)
        shadow_B = self._generate_shadow_sequence(positions, self.shadow_sigma_B)
        
        scenario = {
            "positions": positions,
            "shadow_A": shadow_A,
            "shadow_B": shadow_B,
            "weather": {
                "temperature": float(temperature),
                "humidity": float(humidity),
                "pm25": float(pm25)
            },
            "velocity_mps": float(velocity_mps),
            "seed": seed,
            "config": {
                "track_length_m": self.track_length_m,
                "delta_t_s": self.delta_t_s,
                "position_resolution_m": position_resolution_m,
                "shadow_corr_distance_m": self.shadow_corr_distance_m
            }
        }
        
        return scenario
    
    def _generate_shadow_sequence(self, positions: np.ndarray, sigma: float) -> np.ndarray:
        """
        生成阴影衰落序列（AR(1)模型）
        
        Args:
            positions: 位置序列（米）
            sigma: 阴影衰落标准差（dB）
            
        Returns:
            阴影衰落序列（dB）
        """
        d_corr = self.shadow_corr_distance_m
        shadow = np.zeros(len(positions))
        
        # 初始值
        shadow[0] = np.random.normal(0.0, sigma)
        
        # AR(1) 模型
        for i in range(1, len(positions)):
            delta_d = abs(positions[i] - positions[i-1])
            rho = float(np.exp(-delta_d / max(d_corr, 1e-3)))
            w = np.random.normal(0.0, sigma)
            shadow[i] = rho * shadow[i-1] + np.sqrt(1.0 - rho ** 2) * w
        
        return shadow
    
    def save_scenario(self, scenario: Dict, filepath: str):
        """
        保存场景数据到文件
        
        Args:
            scenario: 场景数据字典
            filepath: 保存路径
        """
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(scenario, f)
        print(f"场景数据已保存至: {filepath}")
    
    @staticmethod
    def load_scenario(filepath: str) -> Dict:
        """
        从文件加载场景数据
        
        Args:
            filepath: 文件路径
            
        Returns:
            场景数据字典
        """
        with open(filepath, 'rb') as f:
            scenario = pickle.load(f)
        print(f"场景数据已从 {filepath} 加载")
        return scenario
    
    def get_shadow_at_position(self, scenario: Dict, x_m: float, 
                               is_cell_A: bool = True) -> float:
        """
        根据位置获取阴影衰落值（通过插值）
        
        Args:
            scenario: 场景数据字典
            x_m: 位置（米）
            is_cell_A: 是否为小区A
            
        Returns:
            阴影衰落值（dB）
        """
        positions = scenario["positions"]
        shadow_seq = scenario["shadow_A"] if is_cell_A else scenario["shadow_B"]
        
        # 限制位置范围
        x_m = np.clip(x_m, positions[0], positions[-1])
        
        # 线性插值
        shadow = np.interp(x_m, positions, shadow_seq)
        
        return float(shadow)
