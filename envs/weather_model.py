"""天气模型：采样与归一化"""
import numpy as np
from typing import Dict


class WeatherModel:
    """天气模型类，负责天气参数的采样和额外损耗计算"""
    
    def __init__(self, config: Dict):
        """
        初始化天气模型
        
        Args:
            config: 配置字典，包含天气相关参数
        """
        self.cfg = config
        self.temperature = None
        self.humidity = None
        self.pm25 = None
    
    def sample_weather(self, seed: int = None):
        """
        采样天气参数
        
        Args:
            seed: 随机种子（如果为None，使用当前随机数生成器状态）
        """
        if seed is not None:
            np.random.seed(seed)
        
        self.temperature = np.random.uniform(
            self.cfg["T_min"], 
            self.cfg["T_max"]
        )
        self.humidity = np.random.uniform(
            self.cfg["H_min"], 
            self.cfg["H_max"]
        )
        self.pm25 = np.random.uniform(
            self.cfg["PM_min"], 
            self.cfg["PM_max"]
        )
    
    def compute_weather_loss_db(self) -> float:
        """
        根据当前温度/湿度/PM2.5，计算额外路径损耗（dB）
        
        Returns:
            额外路径损耗（dB）
        """
        T0 = 20.0  # 参考温度
        a_T = self.cfg["a_T"]
        a_H = self.cfg["a_H"]
        a_PM = self.cfg["a_PM"]
        
        dT = abs(self.temperature - T0)
        excess_H = max(0.0, self.humidity - 70.0)
        pm = self.pm25
        
        Lw = a_T * dT + a_H * excess_H + a_PM * pm
        return Lw
    
    def normalize_weather(self) -> tuple:
        """
        归一化天气参数到 [0, 1]
        
        Returns:
            (T_norm, H_norm, PM_norm)
        """
        def _normalize(x, x_min, x_max):
            return float(np.clip((x - x_min) / (x_max - x_min + 1e-8), 0.0, 1.0))
        
        T_n = _normalize(self.temperature, self.cfg["T_min"], self.cfg["T_max"])
        H_n = _normalize(self.humidity, self.cfg["H_min"], self.cfg["H_max"])
        PM_n = _normalize(self.pm25, self.cfg["PM_min"], self.cfg["PM_max"])
        
        return T_n, H_n, PM_n

