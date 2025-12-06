"""观测构建工具：构建时序窗口观测"""
import numpy as np
from typing import List, Optional
from collections import deque


class ObservationWindow:
    """
    维护时序观测窗口
    
    根据模型方案，需要构建过去 N 步的观测序列
    每个观测包含：RSRP_serv, RSRP_neig, ΔRSRP, SINR_serv, v_norm, pos_norm, 
                 time_since_last_ho_norm, Hys_norm, TTT_norm, 场景特征等
    """
    
    def __init__(self, window_size: int = 15, obs_dim: int = 10):
        """
        Args:
            window_size: 时间窗口长度 N
            obs_dim: 单步观测的特征维度
        """
        self.window_size = window_size
        self.obs_dim = obs_dim
        self.window = deque(maxlen=window_size)
        self.last_ho_time = -1e9
        self.current_time = 0.0
        self.current_hys = 3.0  # 默认 Hys
        self.current_ttt = 160.0  # 默认 TTT (ms)
    
    def reset(self):
        """重置窗口"""
        self.window.clear()
        self.last_ho_time = -1e9
        self.current_time = 0.0
        self.current_hys = 3.0
        self.current_ttt = 160.0
    
    def update_ho_time(self, ho_time: float):
        """更新最后一次切换时间"""
        self.last_ho_time = ho_time
    
    def update_params(self, hys: float, ttt: float):
        """更新当前 Hys 和 TTT 参数"""
        self.current_hys = hys
        self.current_ttt = ttt
    
    def update_time(self, time: float):
        """更新当前时间"""
        self.current_time = time
    
    def build_observation(self, obs_raw: np.ndarray, info: dict, 
                         velocity_mps: float, track_length_m: float,
                         hys_min: float = 1.5, hys_max: float = 5.0,
                         ttt_min: float = 0.0, ttt_max: float = 640.0) -> np.ndarray:
        """
        构建单步观测并添加到窗口
        
        Args:
            obs_raw: 原始观测 [RSRP_serv, RSRP_neig, SINR_serv, pos_norm, T_norm, H_norm, PM_norm]
            info: 信息字典，包含 rsrp_serv_dbm, rsrp_neig_dbm 等
            velocity_mps: 速度（m/s）
            track_length_m: 轨道长度（m）
            hys_min, hys_max: Hys 归一化范围
            ttt_min, ttt_max: TTT 归一化范围
            
        Returns:
            obs_extended: 扩展后的单步观测
        """
        # 提取原始观测
        rsrp_serv_norm = obs_raw[0]
        rsrp_neig_norm = obs_raw[1]
        sinr_serv_norm = obs_raw[2]
        pos_norm = obs_raw[3]
        T_norm = obs_raw[4]
        H_norm = obs_raw[5]
        PM_norm = obs_raw[6]
        
        # 计算 ΔRSRP（归一化）
        # 从 info 中获取原始 RSRP 值（dBm）
        rsrp_serv_dbm = info.get('rsrp_serv_dbm', -90.0)
        rsrp_neig_dbm = info.get('rsrp_neig_dbm', -90.0)
        delta_rsrp_dbm = rsrp_neig_dbm - rsrp_serv_dbm
        
        # 归一化 ΔRSRP（假设范围 [-30, 30] dB）
        delta_rsrp_min = -30.0
        delta_rsrp_max = 30.0
        delta_rsrp_norm = np.clip(
            (delta_rsrp_dbm - delta_rsrp_min) / (delta_rsrp_max - delta_rsrp_min + 1e-8),
            0.0, 1.0
        )
        
        # 速度归一化（假设范围 [0, 100] m/s，对应 0-360 km/h）
        v_max = 100.0  # m/s
        v_norm = np.clip(velocity_mps / v_max, 0.0, 1.0)
        
        # 距离上次切换的时间归一化（假设最大 10 秒）
        time_since_ho = max(0.0, self.current_time - self.last_ho_time)
        time_since_ho_max = 10.0  # 秒
        time_since_ho_norm = np.clip(time_since_ho / time_since_ho_max, 0.0, 1.0)
        
        # Hys 和 TTT 归一化
        hys_norm = np.clip(
            (self.current_hys - hys_min) / (hys_max - hys_min + 1e-8),
            0.0, 1.0
        )
        ttt_norm = np.clip(
            (self.current_ttt - ttt_min) / (ttt_max - ttt_min + 1e-8),
            0.0, 1.0
        )
        
        # 构建扩展观测（10维）
        obs_extended = np.array([
            rsrp_serv_norm,      # 0: RSRP_serv
            rsrp_neig_norm,      # 1: RSRP_neig
            delta_rsrp_norm,     # 2: ΔRSRP
            sinr_serv_norm,      # 3: SINR_serv
            v_norm,              # 4: v_norm
            pos_norm,            # 5: pos_norm
            time_since_ho_norm,  # 6: time_since_last_ho_norm
            hys_norm,            # 7: Hys_norm
            ttt_norm,            # 8: TTT_norm
            T_norm,              # 9: 天气特征（温度）
            # 可以继续添加 H_norm, PM_norm 等，这里简化只保留 T_norm
        ], dtype=np.float32)
        
        # 添加到窗口
        self.window.append(obs_extended)
        
        return obs_extended
    
    def get_window(self) -> np.ndarray:
        """
        获取当前窗口（填充到 window_size）
        
        Returns:
            window: [N, obs_dim] 如果窗口未满，用第一个观测填充
        """
        if len(self.window) == 0:
            # 如果窗口为空，返回零填充
            return np.zeros((self.window_size, self.obs_dim), dtype=np.float32)
        
        # 如果窗口未满，用第一个观测填充
        first_obs = self.window[0]
        while len(self.window) < self.window_size:
            self.window.appendleft(first_obs)
        
        # 转换为 numpy 数组
        window = np.array(list(self.window), dtype=np.float32)
        return window  # [N, obs_dim]
    
    def is_ready(self) -> bool:
        """检查窗口是否已准备好（至少有一个观测）"""
        return len(self.window) > 0
