"""切换逻辑：HO保护时间、A3事件、TTT、outage判断等"""
from typing import Tuple, Optional


class HandoverLogic:
    """切换逻辑类，负责处理切换行为、A3事件、TTT和保护时间"""
    
    def __init__(self, config: dict):
        """
        初始化切换逻辑
        
        Args:
            config: 配置字典
        """
        self.cfg = config
        self.last_ho_time = -1e9  # 很远的过去
        self.last_ho_position = None  # 最后一次切换的位置（用于距离模式）
        
        # A3 事件和 TTT 相关状态
        self.current_hys = 3.0  # 当前 Hys 参数（dB）
        self.current_ttt = 160.0  # 当前 TTT 参数（ms）
        self.ttt_timer = 0.0  # TTT 计时器（秒）
        self.a3_condition_met = False  # A3 事件条件是否满足
    
    def can_handover(self, current_time: float, current_position: float = None) -> bool:
        """
        判断是否可以执行切换（检查保护时间或保护距离）
        
        Args:
            current_time: 当前时间（秒）
            current_position: 当前位置（米），用于距离模式
            
        Returns:
            是否可以切换
        """
        tts_mode = self.cfg.get("TTS_mode", "time")
        
        if tts_mode == "distance" and current_position is not None:
            # 距离模式：检查是否超过保护距离
            if not hasattr(self, 'last_ho_position') or self.last_ho_position is None:
                self.last_ho_position = current_position
                return True  # 首次切换允许
            
            distance_since_last_ho = abs(current_position - self.last_ho_position)
            T_guard_m = self.cfg.get("T_guard_m", 50.0)
            return distance_since_last_ho >= T_guard_m
        else:
            # 时间模式（默认）
            time_since_last_ho = current_time - self.last_ho_time
            T_guard_s = self.cfg.get("T_guard_s", 0.0)
            return time_since_last_ho >= T_guard_s
    
    def update_hys_ttt(self, hys: float, ttt: float):
        """
        更新 Hys 和 TTT 参数
        
        Args:
            hys: 迟滞值（dB）
            ttt: 时间触发阈值（ms）
        """
        # 如果参数改变，重置 TTT 计时器
        if hys != self.current_hys or ttt != self.current_ttt:
            self.current_hys = hys
            self.current_ttt = ttt
            self.ttt_timer = 0.0
            self.a3_condition_met = False
    
    def check_a3_event(self, delta_rsrp: float, dt: float) -> bool:
        """
        检查 A3 事件并更新 TTT 计时器
        
        A3 事件：RSRP_neig - RSRP_serv > Hys
        
        Args:
            delta_rsrp: RSRP 差值（dB），即 RSRP_neig - RSRP_serv
            dt: 时间步长（秒）
            
        Returns:
            是否应该执行切换（A3 条件满足且 TTT 计时器达到阈值）
        """
        # 检查 A3 事件条件
        if delta_rsrp > self.current_hys:
            # A3 条件满足
            if not self.a3_condition_met:
                # 条件刚满足，重置计时器
                self.a3_condition_met = True
                self.ttt_timer = dt
            else:
                # 条件持续满足，累积计时器
                self.ttt_timer += dt
        else:
            # A3 条件不满足，重置状态
            self.a3_condition_met = False
            self.ttt_timer = 0.0
        
        # 检查是否应该执行切换
        ttt_seconds = self.current_ttt / 1000.0  # 转换为秒
        
        if self.a3_condition_met and self.ttt_timer >= ttt_seconds:
            # 达到切换条件，重置计时器
            self.ttt_timer = 0.0
            self.a3_condition_met = False
            return True
        
        return False
    
    def execute_handover(self, current_time: float, serving_cell: int, current_position: float = None) -> Tuple[int, bool]:
        """
        执行切换（旧接口，保持兼容）
        
        Args:
            current_time: 当前时间（秒）
            serving_cell: 当前服务小区 (0: A, 1: B)
            current_position: 当前位置（米），用于距离模式
            
        Returns:
            (新的服务小区, 是否成功执行切换)
        """
        if self.can_handover(current_time, current_position):
            new_cell = 1 - serving_cell  # A<->B 切换
            self.last_ho_time = current_time
            if current_position is not None:
                self.last_ho_position = current_position
            return new_cell, True
        return serving_cell, False
    
    def execute_handover_with_a3(self, current_time: float, serving_cell: int, 
                                  delta_rsrp: float, dt: float, current_position: float = None) -> Tuple[int, bool]:
        """
        执行切换（新接口，支持 A3 事件和 TTT）
        
        Args:
            current_time: 当前时间（秒）
            serving_cell: 当前服务小区 (0: A, 1: B)
            delta_rsrp: RSRP 差值（dB）
            dt: 时间步长（秒）
            current_position: 当前位置（米），用于距离模式
            
        Returns:
            (新的服务小区, 是否成功执行切换)
        """
        # 检查 A3 事件和 TTT
        should_handover = self.check_a3_event(delta_rsrp, dt)
        
        # 如果满足切换条件且通过保护时间/距离检查，执行切换
        if should_handover and self.can_handover(current_time, current_position):
            new_cell = 1 - serving_cell  # A<->B 切换
            self.last_ho_time = current_time
            if current_position is not None:
                self.last_ho_position = current_position
            return new_cell, True
        
        return serving_cell, False
    
    def check_outage(self, sinr_serv: float) -> bool:
        """
        检查是否发生outage
        
        Args:
            sinr_serv: 服务小区SINR（dB）
            
        Returns:
            是否发生outage
        """
        return sinr_serv < self.cfg["sinr_outage_db"]
    
    def reset(self):
        """重置切换逻辑状态"""
        self.last_ho_time = -1e9
        self.last_ho_position = None
        self.current_hys = 3.0
        self.current_ttt = 160.0
        self.ttt_timer = 0.0
        self.a3_condition_met = False
    
    def get_current_params(self) -> Tuple[float, float]:
        """获取当前 Hys 和 TTT 参数"""
        return self.current_hys, self.current_ttt

