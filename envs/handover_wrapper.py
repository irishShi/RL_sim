"""环境包装器：将 Hys/TTT 动作转换为切换决策"""
import numpy as np
from typing import Tuple, Dict, Optional
from .train_ho_env import TrainHandoverEnv
from models.action_space import ActionSpace


class HandoverWrapper:
    """
    环境包装器：将 48 个动作（Hys/TTT 组合）转换为环境可执行的切换逻辑
    
    实现基于 A3 事件和 TTT 的切换算法：
    - A3 事件：RSRP_neig - RSRP_serv > Hys
    - TTT：条件需持续满足一定时间才触发切换
    """
    
    def __init__(self, env: TrainHandoverEnv, action_space: ActionSpace):
        """
        Args:
            env: 原始环境实例
            action_space: 动作空间（用于动作转换）
        """
        self.env = env
        self.action_space = action_space
        
        # 当前生效的 Hys 和 TTT 参数
        self.current_hys = 3.0  # dB
        self.current_ttt = 160.0  # ms
        
        # TTT 计时器状态
        self.ttt_timer = 0.0  # 秒
        self.a3_condition_met = False  # A3 事件条件是否满足
        
        # 保存上一步的观测（用于获取当前 RSRP）
        self.last_obs = None
        self.last_info = None
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """
        重置环境
        
        Returns:
            (obs, info)
        """
        obs, info = self.env.reset(seed=seed, options=options)
        
        # 重置状态
        self.current_hys = 3.0
        self.current_ttt = 160.0
        self.ttt_timer = 0.0
        self.a3_condition_met = False
        self.last_obs = obs
        self.last_info = info.copy()
        
        # 在 info 中添加当前参数
        info['current_hys'] = self.current_hys
        info['current_ttt'] = self.current_ttt
        info['delta_rsrp_dbm'] = 0.0  # 初始时设为 0
        
        return obs, info
    
    def step(self, action_48: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        执行动作：根据 Hys/TTT 参数决定是否切换
        
        Args:
            action_48: 0-47 的动作索引
            
        Returns:
            (obs, reward, terminated, truncated, info)
        """
        # 1. 将动作转换为 Hys/TTT
        hys, ttt = self.action_space.action_to_hys_ttt(action_48)
        
        # 2. 更新参数（如果参数改变，重置 TTT 计时器）
        if hys != self.current_hys or ttt != self.current_ttt:
            self.current_hys = hys
            self.current_ttt = ttt
            self.ttt_timer = 0.0
            self.a3_condition_met = False
        
        # 3. 获取当前 RSRP（从上一步的 info 中获取，因为环境 step 会推进状态）
        if self.last_info is not None:
            rsrp_serv = self.last_info.get('rsrp_serv_dbm', -90.0)
            rsrp_neig = self.last_info.get('rsrp_neig_dbm', -90.0)
            delta_rsrp = rsrp_neig - rsrp_serv
        else:
            delta_rsrp = 0.0
        
        # 4. 检查 A3 事件：RSRP_neig - RSRP_serv > Hys
        dt = self.env.cfg['delta_t_s']
        
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
        
        # 5. 如果 A3 条件满足且持续超过 TTT，执行切换
        env_action = 0  # 默认不切换
        ttt_seconds = self.current_ttt / 1000.0  # 转换为秒
        
        if self.a3_condition_met and self.ttt_timer >= ttt_seconds:
            env_action = 1  # 执行切换
            self.ttt_timer = 0.0
            self.a3_condition_met = False
        
        # 6. 执行环境动作
        obs, reward, terminated, truncated, info = self.env.step(env_action)
        
        # 7. 在 info 中添加当前参数和 ΔRSRP
        info['current_hys'] = self.current_hys
        info['current_ttt'] = self.current_ttt
        info['delta_rsrp_dbm'] = delta_rsrp
        info['a3_condition_met'] = self.a3_condition_met
        info['ttt_timer'] = self.ttt_timer
        
        # 8. 保存当前观测和 info（用于下一步）
        self.last_obs = obs
        self.last_info = info.copy()
        
        return obs, reward, terminated, truncated, info
    
    @property
    def observation_space(self):
        """返回观测空间"""
        return self.env.observation_space
    
    @property
    def action_space(self):
        """返回动作空间（48 个动作）"""
        # 这里应该返回新的动作空间，但为了兼容，先返回原始空间
        # 实际使用时，应该创建新的 Discrete(48) 空间
        return self.env.action_space
    
    def get_num_actions(self) -> int:
        """返回动作数量"""
        return self.action_space.num_actions
