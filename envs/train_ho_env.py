"""铁路切换算法主环境：TrainHandoverEnv"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Tuple, Optional
import yaml
import os

from .weather_model import WeatherModel
from .channel_model import ChannelModel
from .ho_logic import HandoverLogic

# 可选：如果使用 ActionSpace，需要导入
try:
    from models.action_space import ActionSpace
    HAS_ACTION_SPACE = True
except ImportError:
    HAS_ACTION_SPACE = False


class TrainHandoverEnv(gym.Env):
    """
    铁路切换算法环境
    
    遵循 Gym/Gymnasium 接口，用于强化学习训练
    """
    
    metadata = {'render_modes': ['human']}
    
    def __init__(self, config: Optional[Dict] = None, config_path: Optional[str] = None):
        """
        初始化环境
        
        Args:
            config: 配置字典（可选）
            config_path: 配置文件路径（可选）
        """
        super().__init__()
        
        # 1) 加载配置
        self.cfg = self._load_default_config()
        if config_path is not None:
            self._load_config_from_yaml(config_path)
        if config is not None:
            self.cfg.update(config)
        
        # 2) 观测空间：7维
        # [RSRP_serv, RSRP_neig, SINR_serv, pos_norm, T_norm, H_norm, PM_norm]
        high = np.array([1.0] * 7, dtype=np.float32)
        low = np.array([0.0] * 7, dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        
        # 动作空间：48个动作（Hys/TTT 组合）
        # 如果配置中指定了 use_hys_ttt，则使用48个动作，否则保持兼容使用2个动作
        self.use_hys_ttt = self.cfg.get("use_hys_ttt", True)  # 默认使用 Hys/TTT 模式
        
        if self.use_hys_ttt:
            # 从配置读取动作空间参数，或使用默认值
            action_space_cfg = self.cfg.get("action_space", {})
            hys_set = action_space_cfg.get("hys_set", [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0])
            ttt_set = action_space_cfg.get("ttt_set", [0, 40, 80, 160, 320, 640])
            
            # 计算动作数量
            num_actions = len(hys_set) * len(ttt_set)
            
            # 创建动作空间
            self.action_space = spaces.Discrete(num_actions)
            
            # 创建动作空间映射（如果可用）
            if HAS_ACTION_SPACE:
                # 如果 ActionSpace 支持自定义参数，可以传入
                # 目前 ActionSpace 使用硬编码，所以直接创建
                self.action_space_helper = ActionSpace()
                # 验证动作数量是否匹配
                if self.action_space_helper.num_actions != num_actions:
                    print(f"警告：ActionSpace 的动作数量 ({self.action_space_helper.num_actions}) "
                          f"与配置的动作数量 ({num_actions}) 不匹配。"
                          f"将使用配置的动作空间。")
                    self.action_space_helper = None
                    self._init_default_action_space(hys_set, ttt_set)
            else:
                # 如果没有 ActionSpace，使用默认映射
                self.action_space_helper = None
                self._init_default_action_space(hys_set, ttt_set)
        else:
            # 兼容模式：2 个动作（0=不切换, 1=切换）
            self.action_space = spaces.Discrete(2)
            self.action_space_helper = None
        
        # 3) 初始化子模块
        self.weather_model = WeatherModel(self.cfg)
        self.channel_model = ChannelModel(self.cfg, self.weather_model)
        self.ho_logic = HandoverLogic(self.cfg)
        
        # 4) 内部运行状态（每次 reset 会重置）
        self.position_m = None           # 当前列车位置 x
        self.velocity_mps = None         # 当前速度 v
        self.serving_cell = None         # 当前服务小区 (0: A, 1: B)
        self.time_step = None            # 步计数
        
        # 5) L3 滤波用的上一次测量
        self.last_rsrp_serv = None
        self.last_rsrp_neig = None
        self.last_sinr_serv = None
        
        # 6) 切换中断状态
        self.ho_interruption_remaining = 0  # 剩余中断时隙数（0表示无中断）
    
    def _init_default_action_space(self, hys_set=None, ttt_set=None):
        """
        初始化默认动作空间映射（当 ActionSpace 不可用时）
        
        Args:
            hys_set: Hys 集合（如果为 None，使用默认值）
            ttt_set: TTT 集合（如果为 None，使用默认值）
        """
        if hys_set is None:
            hys_set = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
        if ttt_set is None:
            ttt_set = [0, 40, 80, 160, 320, 640]
        
        hys_set = np.array(hys_set, dtype=np.float32)
        ttt_set = np.array(ttt_set, dtype=np.float32)
        
        self.action_to_params = []
        for hys in hys_set:
            for ttt in ttt_set:
                self.action_to_params.append((float(hys), float(ttt)))
        self.action_to_params = np.array(self.action_to_params, dtype=np.float32)
    
    def _action_to_hys_ttt(self, action: int) -> tuple:
        """将动作索引转换为 (Hys, TTT) 参数"""
        if self.action_space_helper is not None:
            return self.action_space_helper.action_to_hys_ttt(action)
        else:
            # 使用默认映射
            hys, ttt = self.action_to_params[action]
            return float(hys), float(ttt)
    
    def _load_default_config(self) -> Dict:
        """加载默认配置"""
        cfg = {
            "track_length_m": 3000.0,
            "delta_t_s": 0.05,           # 50 ms
            "random_speed": False,
            "v_default_kmh": 300.0,
            # 参考 5G‑R 高速铁路场景，车速多在 200~350 km/h
            "v_min_kmh": 120.0,
            "v_max_kmh": 350.0,

            # 5G‑R 宏站典型发射功率（约 40 W，46 dBm）
            "Ptx_A_dbm": 46.0,
            "Ptx_B_dbm": 46.0,
            # 近似 2 GHz 自由空间 + 高架场景的参考路径损耗
            "pl0_A_db": 32.4,
            "pl0_B_db": 32.4,
            # 开阔高架场景，路径损耗指数略大于 2
            "pathloss_exp_A": 2.7,
            "pathloss_exp_B": 2.7,
            # 高速铁路高架场景，阴影衰落略小于典型城市宏站
            "shadow_sigma_A": 6.0,
            "shadow_sigma_B": 6.0,
            # 阴影相关距离（38.901 风格，大尺度衰落随距离相关），高铁场景常取 50~100 m 量级
            "shadow_corr_distance_m": 80.0,
            
            # 约 20 MHz 带宽、5 dB 噪声系数，对应接收热噪声约 -96 dBm
            "noise_dbm": -96.0,
            
            # 同频干扰配置（基于公式 I(d) = 10^(Pr1(d)/10) + 10^(Pr2(d)/10)）
            "use_dynamic_interference": True,  # 启用动态同频干扰（基于位置动态计算）
            "enable_extra_interference": False,  # 是否启用额外同频基站干扰
            "extra_interference_dbm": -105.0,  # 其他同频基站的额外干扰功率（dBm）
            
            # 静态干扰配置（仅当 use_dynamic_interference=False 时使用）
            "interference_A_dbm": -100.0,  # 基站A固定干扰功率（dBm）
            "interference_B_dbm": -100.0,  # 基站B固定干扰功率（dBm）
            
            "l3_alpha": 0.7,
            
            # 关键业务需要较高 SINR，按 5G‑R 控制信令留一定裕度
            "sinr_outage_db": -5.0,
            
            "T_min": -10.0,
            "T_max": 40.0,
            "H_min": 20.0,
            "H_max": 100.0,
            "PM_min": 0.0,
            "PM_max": 300.0,
            
            "a_T": 0.02,
            "a_H": 0.01,
            "a_PM": 0.002,
            
            "rsrp_min_dbm": -120.0,
            "rsrp_max_dbm": -60.0,
            "sinr_min_db": -10.0,
            "sinr_max_db": 20.0,
            
            "C_outage": 10.0,
            "C_ho": 0.3,
            "C_interruption": 0.0,  # 切换中断惩罚系数（默认0，因为中断期间SINR已经很低）
            
            "T_guard_s": 1.0,
            
            # 切换中断配置
            "ho_interruption_slots": 1,  # 切换导致的通信中断时隙数（默认1个时隙，即50ms）
            "ho_interruption_sinr_db": -20.0,  # 中断期间的SINR值（dB），设置为很低的值以模拟通信中断
            "ho_interruption_rsrp_dbm": -120.0,  # 中断期间的RSRP值（dBm），设置为很低的值以模拟通信中断

            # 快衰落控制（38.901 多径的小尺度衰落，简化为 Rayleigh / Rician）
            "enable_fast_fading": False,
            # 若不为 None，则采用 Rician，K 因子单位 dB；否则为 Rayleigh
            "rician_k_factor_db": None,
        }
        return cfg
    
    def _load_config_from_yaml(self, config_path: str):
        """从YAML文件加载配置"""
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                yaml_config = yaml.safe_load(f)
                if yaml_config:
                    self.cfg.update(yaml_config)
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """
        重置环境，初始化一条新轨迹
        
        Args:
            seed: 随机种子
            options: 可选参数
            
        Returns:
            (观测, 信息字典)
        """
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)
        
        # 1) 位置和速度
        self.position_m = 0.0
        if self.cfg["random_speed"]:
            v_kmh = np.random.uniform(self.cfg["v_min_kmh"], self.cfg["v_max_kmh"])
        else:
            v_kmh = self.cfg["v_default_kmh"]
        self.velocity_mps = v_kmh / 3.6
        
        self.time_step = 0
        self.serving_cell = 0  # 默认在 A 小区
        
        # 2) 采样天气
        self.weather_model.sample_weather(seed=seed)
        
        # 3) 初始化上一时刻测量（用于 IIR 滤波）
        rsrp_A, rsrp_B, sinr_A, sinr_B = self.channel_model.compute_link_metrics(self.position_m)
        # 根据 serving_cell 来初始化 serv/neig
        self.last_rsrp_serv = rsrp_A
        self.last_rsrp_neig = rsrp_B
        self.last_sinr_serv = sinr_A
        
        # 4) 重置切换逻辑
        self.ho_logic.reset()
        
        # 5) 重置切换中断状态
        self.ho_interruption_remaining = 0
        
        obs = self._build_observation()
        info = {}
        return obs, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        执行一步仿真
        
        Args:
            action: 动作
                - 如果 use_hys_ttt=True: 0-47 的动作索引（Hys/TTT 组合）
                - 如果 use_hys_ttt=False: 0=不切换, 1=切换（兼容模式）
            
        Returns:
            (观测, 奖励, 终止标志, 截断标志, 信息字典)
        """
        assert self.action_space.contains(action), f"Invalid action: {action}"
        
        # 1) 时间/位置推进
        self.time_step += 1
        dt = self.cfg["delta_t_s"]
        current_time = self.time_step * dt
        self.position_m += self.velocity_mps * dt
        
        # 2) 处理 HO 行为（先处理切换，以便在切换发生的时隙立即应用中断）
        ho_executed = False
        old_serving_cell = self.serving_cell
        need_recompute_link = False  # 标记是否需要重新计算链路
        
        if self.use_hys_ttt:
            # 新模式：根据 Hys/TTT 参数判断是否切换
            # 2.1) 将动作转换为 Hys/TTT 参数
            hys, ttt = self._action_to_hys_ttt(action)
            self.ho_logic.update_hys_ttt(hys, ttt)
            
            # 2.2) 先计算一次链路以获取当前 RSRP（用于判断切换）
            rsrp_A_temp, rsrp_B_temp, _, _ = self.channel_model.compute_link_metrics(self.position_m)
            
            if self.serving_cell == 0:   # A 为服务小区
                rsrp_serv_temp = rsrp_A_temp
                rsrp_neig_temp = rsrp_B_temp
            else:
                rsrp_serv_temp = rsrp_B_temp
                rsrp_neig_temp = rsrp_A_temp
            
            # 2.3) 计算 ΔRSRP
            delta_rsrp = rsrp_neig_temp - rsrp_serv_temp
            
            # 2.4) 根据 A3 事件和 TTT 判断是否切换
            new_cell, ho_executed = self.ho_logic.execute_handover_with_a3(
                current_time, self.serving_cell, delta_rsrp, dt
            )
            
            if ho_executed:
                self.serving_cell = new_cell
                # 切换发生，设置中断时隙数（中断在切换发生的时隙立即开始）
                self.ho_interruption_remaining = self.cfg.get("ho_interruption_slots", 1)
                need_recompute_link = True  # 切换后需要重新计算链路
            else:
                # 未切换，可以复用之前计算的结果
                rsrp_A, rsrp_B = rsrp_A_temp, rsrp_B_temp
                # SINR 需要重新计算（因为服务小区没变，但位置变了，所以还是需要重新计算）
                # 实际上，由于位置变了，RSRP 也会变，所以还是需要重新计算
                # 但我们可以优化：只在切换时重新计算，否则复用
                # 不过为了简化，我们还是统一重新计算
                need_recompute_link = True
        else:
            # 兼容模式：直接根据动作判断
            if action == 1:
                new_cell, ho_executed = self.ho_logic.execute_handover(current_time, self.serving_cell)
                if ho_executed:
                    self.serving_cell = new_cell
                    # 切换发生，设置中断时隙数（中断在切换发生的时隙立即开始）
                    self.ho_interruption_remaining = self.cfg.get("ho_interruption_slots", 1)
        
        # 2.5) 检查并更新切换中断状态（在切换处理后检查，以便在切换发生的时隙立即应用中断）
        in_interruption = False
        if self.ho_interruption_remaining > 0:
            in_interruption = True
            self.ho_interruption_remaining -= 1
        
        # 3) 计算链路（位置/天气 -> 路损 -> RSRP/SINR）
        # 注意：即使未切换，由于位置变化，链路也会变化，所以总是需要重新计算
        rsrp_A, rsrp_B, sinr_A, sinr_B = self.channel_model.compute_link_metrics(self.position_m)
        
        if self.serving_cell == 0:   # A 为服务小区
            rsrp_serv_raw = rsrp_A
            rsrp_neig_raw = rsrp_B
            sinr_serv_raw = sinr_A
        else:
            rsrp_serv_raw = rsrp_B
            rsrp_neig_raw = rsrp_A
            sinr_serv_raw = sinr_B
        
        # 3.5) 如果处于切换中断期间，将SINR和RSRP设置为中断值
        if in_interruption:
            sinr_serv_raw = self.cfg.get("ho_interruption_sinr_db", -20.0)
            rsrp_serv_raw = self.cfg.get("ho_interruption_rsrp_dbm", -120.0)
        
        # 4) L3 IIR 滤波
        # 重要：如果发生切换，需要重置L3滤波状态，避免使用错误的历史值
        if ho_executed:
            # 切换后，直接用新服务小区的原始值初始化滤波状态
            self.last_rsrp_serv = rsrp_serv_raw
            self.last_rsrp_neig = rsrp_neig_raw
            self.last_sinr_serv = sinr_serv_raw
            # 滤波后的值等于原始值（无历史记忆）
            rsrp_serv = rsrp_serv_raw
            rsrp_neig = rsrp_neig_raw
            sinr_serv = sinr_serv_raw
        else:
            # 未切换，正常进行IIR滤波
            alpha = self.cfg["l3_alpha"]
            rsrp_serv = alpha * self.last_rsrp_serv + (1 - alpha) * rsrp_serv_raw
            rsrp_neig = alpha * self.last_rsrp_neig + (1 - alpha) * rsrp_neig_raw
            sinr_serv = alpha * self.last_sinr_serv + (1 - alpha) * sinr_serv_raw
        
        # 更新L3滤波状态（如果未切换，这里会更新；如果切换了，上面已经重置）
        if not ho_executed:
            self.last_rsrp_serv = rsrp_serv
            self.last_rsrp_neig = rsrp_neig
            self.last_sinr_serv = sinr_serv
        
        # 5) outage & 终止判定
        # 检查outage（这里仅用于统计和奖励，不再用于终止，确保轨迹完整到终点）
        outage = self.ho_logic.check_outage(sinr_serv)
        arrived = self.position_m >= self.cfg["track_length_m"]
        terminated = bool(arrived)  # 仅到达终点时终止，不因 outage 终止
        truncated = False  # 可以加最大步数等逻辑
        
        # 6) 奖励
        reward = self._compute_reward(
            sinr_serv=sinr_serv,
            outage=outage,
            ho_executed=ho_executed,
            in_interruption=in_interruption
        )
        
        # 7) 构建下一状态
        obs = self._build_observation()
        
        # 计算 ΔRSRP（用于辅助预测和调试）
        delta_rsrp_dbm = rsrp_neig - rsrp_serv
        
        info = {
            "outage": outage,
            "arrived": arrived,
            "rsrp_serv_dbm": rsrp_serv,
            "rsrp_neig_dbm": rsrp_neig,
            "sinr_serv_db": sinr_serv,
            "ho_executed": ho_executed,
            "position_m": self.position_m,
            "serving_cell": self.serving_cell,
            # 添加原始 A/B RSRP（未经过 L3 滤波，用于可视化）
            "rsrp_A_dbm": rsrp_A,
            "rsrp_B_dbm": rsrp_B,
            # 添加 ΔRSRP（用于辅助预测）
            "delta_rsrp_dbm": delta_rsrp_dbm,
            # 切换中断状态
            "in_interruption": in_interruption,
            "ho_interruption_remaining": self.ho_interruption_remaining,
            # 区分真正的outage和中断期间的outage（中断期间的outage不会终止episode）
            "outage_during_interruption": bool(outage and in_interruption),
        }
        
        # 如果使用 Hys/TTT 模式，添加当前参数
        if self.use_hys_ttt:
            current_hys, current_ttt = self.ho_logic.get_current_params()
            info["current_hys"] = current_hys
            info["current_ttt"] = current_ttt
            info["a3_condition_met"] = self.ho_logic.a3_condition_met
            info["ttt_timer"] = self.ho_logic.ttt_timer
        
        return obs, reward, terminated, truncated, info
    
    def _normalize(self, x: float, x_min: float, x_max: float) -> float:
        """归一化到 [0, 1]"""
        return float(np.clip((x - x_min) / (x_max - x_min + 1e-8), 0.0, 1.0))
    
    def _build_observation(self) -> np.ndarray:
        """构建观测向量"""
        # 使用上一次滤波后的测量
        rsrp_serv = self.last_rsrp_serv
        rsrp_neig = self.last_rsrp_neig
        sinr_serv = self.last_sinr_serv
        
        # 1) 位置归一化
        pos_norm = self.position_m / self.cfg["track_length_m"]
        pos_norm = float(np.clip(pos_norm, 0.0, 1.0))
        
        # 2) RSRP/SINR 归一化
        rsrp_min, rsrp_max = self.cfg["rsrp_min_dbm"], self.cfg["rsrp_max_dbm"]
        sinr_min, sinr_max = self.cfg["sinr_min_db"], self.cfg["sinr_max_db"]
        
        rsrp_serv_n = self._normalize(rsrp_serv, rsrp_min, rsrp_max)
        rsrp_neig_n = self._normalize(rsrp_neig, rsrp_min, rsrp_max)
        sinr_serv_n = self._normalize(sinr_serv, sinr_min, sinr_max)
        
        # 3) 天气归一化
        T_n, H_n, PM_n = self.weather_model.normalize_weather()
        
        obs = np.array([
            rsrp_serv_n,
            rsrp_neig_n,
            sinr_serv_n,
            pos_norm,
            T_n,
            H_n,
            PM_n
        ], dtype=np.float32)
        
        return obs
    
    def _compute_reward(self, sinr_serv: float, outage: bool, ho_executed: bool, in_interruption: bool = False) -> float:
        """
        计算奖励
        
        Args:
            sinr_serv: 服务小区SINR（dB）
            outage: 是否发生outage
            ho_executed: 是否执行了切换
            in_interruption: 是否处于切换中断期间
            
        Returns:
            奖励值
        """
        # 把当前 SINR 按 [sinr_min, sinr_max] 归一化到 [0,1]
        sinr_min, sinr_max = self.cfg["sinr_min_db"], self.cfg["sinr_max_db"]
        sinr_norm = self._normalize(sinr_serv, sinr_min, sinr_max)
        
        reward = sinr_norm
        
        if outage:
            reward -= self.cfg["C_outage"]  # 比如 10.0
        
        if ho_executed:
            reward -= self.cfg["C_ho"]      # 比如 0.3
        
        # 切换中断期间的额外惩罚（中断本身已经导致SINR很低，这里可以额外惩罚）
        if in_interruption:
            C_interruption = self.cfg.get("C_interruption", 0.0)  # 中断惩罚系数（默认0，因为SINR已经很低）
            reward -= C_interruption
        
        return float(reward)
    
    def render(self, mode='human'):
        """渲染环境（可选实现）"""
        if mode == 'human':
            print(f"Step: {self.time_step}, Position: {self.position_m:.2f}m, "
                  f"Serving Cell: {self.serving_cell}, "
                  f"SINR: {self.last_sinr_serv:.2f}dB")

