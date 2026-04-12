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
        
        # 7) Outage/RLF 事件检测状态
        self.outage_start_time = None  # 当前outage开始时间
        self.outage_events = 0  # 累计outage事件数（连续outage >= T_out记一次）
        self.outage_total_time = 0.0  # 累计outage总时长（秒）
        
        # 8) KPI统计（每个episode）
        self.episode_kpis = {
            "ho_count": 0,  # 切换次数
            "outage_time_ratio": 0.0,  # Outage时间占比
            "outage_events": 0,  # Outage事件数
            "sinr_samples": [],  # SINR样本（用于计算分位数）
            "interruption_total_time": 0.0,  # 中断总时长（全程）
            "interruption_time_in_overlap_zone": 0.0,  # 仅统计位于切换重叠区内的断连时长（s）
            "ping_pong_count": 0,  # 乒乓切换次数（1秒内来回切换）
            "overlap_zone_time": 0.0,  # 处于切换重叠区内的累计时间（s）
        }
        self.last_ho_positions = []  # 记录最近切换位置（用于乒乓检测）
        
        # 9) 测量滤波增强（SINR差值滤波）
        self.enable_delta_sinr_filter = self.cfg.get("enable_delta_sinr_filter", True)
        self.delta_sinr_filter_alpha = self.cfg.get("delta_sinr_filter_alpha", 0.7)
        self.last_delta_sinr_filtered = None  # 上一次滤波后的SINR差值
        
        # 10) 历史SINR（用于状态特征增强）
        self.sinr_history = []  # 最近n步的SINR历史
        self.sinr_history_size = 5  # 保留最近5步（250ms）
    
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

            # 切换重叠区 KPI（5G-R：单侧 = 1/2 过渡距离 + 测量区 + 执行区；本仿真用可配置米数 + v×时延近似测量/执行）
            "ho_overlap_kpi_enabled": True,
            "ho_overlap_midpoint_m": None,  # None 表示轨道中点 track_length_m/2（两小区边界）
            "ho_overlap_transition_half_m": 25.0,  # 单侧计入的 1/2 切换过渡距离（m）
            "ho_overlap_meas_exec_delay_s": 0.2,  # 测量+执行等效时延（s），单侧附加距离 = v * delay
            "ho_overlap_meas_exec_extra_m": 0.0,  # 除 v×delay 外再叠加的固定距离（m）

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
            options: 可选参数，可以包含：
                - scenario_data: 预生成的场景数据（如果提供，将使用此场景数据）
            
        Returns:
            (观测, 信息字典)
        """
        super().reset(seed=seed)
        
        # 检查是否使用预生成的场景数据
        scenario_data = None
        if options is not None and "scenario_data" in options:
            scenario_data = options["scenario_data"]
        
        if scenario_data is not None:
            # 使用预生成的场景数据
            # 1) 位置和速度（从场景数据中获取）
            self.position_m = 0.0
            self.velocity_mps = scenario_data["velocity_mps"]
            
            self.time_step = 0
            self.serving_cell = 0  # 默认在 A 小区
            
            # 2) 设置天气参数（从场景数据中获取）
            weather = scenario_data["weather"]
            self.weather_model.temperature = weather["temperature"]
            self.weather_model.humidity = weather["humidity"]
            self.weather_model.pm25 = weather["pm25"]
            
            # 2.5) 重置信道模型的阴影衰落状态（传入场景数据）
            self.channel_model.reset(seed=None, scenario_data=scenario_data)
        else:
            # 使用原来的方法：随机生成场景
            # 关键：只设置一次seed，然后按固定顺序使用随机数
            # 这样可以确保相同的seed总是生成相同的场景
            # 随机数使用顺序（固定）：
            #   1. 速度采样（如果random_speed=True，消耗1个；否则跳过但保持位置）
            #   2-4. 天气采样（温度、湿度、PM2.5，消耗3个）
            #   5-6. 阴影衰落初始值（阴影A、阴影B，消耗2个）
            if seed is not None:
                np.random.seed(seed)
            
            # 1) 位置和速度
            self.position_m = 0.0
            if self.cfg["random_speed"]:
                # 消耗第1个随机数
                v_kmh = np.random.uniform(self.cfg["v_min_kmh"], self.cfg["v_max_kmh"])
            else:
                # 即使不使用随机速度，也消耗一个随机数，确保后续随机数位置一致
                # 这样无论random_speed设置如何，天气和阴影衰落都使用相同的随机数位置
                _ = np.random.uniform(0.0, 1.0)  # 消耗第1个随机数（丢弃）
                v_kmh = self.cfg["v_default_kmh"]
            self.velocity_mps = v_kmh / 3.6
            
            self.time_step = 0
            self.serving_cell = 0  # 默认在 A 小区
            
            # 2) 采样天气（消耗第2-4个随机数：温度、湿度、PM2.5）
            # 注意：不重新设置seed，使用当前随机数生成器的状态
            self.weather_model.sample_weather(seed=None)
            
            # 2.5) 重置信道模型的阴影衰落状态
            # 阴影衰落初始值采样（消耗第5-6个随机数：阴影A、阴影B）
            # 注意：不重新设置seed，使用当前随机数生成器的状态
            self.channel_model.reset(seed=None)
        
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
        
        # 6) 重置Outage/RLF事件检测状态
        self.outage_start_time = None
        self.outage_events = 0
        self.outage_total_time = 0.0
        
        # 7) 重置KPI统计
        self.episode_kpis = {
            "ho_count": 0,
            "outage_time_ratio": 0.0,
            "outage_events": 0,
            "sinr_samples": [],
            "interruption_total_time": 0.0,
            "interruption_time_in_overlap_zone": 0.0,
            "ping_pong_count": 0,
            "overlap_zone_time": 0.0,
        }
        self.last_ho_positions = []
        
        # 8) 重置测量滤波状态
        self.last_delta_sinr_filtered = None
        self.sinr_history = []
        
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
            
            # TTT量化：确保TTT是50ms的倍数
            if self.cfg.get("ttt_quantize_to_50ms", True):
                dt_ms = self.cfg["delta_t_s"] * 1000.0  # 50ms
                ttt = round(ttt / dt_ms) * dt_ms  # 量化到50ms网格
            
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
            
            # 2.4) 根据 A3 事件和 TTT 判断是否切换（传入位置用于距离模式）
            new_cell, ho_executed = self.ho_logic.execute_handover_with_a3(
                current_time, self.serving_cell, delta_rsrp, dt, self.position_m
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
                new_cell, ho_executed = self.ho_logic.execute_handover(current_time, self.serving_cell, self.position_m)
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
        
        # 5) Outage检测（包括切换中断期间）
        # 切换中断期间也计入outage
        outage_sinr = self.ho_logic.check_outage(sinr_serv)
        outage = outage_sinr or in_interruption
        
        # 5.1) Outage事件检测（连续outage >= T_out记一次事件）
        T_out_s = self.cfg.get("T_out_s", 0.2)  # 默认200ms
        if outage:
            if self.outage_start_time is None:
                self.outage_start_time = current_time
            # 累计outage时长
            self.outage_total_time += dt
        else:
            # 检查是否形成outage事件
            if self.outage_start_time is not None:
                outage_duration = current_time - self.outage_start_time
                if outage_duration >= T_out_s:
                    self.outage_events += 1
                self.outage_start_time = None
        
        # 5.2) 终止判定
        arrived = self.position_m >= self.cfg["track_length_m"]
        terminated = bool(arrived)  # 仅到达终点时终止，不因 outage 终止
        truncated = False  # 可以加最大步数等逻辑
        
        # 5.3) KPI统计更新
        self._update_kpis(
            sinr_serv, ho_executed, in_interruption, dt, current_time, self.position_m
        )
        
        # 6) 奖励计算（支持R0/R1/R2消融）
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
        
        # 计算SINR差值（用于滤波）
        sinr_neig = sinr_B if self.serving_cell == 0 else sinr_A
        delta_sinr = sinr_neig - sinr_serv
        
        # SINR差值滤波（如果启用）
        if self.enable_delta_sinr_filter:
            if self.last_delta_sinr_filtered is None:
                self.last_delta_sinr_filtered = delta_sinr
            else:
                alpha = self.delta_sinr_filter_alpha
                self.last_delta_sinr_filtered = alpha * self.last_delta_sinr_filtered + (1 - alpha) * delta_sinr
            delta_sinr_filtered = self.last_delta_sinr_filtered
        else:
            delta_sinr_filtered = delta_sinr
        
        # 更新SINR历史（用于状态特征）
        self.sinr_history.append(sinr_serv)
        if len(self.sinr_history) > self.sinr_history_size:
            self.sinr_history.pop(0)
        
        # 计算最终KPI（如果episode结束）
        kpis = None
        if terminated:
            total_time = current_time
            kpis = self._compute_final_kpis(total_time)
        
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
            # 增强特征：滤波后的SINR差值
            "delta_sinr_filtered_db": delta_sinr_filtered,
            # 增强特征：历史SINR统计
            "sinr_mean_db": np.mean(self.sinr_history) if len(self.sinr_history) > 0 else sinr_serv,
            "sinr_slope_db": (self.sinr_history[-1] - self.sinr_history[0]) / len(self.sinr_history) if len(self.sinr_history) > 1 else 0.0,
            # TTS状态（是否处于保护窗口）
            "in_tts_window": not self.ho_logic.can_handover(current_time, self.position_m),
            "time_since_last_ho": current_time - self.ho_logic.last_ho_time,
        }
        
        # 如果episode结束，添加最终KPI
        if kpis is not None:
            info["kpis"] = kpis
        
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
        计算奖励（支持R0/R1/R2消融设计）
        
        R0: r_t = SINR_t (仅SINR)
        R1: r_t = SINR_t - λ_ho * 1_{HO} (SINR + 切换惩罚)
        R2: r_t = SINR_t - λ_out * 1_{SINR<γ_out} - λ_ho * 1_{HO} (SINR + Outage惩罚 + 切换惩罚)
        
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
        
        # 获取reward类型（R0/R1/R2）
        reward_type = self.cfg.get("reward_type", "R2")
        lambda_ho = self.cfg.get("lambda_ho", 2.0)
        lambda_out = self.cfg.get("lambda_out", 15.0)
        
        if reward_type == "R0":
            # R0: 仅SINR
            reward = sinr_norm
        elif reward_type == "R1":
            # R1: SINR + 切换惩罚
            reward = sinr_norm
            if ho_executed:
                reward -= lambda_ho
        elif reward_type == "R2":
            # R2: SINR + Outage惩罚 + 切换惩罚
            reward = sinr_norm
            if outage:
                reward -= lambda_out
            if ho_executed:
                reward -= lambda_ho
        else:
            # 默认使用R2
            reward = sinr_norm
            if outage:
                reward -= lambda_out
            if ho_executed:
                reward -= lambda_ho
        
        return float(reward)
    
    def _overlap_half_width_m(self) -> float:
        """
        切换重叠区在轨道中点单侧的半宽 W（米）。

        与 5G-R 文献一致：单侧距离 ≈ (1/2)切换过渡距离 + 切换测量距离 + 切换执行距离；
        其中测量/执行区用「固定米数 + 车速×等效时延」在本项目中近似，便于与 delta_t、车速配置对齐。
        """
        if not self.cfg.get("ho_overlap_kpi_enabled", True):
            return 0.0
        v = float(self.velocity_mps) if self.velocity_mps is not None else 0.0
        t_half = float(self.cfg.get("ho_overlap_transition_half_m", 25.0))
        delay = float(self.cfg.get("ho_overlap_meas_exec_delay_s", 0.2))
        extra = float(self.cfg.get("ho_overlap_meas_exec_extra_m", 0.0))
        return max(0.0, t_half + extra + max(v, 0.0) * delay)
    
    def _in_handover_overlap_zone(self, position_m: float) -> bool:
        """列车位置是否落在以两小区几何中点为心的切换重叠区内。"""
        if not self.cfg.get("ho_overlap_kpi_enabled", True):
            return False
        W = self._overlap_half_width_m()
        if W <= 0.0:
            return False
        mid = self.cfg.get("ho_overlap_midpoint_m")
        if mid is None:
            mid = 0.5 * float(self.cfg["track_length_m"])
        else:
            mid = float(mid)
        return abs(float(position_m) - mid) <= W
    
    def _update_kpis(
        self,
        sinr_serv: float,
        ho_executed: bool,
        in_interruption: bool,
        dt: float,
        current_time: float,
        position_m: float,
    ):
        """
        更新KPI统计
        
        Args:
            sinr_serv: 服务小区SINR（dB）
            ho_executed: 是否执行了切换
            in_interruption: 是否处于切换中断期间
            dt: 时间步长（秒）
            current_time: 当前时间（秒）
            position_m: 当前列车位置（米），用于切换重叠区统计
        """
        # 1. 切换次数
        if ho_executed:
            self.episode_kpis["ho_count"] += 1
            self.last_ho_positions.append((current_time, self.position_m))
            # 只保留最近1秒内的切换记录（用于乒乓检测）
            self.last_ho_positions = [(t, p) for t, p in self.last_ho_positions if current_time - t <= 1.0]
        
        # 2. 乒乓检测（1秒内来回切换）
        if ho_executed and len(self.last_ho_positions) >= 2:
            # 检查最近两次切换是否在1秒内且方向相反
            if len(self.last_ho_positions) >= 2:
                t1, p1 = self.last_ho_positions[-2]
                t2, p2 = self.last_ho_positions[-1]
                if current_time - t1 <= 1.0:
                    # 检查是否来回切换（位置变化方向相反）
                    if (p2 - p1) * (self.position_m - p2) < 0:  # 方向相反
                        self.episode_kpis["ping_pong_count"] += 1
        
        # 3. SINR样本收集（用于计算分位数）
        self.episode_kpis["sinr_samples"].append(sinr_serv)
        # 限制历史长度，避免内存过大
        if len(self.episode_kpis["sinr_samples"]) > 10000:
            self.episode_kpis["sinr_samples"] = self.episode_kpis["sinr_samples"][-10000:]
        
        # 4. 中断总时长
        if in_interruption:
            self.episode_kpis["interruption_total_time"] += dt
        
        in_oz = self._in_handover_overlap_zone(position_m)
        # 4.1 切换重叠区内累计时长（通信中断率的分母）
        if in_oz:
            self.episode_kpis["overlap_zone_time"] += dt
        # 4.2 重叠区内的断连时长（分子；避免列车在区外切换导致比值>1 的歧义）
        if in_interruption and in_oz:
            self.episode_kpis["interruption_time_in_overlap_zone"] += dt
        
        # 5. Outage时间占比（在step结束时计算，这里只更新outage_total_time）
        # outage_total_time在outage检测部分已更新
    
    def _compute_final_kpis(self, total_time: float) -> dict:
        """
        计算最终KPI（episode结束时调用）
        
        Args:
            total_time: 总时长（秒）
            
        Returns:
            KPI字典
        """
        kpis = {}
        
        # 1. Outage时间占比
        kpis["outage_time_ratio"] = self.outage_total_time / total_time if total_time > 0 else 0.0
        
        # 2. Outage事件数
        kpis["outage_events"] = self.outage_events
        # 如果episode结束时还在outage，检查是否形成事件
        if self.outage_start_time is not None:
            T_out_s = self.cfg.get("T_out_s", 0.2)
            if total_time - self.outage_start_time >= T_out_s:
                kpis["outage_events"] += 1
        
        # 3. 切换次数
        kpis["ho_count"] = self.episode_kpis["ho_count"]
        
        # 4. 切换次数/公里
        track_length_km = self.cfg["track_length_m"] / 1000.0
        kpis["ho_per_km"] = kpis["ho_count"] / track_length_km if track_length_km > 0 else 0.0
        
        # 5. 乒乓率
        kpis["ping_pong_count"] = self.episode_kpis["ping_pong_count"]
        kpis["ping_pong_ratio"] = kpis["ping_pong_count"] / kpis["ho_count"] if kpis["ho_count"] > 0 else 0.0
        
        # 6. 中断总时长
        kpis["interruption_total_time"] = self.episode_kpis["interruption_total_time"]
        
        # 6.1 切换重叠区与区内通信中断率（断连时间 / 列车处于重叠区的时间）
        ozt = float(self.episode_kpis.get("overlap_zone_time", 0.0))
        kpis["overlap_zone_time_s"] = ozt
        W = self._overlap_half_width_m()
        kpis["overlap_zone_half_width_m"] = float(W)
        kpis["overlap_zone_full_width_m"] = float(2.0 * W)
        int_in_oz = float(self.episode_kpis.get("interruption_time_in_overlap_zone", 0.0))
        kpis["interruption_time_in_overlap_zone_s"] = int_in_oz
        if ozt > 0.0:
            kpis["comm_interruption_ratio_in_overlap_zone"] = int_in_oz / ozt
        else:
            kpis["comm_interruption_ratio_in_overlap_zone"] = 0.0
        
        # 7. SINR统计
        sinr_samples = np.array(self.episode_kpis["sinr_samples"])
        if len(sinr_samples) > 0:
            kpis["sinr_mean_db"] = float(np.mean(sinr_samples))
            kpis["sinr_std_db"] = float(np.std(sinr_samples))
            kpis["sinr_p5_db"] = float(np.percentile(sinr_samples, 5))  # 5%分位
            kpis["sinr_p95_db"] = float(np.percentile(sinr_samples, 95))  # 95%分位
            # SINR低于某门限的时间占比
            kpis["sinr_below_minus3db_ratio"] = float(np.mean(sinr_samples < -3.0))
        else:
            kpis["sinr_mean_db"] = 0.0
            kpis["sinr_std_db"] = 0.0
            kpis["sinr_p5_db"] = 0.0
            kpis["sinr_p95_db"] = 0.0
            kpis["sinr_below_minus3db_ratio"] = 0.0
        
        return kpis
    
    def render(self, mode='human'):
        """渲染环境（可选实现）"""
        if mode == 'human':
            print(f"Step: {self.time_step}, Position: {self.position_m:.2f}m, "
                  f"Serving Cell: {self.serving_cell}, "
                  f"SINR: {self.last_sinr_serv:.2f}dB")

