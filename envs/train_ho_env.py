"""铁路切换算法主环境：TrainHandoverEnv"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Tuple, Optional
import yaml
import os

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
        
        # 2) 观测空间：4维
        # [RSRP_serv, RSRP_neig, SINR_serv, pos_norm]
        high = np.array([1.0] * 4, dtype=np.float32)
        low = np.array([0.0] * 4, dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        
        # 动作空间：48个动作（Hys/TTT 组合）
        # 如果配置中指定了 use_hys_ttt，则使用48个动作，否则保持兼容使用2个动作
        self.use_hys_ttt = self.cfg.get("use_hys_ttt", True)  # 默认使用 Hys/TTT 模式
        
        if self.use_hys_ttt:
            # 从配置读取动作空间参数，或使用默认值
            action_space_cfg = self.cfg.get("action_space", {})
            hys_set = action_space_cfg.get("hys_set", [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0])
            ttt_set = action_space_cfg.get("ttt_set", [0, 50, 100, 150, 300, 650])
            
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
        self.channel_model = ChannelModel(self.cfg)
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
            "ping_pong_count": 0,  # 乒乓切换次数（重叠区内 A->B->A 或 B->A->B 记1次）
            "overlap_zone_time": 0.0,  # 处于切换重叠区内的累计时间（s）
            "overlap_zone_sinr_time_weighted_sum": 0.0,  # 重叠区内SINR按时间积分（dB*s）
        }
        self.last_overlap_ho_transition = None  # 最近一次"重叠区内切换"方向 (from_cell, to_cell)
        
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
            ttt_set = [0, 50, 100, 150, 300, 650]
        
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
            
            "rsrp_min_dbm": -120.0,
            "rsrp_max_dbm": -60.0,
            "sinr_min_db": -10.0,
            "sinr_max_db": 20.0,
            "reward_type": "R4",  # R0/R1/R2(兼容旧版)/R3(分级Outage+中断主惩罚)/R4(事件驱动)
            "lambda_ho": 2.0,  # 旧版HO惩罚（R1/R2）
            "lambda_out": 15.0,  # 旧版outage惩罚（R2）/R3最高等级惩罚
            # R3: 分级SINR/outage惩罚
            "sinr_warn_db": -3.0,  # 轻度劣化阈值
            "sinr_degrade_db": -5.0,  # 中度劣化阈值
            "lambda_sinr_warn": 1.5,  # 轻度惩罚
            "lambda_sinr_degrade": 4.0,  # 中度惩罚
            # R3: 用中断时长替代HO次数作为主惩罚
            "lambda_interruption": 6.0,  # 处于中断时隙时每步惩罚
            "lambda_ho_minor": 0.2,  # 可选小HO惩罚（避免完全忽略HO代价）
            # 奖励尺度系数（可由训练阶段策略动态调整）
            "reward_outage_scale": 1.0,
            "reward_interruption_scale": 1.0,
            "reward_ho_scale": 1.0,
            
            "C_outage": 10.0,
            "C_ho": 0.3,
            "C_interruption": 0.0,  # 切换中断惩罚系数（默认0，因为中断期间SINR已经很低）
            
            "T_guard_s": 0.0,
            
            # 切换中断配置
            "ho_interruption_slots": 1,  # 切换导致的通信中断时隙数（默认1个时隙，即50ms）
            "ho_interruption_sinr_db": -20.0,  # 中断期间的SINR值（dB），设置为很低的值以模拟通信中断
            "ho_interruption_rsrp_dbm": -120.0,  # 中断期间的RSRP值（dBm），设置为很低的值以模拟通信中断

            # 动态切换区 KPI（基于 ΔRSRP 实时判定，参考铁路 5G-R 规划文献）
            "ho_overlap_kpi_enabled": True,
            "ho_overlap_delta_rsrp_db": 5.0,  # ΔRSRP 门限 N（dB），用于切换区左右边界定义

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
        
        # 使用 Gymnasium 管理的 self.np_random（由 super().reset(seed=seed) 设置），
        # 避免污染全局 np.random 状态，确保数据收集策略的随机性不受环境 seed 影响
        rng = self.np_random
        self.channel_model.set_rng(rng)

        if scenario_data is not None:
            # 使用预生成的场景数据
            # 1) 位置和速度（从场景数据中获取）
            self.position_m = 0.0
            self.velocity_mps = scenario_data["velocity_mps"]

            self.time_step = 0
            self.serving_cell = 0  # 默认在 A 小区

            # 2.5) 重置信道模型的阴影衰落状态（传入场景数据）
            self.channel_model.reset(seed=None, scenario_data=scenario_data)
        else:
            # 使用 self.np_random 生成场景，确保同一 seed 生成相同的场景
            # 1) 位置和速度
            self.position_m = 0.0
            if self.cfg["random_speed"]:
                speed_tiers_cfg = self.cfg.get("speed_tiers", {})
                if speed_tiers_cfg.get("enabled", False):
                    # 分层速度采样：按权重从各速度段采样
                    tiers = speed_tiers_cfg["tiers"]
                    weights = np.array([t["weight"] for t in tiers], dtype=np.float64)
                    weights /= weights.sum()
                    tier_idx = int(rng.choice(len(tiers), p=weights))
                    tier = tiers[tier_idx]
                    v_kmh = float(rng.uniform(tier["range"][0], tier["range"][1]))
                else:
                    v_kmh = float(rng.uniform(self.cfg["v_min_kmh"], self.cfg["v_max_kmh"]))
            else:
                _ = rng.uniform(0.0, 1.0)  # 消耗随机数保持一致性
                v_kmh = self.cfg["v_default_kmh"]
            self.velocity_mps = v_kmh / 3.6

            self.time_step = 0
            self.serving_cell = 0  # 默认在 A 小区

            # 2.5) 重置信道模型的阴影衰落状态
            # 阴影衰落初始值采样（消耗第5-6个随机数：阴影A、阴影B）
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
            "overlap_zone_sinr_time_weighted_sum": 0.0,
        }
        self.last_overlap_ho_transition = None
        
        # 8) 重置测量滤波状态
        self.last_delta_sinr_filtered = None
        self.sinr_history = []
        
        # 9) 动态切换区状态（基于 ΔRSRP 实时判定）
        self._current_rsrp_A = None
        self._current_rsrp_B = None
        # 单一连续切换重叠区边界（避免一个场景出现多个重叠区片段）
        # 定义见 _update_handover_overlap_zone_bounds()
        self._overlap_zone_left_m = None
        self._overlap_zone_right_m = None
        self._overlap_zone_closed = False  # 一旦确认离开切换区后闭合右边界
        self._overlap_zone_exit_streak = 0  # 连续满足 delta < -N 的步数（用于抗快衰落抖动）
        
        obs = self._build_observation()
        info = {
            "rsrp_serv_dbm": self.last_rsrp_serv,
            "rsrp_neig_dbm": self.last_rsrp_neig,
            "sinr_serv_db": self.last_sinr_serv,
            "position_m": self.position_m,
            "serving_cell": self.serving_cell,
        }
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

        if self.use_hys_ttt:
            # 新模式：根据 Hys/TTT 参数判断是否切换
            # 2.1) 将动作转换为 Hys/TTT 参数
            hys, ttt = self._action_to_hys_ttt(action)

            # TTT量化：确保TTT是50ms的倍数
            if self.cfg.get("ttt_quantize_to_50ms", True):
                dt_ms = self.cfg["delta_t_s"] * 1000.0  # 50ms
                ttt = round(ttt / dt_ms) * dt_ms  # 量化到50ms网格

            self.ho_logic.update_hys_ttt(hys, ttt)

            # 2.2) 计算链路（仅一次，避免 AR(1) 阴影衰落被重复推进）
            rsrp_A, rsrp_B, sinr_A, sinr_B = self.channel_model.compute_link_metrics(self.position_m)

            if self.serving_cell == 0:   # A 为服务小区
                rsrp_serv_temp = rsrp_A
                rsrp_neig_temp = rsrp_B
            else:
                rsrp_serv_temp = rsrp_B
                rsrp_neig_temp = rsrp_A

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
        else:
            # 兼容模式：直接根据动作判断
            if action == 1:
                new_cell, ho_executed = self.ho_logic.execute_handover(current_time, self.serving_cell, self.position_m)
                if ho_executed:
                    self.serving_cell = new_cell
                    # 切换发生，设置中断时隙数（中断在切换发生的时隙立即开始）
                    self.ho_interruption_remaining = self.cfg.get("ho_interruption_slots", 1)

            # 兼容模式下计算链路
            rsrp_A, rsrp_B, sinr_A, sinr_B = self.channel_model.compute_link_metrics(self.position_m)

        # 2.5) 缓存当前 rsrp_A / rsrp_B，供动态切换区判定使用
        self._current_rsrp_A = rsrp_A
        self._current_rsrp_B = rsrp_B
        self._update_handover_overlap_zone_bounds(self.position_m, rsrp_A, rsrp_B)

        # 2.6) 检查并更新切换中断状态（在切换处理后检查，以便在切换发生的时隙立即应用中断）
        in_interruption = False
        if self.ho_interruption_remaining > 0:
            in_interruption = True
            self.ho_interruption_remaining -= 1
        
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
            sinr_serv,
            ho_executed,
            in_interruption,
            dt,
            current_time,
            self.position_m,
            old_serving_cell,
            self.serving_cell,
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
            # 切换成功率统计辅助字段：
            # ho_triggered: A3条件+TTT达到（可视作"触发尝试"）
            # ho_blocked_by_guard: 触发但被保护时间/距离窗口阻塞
            "ho_triggered": self.ho_logic.last_ho_triggered,
            "ho_blocked_by_guard": self.ho_logic.last_ho_blocked,
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
            # 动态切换区标志（供外部分析和未来观测增强使用）
            "in_overlap_zone": self._in_handover_overlap_zone(self.position_m),
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
        
        obs = np.array([
            rsrp_serv_n,
            rsrp_neig_n,
            sinr_serv_n,
            pos_norm
        ], dtype=np.float32)
        
        return obs
    
    def _compute_tiered_outage_penalty(self, sinr_serv: float, outage: bool) -> float:
        """
        计算分级outage/劣化惩罚（R3用）。

        规则（从重到轻）：
        1) outage=True: 使用最高惩罚 lambda_out
        2) sinr < sinr_degrade_db: 使用中度惩罚 lambda_sinr_degrade
        3) sinr < sinr_warn_db: 使用轻度惩罚 lambda_sinr_warn
        """
        if outage:
            return float(self.cfg.get("lambda_out", 15.0))
        if sinr_serv < float(self.cfg.get("sinr_degrade_db", -5.0)):
            return float(self.cfg.get("lambda_sinr_degrade", 4.0))
        if sinr_serv < float(self.cfg.get("sinr_warn_db", -3.0)):
            return float(self.cfg.get("lambda_sinr_warn", 1.5))
        return 0.0

    def _compute_reward(self, sinr_serv: float, outage: bool, ho_executed: bool, in_interruption: bool = False) -> float:
        """
        计算奖励（支持R0/R1/R2/R3/R4）
        
        R0: r_t = SINR_t (仅SINR)
        R1: r_t = SINR_t - λ_ho * 1_{HO} (SINR + 切换惩罚)
        R2: r_t = SINR_t - λ_out * 1_{SINR<γ_out} - λ_ho * 1_{HO} (SINR + Outage惩罚 + 切换惩罚)
        R3: r_t = SINR_t - 分级SINR/Outage惩罚 - λ_int*1_{interruption} - λ_ho_minor*1_{HO}
            （以中断时长惩罚为主，HO次数惩罚为辅）
        R4: 事件驱动型奖励，以零为基线。正常时隙奖励为零，仅在关键事件（outage/中断/
            良好SINR余量/切换重叠区表现）时给出非零信号，拉大好坏动作的差距。
        
        Args:
            sinr_serv: 服务小区SINR（dB）
            outage: 是否发生outage
            ho_executed: 是否执行了切换
            in_interruption: 是否处于切换中断期间
            
        Returns:
            奖励值
        """
        sinr_min, sinr_max = self.cfg["sinr_min_db"], self.cfg["sinr_max_db"]
        sinr_norm = self._normalize(sinr_serv, sinr_min, sinr_max)
        
        reward_type = self.cfg.get("reward_type", "R4")
        lambda_ho = self.cfg.get("lambda_ho", 2.0)
        lambda_out = self.cfg.get("lambda_out", 15.0)
        
        if reward_type == "R0":
            reward = sinr_norm
        elif reward_type == "R1":
            reward = sinr_norm
            if ho_executed:
                reward -= lambda_ho
        elif reward_type == "R2":
            reward = sinr_norm
            if outage:
                reward -= lambda_out
            if ho_executed:
                reward -= lambda_ho
        elif reward_type == "R3":
            reward = sinr_norm
            out_scale = float(self.cfg.get("reward_outage_scale", 1.0))
            int_scale = float(self.cfg.get("reward_interruption_scale", 1.0))
            ho_scale = float(self.cfg.get("reward_ho_scale", 1.0))
            lambda_int = float(self.cfg.get("lambda_interruption", 6.0))
            lambda_ho_minor = float(self.cfg.get("lambda_ho_minor", 0.2))

            reward -= out_scale * self._compute_tiered_outage_penalty(sinr_serv, outage)
            if in_interruption:
                reward -= int_scale * lambda_int
            if ho_executed:
                reward -= ho_scale * lambda_ho_minor
        elif reward_type == "R4":
            reward = self._compute_reward_r4(sinr_serv, outage, ho_executed, in_interruption)
        elif reward_type == "R5":
            reward = self._compute_reward_r5(sinr_serv, outage, ho_executed, in_interruption)
        else:
            reward = self._compute_reward_r5(sinr_serv, outage, ho_executed, in_interruption)
        
        return float(reward)

    def _compute_reward_r4(self, sinr_serv: float, outage: bool,
                           ho_executed: bool, in_interruption: bool) -> float:
        """
        R4: 事件驱动型奖励函数。

        设计原则：
        - 基线为零：正常时隙（SINR 在 warn 阈值以上、无中断、无切换）奖励 = 0。
          这消除了 R3 中 sinr_norm 正向偏置导致好坏动作差距被稀释的问题。
        - 惩罚信号显著：outage / SINR劣化 / 切换中断 给出大幅负奖励，确保信噪比远高于
          环境本身的随机方差。
        - 正向激励稀疏但明确：只在"切换重叠区 SINR 维持良好"或"高速场景保持高 SINR"
          时给正奖励，引导策略学习场景自适应。
        - 乒乓重惩：在重叠区内发生的乒乓方向切换，给额外惩罚。
        """
        cfg = self.cfg
        reward = 0.0

        # ── 1. 分级 SINR / Outage 惩罚 ──
        out_scale = float(cfg.get("reward_outage_scale", 1.0))
        int_scale = float(cfg.get("reward_interruption_scale", 1.0))
        ho_scale = float(cfg.get("reward_ho_scale", 1.0))

        if outage:
            reward -= out_scale * float(cfg.get("r4_lambda_outage", 10.0))
        elif sinr_serv < float(cfg.get("sinr_degrade_db", -5.0)):
            reward -= out_scale * float(cfg.get("r4_lambda_degrade", 3.0))
        elif sinr_serv < float(cfg.get("sinr_warn_db", -3.0)):
            reward -= out_scale * float(cfg.get("r4_lambda_warn", 1.0))

        # ── 2. 切换中断惩罚 ──
        if in_interruption:
            reward -= int_scale * float(cfg.get("r4_lambda_interruption", 5.0))

        # ── 3. 切换惩罚 ──
        if ho_executed:
            reward -= ho_scale * float(cfg.get("r4_lambda_ho", 1.5))

        # ── 4. 切换重叠区质量奖励/惩罚 ──
        # 在切换关键区域，SINR 维持良好应得到正向激励
        in_oz = self._in_handover_overlap_zone(self.position_m)
        if in_oz:
            sinr_good_thresh = float(cfg.get("r4_overlap_sinr_good_db", 5.0))
            sinr_bad_thresh = float(cfg.get("r4_overlap_sinr_bad_db", -2.0))
            if sinr_serv >= sinr_good_thresh:
                reward += float(cfg.get("r4_overlap_bonus", 0.3))
            elif sinr_serv < sinr_bad_thresh:
                reward -= float(cfg.get("r4_overlap_penalty", 1.0))

        # ── 5. SINR 余量 bonus（仅在高速场景时给，鼓励场景自适应） ──
        v_mps = self.velocity_mps if self.velocity_mps is not None else 0.0
        v_fast_thresh = float(cfg.get("r4_fast_speed_mps", 83.3))  # 约 300 km/h
        sinr_margin_db = float(cfg.get("r4_sinr_margin_db", 8.0))
        if v_mps >= v_fast_thresh and sinr_serv >= sinr_margin_db and not outage:
            reward += float(cfg.get("r4_sinr_margin_bonus", 0.2))

        return reward
    
    def _compute_reward_r5(self, sinr_serv: float, outage: bool,
                           ho_executed: bool, in_interruption: bool) -> float:
        """
        R5: 切换区聚焦型奖励函数。

        设计原则：
        - 切换区外（单基站主导区域）agent 决策几乎无影响 → 奖励≈0，仅保留安全网
        - 切换区内（由 ΔRSRP 门限 N 定义的“单一连续重叠区”）是切换决策真正发挥作用的区域：
          a) "次优服务小区"连续惩罚（动作依赖）：当邻区 RSRP > 服务区 RSRP，
             说明 agent 的 (Hys, TTT) 太保守，该切未切 → 每步给持续惩罚
          b) 事件惩罚：outage / 中断 / 切换 / 乒乓
        - 好策略：及时切换到强小区（次优惩罚少）+ 切换次数适度（事件惩罚少）
        - 坏策略：长期留在弱小区（次优惩罚大）或频繁乒乓（事件惩罚大）
        """
        cfg = self.cfg
        in_oz = self._in_handover_overlap_zone(self.position_m)
        out_scale = float(cfg.get("reward_outage_scale", 1.0))
        int_scale = float(cfg.get("reward_interruption_scale", 1.0))
        ho_scale = float(cfg.get("reward_ho_scale", 1.0))

        if not in_oz:
            # ── 切换区外：奖励≈0，仅保留安全网 ──
            reward = 0.0
            if outage:
                reward -= out_scale * float(cfg.get("r5_outside_outage_penalty", 1.0))
            if in_interruption:
                reward -= int_scale * float(cfg.get("r5_outside_interruption_penalty", 0.5))
            return reward

        # ── 切换区内 ──
        reward = 0.0

        # 1. 次优服务小区惩罚（动作依赖的连续信号）
        #    当邻区信号优于服务区，说明 agent 应该已经切换但没有 →
        #    动作的 (Hys, TTT) 决定了切换时机，因此这个惩罚是动作因果相关的
        if not in_interruption and self._current_rsrp_A is not None:
            if self.serving_cell == 0:
                rsrp_serv_now = self._current_rsrp_A
                rsrp_neig_now = self._current_rsrp_B
            else:
                rsrp_serv_now = self._current_rsrp_B
                rsrp_neig_now = self._current_rsrp_A
            delta = rsrp_neig_now - rsrp_serv_now  # 正值 = 邻区更强
            if delta > 0:
                subopt_range = float(cfg.get("r5_subopt_range_db", 6.0))
                subopt_alpha = float(cfg.get("r5_subopt_alpha", 0.4))
                subopt = min(delta / subopt_range, 1.0)
                reward -= subopt_alpha * subopt

        # 2. SINR shaping（可选，默认关闭）
        alpha = float(cfg.get("r5_sinr_shaping_alpha", 0.0))
        if alpha > 0:
            sinr_target = float(cfg.get("r5_sinr_target_db", 2.0))
            sinr_range = float(cfg.get("r5_sinr_range_db", 10.0))
            sinr_shaped = max(-1.0, min(1.0, (sinr_serv - sinr_target) / sinr_range))
            reward += alpha * sinr_shaped

        # 3. Outage 惩罚（最严重）
        if outage:
            reward -= out_scale * float(cfg.get("r5_outage_penalty", 5.0))

        # 4. 切换中断惩罚
        if in_interruption:
            reward -= int_scale * float(cfg.get("r5_interruption_penalty", 2.0))

        # 5. 切换执行惩罚
        if ho_executed:
            reward -= ho_scale * float(cfg.get("r5_ho_penalty", 1.0))
            # 乒乓切换额外惩罚（_update_kpis 在 reward 之前调用，已设置标志）
            if self._r5_pingpong_this_step:
                reward -= ho_scale * float(cfg.get("r5_pingpong_extra_penalty", 2.0))

        return reward

    def _update_handover_overlap_zone_bounds(self, position_m: float, rsrp_A_dbm: float, rsrp_B_dbm: float):
        """
        更新“单一连续切换重叠区”的左右边界。

        目标：避免按 |ΔRSRP| ≤ N 逐点判定导致一个场景出现多个重叠区片段。

        定义（N = ho_overlap_delta_rsrp_db）：
        - 左边界：从左向右扫描，首次满足 (RSRP_A - RSRP_B) ≤ +N 的位置
          （即 A 不再“比 B 强超过 N dB” 的最早点）
        - 右边界：从左向右扫描，在左边界之后，满足 (RSRP_A - RSRP_B) ≥ -N 的最右位置
          （即 A 尚未“比 B 弱超过 N dB” 的最右点）

        最终切换区为区间 [x_left, x_right]（若边界不存在则视为无切换区）。
        """
        if not self.cfg.get("ho_overlap_kpi_enabled", True):
            return
        if rsrp_A_dbm is None or rsrp_B_dbm is None:
            return
        N = float(self.cfg.get("ho_overlap_delta_rsrp_db", 5.0))
        delta = float(rsrp_A_dbm - rsrp_B_dbm)

        # 左边界：首次进入 (delta <= +N)
        if self._overlap_zone_left_m is None:
            if delta <= N:
                self._overlap_zone_left_m = float(position_m)
                # 进入后默认认为切换区“连续存在”，右边界先随位置推进，直到确认离开再闭合
                self._overlap_zone_right_m = float(position_m)
                self._overlap_zone_closed = False
                self._overlap_zone_exit_streak = 0
            return

        # 已闭合则不再更新
        if self._overlap_zone_closed:
            return

        # 右边界闭合判定：为避免快衰落造成 delta 短暂跌破 -N 而形成多个碎片区，
        # 采用“连续 confirm_steps 步 delta < -N”才认为真正离开切换区。
        confirm_steps = int(self.cfg.get("ho_overlap_exit_confirm_steps", 3))
        if delta < -N:
            self._overlap_zone_exit_streak += 1
            if self._overlap_zone_exit_streak >= max(1, confirm_steps):
                # 右边界保持为“最后一次更新的位置”，闭合区间
                self._overlap_zone_closed = True
            return

        # 仍处于 (-N) 右侧：持续推进右边界并清空离开计数
        self._overlap_zone_exit_streak = 0
        self._overlap_zone_right_m = float(position_m)

    def _in_handover_overlap_zone(self, position_m: float = None) -> bool:
        """
        单一连续切换重叠区判定（基于左右边界区间）。

        边界由 _update_handover_overlap_zone_bounds() 在线更新，N 由配置 ho_overlap_delta_rsrp_db 控制。
        """
        if not self.cfg.get("ho_overlap_kpi_enabled", True):
            return False
        if position_m is None:
            position_m = float(self.position_m)
        if self._overlap_zone_left_m is None:
            return False
        if float(position_m) < float(self._overlap_zone_left_m):
            return False
        # 未闭合前：保证切换区为单一连续区间（进入后不因快衰落抖动离开）
        if not self._overlap_zone_closed:
            return True
        if self._overlap_zone_right_m is None:
            return False
        return float(position_m) <= float(self._overlap_zone_right_m)
    
    def _update_kpis(
        self,
        sinr_serv: float,
        ho_executed: bool,
        in_interruption: bool,
        dt: float,
        current_time: float,
        position_m: float,
        old_serving_cell: int,
        new_serving_cell: int,
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
            old_serving_cell: 切换前服务小区
            new_serving_cell: 当前服务小区（若发生切换则为切换后）
        """
        in_oz = self._in_handover_overlap_zone(position_m)

        # R5 乒乓标志（每步重置，供奖励函数读取）
        self._r5_pingpong_this_step = False

        # 1. 切换次数
        if ho_executed:
            self.episode_kpis["ho_count"] += 1
            # 2. 乒乓检测：仅考虑"重叠区内"切换，且相邻两次切换方向相反（A->B->A / B->A->B）
            if in_oz and old_serving_cell != new_serving_cell:
                current_transition = (int(old_serving_cell), int(new_serving_cell))
                prev_transition = self.last_overlap_ho_transition
                if prev_transition is not None:
                    # 上一次 from->to 与本次完全反向，即构成一次乒乓
                    if prev_transition[0] == current_transition[1] and prev_transition[1] == current_transition[0]:
                        self.episode_kpis["ping_pong_count"] += 1
                        self._r5_pingpong_this_step = True
                        # 清空可避免连续序列被重复配对计数
                        self.last_overlap_ho_transition = None
                    else:
                        self.last_overlap_ho_transition = current_transition
                else:
                    self.last_overlap_ho_transition = current_transition
            elif not in_oz:
                # 离开重叠区后，按你的定义不跨区间拼接乒乓
                self.last_overlap_ho_transition = None
        
        # 3. SINR样本收集（用于计算分位数）
        self.episode_kpis["sinr_samples"].append(sinr_serv)
        # 限制历史长度，避免内存过大
        if len(self.episode_kpis["sinr_samples"]) > 10000:
            self.episode_kpis["sinr_samples"] = self.episode_kpis["sinr_samples"][-10000:]
        
        # 4. 中断总时长
        if in_interruption:
            self.episode_kpis["interruption_total_time"] += dt
        
        # 4.1 切换重叠区内累计时长（通信中断率的分母）
        if in_oz:
            self.episode_kpis["overlap_zone_time"] += dt
            self.episode_kpis["overlap_zone_sinr_time_weighted_sum"] += sinr_serv * dt
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
        
        # 6.1 动态切换区与区内通信中断率
        ozt = float(self.episode_kpis.get("overlap_zone_time", 0.0))
        kpis["overlap_zone_time_s"] = ozt
        N = float(self.cfg.get("ho_overlap_delta_rsrp_db", 5.0))
        kpis["overlap_zone_delta_rsrp_threshold_db"] = N
        int_in_oz = float(self.episode_kpis.get("interruption_time_in_overlap_zone", 0.0))
        kpis["interruption_time_in_overlap_zone_s"] = int_in_oz
        oz_sinr_time_sum = float(self.episode_kpis.get("overlap_zone_sinr_time_weighted_sum", 0.0))
        if ozt > 0.0:
            kpis["overlap_zone_sinr_mean_db"] = oz_sinr_time_sum / ozt
        else:
            kpis["overlap_zone_sinr_mean_db"] = 0.0
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
