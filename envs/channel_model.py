"""信道模型：路径损耗、阴影衰落、RSRP/SINR计算（含 3GPP TR 38.901 风格增强）"""
import numpy as np
from typing import Tuple, Dict
from .weather_model import WeatherModel


class ChannelModel:
    """信道模型类，负责路径损耗、RSRP、SINR等计算

    当前实现参考 3GPP TR 38.901 思路，包含：
    - 基于对数组的路径损耗（可近似 RMa/UMa）
    - 相关阴影衰落（按距离相关，非独立高斯）
    - 可选的小尺度快衰落（Rayleigh / Rician）
    """

    def __init__(self, config: Dict, weather_model: WeatherModel):
        """
        初始化信道模型

        Args:
            config: 配置字典
            weather_model: 天气模型实例
        """
        self.cfg = config
        self.weather_model = weather_model

        # 相关阴影衰落状态（按距离相关，而不是每步独立采样）
        self._shadow_A_db = 0.0
        self._shadow_B_db = 0.0
        self._last_pos_m = None

    # -----------------------
    # 大尺度衰落：路径损耗 + 相关阴影
    # -----------------------
    def _update_shadowing(self, x_m: float):
        """根据 3GPP 38.901 思路实现距离相关的阴影衰落（简化版）"""
        d_corr = self.cfg.get("shadow_corr_distance_m", 50.0)  # 相关距离，默认 50 m

        if self._last_pos_m is None:
            # 首次：直接采样
            self._shadow_A_db = np.random.normal(0.0, self.cfg["shadow_sigma_A"])
            self._shadow_B_db = np.random.normal(0.0, self.cfg["shadow_sigma_B"])
            self._last_pos_m = x_m
            return

        delta_d = abs(x_m - self._last_pos_m)
        # 相关系数：指数衰减，近似 38.901 的距离相关模型
        rho = float(np.exp(-delta_d / max(d_corr, 1e-3)))

        # AR(1) 模型：X_k = rho * X_{k-1} + sqrt(1-rho^2) * w_k
        sigma_A = self.cfg["shadow_sigma_A"]
        sigma_B = self.cfg["shadow_sigma_B"]

        w_A = np.random.normal(0.0, sigma_A)
        w_B = np.random.normal(0.0, sigma_B)

        self._shadow_A_db = rho * self._shadow_A_db + np.sqrt(1.0 - rho ** 2) * w_A
        self._shadow_B_db = rho * self._shadow_B_db + np.sqrt(1.0 - rho ** 2) * w_B
        self._last_pos_m = x_m

    def _basic_pathloss_db(self, d_m: float, is_cell_A: bool) -> float:
        """基础路径损耗模型（对数距离模型，可视作 38.901 中 PL0 + 10nlog10(d/d0) 的简化）"""
        d0 = 1.0  # 参考1米
        d = max(d_m, d0)

        if is_cell_A:
            pl0 = self.cfg["pl0_A_db"]
            n = self.cfg["pathloss_exp_A"]
        else:
            pl0 = self.cfg["pl0_B_db"]
            n = self.cfg["pathloss_exp_B"]

        return pl0 + 10 * n * np.log10(d / d0)

    def pathloss_db(self, x_m: float, d_m: float, is_cell_A: bool) -> float:
        """
        计算路径损耗（dB），包含：
        - 基础路径损耗（近似 38.901 大尺度路径损耗）
        - 距离相关的阴影衰落
        - 天气引起的额外损耗

        Args:
            x_m: 沿轨道的位置（用于相关阴影衰落）
            d_m: UE 到该基站的直线距离（米）
            is_cell_A: 是否为小区A

        Returns:
            路径损耗（dB）
        """
        # 更新阴影衰落（和位置相关）
        self._update_shadowing(x_m)

        # 基础路径损耗
        pl_mean = self._basic_pathloss_db(d_m, is_cell_A=is_cell_A)

        # 选择对应小区的阴影值
        shadow = self._shadow_A_db if is_cell_A else self._shadow_B_db

        # 天气额外损耗
        Lw = self.weather_model.compute_weather_loss_db()

        pl_total = pl_mean + shadow + Lw

        # 小尺度快衰落（可选，类似 38.901 的多径快衰落的功率波动，简化为 Rayleigh/Rician 振幅）
        if self.cfg.get("enable_fast_fading", False):
            k_factor_db = self.cfg.get("rician_k_factor_db", None)
            if k_factor_db is None:
                # Rayleigh：CN(0,1)
                h = (np.random.normal(0.0, 1.0) + 1j * np.random.normal(0.0, 1.0)) / np.sqrt(2.0)
            else:
                # Rician：有 LOS 分量 + NLOS 分量的组合（简化实现）
                K = 10 ** (k_factor_db / 10.0)
                sigma2 = 1.0 / (2.0 * (K + 1.0))
                h_los = np.sqrt(K / (K + 1.0))  # 实数 LOS 分量
                h_nlos = (np.random.normal(0.0, np.sqrt(sigma2)) +
                          1j * np.random.normal(0.0, np.sqrt(sigma2)))
                h = h_los + h_nlos

            # 振幅平方对应功率增益，转换为 dB
            power_lin = np.abs(h) ** 2 + 1e-12
            fading_db = 10 * np.log10(power_lin)
            pl_total -= fading_db  # 增益 -> 等效减少路径损耗

        return pl_total

    # -----------------------
    # RSRP / SINR 计算
    # -----------------------
    def compute_link_metrics(self, x_m: float) -> Tuple[float, float, float, float]:
        """
        给定位置 x（0~D），计算 RSRP 和 SINR
        
        基于 5G-R 高速铁路场景，同频干扰动态建模：
        根据公式 I(d) = 10^(Pr1(d)/10) + 10^(Pr2(d)/10)
        - 当UE连接到基站A时，基站B的同频信号产生干扰：I_A = 10^(rsrp_B/10)
        - 当UE连接到基站B时，基站A的同频信号产生干扰：I_B = 10^(rsrp_A/10)
        - 干扰功率与UE位置动态相关，随两个基站信号强度变化

        Args:
            x_m: 列车位置（米）

        Returns:
            (rsrp_A_dbm, rsrp_B_dbm, sinr_A_db, sinr_B_db)
        """
        D = self.cfg["track_length_m"]
        # 基站 A 在 0，B 在 D
        d_A = max(x_m - 0.0, 1.0)
        d_B = max(D - x_m, 1.0)

        # 1) 路径损耗（包含阴影 + 天气 + 可选快衰落）
        pl_A = self.pathloss_db(x_m, d_A, is_cell_A=True)
        pl_B = self.pathloss_db(x_m, d_B, is_cell_A=False)

        # 2) RSRP = Ptx - PL (单位 dBm)
        Ptx_A = self.cfg["Ptx_A_dbm"]
        Ptx_B = self.cfg["Ptx_B_dbm"]
        rsrp_A = Ptx_A - pl_A
        rsrp_B = Ptx_B - pl_B

        # 3) 噪声功率
        noise_dbm = self.cfg["noise_dbm"]
        
        # 线性功率转换函数 (dBm -> mW)
        def dbm_to_mw(dbm):
            return 10 ** (dbm / 10.0)
        
        def linear_to_db(x):
            return 10 * np.log10(x + 1e-12)

        noise_mw = dbm_to_mw(noise_dbm)
        
        # 4) 动态同频干扰建模（基于公式 I(d) = 10^(Pr1(d)/10) + 10^(Pr2(d)/10)）
        # 是否启用动态干扰（默认启用）
        use_dynamic_interference = self.cfg.get("use_dynamic_interference", True)
        
        if use_dynamic_interference:
            # 根据公式：同频干扰 = 10^(Pr1/10) + 10^(Pr2/10)
            # 对于基站A：干扰来自基站B的信号（Pr2 = rsrp_B）
            # 对于基站B：干扰来自基站A的信号（Pr1 = rsrp_A）
            
            # 基站A受到的干扰 = 基站B的信号功率（线性域）
            interference_A_mw = dbm_to_mw(rsrp_B)
            
            # 基站B受到的干扰 = 基站A的信号功率（线性域）
            interference_B_mw = dbm_to_mw(rsrp_A)
            
            # 如果配置了额外的同频基站干扰，可以叠加
            if self.cfg.get("enable_extra_interference", False):
                extra_interference_dbm = self.cfg.get("extra_interference_dbm", -105.0)
                extra_interference_mw = dbm_to_mw(extra_interference_dbm)
                interference_A_mw += extra_interference_mw
                interference_B_mw += extra_interference_mw
            
            # 总噪声+干扰（线性功率叠加）
            noise_plus_interference_A_mw = noise_mw + interference_A_mw
            noise_plus_interference_B_mw = noise_mw + interference_B_mw
        else:
            # 静态干扰：使用固定干扰功率（向后兼容）
            inter_A_dbm = self.cfg.get("interference_A_dbm", -100.0)
            inter_B_dbm = self.cfg.get("interference_B_dbm", -100.0)
            
            noise_plus_interference_A_mw = noise_mw + dbm_to_mw(inter_A_dbm)
            noise_plus_interference_B_mw = noise_mw + dbm_to_mw(inter_B_dbm)

        # 5) 计算 SINR
        sig_A_mw = dbm_to_mw(rsrp_A)
        sig_B_mw = dbm_to_mw(rsrp_B)

        sinr_A = sig_A_mw / noise_plus_interference_A_mw
        sinr_B = sig_B_mw / noise_plus_interference_B_mw

        sinr_A_db = linear_to_db(sinr_A)
        sinr_B_db = linear_to_db(sinr_B)

        return rsrp_A, rsrp_B, sinr_A_db, sinr_B_db

