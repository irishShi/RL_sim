"""
批量对比测试：多个场景下对比不同A3策略和RL策略的性能

功能：
1. 生成100个不同的场景
2. 定义多个A3策略的(Hys, TTT)组合
3. 对每个场景运行所有策略（包括RL策略）
4. 收集性能指标并生成对比图表
"""
import argparse
import os
import sys
import json
from datetime import datetime
from collections import Counter
import numpy as np
import torch
import yaml
import matplotlib.pyplot as plt
from tqdm import tqdm

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from envs.train_ho_env import TrainHandoverEnv
from models import RainbowWithForecast, ActionSpace, ObservationWindow
from utils.scenario_generator import ScenarioGenerator
from test_simple import TraditionalA3Policy, resolve_checkpoint_path, run_episode

# 配置 matplotlib 支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

RL_POLICY_NAME = "RL策略 (Rainbow DQN)"
GLOBAL_ORACLE_A3_NAME = "全局最优A3 (逐场景Oracle)"
FIXED_A3_REFERENCE_NAME = "固定A3参考 (全场景同参)"


def _extract_handover_points(trajectory: list):
    """从轨迹中提取切换点坐标（x, y）。"""
    ho_x = []
    ho_y = []
    for p in trajectory:
        if p.get("ho_executed", False):
            ho_x.append(float(p.get("x", 0.0)))
            serving_cell = int(p.get("serving_cell", 0))
            if serving_cell == 0:
                ho_y.append(float(p.get("rsrp_A_dbm", p.get("rsrp_serv_dbm", 0.0))))
            else:
                ho_y.append(float(p.get("rsrp_B_dbm", p.get("rsrp_serv_dbm", 0.0))))
    return ho_x, ho_y


def plot_scenario_rsrp_with_handover_points(
    scenario_idx: int,
    trajectories_by_policy: dict,
    output_dir: str,
    overlap_delta_rsrp_db: float = 5.0,
    overlap_zone_enabled: bool = True,
):
    """
    为单个场景绘制“分子图样式”RSRP-距离曲线，并标注各策略切换点。
    """
    if not trajectories_by_policy:
        return

    policy_items = [(name, traj) for name, traj in trajectories_by_policy.items() if traj]
    if not policy_items:
        return

    n_policies = len(policy_items)
    fig, axes = plt.subplots(n_policies, 1, figsize=(12, max(4.0 * n_policies, 6.0)), sharex=True)
    if n_policies == 1:
        axes = [axes]

    marker_list = ['o', '^', 's', 'D', 'P', 'X', 'v', '<', '>', '*', 'h', '8']
    colors = plt.get_cmap('tab20')(np.linspace(0, 1, max(2, n_policies)))

    for idx, (policy_name, traj) in enumerate(policy_items):
        ax = axes[idx]
        xs = np.array([p.get("x", 0.0) for p in traj], dtype=np.float32)
        rsrp_A = np.array([p.get("rsrp_A_dbm", 0.0) for p in traj], dtype=np.float32)
        rsrp_B = np.array([p.get("rsrp_B_dbm", 0.0) for p in traj], dtype=np.float32)

        # 标注“切换重叠区”（单一连续区间）：避免 |ΔRSRP|≤N 在快衰落下形成多个碎片区
        # 定义与 envs/train_ho_env.py 保持一致：
        #   x_left = 首次满足 (RSRP_A - RSRP_B) ≤ +N 的位置
        #   x_right = （在 x_left 之后）满足 (RSRP_A - RSRP_B) ≥ -N 的最右位置
        if overlap_zone_enabled and len(xs) > 1:
            N = float(overlap_delta_rsrp_db)
            delta = rsrp_A - rsrp_B
            left_candidates = np.where(delta <= N)[0]
            if left_candidates.size > 0:
                left_idx = int(left_candidates[0])
                right_candidates = np.where(delta >= -N)[0]
                right_candidates = right_candidates[right_candidates >= left_idx]
                if right_candidates.size > 0:
                    right_idx = int(right_candidates[-1])
                    x0 = float(xs[left_idx])
                    x1 = float(xs[right_idx])
                    if x1 < x0:
                        x0, x1 = x1, x0
                    label = f"重叠切换区(N={N:.1f}dB)"
                    ax.axvspan(x0, x1, color="grey", alpha=0.12, zorder=0, label=label)

        ax.plot(xs, rsrp_A, label="RSRP A小区", color="tab:blue", alpha=0.65, linewidth=1.8)
        ax.plot(xs, rsrp_B, label="RSRP B小区", color="tab:orange", alpha=0.65, linewidth=1.8)

        ho_x, ho_y = _extract_handover_points(traj)
        if ho_x:
            ax.scatter(
                ho_x,
                ho_y,
                s=85,
                marker=marker_list[idx % len(marker_list)],
                color=colors[idx],
                edgecolors='black',
                linewidths=0.9,
                label=f"{policy_name} 切换点",
                zorder=6,
            )

        ax.set_ylabel("RSRP / dBm")
        ax.set_title(f"Episode {scenario_idx} - {policy_name}")
        ax.grid(True, alpha=0.25)
        ax.legend(loc='upper left', fontsize=8)

    axes[-1].set_xlabel("距离 x / m")
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"episode_{scenario_idx:03d}.png")
    plt.savefig(output_path, dpi=220, bbox_inches='tight')
    plt.close()


def collect_kpis(trajectory: list, kpis: dict = None) -> dict:
    """
    从轨迹数据中收集KPI指标
    
    Args:
        trajectory: 轨迹数据列表
        kpis: 环境返回的KPI字典（如果有）
        
    Returns:
        KPI字典
    """
    if kpis is None:
        kpis = {}
    
    # 从轨迹中提取数据
    sinr_samples = [p['sinr_serv_db'] for p in trajectory]
    ho_count = sum(1 for p in trajectory if p['ho_executed'])
    # 切换成功率（结合文档思路做离散化）：
    # P_s ≈ 成功执行切换次数 / 切换触发次数
    ho_trigger_count = sum(1 for p in trajectory if p.get('ho_triggered', p.get('ho_executed', False)))
    ho_success_count = ho_count
    ho_success_rate = (ho_success_count / ho_trigger_count) if ho_trigger_count > 0 else 0.0
    ho_blocked_count = sum(1 for p in trajectory if p.get('ho_blocked_by_guard', False))
    ho_blocked_ratio = (ho_blocked_count / ho_trigger_count) if ho_trigger_count > 0 else 0.0
    
    # 计算SINR统计
    if sinr_samples:
        sinr_mean = np.mean(sinr_samples)
        sinr_std = np.std(sinr_samples)
        sinr_p5 = np.percentile(sinr_samples, 5)
        sinr_p95 = np.percentile(sinr_samples, 95)
        sinr_below_minus3db_ratio = np.mean(np.array(sinr_samples) < -3.0)
    else:
        sinr_mean = 0.0
        sinr_std = 0.0
        sinr_p5 = 0.0
        sinr_p95 = 0.0
        sinr_below_minus3db_ratio = 0.0
    
    # 计算切换次数/公里（假设轨迹长度为3000m）
    track_length_km = 3.0  # 3km
    ho_per_km = ho_count / track_length_km if track_length_km > 0 else 0.0
    
    # 合并KPI
    result = {
        'ho_count': ho_count,
        'ho_per_km': ho_per_km,
        'ho_trigger_count': ho_trigger_count,
        'ho_success_count': ho_success_count,
        'ho_success_rate': ho_success_rate,
        'ho_blocked_by_guard_ratio': ho_blocked_ratio,
        'sinr_mean_db': sinr_mean,
        'sinr_std_db': sinr_std,
        'sinr_p5_db': sinr_p5,
        'sinr_p95_db': sinr_p95,
        'sinr_below_minus3db_ratio': sinr_below_minus3db_ratio,
        # 默认值
        'outage_time_ratio': 0.0,
        'outage_events': 0,
        'ping_pong_count': 0,
        'ping_pong_ratio': 0.0,
        'interruption_total_time': 0.0,
        'interruption_time_in_overlap_zone_s': 0.0,
        'overlap_zone_time_s': 0.0,
        'overlap_zone_sinr_mean_db': 0.0,
        'overlap_zone_delta_rsrp_threshold_db': 0.0,
        'comm_interruption_ratio_in_overlap_zone': 0.0,
    }
    
    # 添加环境返回的KPI（如果存在）
    if kpis:
        result.update({
            'outage_time_ratio': kpis.get('outage_time_ratio', result['outage_time_ratio']),
            'outage_events': kpis.get('outage_events', result['outage_events']),
            'ping_pong_count': kpis.get('ping_pong_count', result['ping_pong_count']),
            'ping_pong_ratio': kpis.get('ping_pong_ratio', result['ping_pong_ratio']),
            'interruption_total_time': kpis.get('interruption_total_time', result['interruption_total_time']),
            'interruption_time_in_overlap_zone_s': kpis.get(
                'interruption_time_in_overlap_zone_s', result['interruption_time_in_overlap_zone_s']
            ),
            'overlap_zone_time_s': kpis.get('overlap_zone_time_s', result['overlap_zone_time_s']),
            'overlap_zone_sinr_mean_db': kpis.get('overlap_zone_sinr_mean_db', result['overlap_zone_sinr_mean_db']),
            'overlap_zone_delta_rsrp_threshold_db': kpis.get(
                'overlap_zone_delta_rsrp_threshold_db', result['overlap_zone_delta_rsrp_threshold_db']
            ),
            'comm_interruption_ratio_in_overlap_zone': kpis.get(
                'comm_interruption_ratio_in_overlap_zone',
                result['comm_interruption_ratio_in_overlap_zone'],
            ),
        })
    
    return result


def summarize_policy_results(results_by_policy: dict) -> dict:
    """按策略汇总KPI均值、标准差、最小值和最大值。"""
    stats = {}
    for policy_name, policy_results in results_by_policy.items():
        if not policy_results:
            continue

        stats[policy_name] = {}
        for kpi_name in policy_results[0].keys():
            values = [r[kpi_name] for r in policy_results]
            stats[policy_name][f"{kpi_name}_mean"] = np.mean(values)
            stats[policy_name][f"{kpi_name}_std"] = np.std(values)
            stats[policy_name][f"{kpi_name}_min"] = np.min(values)
            stats[policy_name][f"{kpi_name}_max"] = np.max(values)
    return stats


def run_batch_test(num_scenarios: int = 100,
                   a3_configs: list = None,
                   base_seed: int = 10000,
                   device: str = 'cpu',
                   top_k_a3: int = 5,
                   oracle_use_full_grid: bool = True,
                   oracle_ho_penalty_weight: float = 0.5,
                   checkpoint_path: str = None,
                   run_dir: str = None):
    """
    运行批量测试

    Args:
        num_scenarios: 场景数量
        a3_configs: A3策略配置列表，每个元素为 (hys, ttt, name)。
                    必须从动作空间有效值构建（通过 build_a3_grid_configs）。
        base_seed: 基础随机种子
        device: 计算设备
        top_k_a3: 从RL动作分布中选择的Top-K A3策略数量
        oracle_use_full_grid: Oracle是否在全部候选中选优
        oracle_ho_penalty_weight: Oracle评分中的切换惩罚权重
        checkpoint_path: 待评估模型路径
        run_dir: 输出目录

    Returns:
        测试结果字典
    """
    if a3_configs is None:
        raise ValueError("a3_configs 不能为 None，请通过 build_a3_grid_configs() 构建")
    oracle_candidate_configs = list(a3_configs)
    
    print("=" * 80)
    print(
        f"批量对比测试：{num_scenarios}个场景，"
        f"A3候选{len(a3_configs)}组，最终显式对比RL分布Top-{top_k_a3}的A3策略 + 1个RL策略"
    )
    print("=" * 80)
    
    # 1. 加载配置
    base_dir = PROJECT_ROOT
    env_config_path = os.path.join(base_dir, "configs", "default_env_config.yaml")
    model_config_path = os.path.join(base_dir, "configs", "model_config.yaml")
    
    with open(model_config_path, 'r', encoding='utf-8') as f:
        model_config = yaml.safe_load(f)
    
    # 2. 加载RL模型
    checkpoint_path = resolve_checkpoint_path(base_dir, checkpoint_path)

    if not os.path.exists(checkpoint_path):
        print(f"\n错误：未找到离线训练模型！")
        print(f"查找路径：{checkpoint_path}")
        print(f"请先运行 train_rainbow_offline.py 生成 checkpoint")
        return None

    print(f"\n加载离线训练模型: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    hys_set = model_config['action_space']['hys_set']
    ttt_set = model_config['action_space']['ttt_set']
    num_actions = len(hys_set) * len(ttt_set)
    obs_dim = model_config['observation']['obs_dim']
    window_size = model_config['observation']['window_size']
    
    use_noisy = checkpoint.get('config', {}).get('use_noisy', False)
    if 'use_cql' in checkpoint or 'rainbow_offline' in checkpoint_path:
        use_noisy = False
    
    model = RainbowWithForecast(
        obs_dim=obs_dim,
        num_actions=num_actions,
        n_steps=window_size,
        feature_hidden=model_config['network']['shared']['hidden_dim'],
        encoder_hidden=model_config['network']['encoder']['hidden_dim'],
        num_atoms=model_config['network']['rainbow']['num_atoms'],
        v_min=model_config['network']['rainbow']['v_min'],
        v_max=model_config['network']['rainbow']['v_max'],
        use_noisy=use_noisy
    ).to(device)
    
    try:
        model.load_state_dict(checkpoint['online_net_state_dict'])
    except RuntimeError as exc:
        raise RuntimeError(
            "Checkpoint 与当前模型输入维度不兼容。当前版本已移除天气/温度特征，"
            "并从策略输入中移除当前 Hys/TTT，obs_dim 已调整为 7；"
            "需要重新采集离线数据并重新训练模型。"
        ) from exc
    model.eval()
    print("模型加载成功！")
    
    # 3. 创建环境和动作空间
    env_template = TrainHandoverEnv(config_path=env_config_path)
    action_space = ActionSpace()
    overlap_zone_enabled = bool(env_template.cfg.get("ho_overlap_kpi_enabled", True))
    overlap_delta_rsrp_db = float(env_template.cfg.get("ho_overlap_delta_rsrp_db", 5.0))
    
    # 4. 创建观测窗口（用于RL）
    obs_window = ObservationWindow(window_size=window_size, obs_dim=obs_dim)
    
    # 5. 生成场景数据
    print(f"\n生成{num_scenarios}个场景数据...")
    scenario_gen = ScenarioGenerator(env_template.cfg)
    scenario_dir = os.path.join(base_dir, "data", "scenarios")
    os.makedirs(scenario_dir, exist_ok=True)
    
    scenarios = []
    for i in tqdm(range(num_scenarios), desc="生成场景"):
        seed = base_seed + i
        scenario_data = scenario_gen.generate_scenario(seed=seed, position_resolution_m=1.0)
        scenarios.append(scenario_data)
    
    # 6. 第一阶段：仅运行RL，统计动作分布并收集RL KPI
    all_results = {RL_POLICY_NAME: []}
    rl_action_counter = Counter()
    rl_total_steps = 0
    rl_trajectories = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if run_dir:
        run_dir = run_dir if os.path.isabs(run_dir) else os.path.join(base_dir, run_dir)
    else:
        run_dir = os.path.join(base_dir, "experiments", "runs", f"{timestamp}_batch_comparison")
    scenario_plot_dir = os.path.join(
        run_dir, "figures", "episodes"
    )
    os.makedirs(scenario_plot_dir, exist_ok=True)
    print(f"\n场景级RSRP衰减图输出目录: {scenario_plot_dir}")

    print(f"\n第一阶段：运行RL策略并统计动作分布（{num_scenarios}个场景）...")
    for scenario_data in tqdm(scenarios, desc="RL测试进度"):
        env_rl = TrainHandoverEnv(config_path=env_config_path)
        obs_window.reset()
        traj_rl = run_episode(
            env_rl, None, RL_POLICY_NAME,
            obs_window=obs_window, model=model, device=device,
            seed=None, scenario_data=scenario_data
        )
        rl_trajectories.append(traj_rl.get('trajectory', []))

        # 获取环境返回的KPI（如果episode结束）
        kpis_from_env_rl = None
        if traj_rl['trajectory']:
            last_info_rl = traj_rl.get('last_info', {})
            if 'kpis' in last_info_rl:
                kpis_from_env_rl = last_info_rl['kpis']

        # 收集RL KPI
        kpis_rl = collect_kpis(traj_rl['trajectory'], kpis_from_env_rl)
        all_results[RL_POLICY_NAME].append(kpis_rl)

        # 统计RL动作分布（映射回离散动作集合）
        for p in traj_rl['trajectory']:
            h = float(p.get("current_hys", 3.0))
            t = float(p.get("current_ttt", 150.0))
            a = action_space.hys_ttt_to_action(h, t)
            h_c, t_c = action_space.action_to_hys_ttt(a)
            rl_action_counter[(float(h_c), float(t_c))] += 1
            rl_total_steps += 1

    # 7. 根据RL分布选Top-K A3策略
    if not rl_action_counter:
        print("\n警告：RL动作分布为空，回退到传入的前5个A3策略。")
        a3_configs = a3_configs[:max(1, int(top_k_a3))]
    else:
        top_k_a3 = max(1, int(top_k_a3))
        top_items = rl_action_counter.most_common(top_k_a3)
        a3_configs = []
        print(f"\n基于RL动作分布选择 Top-{len(top_items)} A3策略：")
        for (h, t), cnt in top_items:
            ratio = cnt / max(rl_total_steps, 1)
            name = f"A3 (Hys={h:.1f}dB, TTT={int(round(t))}ms)"
            a3_configs.append((float(h), float(t), name))
            print(f"  {name}: count={cnt}, ratio={ratio:.4f}")
    print(
        f"全局最优A3候选集: {'全48组' if oracle_use_full_grid else 'Top-K子集'} "
        f"| ho惩罚权重={oracle_ho_penalty_weight}"
    )

    # 初始化A3结果存储
    fixed_a3_configs = oracle_candidate_configs if oracle_use_full_grid else list(a3_configs)
    fixed_a3_results = {name: [] for _, _, name in fixed_a3_configs}
    for _, _, name in a3_configs:
        all_results[name] = []
    all_results[GLOBAL_ORACLE_A3_NAME] = []
    oracle_selection_records = []

    # 8. 第二阶段：运行Top-K A3策略 + Oracle候选
    print(f"\n第二阶段：运行测试（{num_scenarios}个场景 × {len(a3_configs)}个A3策略）...")
    for scenario_idx, scenario_data in enumerate(tqdm(scenarios, desc="A3测试进度"), start=1):
        scenario_trajectories = {
            RL_POLICY_NAME: rl_trajectories[scenario_idx - 1] if scenario_idx - 1 < len(rl_trajectories) else []
        }
        per_scenario_oracle_kpis = {}
        per_scenario_trajs = {}

        # 运行Top-K A3策略（复用单个 env 实例，通过 reset 切换场景）
        env_a3 = TrainHandoverEnv(config_path=env_config_path)
        for hys, ttt, name in a3_configs:
            policy = TraditionalA3Policy(hys, ttt, action_space)
            policy.reset()

            traj = run_episode(
                env_a3, lambda obs, info, dt, p=policy: p.decide(obs, info, dt), name,
                obs_window=None, model=None, device=device,
                seed=None, scenario_data=scenario_data, a3_policy=policy
            )

            kpis_from_env = None
            if traj['trajectory']:
                last_info = traj.get('last_info', {})
                if 'kpis' in last_info:
                    kpis_from_env = last_info['kpis']
            kpis = collect_kpis(traj['trajectory'], kpis_from_env)
            all_results[name].append(kpis)
            if name in fixed_a3_results:
                fixed_a3_results[name].append(kpis)
            per_scenario_oracle_kpis[name] = kpis
            per_scenario_trajs[name] = traj['trajectory']
            scenario_trajectories[name] = traj['trajectory']

        # 若Oracle使用全网格，补跑Top-K之外的A3组合（仅用于Oracle选优，不加入显式对比表）
        if oracle_use_full_grid:
            selected_names = {name for _, _, name in a3_configs}
            for hys, ttt, name in oracle_candidate_configs:
                if name in selected_names:
                    continue
                policy_oracle = TraditionalA3Policy(hys, ttt, action_space)
                policy_oracle.reset()

                traj_oracle = run_episode(
                    env_a3, lambda obs, info, dt, p=policy_oracle: p.decide(obs, info, dt), name,
                    obs_window=None, model=None, device=device,
                    seed=None, scenario_data=scenario_data, a3_policy=policy_oracle
                )
                kpis_from_env_oracle = None
                if traj_oracle['trajectory']:
                    last_info_oracle = traj_oracle.get('last_info', {})
                    if 'kpis' in last_info_oracle:
                        kpis_from_env_oracle = last_info_oracle['kpis']
                kpis_oracle = collect_kpis(traj_oracle['trajectory'], kpis_from_env_oracle)
                per_scenario_oracle_kpis[name] = kpis_oracle
                if name in fixed_a3_results:
                    fixed_a3_results[name].append(kpis_oracle)
                # 关键：保存Oracle候选的轨迹，便于场景级绘图时把逐场景最优A3也画出来
                per_scenario_trajs[name] = traj_oracle['trajectory']

        # 全局最优A3（逐场景Oracle）：
        # 对每个场景单独枚举候选A3组合，并利用该场景完整结果选择自己的最优参数。
        # 这不是跨场景共用同一个固定解，而是“上帝视角”的逐场景选优基线。
        oracle_cfgs_for_selection = oracle_candidate_configs if oracle_use_full_grid else a3_configs
        if per_scenario_oracle_kpis:
            scenario_best = select_best_a3_for_single_scenario(
                per_scenario_oracle_kpis,
                oracle_cfgs_for_selection,
                ho_penalty_weight=float(oracle_ho_penalty_weight),
            )
            if scenario_best is not None:
                best_name, best_kpis, best_hys, best_ttt = scenario_best
                all_results[GLOBAL_ORACLE_A3_NAME].append(best_kpis)
                oracle_selection_records.append({
                    "scenario_idx": int(scenario_idx),
                    "policy_name": best_name,
                    "hys_db": float(best_hys),
                    "ttt_ms": float(best_ttt),
                })
                if best_name in per_scenario_trajs:
                    # 仅用于场景图展示：把Oracle选中的参数也标注出来（每个场景可能不同）
                    oracle_plot_name = f"全局最优A3 (Hys={best_hys:.1f}dB, TTT={best_ttt:.0f}ms)"
                    scenario_trajectories[oracle_plot_name] = per_scenario_trajs[best_name]

        # 每个场景输出1张图：双基站RSRP曲线 + 各策略切换点
        plot_scenario_rsrp_with_handover_points(
            scenario_idx=scenario_idx,
            trajectories_by_policy=scenario_trajectories,
            output_dir=scenario_plot_dir,
            overlap_delta_rsrp_db=overlap_delta_rsrp_db,
            overlap_zone_enabled=overlap_zone_enabled,
        )
    
    # 8. 计算统计结果
    print("\n计算统计结果...")
    stats = summarize_policy_results(all_results)
    fixed_a3_stats = summarize_policy_results(fixed_a3_results)
    
    return {
        'all_results': all_results,
        'stats': stats,
        'fixed_a3_results': fixed_a3_results,
        'fixed_a3_stats': fixed_a3_stats,
        'fixed_a3_configs': fixed_a3_configs,
        'num_scenarios': num_scenarios,
        'a3_configs': a3_configs,
        'rl_action_distribution_topk': [
            {
                'hys_db': h,
                'ttt_ms': t,
                'count': int(c),
                'ratio': float(c / max(rl_total_steps, 1)),
            }
            for (h, t), c in rl_action_counter.most_common(max(1, int(top_k_a3)))
        ],
        'oracle_selection_records': oracle_selection_records,
        'global_oracle_policy_name': GLOBAL_ORACLE_A3_NAME,
        'fixed_a3_reference_name': FIXED_A3_REFERENCE_NAME,
        'oracle_use_full_grid': bool(oracle_use_full_grid),
        'oracle_ho_penalty_weight': float(oracle_ho_penalty_weight),
        'checkpoint_path': os.path.abspath(checkpoint_path),
        'run_dir': run_dir,
    }


def build_a3_grid_configs(hys_set: list, ttt_set: list) -> list:
    """构建 A3 参数全搜索网格。"""
    a3_configs = []
    for hys in hys_set:
        for ttt in ttt_set:
            h = float(hys)
            t = float(ttt)
            name = f"A3 (Hys={h:.1f}dB, TTT={int(round(t))}ms)"
            a3_configs.append((h, t, name))
    return a3_configs


def select_best_a3_lexicographic(stats: dict, a3_configs: list, eps: float = 1e-12) -> dict:
    """
    词典序选择最优A3参数：
    1) 最小 ho_count_mean
    2) 最大 overlap_zone_sinr_mean_db_mean
    3) 最小 outage_time_ratio_mean（tie-break）
    """
    candidates = []
    for hys, ttt, name in a3_configs:
        s = stats.get(name)
        if not s:
            continue
        candidates.append({
            "policy_name": name,
            "hys_db": float(hys),
            "ttt_ms": float(ttt),
            "ho_count_mean": float(s.get("ho_count_mean", 0.0)),
            "ho_success_rate_mean": float(s.get("ho_success_rate_mean", 0.0)),
            "overlap_zone_sinr_mean_db_mean": float(s.get("overlap_zone_sinr_mean_db_mean", 0.0)),
            "outage_time_ratio_mean": float(s.get("outage_time_ratio_mean", 0.0)),
            "ping_pong_ratio_mean": float(s.get("ping_pong_ratio_mean", 0.0)),
            "sinr_mean_db_mean": float(s.get("sinr_mean_db_mean", 0.0)),
            "comm_interruption_ratio_in_overlap_zone_mean": float(
                s.get("comm_interruption_ratio_in_overlap_zone_mean", 0.0)
            ),
        })
    
    if not candidates:
        return {}
    
    min_ho = min(c["ho_count_mean"] for c in candidates)
    ho_best = [c for c in candidates if abs(c["ho_count_mean"] - min_ho) <= eps]
    
    max_overlap_sinr = max(c["overlap_zone_sinr_mean_db_mean"] for c in ho_best)
    quality_best = [c for c in ho_best if abs(c["overlap_zone_sinr_mean_db_mean"] - max_overlap_sinr) <= eps]
    
    best = min(quality_best, key=lambda c: c["outage_time_ratio_mean"])
    ranking = sorted(
        candidates,
        key=lambda c: (
            c["ho_count_mean"],
            -c["overlap_zone_sinr_mean_db_mean"],
            c["outage_time_ratio_mean"],
        ),
    )
    
    return {
        "best": best,
        "ranking": ranking,
    }


def select_best_a3_for_single_scenario(
    per_scenario_a3_kpis: dict,
    a3_configs: list,
    ho_penalty_weight: float = 0.5,
):
    """
    在单个场景内选择最优A3（逐场景Oracle）：
    对同一个预生成场景枚举候选A3组合，利用完整场景结果为该场景单独选优。
    因此不同场景可以选择不同的 (Hys, TTT)，不是全场景共享同一组固定参数。

    score = overlap_zone_sinr_mean_db - ho_penalty_weight * ho_count - 10 * outage_time_ratio
    再以 (更小ho_count, 更小outage) 做稳定tie-break。
    """
    candidates = []
    for hys, ttt, name in a3_configs:
        if name not in per_scenario_a3_kpis:
            continue
        k = per_scenario_a3_kpis[name]
        candidates.append((name, k, float(hys), float(ttt)))
    if not candidates:
        return None
    def scenario_score(item):
        _, k, _, _ = item
        return (
            float(k.get("overlap_zone_sinr_mean_db", 0.0))
            - float(ho_penalty_weight) * float(k.get("ho_count", 0.0))
            - 10.0 * float(k.get("outage_time_ratio", 0.0))
        )

    candidates.sort(
        key=lambda x: (
            -scenario_score(x),
            float(x[1].get("ho_count", 0.0)),
            float(x[1].get("outage_time_ratio", 0.0)),
        )
    )
    return candidates[0]


def save_fixed_a3_reference_result(output_path: str, selection: dict, num_scenarios: int):
    """保存全场景同一组固定A3参数的参考结果。"""
    if not selection:
        return
    
    payload = {
        "policy_name": FIXED_A3_REFERENCE_NAME,
        "meaning": "在所有测试场景上共用同一组固定A3参数，并按跨场景平均KPI做词典序排序；该结果只是传统固定参数参考，不代表逐场景全局最优。",
        "selection_rule": [
            "minimize ho_count_mean",
            "maximize overlap_zone_sinr_mean_db_mean",
            "minimize outage_time_ratio_mean as tie-break",
        ],
        "num_scenarios": int(num_scenarios),
        "best": selection["best"],
        "top10": selection["ranking"][:10],
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def save_fixed_a3_grid_result(output_path: str, selection: dict, num_scenarios: int):
    """保存完整固定A3网格的词典序排名结果。"""
    if not selection:
        return

    payload = {
        "policy_name": "固定A3全网格排名",
        "meaning": "每一行都是一组在所有场景上保持不变的固定A3参数，用于传统基线敏感性分析。",
        "selection_rule": [
            "minimize ho_count_mean",
            "maximize overlap_zone_sinr_mean_db_mean",
            "minimize outage_time_ratio_mean as tie-break",
        ],
        "num_scenarios": int(num_scenarios),
        "num_candidates": int(len(selection.get("ranking", []))),
        "best": selection["best"],
        "ranking": selection["ranking"],
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def save_oracle_a3_result(output_path: str, results: dict):
    """保存逐场景全局最优A3选择分布与汇总KPI。"""
    records = results.get("oracle_selection_records", [])
    if not records:
        return
    counter = Counter((r["hys_db"], r["ttt_ms"], r["policy_name"]) for r in records)
    total = len(records)
    distribution = []
    for (hys, ttt, name), c in counter.most_common():
        distribution.append({
            "policy_name": name,
            "hys_db": float(hys),
            "ttt_ms": float(ttt),
            "count": int(c),
            "ratio": float(c / total),
        })
    oracle_stats_raw = results.get("stats", {}).get(GLOBAL_ORACLE_A3_NAME, {})
    oracle_stats = {}
    for k, v in oracle_stats_raw.items():
        if isinstance(v, (np.floating, float)):
            oracle_stats[k] = float(v)
        elif isinstance(v, (np.integer, int)):
            oracle_stats[k] = int(v)
        else:
            oracle_stats[k] = v
    candidate_configs = results.get("fixed_a3_configs") if results.get("oracle_use_full_grid", True) else results.get("a3_configs")
    candidate_configs = candidate_configs or []
    payload = {
        "policy_name": GLOBAL_ORACLE_A3_NAME,
        "meaning": "逐场景上帝视角：每个场景都基于该场景完整信号/结果枚举候选A3组合并单独选择最优解，不要求所有场景共用同一组参数。",
        "selection_rule": [
            "for each scenario, evaluate every candidate A3 config on the same pre-generated scenario",
            "per-scenario maximize score",
            "score = overlap_zone_sinr_mean_db - ho_penalty_weight * ho_count - 10 * outage_time_ratio",
            "tie-break: lower ho_count, then lower outage_time_ratio",
        ],
        "num_scenarios": int(results.get("num_scenarios", total)),
        "num_candidates_per_scenario": int(len(candidate_configs)),
        "oracle_use_full_grid": bool(results.get("oracle_use_full_grid", True)),
        "oracle_ho_penalty_weight": float(results.get("oracle_ho_penalty_weight", 0.5)),
        "global_optimal_summary_kpis": oracle_stats,
        "global_optimal_selection_distribution": distribution,
        "scenario_selections": records,
        # 兼容旧字段名
        "oracle_summary_kpis": oracle_stats,
        "oracle_selection_distribution": distribution,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def plot_comparison(results: dict, output_dir: str = "."):
    """
    绘制对比图表
    
    Args:
        results: 测试结果字典
        output_dir: 输出目录
    """
    all_results = results['all_results']
    stats = results['stats']
    num_scenarios = results['num_scenarios']
    os.makedirs(output_dir, exist_ok=True)
    
    # 选择代表性的KPI指标
    kpi_names = [
        ('ho_count', '切换次数'),
        ('ho_success_rate', '切换成功率'),
        ('ho_per_km', '切换次数/公里'),
        ('sinr_mean_db', '平均SINR (dB)'),
        ('sinr_p5_db', 'SINR 5%分位 (dB)'),
        ('outage_time_ratio', 'Outage时间占比'),
        ('ping_pong_ratio', '乒乓切换率'),
        ('overlap_zone_sinr_mean_db', '重叠区平均SINR (dB)'),
        ('comm_interruption_ratio_in_overlap_zone', '重叠区内通信中断率'),
    ]
    
    # 创建图表
    n_kpis = len(kpi_names)
    n_cols = 3
    n_rows = (n_kpis + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    axes = axes.flatten() if n_kpis > 1 else [axes]
    
    policy_names = list(stats.keys())
    n_policies = len(policy_names)
    x_pos = np.arange(n_policies)
    width = 0.6
    
    for idx, (kpi_key, kpi_label) in enumerate(kpi_names):
        ax = axes[idx]
        
        # 提取数据
        means = []
        stds = []
        for policy_name in policy_names:
            mean_key = f"{kpi_key}_mean"
            std_key = f"{kpi_key}_std"
            if mean_key in stats[policy_name]:
                means.append(stats[policy_name][mean_key])
                stds.append(stats[policy_name][std_key])
            else:
                means.append(0.0)
                stds.append(0.0)
        
        # 绘制柱状图
        bars = ax.bar(x_pos, means, width, yerr=stds, capsize=5, 
                     alpha=0.7, edgecolor='black', linewidth=1)
        
        # 设置颜色（RL、逐场景全局最优、普通A3分开）
        for i, (bar, policy_name) in enumerate(zip(bars, policy_names)):
            if "RL" in policy_name:
                bar.set_color('red')
            elif policy_name == GLOBAL_ORACLE_A3_NAME:
                bar.set_color('seagreen')
            else:
                bar.set_color('steelblue')
        
        ax.set_xlabel('策略')
        ax.set_ylabel(kpi_label)
        ax.set_title(f'{kpi_label}对比 (n={num_scenarios})')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(policy_names, rotation=45, ha='right')
        ax.grid(True, alpha=0.3, axis='y')
    
    # 隐藏多余的子图
    for idx in range(n_kpis, len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, "batch_comparison.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n对比图已保存至: {output_path}")
    plt.close()
    
    # 绘制箱线图（更详细的分布对比）
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    axes = axes.flatten() if n_kpis > 1 else [axes]
    
    for idx, (kpi_key, kpi_label) in enumerate(kpi_names):
        ax = axes[idx]
        
        # 提取数据
        data_list = []
        labels = []
        for policy_name in policy_names:
            values = [r[kpi_key] for r in all_results[policy_name] if kpi_key in r]
            if values:
                data_list.append(values)
                labels.append(policy_name)
        
        if data_list:
            # Matplotlib ≥3.9：labels 改名为 tick_labels
            try:
                bp = ax.boxplot(data_list, tick_labels=labels, patch_artist=True)
            except TypeError:
                bp = ax.boxplot(data_list, labels=labels, patch_artist=True)
            
            # 设置颜色
            for i, patch in enumerate(bp['boxes']):
                if "RL" in labels[i]:
                    patch.set_facecolor('lightcoral')
                elif labels[i] == GLOBAL_ORACLE_A3_NAME:
                    patch.set_facecolor('lightgreen')
                else:
                    patch.set_facecolor('lightblue')
                patch.set_alpha(0.7)
            
            ax.set_xlabel('策略')
            ax.set_ylabel(kpi_label)
            ax.set_title(f'{kpi_label}分布对比 (n={num_scenarios})')
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3, axis='y')
    
    # 隐藏多余的子图
    for idx in range(n_kpis, len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, "batch_comparison_boxplot.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"箱线图已保存至: {output_path}")
    plt.close()


def print_summary(results: dict, fixed_a3_reference: dict = None):
    """
    打印统计摘要
    
    Args:
        results: 测试结果字典
    """
    stats = results['stats']
    num_scenarios = results['num_scenarios']
    
    print("\n" + "=" * 80)
    print("统计摘要")
    print("=" * 80)
    
    # 打印表格（增加切换成功率，与文档定义离散化版本一致）
    hdr = (
        f"\n{'策略':<30} {'切换次数':<10} {'切换成功率':<10} {'SINR均值':<10} "
        f"{'Outage占比':<12} {'乒乓率':<8} {'重叠区SINR':<12} {'区内中断率':<12}"
    )
    print(hdr)
    print("-" * 122)
    
    for policy_name in sorted(stats.keys()):
        s = stats[policy_name]
        ho_mean = s.get('ho_count_mean', 0.0)
        ho_succ_rate = s.get('ho_success_rate_mean', 0.0)
        sinr_mean = s.get('sinr_mean_db_mean', 0.0)
        outage_ratio = s.get('outage_time_ratio_mean', 0.0)
        ping_pong = s.get('ping_pong_ratio_mean', 0.0)
        overlap_sinr = s.get('overlap_zone_sinr_mean_db_mean', 0.0)
        int_in_oz = s.get('comm_interruption_ratio_in_overlap_zone_mean', 0.0)
        
        print(
            f"{policy_name:<30} {ho_mean:<10.2f} {ho_succ_rate:<10.4f} {sinr_mean:<10.2f} "
            f"{outage_ratio:<12.4f} {ping_pong:<8.4f} {overlap_sinr:<12.3f} {int_in_oz:<12.4f}"
        )
    
    # 额外输出词典序选中的固定A3参考，便于和全表对照
    if fixed_a3_reference:
        print("-" * 122)
        print(
            f"{FIXED_A3_REFERENCE_NAME:<30} "
            f"{fixed_a3_reference.get('ho_count_mean', 0.0):<10.2f} "
            f"{fixed_a3_reference.get('ho_success_rate_mean', 0.0):<10.4f} "
            f"{fixed_a3_reference.get('sinr_mean_db_mean', 0.0):<10.2f} "
            f"{fixed_a3_reference.get('outage_time_ratio_mean', 0.0):<12.4f} "
            f"{fixed_a3_reference.get('ping_pong_ratio_mean', 0.0):<8.4f} "
            f"{fixed_a3_reference.get('overlap_zone_sinr_mean_db_mean', 0.0):<12.3f} "
            f"{fixed_a3_reference.get('comm_interruption_ratio_in_overlap_zone_mean', 0.0):<12.4f}"
        )
        print(
            f"  -> 固定参数: Hys={fixed_a3_reference.get('hys_db', 0.0):.1f} dB, "
            f"TTT={fixed_a3_reference.get('ttt_ms', 0.0):.0f} ms, "
            f"对应策略: {fixed_a3_reference.get('policy_name', 'N/A')}"
        )
    
    print(f"\n测试场景数: {num_scenarios}")

    # 逐场景全局最优A3参数分布（每场景最优的 (Hys, TTT) 可能不同）
    records = results.get("oracle_selection_records", []) or []
    if records:
        counter = Counter((float(r.get("hys_db", 0.0)), float(r.get("ttt_ms", 0.0))) for r in records)
        print("\n全局最优A3逐场景参数分布（Top-5）:")
        for (hys, ttt), c in counter.most_common(5):
            ratio = c / max(1, len(records))
            print(f"  Hys={hys:.1f} dB, TTT={ttt:.0f} ms: count={c}, ratio={ratio:.3f}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="批量测试：A3策略网格 vs 离线训练 Rainbow DQN")
    parser.add_argument("--checkpoint_path", type=str, default=None,
                        help="待评估 checkpoint 路径；默认自动选择最新非 legacy 离线 best/final 模型")
    parser.add_argument("--num_scenarios", type=int, default=100, help="测试场景数量")
    parser.add_argument("--base_seed", type=int, default=10000, help="场景随机种子起点")
    parser.add_argument("--top_k_a3", type=int, default=5, help="显式对比RL动作分布Top-K A3策略")
    parser.add_argument("--oracle_ho_penalty_weight", type=float, default=0.5,
                        help="逐场景全局最优A3评分中的切换次数惩罚权重")
    parser.add_argument("--oracle_topk_only", action="store_true",
                        help="逐场景全局最优A3只在Top-K A3中选优；默认每个场景都在48组全网格中选优")
    parser.add_argument("--run_dir", type=str, default=None,
                        help="输出目录；默认写入 experiments/runs/<timestamp>_batch_comparison")
    args = parser.parse_args()

    base_dir = PROJECT_ROOT
    env_config_path = os.path.join(base_dir, "configs", "default_env_config.yaml")
    with open(env_config_path, 'r', encoding='utf-8') as f:
        env_cfg = yaml.safe_load(f)
    action_space_cfg = env_cfg.get("action_space", {})
    hys_set = action_space_cfg.get("hys_set", [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0])
    ttt_set = action_space_cfg.get("ttt_set", [0, 50, 100, 150, 300, 650])
    # 定义A3参数全搜索配置（默认48组）
    a3_configs = build_a3_grid_configs(hys_set, ttt_set)
    
    # 设置设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n使用设备: {device}")
    
    # 运行批量测试
    results = run_batch_test(
        num_scenarios=args.num_scenarios,
        a3_configs=a3_configs,
        base_seed=args.base_seed,
        device=device,
        top_k_a3=args.top_k_a3,
        oracle_use_full_grid=not args.oracle_topk_only,
        oracle_ho_penalty_weight=args.oracle_ho_penalty_weight,
        checkpoint_path=args.checkpoint_path,
        run_dir=args.run_dir,
    )
    
    if results is None:
        return

    output_dir = results.get("run_dir") or os.path.join(base_dir, "experiments", "runs", "latest_batch_comparison")
    metrics_dir = os.path.join(output_dir, "metrics")
    figures_dir = os.path.join(output_dir, "figures")
    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    # 逐场景全局最优A3结果：每个场景单独枚举候选组合并选优
    oracle_stats = results.get("stats", {}).get(GLOBAL_ORACLE_A3_NAME, {})
    if oracle_stats:
        print("\n" + "=" * 80)
        print(f"{GLOBAL_ORACLE_A3_NAME} ({'全48组' if results.get('oracle_use_full_grid', True) else 'Top-K子集'}枚举)")
        print("=" * 80)
        print(
            "ho_count_mean={:.4f}, overlap_zone_sinr_mean_db_mean={:.4f}, outage_time_ratio_mean={:.4f}".format(
                oracle_stats.get("ho_count_mean", 0.0),
                oracle_stats.get("overlap_zone_sinr_mean_db_mean", 0.0),
                oracle_stats.get("outage_time_ratio_mean", 0.0),
            )
        )
        global_optimal_result_path = os.path.join(metrics_dir, "global_optimal_a3_result.json")
        save_oracle_a3_result(global_optimal_result_path, results)
        print(f"逐场景全局最优A3结果已保存至: {global_optimal_result_path}")

        # 兼容旧入口：best_a3_result.json 现在也指向逐场景全局最优，而不是固定同参A3。
        best_result_path = os.path.join(metrics_dir, "best_a3_result.json")
        save_oracle_a3_result(best_result_path, results)
        print(f"best_a3_result.json 已更新为逐场景全局最优结果: {best_result_path}")

        oracle_alias_path = os.path.join(metrics_dir, "oracle_a3_result.json")
        save_oracle_a3_result(oracle_alias_path, results)
        print(f"兼容旧文件名的Oracle结果已保存至: {oracle_alias_path}")
    
    # 词典序最优 A3 参数选择（全48组固定参数，仅作参考）
    fixed_a3_stats = results.get("fixed_a3_stats", results["stats"])
    fixed_a3_configs = results.get("fixed_a3_configs", results["a3_configs"])
    selection = select_best_a3_lexicographic(fixed_a3_stats, fixed_a3_configs)
    if selection:
        best = selection["best"]
        print("\n" + "=" * 80)
        print(f"{FIXED_A3_REFERENCE_NAME} ({len(fixed_a3_configs)}组固定参数词典序排名)")
        print("=" * 80)
        print(
            f"Hys={best['hys_db']:.1f} dB, TTT={best['ttt_ms']:.0f} ms | "
            f"ho_count_mean={best['ho_count_mean']:.4f}, "
            f"overlap_zone_sinr_mean_db_mean={best['overlap_zone_sinr_mean_db_mean']:.4f}, "
            f"outage_time_ratio_mean={best['outage_time_ratio_mean']:.4f}"
        )
        fixed_reference_result_path = os.path.join(metrics_dir, "fixed_a3_reference_result.json")
        save_fixed_a3_reference_result(fixed_reference_result_path, selection, results["num_scenarios"])
        print(f"固定同参A3参考结果已保存至: {fixed_reference_result_path}")
        fixed_grid_result_path = os.path.join(metrics_dir, "fixed_a3_grid_ranking.json")
        save_fixed_a3_grid_result(fixed_grid_result_path, selection, results["num_scenarios"])
        print(f"完整固定A3网格排名已保存至: {fixed_grid_result_path}")
    
    # 打印摘要
    print_summary(results, selection.get("best") if selection else None)
    
    # 绘制对比图
    plot_comparison(results, figures_dir)
    
    print("\n" + "=" * 80)
    print("批量测试完成！")
    print("=" * 80)


if __name__ == "__main__":
    main()
