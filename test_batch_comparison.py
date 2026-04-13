"""
批量对比测试：多个场景下对比不同A3策略和RL策略的性能

功能：
1. 生成100个不同的场景
2. 定义多个A3策略的(Hys, TTT)组合
3. 对每个场景运行所有策略（包括RL策略）
4. 收集性能指标并生成对比图表
"""
import os
import json
from collections import Counter
import numpy as np
import torch
import yaml
import matplotlib.pyplot as plt
from tqdm import tqdm
from envs.train_ho_env import TrainHandoverEnv
from models import RainbowWithForecast, ActionSpace, ObservationWindow
from utils.scenario_generator import ScenarioGenerator
from test_simple import TraditionalA3Policy, run_episode

# 配置 matplotlib 支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


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
        'overlap_zone_half_width_m': 0.0,
        'overlap_zone_full_width_m': 0.0,
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
            'overlap_zone_half_width_m': kpis.get('overlap_zone_half_width_m', result['overlap_zone_half_width_m']),
            'overlap_zone_full_width_m': kpis.get('overlap_zone_full_width_m', result['overlap_zone_full_width_m']),
            'comm_interruption_ratio_in_overlap_zone': kpis.get(
                'comm_interruption_ratio_in_overlap_zone',
                result['comm_interruption_ratio_in_overlap_zone'],
            ),
        })
    
    return result


def run_batch_test(num_scenarios: int = 100, 
                   a3_configs: list = None,
                   base_seed: int = 10000,
                   device: str = 'cpu',
                   top_k_a3: int = 5,
                   oracle_use_full_grid: bool = True,
                   oracle_ho_penalty_weight: float = 0.5):
    """
    运行批量测试
    
    Args:
        num_scenarios: 场景数量
        a3_configs: A3策略配置列表，每个元素为 (hys, ttt, name)
        base_seed: 基础随机种子
        device: 计算设备
        
    Returns:
        测试结果字典
    """
    # 默认A3策略配置（作为候选空间输入）
    if a3_configs is None:
        a3_configs = [
            (2.0, 80.0, "A3 (Hys=2.0dB, TTT=80ms)"),
            (2.5, 160.0, "A3 (Hys=2.5dB, TTT=160ms)"),
            (3.0, 160.0, "A3 (Hys=3.0dB, TTT=160ms)"),
            (3.5, 160.0, "A3 (Hys=3.5dB, TTT=160ms)"),
            (4.0, 160.0, "A3 (Hys=4.0dB, TTT=160ms)"),
            (3.0, 80.0, "A3 (Hys=3.0dB, TTT=80ms)"),
            (3.0, 320.0, "A3 (Hys=3.0dB, TTT=320ms)"),
        ]
    oracle_candidate_configs = list(a3_configs)
    
    print("=" * 80)
    print(
        f"批量对比测试：{num_scenarios}个场景，"
        f"A3候选{len(a3_configs)}组，最终显式对比RL分布Top-{top_k_a3}的A3策略 + 1个RL策略"
    )
    print("=" * 80)
    
    # 1. 加载配置
    base_dir = os.path.dirname(__file__)
    env_config_path = os.path.join(base_dir, "configs", "default_env_config.yaml")
    model_config_path = os.path.join(base_dir, "configs", "model_config.yaml")
    
    with open(model_config_path, 'r', encoding='utf-8') as f:
        model_config = yaml.safe_load(f)
    
    # 2. 加载RL模型
    checkpoints_dir = os.path.join(base_dir, "checkpoints")
    checkpoint_path = os.path.join(checkpoints_dir, "rainbow_offline_final.pth")
    
    if not os.path.exists(checkpoint_path):
        print(f"\n错误：未找到离线训练模型！")
        print(f"请确保已运行 train_rainbow_offline.py 并生成了模型文件")
        return None
    
    print(f"\n加载离线训练模型: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
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
    
    model.load_state_dict(checkpoint['online_net_state_dict'])
    model.eval()
    print("模型加载成功！")
    
    # 3. 创建环境和动作空间
    env_template = TrainHandoverEnv(config_path=env_config_path)
    action_space = ActionSpace()
    
    # 4. 创建观测窗口（用于RL）
    obs_window = ObservationWindow(window_size=window_size, obs_dim=obs_dim)
    
    # 5. 生成场景数据
    print(f"\n生成{num_scenarios}个场景数据...")
    scenario_gen = ScenarioGenerator(env_template.cfg)
    scenario_dir = os.path.join(base_dir, "scenarios")
    os.makedirs(scenario_dir, exist_ok=True)
    
    scenarios = []
    for i in tqdm(range(num_scenarios), desc="生成场景"):
        seed = base_seed + i
        scenario_data = scenario_gen.generate_scenario(seed=seed, position_resolution_m=1.0)
        scenarios.append(scenario_data)
    
    # 6. 第一阶段：仅运行RL，统计动作分布并收集RL KPI
    all_results = {"RL策略 (Rainbow DQN)": []}
    rl_action_counter = Counter()
    rl_total_steps = 0

    print(f"\n第一阶段：运行RL策略并统计动作分布（{num_scenarios}个场景）...")
    for scenario_data in tqdm(scenarios, desc="RL测试进度"):
        env_rl = TrainHandoverEnv(config_path=env_config_path)
        obs_window.reset()
        traj_rl = run_episode(
            env_rl, None, "RL策略 (Rainbow DQN)",
            obs_window=obs_window, model=model, device=device,
            seed=None, scenario_data=scenario_data
        )

        # 获取环境返回的KPI（如果episode结束）
        kpis_from_env_rl = None
        if traj_rl['trajectory']:
            last_info_rl = traj_rl.get('last_info', {})
            if 'kpis' in last_info_rl:
                kpis_from_env_rl = last_info_rl['kpis']

        # 收集RL KPI
        kpis_rl = collect_kpis(traj_rl['trajectory'], kpis_from_env_rl)
        all_results["RL策略 (Rainbow DQN)"].append(kpis_rl)

        # 统计RL动作分布（映射回离散动作集合）
        for p in traj_rl['trajectory']:
            h = float(p.get("current_hys", 3.0))
            t = float(p.get("current_ttt", 160.0))
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
        f"Oracle候选集: {'全48组' if oracle_use_full_grid else 'Top-K子集'} "
        f"| ho惩罚权重={oracle_ho_penalty_weight}"
    )

    # 初始化A3结果存储
    for _, _, name in a3_configs:
        all_results[name] = []
    all_results["Oracle A3 (per-scenario best)"] = []
    oracle_selection_records = []

    # 8. 第二阶段：仅运行Top-K A3策略
    print(f"\n第二阶段：运行测试（{num_scenarios}个场景 × {len(a3_configs)}个A3策略）...")
    for scenario_data in tqdm(scenarios, desc="A3测试进度"):
        envs = {}
        policies = {}

        for hys, ttt, name in a3_configs:
            env = TrainHandoverEnv(config_path=env_config_path)
            policy = TraditionalA3Policy(hys, ttt, action_space)
            policy.reset()

            def make_policy_func(p):
                def policy_func(obs, info, dt):
                    return p.decide(obs, info, dt)
                return policy_func

            envs[name] = env
            policies[name] = (make_policy_func(policy), policy)

        per_scenario_a3_kpis = {}
        per_scenario_oracle_kpis = {}
        for hys, ttt, name in a3_configs:
            env = envs[name]
            policy_func, policy = policies[name]
            traj = run_episode(
                env, policy_func, name,
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
            per_scenario_a3_kpis[name] = kpis
            per_scenario_oracle_kpis[name] = kpis

        # 若Oracle使用全网格，则补跑Top-K之外的A3组合（仅用于Oracle选优，不加入显式对比表）
        if oracle_use_full_grid:
            selected_names = {name for _, _, name in a3_configs}
            for hys, ttt, name in oracle_candidate_configs:
                if name in selected_names:
                    continue
                env_oracle = TrainHandoverEnv(config_path=env_config_path)
                policy_oracle = TraditionalA3Policy(hys, ttt, action_space)
                policy_oracle.reset()

                def oracle_policy_func(obs, info, dt, p=policy_oracle):
                    return p.decide(obs, info, dt)

                traj_oracle = run_episode(
                    env_oracle, oracle_policy_func, name,
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

        # 场景级Oracle：每个场景单独按词典序选择最优A3参数
        oracle_cfgs_for_selection = oracle_candidate_configs if oracle_use_full_grid else a3_configs
        if per_scenario_oracle_kpis:
            scenario_best = select_best_a3_for_single_scenario(
                per_scenario_oracle_kpis,
                oracle_cfgs_for_selection,
                ho_penalty_weight=float(oracle_ho_penalty_weight),
            )
            if scenario_best is not None:
                best_name, best_kpis, best_hys, best_ttt = scenario_best
                all_results["Oracle A3 (per-scenario best)"].append(best_kpis)
                oracle_selection_records.append({
                    "policy_name": best_name,
                    "hys_db": float(best_hys),
                    "ttt_ms": float(best_ttt),
                })
    
    # 8. 计算统计结果
    print("\n计算统计结果...")
    stats = {}
    for policy_name, results in all_results.items():
        if not results:
            continue
        
        # 计算平均值和标准差
        stats[policy_name] = {}
        for kpi_name in results[0].keys():
            values = [r[kpi_name] for r in results]
            stats[policy_name][f"{kpi_name}_mean"] = np.mean(values)
            stats[policy_name][f"{kpi_name}_std"] = np.std(values)
            stats[policy_name][f"{kpi_name}_min"] = np.min(values)
            stats[policy_name][f"{kpi_name}_max"] = np.max(values)
    
    return {
        'all_results': all_results,
        'stats': stats,
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
        'oracle_use_full_grid': bool(oracle_use_full_grid),
        'oracle_ho_penalty_weight': float(oracle_ho_penalty_weight),
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
            "overlap_zone_sinr_mean_db_mean": float(s.get("overlap_zone_sinr_mean_db_mean", 0.0)),
            "outage_time_ratio_mean": float(s.get("outage_time_ratio_mean", 0.0)),
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
    在单个场景内选择最优A3（含切换次数惩罚）：
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


def save_best_a3_result(output_path: str, selection: dict, num_scenarios: int):
    """保存最优 A3 参数结果到 JSON 文件。"""
    if not selection:
        return
    
    payload = {
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


def save_oracle_a3_result(output_path: str, results: dict):
    """保存场景级Oracle A3选择分布与汇总KPI。"""
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
    oracle_stats_raw = results.get("stats", {}).get("Oracle A3 (per-scenario best)", {})
    oracle_stats = {}
    for k, v in oracle_stats_raw.items():
        if isinstance(v, (np.floating, float)):
            oracle_stats[k] = float(v)
        elif isinstance(v, (np.integer, int)):
            oracle_stats[k] = int(v)
        else:
            oracle_stats[k] = v
    payload = {
        "selection_rule": [
            "per-scenario maximize score",
            "score = overlap_zone_sinr_mean_db - ho_penalty_weight * ho_count - 10 * outage_time_ratio",
            "tie-break: lower ho_count, then lower outage_time_ratio",
        ],
        "num_scenarios": int(results.get("num_scenarios", total)),
        "oracle_use_full_grid": bool(results.get("oracle_use_full_grid", True)),
        "oracle_ho_penalty_weight": float(results.get("oracle_ho_penalty_weight", 0.5)),
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
    
    # 选择代表性的KPI指标
    kpi_names = [
        ('ho_count', '切换次数'),
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
        
        # 设置颜色（RL策略用不同颜色）
        for i, (bar, policy_name) in enumerate(zip(bars, policy_names)):
            if "RL" in policy_name:
                bar.set_color('red')
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


def print_summary(results: dict, best_a3: dict = None):
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
    
    # 打印表格（含切换重叠区内通信中断率，与 env kpis / 图表一致）
    hdr = (
        f"\n{'策略':<30} {'切换次数':<10} {'SINR均值':<10} "
        f"{'Outage占比':<12} {'乒乓率':<8} {'重叠区SINR':<12} {'区内中断率':<12}"
    )
    print(hdr)
    print("-" * 110)
    
    for policy_name in sorted(stats.keys()):
        s = stats[policy_name]
        ho_mean = s.get('ho_count_mean', 0.0)
        sinr_mean = s.get('sinr_mean_db_mean', 0.0)
        outage_ratio = s.get('outage_time_ratio_mean', 0.0)
        ping_pong = s.get('ping_pong_ratio_mean', 0.0)
        overlap_sinr = s.get('overlap_zone_sinr_mean_db_mean', 0.0)
        int_in_oz = s.get('comm_interruption_ratio_in_overlap_zone_mean', 0.0)
        
        print(
            f"{policy_name:<30} {ho_mean:<10.2f} {sinr_mean:<10.2f} "
            f"{outage_ratio:<12.4f} {ping_pong:<8.4f} {overlap_sinr:<12.3f} {int_in_oz:<12.4f}"
        )
    
    # 额外输出词典序选中的 Best A3，便于和全表对照
    if best_a3:
        print("-" * 110)
        print(
            f"{'Best A3 (global fixed)':<30} "
            f"{best_a3.get('ho_count_mean', 0.0):<10.2f} "
            f"{best_a3.get('sinr_mean_db_mean', 0.0):<10.2f} "
            f"{best_a3.get('outage_time_ratio_mean', 0.0):<12.4f} "
            f"{best_a3.get('ping_pong_ratio_mean', 0.0):<8.4f} "
            f"{best_a3.get('overlap_zone_sinr_mean_db_mean', 0.0):<12.3f} "
            f"{best_a3.get('comm_interruption_ratio_in_overlap_zone_mean', 0.0):<12.4f}"
        )
        print(
            f"  -> 参数: Hys={best_a3.get('hys_db', 0.0):.1f} dB, "
            f"TTT={best_a3.get('ttt_ms', 0.0):.0f} ms, "
            f"对应策略: {best_a3.get('policy_name', 'N/A')}"
        )
    
    print(f"\n测试场景数: {num_scenarios}")


def main():
    """主函数"""
    base_dir = os.path.dirname(__file__)
    env_config_path = os.path.join(base_dir, "configs", "default_env_config.yaml")
    with open(env_config_path, 'r', encoding='utf-8') as f:
        env_cfg = yaml.safe_load(f)
    action_space_cfg = env_cfg.get("action_space", {})
    hys_set = action_space_cfg.get("hys_set", [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0])
    ttt_set = action_space_cfg.get("ttt_set", [0, 40, 80, 160, 320, 640])
    # 定义A3参数全搜索配置（默认48组）
    a3_configs = build_a3_grid_configs(hys_set, ttt_set)
    
    # 设置设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n使用设备: {device}")
    
    # 运行批量测试
    results = run_batch_test(
        num_scenarios=100,
        a3_configs=a3_configs,
        base_seed=10000,
        device=device
    )
    
    if results is None:
        return
    
    # 词典序最优 A3 参数选择（全局固定参数，仅作参考）
    selection = select_best_a3_lexicographic(results["stats"], results["a3_configs"])
    if selection:
        best = selection["best"]
        print("\n" + "=" * 80)
        print("Best A3 (global fixed, lexicographic)")
        print("=" * 80)
        print(
            f"Hys={best['hys_db']:.1f} dB, TTT={best['ttt_ms']:.0f} ms | "
            f"ho_count_mean={best['ho_count_mean']:.4f}, "
            f"overlap_zone_sinr_mean_db_mean={best['overlap_zone_sinr_mean_db_mean']:.4f}, "
            f"outage_time_ratio_mean={best['outage_time_ratio_mean']:.4f}"
        )
        best_result_path = os.path.join(base_dir, "best_a3_result.json")
        save_best_a3_result(best_result_path, selection, results["num_scenarios"])
        print(f"全局固定最优A3结果已保存至: {best_result_path}")

    # 场景级Oracle结果输出
    oracle_stats = results.get("stats", {}).get("Oracle A3 (per-scenario best)", {})
    if oracle_stats:
        print("\n" + "=" * 80)
        print("Oracle A3 (per-scenario best)")
        print("=" * 80)
        print(
            "ho_count_mean={:.4f}, overlap_zone_sinr_mean_db_mean={:.4f}, outage_time_ratio_mean={:.4f}".format(
                oracle_stats.get("ho_count_mean", 0.0),
                oracle_stats.get("overlap_zone_sinr_mean_db_mean", 0.0),
                oracle_stats.get("outage_time_ratio_mean", 0.0),
            )
        )
        oracle_result_path = os.path.join(base_dir, "oracle_a3_result.json")
        save_oracle_a3_result(oracle_result_path, results)
        print(f"场景级Oracle A3结果已保存至: {oracle_result_path}")
    
    # 打印摘要
    print_summary(results, selection.get("best") if selection else None)
    
    # 绘制对比图
    plot_comparison(results, base_dir)
    
    print("\n" + "=" * 80)
    print("批量测试完成！")
    print("=" * 80)


if __name__ == "__main__":
    main()
