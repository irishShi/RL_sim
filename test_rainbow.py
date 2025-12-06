"""
Rainbow DQN 测试脚本

测试训练好的模型，并与传统固定参数的 A3 切换算法对比
"""
import os
import numpy as np
import torch
import yaml
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple

from envs.train_ho_env import TrainHandoverEnv
from models import RainbowWithForecast, ActionSpace, ObservationWindow

# 配置 matplotlib 支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class TraditionalA3Policy:
    """
    传统 A3 切换算法策略
    
    固定参数：
    - Hys: 迟滞值（dB）
    - TTT: 时间触发阈值（ms）
    """
    
    def __init__(self, hys_db: float, ttt_ms: float, action_space: ActionSpace):
        """
        Args:
            hys_db: 迟滞值（dB）
            ttt_ms: 时间触发阈值（ms）
            action_space: 动作空间（用于将 Hys/TTT 转换为动作索引）
        """
        self.hys_db = hys_db
        self.ttt_ms = ttt_ms
        self.action_space = action_space
        self.ttt_timer = 0.0  # 秒
        self.a3_condition_met = False
    
    def reset(self):
        """重置状态"""
        self.ttt_timer = 0.0
        self.a3_condition_met = False
    
    def decide(self, obs, info, dt: float) -> int:
        """
        决策函数
        
        Args:
            obs: 观测（未使用）
            info: 信息字典，包含 rsrp_serv_dbm 和 rsrp_neig_dbm
            dt: 时间步长（秒）
            
        Returns:
            action: 动作索引（0-47）
        """
        rsrp_serv = info.get('rsrp_serv_dbm', -90.0)
        rsrp_neig = info.get('rsrp_neig_dbm', -90.0)
        delta_rsrp = rsrp_neig - rsrp_serv
        
        # 检查 A3 事件
        if delta_rsrp > self.hys_db:
            # A3 条件满足
            if not self.a3_condition_met:
                self.a3_condition_met = True
                self.ttt_timer = dt
            else:
                self.ttt_timer += dt
        else:
            # A3 条件不满足，重置
            self.a3_condition_met = False
            self.ttt_timer = 0.0
        
        # 如果满足条件且达到 TTT，选择对应的动作
        ttt_seconds = self.ttt_ms / 1000.0
        if self.a3_condition_met and self.ttt_timer >= ttt_seconds:
            # 重置计时器
            self.ttt_timer = 0.0
            self.a3_condition_met = False
        
        # 将 Hys/TTT 转换为动作索引
        action = self.action_space.hys_ttt_to_action(self.hys_db, self.ttt_ms)
        return action


def run_episode(env, policy_func, policy_name: str, obs_window: ObservationWindow = None,
                model: RainbowWithForecast = None, device: str = 'cpu', seed: int = 42,
                verbose: bool = False) -> Dict:
    """
    运行一个 episode
    
    Args:
        env: 环境实例
        policy_func: 策略函数，输入 (obs, info, dt)，输出 action
        policy_name: 策略名称
        obs_window: 观测窗口（用于 Rainbow 模型）
        model: Rainbow 模型（如果使用）
        device: 设备
        seed: 随机种子
        verbose: 是否打印详细信息
        
    Returns:
        统计字典
    """
    obs_raw, info = env.reset(seed=seed)
    
    # 初始化观测窗口（如果使用）
    if obs_window is not None:
        obs_window.reset()
        obs_window.update_time(0.0)
        # 填充窗口
        for _ in range(obs_window.window_size):
            obs_extended = obs_window.build_observation(
                obs_raw, info,
                velocity_mps=env.velocity_mps,
                track_length_m=env.cfg['track_length_m']
            )
    
    done = False
    total_reward = 0.0
    step_count = 0
    ho_count = 0
    outage_count = 0
    sinr_list = []
    trajectory = []
    dt = env.cfg['delta_t_s']
    
    if verbose:
        print(f"\n运行策略: {policy_name}")
        print(f"初始位置: {env.position_m:.2f} m")
        print(f"初始服务小区: {'A' if env.serving_cell == 0 else 'B'}")
    
    while not done:
        # 选择动作
        if model is not None and obs_window is not None:
            # 使用 Rainbow 模型
            window = obs_window.get_window()
            window_tensor = torch.FloatTensor(window).unsqueeze(0).to(device)
            with torch.no_grad():
                action = model.act(window_tensor, epsilon=0.0)
        else:
            # 使用传统策略
            action = policy_func(obs_raw, info, dt)
        
        # 执行动作
        next_obs_raw, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        step_count += 1
        
        sinr = info.get("sinr_serv_db", 0.0)
        sinr_list.append(sinr)
        
        if info.get("ho_executed", False):
            ho_count += 1
        
        if info.get("outage", False):
            outage_count += 1
        
        # 更新观测窗口
        if obs_window is not None:
            current_time = env.time_step * dt
            obs_window.update_time(current_time)
            if info.get('ho_executed', False):
                obs_window.update_ho_time(current_time)
            obs_window.update_params(
                info.get('current_hys', 3.0),
                info.get('current_ttt', 160.0)
            )
            obs_window.build_observation(
                next_obs_raw, info,
                velocity_mps=env.velocity_mps,
                track_length_m=env.cfg['track_length_m']
            )
        
        # 记录轨迹
        trajectory.append({
            "step": step_count,
            "x": info.get("position_m", 0.0),
            "serving_cell": info.get("serving_cell", 0),
            "rsrp_A_dbm": info.get("rsrp_A_dbm", 0.0),
            "rsrp_B_dbm": info.get("rsrp_B_dbm", 0.0),
            "rsrp_serv_dbm": info.get("rsrp_serv_dbm", 0.0),
            "rsrp_neig_dbm": info.get("rsrp_neig_dbm", 0.0),
            "sinr_serv_db": sinr,
            "ho_executed": info.get("ho_executed", False),
            "current_hys": info.get("current_hys", 3.0),
            "current_ttt": info.get("current_ttt", 160.0),
        })
        
        obs_raw = next_obs_raw
        
        if step_count >= 10000:
            break
    
    avg_sinr = np.mean(sinr_list) if sinr_list else 0.0
    min_sinr = np.min(sinr_list) if sinr_list else 0.0
    max_sinr = np.max(sinr_list) if sinr_list else 0.0
    
    stats = {
        "policy_name": policy_name,
        "total_steps": step_count,
        "total_reward": total_reward,
        "ho_count": ho_count,
        "outage_count": outage_count,
        "min_sinr": min_sinr,
        "max_sinr": max_sinr,
        "avg_sinr": avg_sinr,
        "trajectory": trajectory,
    }
    
    if verbose:
        print(f"Episode 结束: 步数={step_count}, 奖励={total_reward:.2f}, "
              f"切换次数={ho_count}, 平均SINR={avg_sinr:.2f}dB")
    
    return stats


def main():
    """主测试函数"""
    print("=" * 80)
    print("Rainbow DQN 模型测试 - 与传统 A3 算法对比")
    print("=" * 80)
    
    # 1. 加载配置
    base_dir = os.path.dirname(__file__)
    env_config_path = os.path.join(base_dir, "configs", "default_env_config.yaml")
    model_config_path = os.path.join(base_dir, "configs", "model_config.yaml")
    
    with open(model_config_path, 'r', encoding='utf-8') as f:
        model_config = yaml.safe_load(f)
    
    # 2. 设置设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n使用设备: {device}")
    
    # 3. 创建环境和动作空间
    env = TrainHandoverEnv(config_path=env_config_path)
    action_space = ActionSpace()
    
    # 4. 加载训练好的模型
    checkpoint_path = os.path.join(base_dir, "checkpoints", "rainbow_final.pth")
    if not os.path.exists(checkpoint_path):
        # 尝试加载最新的 checkpoint
        checkpoints_dir = os.path.join(base_dir, "checkpoints")
        if os.path.exists(checkpoints_dir):
            checkpoints = [f for f in os.listdir(checkpoints_dir) if f.endswith('.pth')]
            if checkpoints:
                checkpoint_path = os.path.join(checkpoints_dir, sorted(checkpoints)[-1])
                print(f"未找到 rainbow_final.pth，使用最新的 checkpoint: {checkpoint_path}")
            else:
                print(f"错误：未找到训练好的模型！请先运行 train_rainbow.py")
                return
        else:
            print(f"错误：未找到训练好的模型！请先运行 train_rainbow.py")
            return
    
    print(f"\n加载模型: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # 创建模型
    hys_set = model_config['action_space']['hys_set']
    ttt_set = model_config['action_space']['ttt_set']
    num_actions = len(hys_set) * len(ttt_set)
    obs_dim = model_config['observation']['obs_dim']
    window_size = model_config['observation']['window_size']
    
    model = RainbowWithForecast(
        obs_dim=obs_dim,
        num_actions=num_actions,
        n_steps=window_size,
        feature_hidden=model_config['network']['shared']['hidden_dim'],
        encoder_hidden=model_config['network']['encoder']['hidden_dim'],
        num_atoms=model_config['network']['rainbow']['num_atoms'],
        v_min=model_config['network']['rainbow']['v_min'],
        v_max=model_config['network']['rainbow']['v_max']
    ).to(device)
    
    model.load_state_dict(checkpoint['online_net_state_dict'])
    model.eval()
    print(f"模型加载成功（训练到 Episode {checkpoint.get('episode', '?')}）")
    
    # 5. 创建观测窗口
    obs_window = ObservationWindow(
        window_size=window_size,
        obs_dim=obs_dim
    )
    
    # 6. 定义测试策略
    test_seed = 42
    num_test_episodes = 5  # 每个策略测试 5 个 episode
    
    results = []
    
    # 6.1 Rainbow DQN 模型
    print(f"\n{'='*80}")
    print("测试 Rainbow DQN 模型")
    print(f"{'='*80}")
    
    rainbow_rewards = []
    for episode in range(num_test_episodes):
        stats = run_episode(
            env, None, f"Rainbow DQN (episode {episode+1})",
            obs_window=obs_window, model=model, device=device,
            seed=test_seed + episode, verbose=False
        )
        rainbow_rewards.append(stats['total_reward'])
        results.append(stats)
        print(f"Episode {episode+1}: 奖励={stats['total_reward']:.2f}, "
              f"切换次数={stats['ho_count']}, 平均SINR={stats['avg_sinr']:.2f}dB")
    
    avg_rainbow_reward = np.mean(rainbow_rewards)
    print(f"\nRainbow DQN 平均奖励: {avg_rainbow_reward:.2f}")
    
    # 6.2 传统 A3 算法（多个参数组合）
    print(f"\n{'='*80}")
    print("测试传统 A3 切换算法（固定参数）")
    print(f"{'='*80}")
    
    traditional_configs = [
        (3.0, 160.0, "A3 (Hys=3.0dB, TTT=160ms)"),
        (3.0, 320.0, "A3 (Hys=3.0dB, TTT=320ms)"),
        (4.0, 160.0, "A3 (Hys=4.0dB, TTT=160ms)"),
        (4.0, 320.0, "A3 (Hys=4.0dB, TTT=320ms)"),
        (5.0, 160.0, "A3 (Hys=5.0dB, TTT=160ms)"),
    ]
    
    for hys, ttt, name in traditional_configs:
        policy = TraditionalA3Policy(hys, ttt, action_space)
        
        def policy_func(obs, info, dt):
            policy.reset()  # 每次 episode 开始时重置
            return policy.decide(obs, info, dt)
        
        # 创建新的观测窗口（不使用模型）
        test_obs_window = ObservationWindow(window_size=window_size, obs_dim=obs_dim)
        
        rewards = []
        for episode in range(num_test_episodes):
            policy.reset()  # 重置策略状态
            stats = run_episode(
                env, policy_func, f"{name} (episode {episode+1})",
                obs_window=test_obs_window, model=None, device=device,
                seed=test_seed + episode, verbose=False
            )
            rewards.append(stats['total_reward'])
            results.append(stats)
        
        avg_reward = np.mean(rewards)
        print(f"{name:30s}: 平均奖励={avg_reward:7.2f}, "
              f"切换次数={np.mean([r['ho_count'] for r in results[-num_test_episodes:]]):.1f}, "
              f"平均SINR={np.mean([r['avg_sinr'] for r in results[-num_test_episodes:]]):.2f}dB")
    
    # 7. 对比结果汇总
    print(f"\n{'='*80}")
    print("性能对比总结")
    print(f"{'='*80}")
    print(f"{'策略名称':<35} | {'平均奖励':<10} | {'切换次数':<10} | {'平均SINR':<10} | {'最小SINR':<10}")
    print("-" * 80)
    
    # 按策略名称分组
    policy_groups = {}
    for stats in results:
        name = stats['policy_name'].split(' (')[0]  # 去掉 episode 编号
        if name not in policy_groups:
            policy_groups[name] = []
        policy_groups[name].append(stats)
    
    for name, stats_list in policy_groups.items():
        avg_reward = np.mean([s['total_reward'] for s in stats_list])
        avg_ho = np.mean([s['ho_count'] for s in stats_list])
        avg_sinr = np.mean([s['avg_sinr'] for s in stats_list])
        min_sinr = np.min([s['min_sinr'] for s in stats_list])
        print(f"{name:<35} | {avg_reward:>10.2f} | {avg_ho:>10.1f} | {avg_sinr:>10.2f} | {min_sinr:>10.2f}")
    
    # 8. 可视化对比
    print(f"\n生成可视化对比图...")
    
    # 选择第一个 Rainbow 和第一个传统算法的轨迹
    rainbow_traj = None
    traditional_traj = None
    
    for stats in results:
        if "Rainbow DQN" in stats['policy_name'] and rainbow_traj is None:
            rainbow_traj = stats['trajectory']
        if "A3 (Hys=3.0dB, TTT=160ms)" in stats['policy_name'] and traditional_traj is None:
            traditional_traj = stats['trajectory']
    
    if rainbow_traj and traditional_traj:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 子图1: RSRP 曲线和切换点
        ax1 = axes[0, 0]
        xs = np.array([p['x'] for p in rainbow_traj])
        rsrp_A = np.array([p['rsrp_A_dbm'] for p in rainbow_traj])
        rsrp_B = np.array([p['rsrp_B_dbm'] for p in rainbow_traj])
        
        ax1.plot(xs, rsrp_A, label="RSRP A小区", color="tab:blue", alpha=0.7)
        ax1.plot(xs, rsrp_B, label="RSRP B小区", color="tab:orange", alpha=0.7)
        
        # Rainbow 切换点
        ho_x_rainbow = [p['x'] for p in rainbow_traj if p['ho_executed']]
        ho_y_rainbow = []
        for p in rainbow_traj:
            if p['ho_executed']:
                if p['serving_cell'] == 0:
                    ho_y_rainbow.append(p['rsrp_A_dbm'])
                else:
                    ho_y_rainbow.append(p['rsrp_B_dbm'])
        
        if ho_x_rainbow:
            ax1.scatter(ho_x_rainbow, ho_y_rainbow, marker='o', s=100, 
                       color='red', label='Rainbow 切换点', zorder=5)
        
        # 传统算法切换点
        ho_x_trad = [p['x'] for p in traditional_traj if p['ho_executed']]
        ho_y_trad = []
        for p in traditional_traj:
            if p['ho_executed']:
                if p['serving_cell'] == 0:
                    ho_y_trad.append(p['rsrp_A_dbm'])
                else:
                    ho_y_trad.append(p['rsrp_B_dbm'])
        
        if ho_x_trad:
            ax1.scatter(ho_x_trad, ho_y_trad, marker='s', s=100,
                       color='green', label='传统A3 切换点', zorder=5)
        
        ax1.set_xlabel("距离 x / m")
        ax1.set_ylabel("RSRP / dBm")
        ax1.set_title("RSRP 曲线与切换点对比")
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # 子图2: SINR 对比
        ax2 = axes[0, 1]
        sinr_rainbow = [p['sinr_serv_db'] for p in rainbow_traj]
        sinr_trad = [p['sinr_serv_db'] for p in traditional_traj]
        
        ax2.plot(xs[:len(sinr_rainbow)], sinr_rainbow, label="Rainbow DQN", 
                color="red", linewidth=2)
        ax2.plot(xs[:len(sinr_trad)], sinr_trad, label="传统 A3", 
                color="green", linewidth=2, linestyle='--')
        ax2.axhline(y=env.cfg['sinr_outage_db'], color='black', linestyle=':', 
                   label=f"Outage阈值 ({env.cfg['sinr_outage_db']}dB)")
        ax2.set_xlabel("距离 x / m")
        ax2.set_ylabel("SINR / dB")
        ax2.set_title("SINR 对比")
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # 子图3: 参数选择对比（Rainbow）
        ax3 = axes[1, 0]
        hys_rainbow = [p['current_hys'] for p in rainbow_traj]
        ttt_rainbow = [p['current_ttt'] for p in rainbow_traj]
        
        ax3_twin = ax3.twinx()
        line1 = ax3.plot(xs[:len(hys_rainbow)], hys_rainbow, label="Hys (Rainbow)", 
                        color="blue", linewidth=2)
        line2 = ax3_twin.plot(xs[:len(ttt_rainbow)], ttt_rainbow, label="TTT (Rainbow)", 
                             color="orange", linewidth=2)
        
        ax3.set_xlabel("距离 x / m")
        ax3.set_ylabel("Hys / dB", color="blue")
        ax3_twin.set_ylabel("TTT / ms", color="orange")
        ax3.set_title("Rainbow DQN 参数选择")
        ax3.tick_params(axis='y', labelcolor="blue")
        ax3_twin.tick_params(axis='y', labelcolor="orange")
        ax3.grid(True, alpha=0.3)
        
        # 合并图例
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax3.legend(lines, labels, loc='upper left')
        
        # 子图4: 性能对比柱状图
        ax4 = axes[1, 1]
        policy_names = []
        avg_rewards = []
        avg_hos = []
        
        for name, stats_list in list(policy_groups.items())[:6]:  # 只显示前6个
            policy_names.append(name.split('(')[0].strip())
            avg_rewards.append(np.mean([s['total_reward'] for s in stats_list]))
            avg_hos.append(np.mean([s['ho_count'] for s in stats_list]))
        
        x = np.arange(len(policy_names))
        width = 0.35
        
        bars1 = ax4.bar(x - width/2, avg_rewards, width, label='平均奖励', alpha=0.8)
        ax4_twin = ax4.twinx()
        bars2 = ax4_twin.bar(x + width/2, avg_hos, width, label='切换次数', 
                            color='orange', alpha=0.8)
        
        ax4.set_xlabel("策略")
        ax4.set_ylabel("平均奖励", color="blue")
        ax4_twin.set_ylabel("切换次数", color="orange")
        ax4.set_title("性能对比")
        ax4.set_xticks(x)
        ax4.set_xticklabels(policy_names, rotation=45, ha='right')
        ax4.tick_params(axis='y', labelcolor="blue")
        ax4_twin.tick_params(axis='y', labelcolor="orange")
        ax4.grid(True, alpha=0.3, axis='y')
        
        # 合并图例
        lines = [bars1, bars2]
        labels = ['平均奖励', '切换次数']
        ax4.legend(lines, labels, loc='upper left')
        
        plt.tight_layout()
        
        # 保存图片
        save_path = os.path.join(base_dir, "test_results_comparison.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"对比图已保存: {save_path}")
        
        plt.show()
    
    print(f"\n{'='*80}")
    print("测试完成！")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
