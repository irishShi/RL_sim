"""
Rainbow DQN 离线训练模型测试脚本

专门用于测试离线训练得到的模型，支持：
- 自动检测离线训练模型（use_noisy=False）
- 显示离线训练相关信息（CQL、epoch等）
- 与传统算法和在线训练模型对比
"""
import os
import numpy as np
import torch
import yaml
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import argparse
import glob

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


def find_offline_checkpoint(checkpoints_dir: str, checkpoint_name: Optional[str] = None) -> Optional[str]:
    """
    查找离线训练的模型文件
    
    Args:
        checkpoints_dir: 检查点目录
        checkpoint_name: 指定的检查点名称（可选）
        
    Returns:
        检查点路径，如果未找到返回 None
    """
    if checkpoint_name:
        # 如果指定了名称，直接查找
        checkpoint_path = os.path.join(checkpoints_dir, checkpoint_name)
        if os.path.exists(checkpoint_path):
            return checkpoint_path
    
    # 优先查找离线训练的最终模型
    offline_final = os.path.join(checkpoints_dir, "rainbow_offline_final.pth")
    if os.path.exists(offline_final):
        return offline_final
    
    # 查找离线训练的 epoch 模型
    offline_epochs = glob.glob(os.path.join(checkpoints_dir, "rainbow_offline_epoch_*.pth"))
    if offline_epochs:
        # 返回最新的 epoch
        return sorted(offline_epochs, key=lambda x: int(x.split('_')[-1].split('.')[0]))[-1]
    
    return None


def load_offline_model(checkpoint_path: str, model_config: Dict, device: str) -> Tuple[RainbowWithForecast, Dict]:
    """
    加载离线训练的模型
    
    Args:
        checkpoint_path: 检查点路径
        model_config: 模型配置
        device: 设备
        
    Returns:
        (模型, 检查点信息)
    """
    print(f"\n加载离线训练模型: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # 创建模型（离线训练使用 use_noisy=False）
    hys_set = model_config['action_space']['hys_set']
    ttt_set = model_config['action_space']['ttt_set']
    num_actions = len(hys_set) * len(ttt_set)
    obs_dim = model_config['observation']['obs_dim']
    window_size = model_config['observation']['window_size']
    
    # 检查检查点中是否有 use_noisy 信息，如果没有则默认为 False（离线训练）
    use_noisy = checkpoint.get('config', {}).get('use_noisy', False)
    # 如果检查点中有明确的离线训练标记，强制使用 False
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
    
    # 显示模型信息
    print(f"模型加载成功！")
    print(f"  训练轮数: {checkpoint.get('epoch', '?')}")
    print(f"  训练步数: {checkpoint.get('step', '?')}")
    print(f"  使用 NoisyNet: {use_noisy}")
    if 'use_cql' in checkpoint:
        print(f"  使用 CQL: {checkpoint.get('use_cql', False)}")
        if checkpoint.get('use_cql', False):
            print(f"  CQL Alpha: {checkpoint.get('cql_alpha', '?')}")
    
    return model, checkpoint


def main():
    """主测试函数"""
    parser = argparse.ArgumentParser(description='Rainbow DQN 离线训练模型测试')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='检查点文件路径（默认：自动查找 rainbow_offline_final.pth）')
    parser.add_argument('--num_episodes', type=int, default=10,
                       help='每个策略测试的 episode 数量（默认：10）')
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子（默认：42）')
    parser.add_argument('--compare_online', action='store_true',
                       help='是否与在线训练模型对比')
    parser.add_argument('--compare_traditional', action='store_true', default=True,
                       help='是否与传统 A3 算法对比（默认：True）')
    parser.add_argument('--save_plots', action='store_true',
                       help='是否保存对比图表')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("Rainbow DQN 离线训练模型测试")
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
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # 3. 创建环境和动作空间
    env = TrainHandoverEnv(config_path=env_config_path)
    action_space = ActionSpace()
    
    # 4. 查找并加载离线训练模型
    checkpoints_dir = os.path.join(base_dir, "checkpoints")
    checkpoint_path = args.checkpoint
    
    if checkpoint_path is None:
        checkpoint_path = find_offline_checkpoint(checkpoints_dir)
    
    if checkpoint_path is None or not os.path.exists(checkpoint_path):
        print(f"\n错误：未找到离线训练模型！")
        print(f"请确保已运行 train_rainbow_offline.py 并生成了模型文件")
        print(f"查找路径: {checkpoints_dir}")
        return
    
    offline_model, checkpoint_info = load_offline_model(checkpoint_path, model_config, device)
    
    # 5. 创建观测窗口
    obs_dim = model_config['observation']['obs_dim']
    window_size = model_config['observation']['window_size']
    obs_window = ObservationWindow(
        window_size=window_size,
        obs_dim=obs_dim
    )
    
    # 6. 测试离线训练模型
    print(f"\n{'='*80}")
    print("测试离线训练的 Rainbow DQN 模型")
    print(f"{'='*80}")
    
    results = []
    offline_rewards = []
    
    for episode in range(args.num_episodes):
        stats = run_episode(
            env, None, f"离线训练 Rainbow DQN (episode {episode+1})",
            obs_window=obs_window, model=offline_model, device=device,
            seed=args.seed + episode, verbose=False
        )
        offline_rewards.append(stats['total_reward'])
        results.append(stats)
        print(f"Episode {episode+1:2d}: 奖励={stats['total_reward']:7.2f}, "
              f"切换次数={stats['ho_count']:3d}, "
              f"平均SINR={stats['avg_sinr']:6.2f}dB, "
              f"最小SINR={stats['min_sinr']:6.2f}dB")
    
    avg_offline_reward = np.mean(offline_rewards)
    std_offline_reward = np.std(offline_rewards)
    avg_offline_ho = np.mean([r['ho_count'] for r in results])
    avg_offline_sinr = np.mean([r['avg_sinr'] for r in results])
    
    print(f"\n离线训练模型统计:")
    print(f"  平均奖励: {avg_offline_reward:.2f} ± {std_offline_reward:.2f}")
    print(f"  平均切换次数: {avg_offline_ho:.1f}")
    print(f"  平均SINR: {avg_offline_sinr:.2f}dB")
    
    # 7. 对比测试（可选）
    if args.compare_traditional:
        print(f"\n{'='*80}")
        print("与传统 A3 算法对比")
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
                policy.reset()
                return policy.decide(obs, info, dt)
            
            test_obs_window = ObservationWindow(window_size=window_size, obs_dim=obs_dim)
            
            rewards = []
            for episode in range(args.num_episodes):
                policy.reset()
                stats = run_episode(
                    env, policy_func, f"{name} (episode {episode+1})",
                    obs_window=test_obs_window, model=None, device=device,
                    seed=args.seed + episode, verbose=False
                )
                rewards.append(stats['total_reward'])
                results.append(stats)
            
            avg_reward = np.mean(rewards)
            avg_ho = np.mean([r['ho_count'] for r in results[-args.num_episodes:]])
            avg_sinr = np.mean([r['avg_sinr'] for r in results[-args.num_episodes:]])
            
            print(f"{name:30s}: 平均奖励={avg_reward:7.2f}, "
                  f"切换次数={avg_ho:5.1f}, "
                  f"平均SINR={avg_sinr:6.2f}dB")
    
    # 8. 与在线训练模型对比（可选）
    if args.compare_online:
        print(f"\n{'='*80}")
        print("与在线训练模型对比")
        print(f"{'='*80}")
        
        online_checkpoint = os.path.join(checkpoints_dir, "rainbow_final.pth")
        if os.path.exists(online_checkpoint):
            try:
                online_model, _ = load_offline_model(online_checkpoint, model_config, device)
                online_rewards = []
                
                for episode in range(args.num_episodes):
                    stats = run_episode(
                        env, None, f"在线训练 Rainbow DQN (episode {episode+1})",
                        obs_window=obs_window, model=online_model, device=device,
                        seed=args.seed + episode, verbose=False
                    )
                    online_rewards.append(stats['total_reward'])
                    results.append(stats)
                
                avg_online_reward = np.mean(online_rewards)
                print(f"在线训练模型平均奖励: {avg_online_reward:.2f}")
                print(f"离线训练模型平均奖励: {avg_offline_reward:.2f}")
                print(f"性能差异: {avg_offline_reward - avg_online_reward:.2f}")
            except Exception as e:
                print(f"加载在线训练模型失败: {e}")
        else:
            print("未找到在线训练模型，跳过对比")
    
    # 9. 性能对比总结
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
    
    # 10. 可视化（如果请求）
    if args.save_plots or len(results) > 0:
        print(f"\n生成可视化对比图...")
        
        # 选择第一个离线训练和第一个传统算法的轨迹
        offline_traj = None
        traditional_traj = None
        
        for stats in results:
            if "离线训练 Rainbow DQN" in stats['policy_name'] and offline_traj is None:
                offline_traj = stats['trajectory']
            if "A3 (Hys=3.0dB, TTT=160ms)" in stats['policy_name'] and traditional_traj is None:
                traditional_traj = stats['trajectory']
        
        if offline_traj:
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            
            # 子图1: RSRP 曲线和切换点
            ax1 = axes[0, 0]
            xs = np.array([p['x'] for p in offline_traj])
            rsrp_A = np.array([p['rsrp_A_dbm'] for p in offline_traj])
            rsrp_B = np.array([p['rsrp_B_dbm'] for p in offline_traj])
            
            ax1.plot(xs, rsrp_A, label="RSRP A小区", color="tab:blue", alpha=0.7)
            ax1.plot(xs, rsrp_B, label="RSRP B小区", color="tab:orange", alpha=0.7)
            
            # 离线训练切换点
            ho_x_offline = [p['x'] for p in offline_traj if p['ho_executed']]
            ho_y_offline = []
            for p in offline_traj:
                if p['ho_executed']:
                    if p['serving_cell'] == 0:
                        ho_y_offline.append(p['rsrp_A_dbm'])
                    else:
                        ho_y_offline.append(p['rsrp_B_dbm'])
            
            if ho_x_offline:
                ax1.scatter(ho_x_offline, ho_y_offline, marker='o', s=100, 
                           color='red', label='离线训练 切换点', zorder=5)
            
            if traditional_traj:
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
            sinr_offline = [p['sinr_serv_db'] for p in offline_traj]
            ax2.plot(xs[:len(sinr_offline)], sinr_offline, label="离线训练 Rainbow DQN", 
                    color="red", linewidth=2)
            
            if traditional_traj:
                sinr_trad = [p['sinr_serv_db'] for p in traditional_traj]
                ax2.plot(xs[:len(sinr_trad)], sinr_trad, label="传统 A3", 
                        color="green", linewidth=2, linestyle='--')
            
            ax2.axhline(y=env.cfg['sinr_outage_db'], color='black', linestyle=':', 
                       label=f"Outage阈值 ({env.cfg['sinr_outage_db']}dB)")
            ax2.set_xlabel("距离 x / m")
            ax2.set_ylabel("SINR / dB")
            ax2.set_title("SINR 对比")
            ax2.grid(True, alpha=0.3)
            ax2.legend()
            
            # 子图3: 参数选择（离线训练）
            ax3 = axes[1, 0]
            hys_offline = [p['current_hys'] for p in offline_traj]
            ttt_offline = [p['current_ttt'] for p in offline_traj]
            
            ax3_twin = ax3.twinx()
            line1 = ax3.plot(xs[:len(hys_offline)], hys_offline, label="Hys (离线训练)", 
                            color="blue", linewidth=2)
            line2 = ax3_twin.plot(xs[:len(ttt_offline)], ttt_offline, label="TTT (离线训练)", 
                                 color="orange", linewidth=2)
            
            ax3.set_xlabel("距离 x / m")
            ax3.set_ylabel("Hys / dB", color="blue")
            ax3_twin.set_ylabel("TTT / ms", color="orange")
            ax3.set_title("离线训练模型参数选择")
            ax3.tick_params(axis='y', labelcolor="blue")
            ax3_twin.tick_params(axis='y', labelcolor="orange")
            ax3.grid(True, alpha=0.3)
            
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
            
            lines = [bars1, bars2]
            labels = ['平均奖励', '切换次数']
            ax4.legend(lines, labels, loc='upper left')
            
            plt.tight_layout()
            
            # 保存图片
            if args.save_plots:
                save_path = os.path.join(base_dir, "test_results_offline_comparison.png")
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"对比图已保存: {save_path}")
            
            plt.show()
    
    print(f"\n{'='*80}")
    print("测试完成！")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()

