"""
切换算法对比测试脚本

对每个episode分别运行传统A3算法和强化学习算法，
如果A3算法切换次数为0，则绘制该episode的信号强度变化图和切换点。
"""
import os
import numpy as np
import torch
import yaml
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import argparse
import glob
import random

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
        统计字典，包含 trajectory（轨迹数据）
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
    trajectory = []  # 记录轨迹数据
    dt = env.cfg['delta_t_s']
    
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
        
        # 记录轨迹数据
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
    
    return stats


def plot_rsrp_comparison(trajectory_a3: List[Dict], trajectory_rl: List[Dict], 
                        episode_num: int, output_dir: str = "plots"):
    """
    绘制RSRP对比图和切换点
    
    Args:
        trajectory_a3: A3算法的轨迹数据
        trajectory_rl: RL算法的轨迹数据
        episode_num: episode编号
        output_dir: 输出目录
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # 提取数据
    xs_a3 = np.array([p['x'] for p in trajectory_a3])
    rsrp_A_a3 = np.array([p['rsrp_A_dbm'] for p in trajectory_a3])
    rsrp_B_a3 = np.array([p['rsrp_B_dbm'] for p in trajectory_a3])
    
    xs_rl = np.array([p['x'] for p in trajectory_rl])
    rsrp_A_rl = np.array([p['rsrp_A_dbm'] for p in trajectory_rl])
    rsrp_B_rl = np.array([p['rsrp_B_dbm'] for p in trajectory_rl])
    
    # 子图1: A3算法
    ax1 = axes[0]
    ax1.plot(xs_a3, rsrp_A_a3, label="RSRP A小区", color="tab:blue", linewidth=2, alpha=0.7)
    ax1.plot(xs_a3, rsrp_B_a3, label="RSRP B小区", color="tab:orange", linewidth=2, alpha=0.7)
    
    # 标注A3切换点
    ho_x_a3 = [p['x'] for p in trajectory_a3 if p['ho_executed']]
    ho_y_a3 = []
    for p in trajectory_a3:
        if p['ho_executed']:
            if p['serving_cell'] == 0:
                ho_y_a3.append(p['rsrp_A_dbm'])
            else:
                ho_y_a3.append(p['rsrp_B_dbm'])
    
    if ho_x_a3:
        ax1.scatter(ho_x_a3, ho_y_a3, marker='s', s=150, color='red', 
                   label='A3 切换点', zorder=5, edgecolors='black', linewidths=1.5)
    else:
        ax1.text(0.5, 0.95, 'A3算法：无切换', transform=ax1.transAxes,
                fontsize=12, ha='center', va='top', 
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
    
    ax1.set_xlabel("距离 x / m", fontsize=12)
    ax1.set_ylabel("RSRP / dBm", fontsize=12)
    ax1.set_title(f"Episode {episode_num} - 传统A3算法 (切换次数: 0)", fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='best', fontsize=10)
    
    # 子图2: RL算法
    ax2 = axes[1]
    ax2.plot(xs_rl, rsrp_A_rl, label="RSRP A小区", color="tab:blue", linewidth=2, alpha=0.7)
    ax2.plot(xs_rl, rsrp_B_rl, label="RSRP B小区", color="tab:orange", linewidth=2, alpha=0.7)
    
    # 标注RL切换点
    ho_x_rl = [p['x'] for p in trajectory_rl if p['ho_executed']]
    ho_y_rl = []
    for p in trajectory_rl:
        if p['ho_executed']:
            if p['serving_cell'] == 0:
                ho_y_rl.append(p['rsrp_A_dbm'])
            else:
                ho_y_rl.append(p['rsrp_B_dbm'])
    
    if ho_x_rl:
        ax2.scatter(ho_x_rl, ho_y_rl, marker='o', s=150, color='green', 
                   label='RL 切换点', zorder=5, edgecolors='black', linewidths=1.5)
    
    ax2.set_xlabel("距离 x / m", fontsize=12)
    ax2.set_ylabel("RSRP / dBm", fontsize=12)
    ax2.set_title(f"Episode {episode_num} - 强化学习算法 (切换次数: {len(ho_x_rl)})", 
                  fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='best', fontsize=10)
    
    plt.tight_layout()
    
    # 保存图片
    output_path = os.path.join(output_dir, f"episode_{episode_num}_rsrp_comparison.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  已保存图表: {output_path}")
    plt.close()


def load_offline_model(checkpoint_path: str, model_config: Dict, device: str) -> Tuple[RainbowWithForecast, Dict]:
    """
    加载离线训练模型
    
    Args:
        checkpoint_path: 模型检查点路径
        model_config: 模型配置
        device: 设备
        
    Returns:
        (模型, 检查点信息)
    """
    print(f"加载模型: {checkpoint_path}")
    
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
    
    # 返回检查点信息
    checkpoint_info = {
        'epoch': checkpoint.get('epoch', 'N/A'),
        'step': checkpoint.get('step', 'N/A'),
        'use_noisy': use_noisy,
        'use_cql': checkpoint.get('use_cql', False),
        'cql_alpha': checkpoint.get('cql_alpha', None)
    }
    
    return model, checkpoint_info


def find_offline_checkpoint(checkpoints_dir: str) -> Optional[str]:
    """查找离线训练模型检查点"""
    final_path = os.path.join(checkpoints_dir, "rainbow_offline_final.pth")
    if os.path.exists(final_path):
        return final_path
    
    # 查找最新的epoch检查点
    pattern = os.path.join(checkpoints_dir, "rainbow_offline_epoch_*.pth")
    checkpoints = glob.glob(pattern)
    if checkpoints:
        # 按修改时间排序，返回最新的
        checkpoints.sort(key=os.path.getmtime, reverse=True)
        return checkpoints[0]
    
    return None


def main():
    parser = argparse.ArgumentParser(description="切换算法对比测试")
    parser.add_argument("--num_episodes", type=int, default=10, help="测试的episode数量")
    parser.add_argument("--a3_hys", type=float, default=3.0, help="A3算法迟滞值（dB）")
    parser.add_argument("--a3_ttt", type=float, default=160.0, help="A3算法时间触发阈值（ms）")
    parser.add_argument("--checkpoint", type=str, default=None, help="模型检查点路径")
    parser.add_argument("--output_dir", type=str, default="plots", help="图表输出目录")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--env_config", type=str, default="configs/default_env_config.yaml", 
                       help="环境配置文件路径")
    parser.add_argument("--model_config", type=str, default="configs/model_config.yaml",
                       help="模型配置文件路径")
    
    args = parser.parse_args()
    
    # 1. 加载配置
    base_dir = os.path.dirname(os.path.abspath(__file__))
    env_config_path = os.path.join(base_dir, args.env_config)
    model_config_path = os.path.join(base_dir, args.model_config)
    
    with open(model_config_path, 'r', encoding='utf-8') as f:
        model_config = yaml.safe_load(f)
    
    # 2. 检测设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # 3. 创建环境和动作空间
    env = TrainHandoverEnv(config_path=env_config_path)
    action_space = ActionSpace()
    
    # 4. 加载RL模型
    checkpoints_dir = os.path.join(base_dir, "checkpoints")
    checkpoint_path = args.checkpoint
    
    if checkpoint_path is None:
        checkpoint_path = find_offline_checkpoint(checkpoints_dir)
    
    if checkpoint_path is None or not os.path.exists(checkpoint_path):
        print(f"\n错误：未找到离线训练模型！")
        print(f"请确保已运行 train_rainbow_offline.py 并生成了模型文件")
        print(f"查找路径: {checkpoints_dir}")
        return
    
    rl_model, checkpoint_info = load_offline_model(checkpoint_path, model_config, device)
    print(f"模型加载成功！")
    if checkpoint_info:
        print(f"  训练轮数: {checkpoint_info.get('epoch', 'N/A')}")
        print(f"  训练步数: {checkpoint_info.get('step', 'N/A')}")
    
    # 5. 创建观测窗口
    obs_dim = model_config['observation']['obs_dim']
    window_size = model_config['observation']['window_size']
    obs_window = ObservationWindow(
        window_size=window_size,
        obs_dim=obs_dim
    )
    
    # 6. 创建A3策略
    a3_policy = TraditionalA3Policy(args.a3_hys, args.a3_ttt, action_space)
    
    def a3_policy_func(obs, info, dt):
        a3_policy.reset()
        return a3_policy.decide(obs, info, dt)
    
    def rl_policy_func(obs, info, dt):
        # RL策略通过模型在run_episode中处理
        return 0  # 占位符，实际不使用
    
    # 7. 初始化随机数生成器（使用args.seed作为随机数生成器的种子，保证可复现性）
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # 8. 运行测试
    print(f"\n{'='*80}")
    print(f"开始测试 {args.num_episodes} 个episode")
    print(f"A3参数: Hys={args.a3_hys}dB, TTT={args.a3_ttt}ms")
    print(f"随机种子基础值: {args.seed}")
    print(f"{'='*80}\n")
    
    plots_generated = 0
    
    for episode in range(args.num_episodes):
        # 为每个episode生成随机seed
        seed = random.randint(0, 2**31 - 1)
        print(f"Episode {episode+1}/{args.num_episodes} (seed={seed})...")
        
        # 运行A3算法
        a3_policy.reset()
        a3_obs_window = ObservationWindow(window_size=window_size, obs_dim=obs_dim)
        stats_a3 = run_episode(
            env, a3_policy_func, f"A3 (episode {episode+1})",
            obs_window=a3_obs_window, model=None, device=device,
            seed=seed, verbose=False
        )
        
        # 运行RL算法
        rl_obs_window = ObservationWindow(window_size=window_size, obs_dim=obs_dim)
        stats_rl = run_episode(
            env, rl_policy_func, f"RL (episode {episode+1})",
            obs_window=rl_obs_window, model=rl_model, device=device,
            seed=seed, verbose=False
        )
        
        ho_count_a3 = stats_a3['ho_count']
        ho_count_rl = stats_rl['ho_count']
        
        print(f"  A3切换次数: {ho_count_a3}, RL切换次数: {ho_count_rl}")
        
        # 如果A3切换次数为0，生成图表
        if ho_count_a3 == 0:
            print(f"  A3切换次数为0，生成对比图表...")
            plot_rsrp_comparison(
                stats_a3['trajectory'],
                stats_rl['trajectory'],
                episode + 1,
                args.output_dir
            )
            plots_generated += 1
    
    # 9. 总结
    print(f"\n{'='*80}")
    print("测试完成")
    print(f"{'='*80}")
    print(f"总episode数: {args.num_episodes}")
    print(f"生成图表数: {plots_generated} (A3切换次数为0的episode)")
    print(f"图表保存目录: {args.output_dir}")


if __name__ == "__main__":
    main()
