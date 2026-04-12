"""
简单测试脚本：对比传统A3切换算法和离线训练的Rainbow DQN模型

功能：
1. 生成一个episode
2. 使用相同seed分别运行传统A3和离线Rainbow DQN
3. 绘制RSRP曲线和切换点对比图
"""
import os
import numpy as np
import torch
import yaml
import matplotlib.pyplot as plt
from envs.train_ho_env import TrainHandoverEnv
from models import RainbowWithForecast, ActionSpace, ObservationWindow
from utils.scenario_generator import ScenarioGenerator

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
                scenario_data: dict = None, a3_policy: TraditionalA3Policy = None) -> dict:
    """
    运行一个 episode
    
    Args:
        env: 环境实例
        policy_func: 策略函数，输入 (obs, info, dt)，输出 action（用于A3）
        policy_name: 策略名称
        obs_window: 观测窗口（用于 Rainbow 模型）
        model: Rainbow 模型（如果使用）
        device: 设备
        seed: 随机种子（如果scenario_data为None，则使用此seed生成场景）
        scenario_data: 预生成的场景数据（如果提供，将使用此场景数据）
        
    Returns:
        轨迹数据字典
    """
    # 如果提供了场景数据，使用场景数据；否则使用seed
    if scenario_data is not None:
        obs_raw, info = env.reset(seed=None, options={"scenario_data": scenario_data})
    else:
        obs_raw, info = env.reset(seed=seed)
    
    # 初始化观测窗口（如果使用）
    if obs_window is not None:
        obs_window.reset()
        obs_window.update_time(0.0)
        # 填充窗口（用第一个观测填充）
        for _ in range(obs_window.window_size):
            obs_extended = obs_window.build_observation(
                obs_raw, info,
                velocity_mps=env.velocity_mps,
                track_length_m=env.cfg['track_length_m']
            )
    
    done = False
    step_count = 0
    trajectory = []
    dt = env.cfg['delta_t_s']
    
    while not done:
        # 选择动作
        if model is not None and obs_window is not None:
            # 使用 Rainbow 模型：获取当前窗口
            window = obs_window.get_window()  # [N, F]
            window_tensor = torch.FloatTensor(window).unsqueeze(0).to(device)  # [1, N, F]
            with torch.no_grad():
                action = model.act(window_tensor, epsilon=0.0)
        else:
            # 使用传统策略
            action = policy_func(obs_raw, info, dt)
        
        # 执行动作
        next_obs_raw, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        step_count += 1
        
        # 更新观测窗口（在动作执行之后）
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
        # 对于A3策略，如果info中没有参数，从policy对象中获取
        current_hys = info.get("current_hys", None)
        current_ttt = info.get("current_ttt", None)
        if current_hys is None and a3_policy is not None:
            current_hys = a3_policy.hys_db
        if current_ttt is None and a3_policy is not None:
            current_ttt = a3_policy.ttt_ms
        
        trajectory.append({
            "step": step_count,
            "x": info.get("position_m", 0.0),
            "serving_cell": info.get("serving_cell", 0),
            "rsrp_A_dbm": info.get("rsrp_A_dbm", 0.0),
            "rsrp_B_dbm": info.get("rsrp_B_dbm", 0.0),
            "rsrp_serv_dbm": info.get("rsrp_serv_dbm", 0.0),
            "rsrp_neig_dbm": info.get("rsrp_neig_dbm", 0.0),
            "sinr_serv_db": info.get("sinr_serv_db", 0.0),
            "ho_executed": info.get("ho_executed", False),
            "current_hys": current_hys,
            "current_ttt": current_ttt,
        })
        
        obs_raw = next_obs_raw
        
        if step_count >= 10000:
            break
    
    return {
        "policy_name": policy_name,
        "trajectory": trajectory,
        "last_info": info  # 保存最后一步的info，可能包含KPI
    }


def plot_comparison(traj_a3: dict, traj_rl: dict, output_path: str = "test_comparison.png",
                    a3_hys: float = 3.0, a3_ttt: float = 160.0):
    """
    绘制A3和RL策略的RSRP曲线和切换点对比图
    
    Args:
        traj_a3: A3策略的轨迹数据
        traj_rl: RL策略的轨迹数据
        output_path: 输出图片路径
        a3_hys: A3策略的Hys参数（dB）
        a3_ttt: A3策略的TTT参数（ms）
    """
    traj_a3_list = traj_a3['trajectory']
    traj_rl_list = traj_rl['trajectory']
    
    # 提取数据
    xs_a3 = np.array([p['x'] for p in traj_a3_list])
    rsrp_A_a3 = np.array([p['rsrp_A_dbm'] for p in traj_a3_list])
    rsrp_B_a3 = np.array([p['rsrp_B_dbm'] for p in traj_a3_list])
    
    xs_rl = np.array([p['x'] for p in traj_rl_list])
    rsrp_A_rl = np.array([p['rsrp_A_dbm'] for p in traj_rl_list])
    rsrp_B_rl = np.array([p['rsrp_B_dbm'] for p in traj_rl_list])
    
    # 提取切换点
    ho_x_a3 = [p['x'] for p in traj_a3_list if p['ho_executed']]
    ho_y_a3 = []
    for p in traj_a3_list:
        if p['ho_executed']:
            ho_y_a3.append(p['rsrp_A_dbm'] if p['serving_cell'] == 0 else p['rsrp_B_dbm'])
    
    # 提取RL策略的切换点及其对应的Hys和TTT参数
    ho_x_rl = []
    ho_y_rl = []
    ho_hys_rl = []  # 每个切换点对应的Hys
    ho_ttt_rl = []  # 每个切换点对应的TTT
    for p in traj_rl_list:
        if p['ho_executed']:
            ho_x_rl.append(p['x'])
            ho_y_rl.append(p['rsrp_A_dbm'] if p['serving_cell'] == 0 else p['rsrp_B_dbm'])
            # 获取切换点时的Hys和TTT（使用当前步的参数）
            ho_hys_rl.append(p.get('current_hys', None))
            ho_ttt_rl.append(p.get('current_ttt', None))
    
    # 创建图形
    fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
    
    # A3策略
    ax1 = axes[0]
    ax1.plot(xs_a3, rsrp_A_a3, label="RSRP A小区", color="tab:blue", alpha=0.7, linewidth=2)
    ax1.plot(xs_a3, rsrp_B_a3, label="RSRP B小区", color="tab:orange", alpha=0.7, linewidth=2)
    if ho_x_a3:
        ax1.scatter(ho_x_a3, ho_y_a3, s=100, color='green', edgecolors='black', 
                   linewidths=1.0, label='A3 切换点', zorder=5)
    ax1.set_ylabel("RSRP / dBm")
    
    # 添加A3策略的参数标注
    a3_title = f"传统A3切换算法 (Hys={a3_hys:.1f}dB, TTT={a3_ttt:.0f}ms)"
    ax1.set_title(a3_title)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=9, loc='best')
    
    # RL策略
    ax2 = axes[1]
    ax2.plot(xs_rl, rsrp_A_rl, label="RSRP A小区", color="tab:blue", alpha=0.7, linewidth=2)
    ax2.plot(xs_rl, rsrp_B_rl, label="RSRP B小区", color="tab:orange", alpha=0.7, linewidth=2)
    if ho_x_rl:
        ax2.scatter(ho_x_rl, ho_y_rl, s=100, color='red', edgecolors='black', 
                   linewidths=1.0, label='RL 切换点', zorder=5)
        
        # 在每个切换点附近标注Hys和TTT参数
        y_min = min(min(rsrp_A_rl), min(rsrp_B_rl))
        y_max = max(max(rsrp_A_rl), max(rsrp_B_rl))
        y_range = y_max - y_min
        text_offset_y = y_range * 0.08  # 文本垂直偏移量（相对于y轴范围）
        
        for i, (x, y, hys, ttt) in enumerate(zip(ho_x_rl, ho_y_rl, ho_hys_rl, ho_ttt_rl)):
            # 构建标注文本
            if hys is not None and ttt is not None:
                label_text = f"Hys={hys:.1f}dB\nTTT={ttt:.0f}ms"
            elif hys is not None:
                label_text = f"Hys={hys:.1f}dB"
            elif ttt is not None:
                label_text = f"TTT={ttt:.0f}ms"
            else:
                label_text = ""
            
            if label_text:
                # 根据切换点位置调整文本位置，避免重叠
                # 交替在上下方显示，避免重叠
                if i % 2 == 0:
                    text_y = y + text_offset_y
                    va = 'bottom'
                else:
                    text_y = y - text_offset_y
                    va = 'top'
                
                # 添加文本标注
                ax2.annotate(
                    label_text,
                    xy=(x, y),
                    xytext=(x, text_y),
                    fontsize=8,
                    ha='center',
                    va=va,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7, edgecolor='black', linewidth=0.5),
                    arrowprops=dict(arrowstyle='->', color='black', lw=0.5, alpha=0.5),
                    zorder=6
                )
    
    ax2.set_xlabel("距离 x / m")
    ax2.set_ylabel("RSRP / dBm")
    ax2.set_title("离线训练 Rainbow DQN")
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=9, loc='best')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"对比图已保存至: {output_path}")
    plt.close()


def main():
    """主函数"""
    print("=" * 80)
    print("简单测试：传统A3 vs 离线训练 Rainbow DQN")
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
    
    # 3. 查找离线训练模型
    checkpoints_dir = os.path.join(base_dir, "checkpoints")
    checkpoint_path = os.path.join(checkpoints_dir, "rainbow_offline_final.pth")
    
    if not os.path.exists(checkpoint_path):
        print(f"\n错误：未找到离线训练模型！")
        print(f"请确保已运行 train_rainbow_offline.py 并生成了模型文件")
        print(f"查找路径: {checkpoint_path}")
        return
    
    # 4. 加载模型
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
    
    # 5. 创建环境和动作空间
    env_a3 = TrainHandoverEnv(config_path=env_config_path)
    env_rl = TrainHandoverEnv(config_path=env_config_path)
    action_space = ActionSpace()
    
    # 6. 创建观测窗口（用于RL）
    obs_window = ObservationWindow(window_size=window_size, obs_dim=obs_dim)
    
    # 7. 创建A3策略
    a3_hys = 3.0  # dB
    a3_ttt = 160.0  # ms
    a3_policy = TraditionalA3Policy(a3_hys, a3_ttt, action_space)
    a3_policy.reset()
    
    def a3_policy_func(obs, info, dt):
        return a3_policy.decide(obs, info, dt)
    
    # 8. 生成场景数据（确保两个策略使用相同的场景）
    seed = 10000
    print(f"\n生成场景数据（seed={seed}）...")
    
    # 创建场景生成器
    scenario_gen = ScenarioGenerator(env_a3.cfg)
    scenario_data = scenario_gen.generate_scenario(seed=seed, position_resolution_m=1.0)
    
    # 可选：保存场景数据
    scenario_dir = os.path.join(base_dir, "scenarios")
    os.makedirs(scenario_dir, exist_ok=True)
    scenario_path = os.path.join(scenario_dir, f"scenario_seed_{seed}.pkl")
    scenario_gen.save_scenario(scenario_data, scenario_path)
    print(f"场景数据已保存至: {scenario_path}")
    
    print(f"\n使用相同场景数据运行两个策略（确保环境状态完全一致）")
    
    print("\n运行传统A3策略...")
    traj_a3 = run_episode(
        env_a3, a3_policy_func,
        "A3 (Hys=3.0dB, TTT=160ms)",
        obs_window=None, model=None, device=device, seed=None, scenario_data=scenario_data,
        a3_policy=a3_policy
    )
    
    print("运行离线训练 Rainbow DQN...")
    traj_rl = run_episode(
        env_rl, None,
        "离线训练 Rainbow DQN",
        obs_window=obs_window, model=model, device=device, seed=None, scenario_data=scenario_data
    )
    
    # 9. 打印统计信息
    ho_count_a3 = sum(1 for p in traj_a3['trajectory'] if p['ho_executed'])
    ho_count_rl = sum(1 for p in traj_rl['trajectory'] if p['ho_executed'])
    
    print(f"\n统计信息:")
    print(f"  A3策略: 切换次数={ho_count_a3}, 步数={len(traj_a3['trajectory'])}")
    print(f"  RL策略: 切换次数={ho_count_rl}, 步数={len(traj_rl['trajectory'])}")
    
    # 10. 绘制对比图
    output_path = os.path.join(base_dir, "test_comparison.png")
    plot_comparison(traj_a3, traj_rl, output_path, a3_hys=a3_hys, a3_ttt=a3_ttt)
    
    print("\n" + "=" * 80)
    print("测试完成！")
    print("=" * 80)


if __name__ == "__main__":
    main()
