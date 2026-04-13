"""
数据收集脚本

使用环境生成离线学习所需的数据集
支持多种收集策略：随机策略、epsilon-greedy策略、固定策略等
"""
import os
import numpy as np
import yaml
from typing import Dict, List, Optional
import argparse
from tqdm import tqdm

from envs.train_ho_env import TrainHandoverEnv
from models import ActionSpace, ObservationWindow
from utils import NStepBuffer


def normalize_delta_rsrp(delta_rsrp_dbm: float, delta_min: float = -30.0, delta_max: float = 30.0) -> float:
    """归一化 ΔRSRP 到 [0, 1]"""
    return float(np.clip((delta_rsrp_dbm - delta_min) / (delta_max - delta_min + 1e-8), 0.0, 1.0))


class DataCollector:
    """数据收集器"""
    
    def __init__(self, env: TrainHandoverEnv, action_space: ActionSpace, 
                 obs_window: ObservationWindow, n_step_buffer: NStepBuffer,
                 config: Dict):
        """
        Args:
            env: 环境实例
            action_space: 动作空间
            obs_window: 观测窗口
            n_step_buffer: N-step缓冲区
            config: 配置字典
        """
        self.env = env
        self.action_space = action_space
        self.obs_window = obs_window
        self.n_step_buffer = n_step_buffer
        self.config = config
        
        self.window_size = config['observation']['window_size']
        self.obs_dim = config['observation']['obs_dim']
        self.n_steps = config['training']['n_steps']
        self.gamma = config['training']['gamma']
        self._last_phase_idx = None

    def _apply_reward_phase_for_episode(self, episode_idx: int, num_episodes: int):
        """
        按训练进度分阶段调整奖励权重（用于离线数据收集阶段）。

        阶段配置来自 env.cfg['reward_training_phases']。
        """
        phase_cfg = self.env.cfg.get('reward_training_phases', {})
        if not phase_cfg or not phase_cfg.get('enabled', False):
            return
        phases = phase_cfg.get('phases', [])
        if not phases:
            return

        # 训练进度（0~1]
        progress = float(episode_idx + 1) / max(int(num_episodes), 1)
        selected_idx = None
        selected = None
        for i, ph in enumerate(phases):
            end_ratio = float(ph.get('end_episode_ratio', 1.0))
            if progress <= end_ratio:
                selected_idx = i
                selected = ph
                break
        if selected is None:
            selected_idx = len(phases) - 1
            selected = phases[-1]

        # 仅在阶段切换时打印一次，避免刷屏
        if self._last_phase_idx != selected_idx:
            print(
                f"\n[RewardPhase] 进入阶段 {selected_idx + 1}/{len(phases)} "
                f"(progress={progress:.2%}) | "
                f"outage_scale={selected.get('reward_outage_scale', self.env.cfg.get('reward_outage_scale', 1.0))}, "
                f"interruption_scale={selected.get('reward_interruption_scale', self.env.cfg.get('reward_interruption_scale', 1.0))}, "
                f"ho_scale={selected.get('reward_ho_scale', self.env.cfg.get('reward_ho_scale', 1.0))}"
            )
            self._last_phase_idx = selected_idx

        # 覆盖环境中的奖励尺度
        if 'reward_outage_scale' in selected:
            self.env.cfg['reward_outage_scale'] = float(selected['reward_outage_scale'])
        if 'reward_interruption_scale' in selected:
            self.env.cfg['reward_interruption_scale'] = float(selected['reward_interruption_scale'])
        if 'reward_ho_scale' in selected:
            self.env.cfg['reward_ho_scale'] = float(selected['reward_ho_scale'])
        
    def collect_episode(self, policy_type: str = 'random', epsilon: float = 1.0, 
                       seed: Optional[int] = None) -> List[Dict]:
        """
        收集一个episode的数据
        
        Args:
            policy_type: 策略类型 ('random', 'epsilon_greedy', 'fixed')
            epsilon: epsilon-greedy的探索率（仅当policy_type='epsilon_greedy'时使用）
            seed: 随机种子
            
        Returns:
            experiences: 经验列表
        """
        # 重置环境
        obs_raw, info = self.env.reset(seed=seed)
        self.obs_window.reset()
        self.obs_window.update_time(0.0)
        
        # 初始化窗口（用第一个观测填充）
        for _ in range(self.window_size):
            obs_extended = self.obs_window.build_observation(
                obs_raw, info,
                velocity_mps=self.env.velocity_mps,
                track_length_m=self.env.cfg['track_length_m']
            )
        
        experiences = []
        done = False
        step = 0
        
        while not done:
            # 获取当前窗口
            window = self.obs_window.get_window()
            
            # 选择动作
            action = self._select_action(policy_type, epsilon, window)
            
            # 执行动作
            next_obs_raw, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            
            # 更新观测窗口
            current_time = self.env.time_step * self.env.cfg['delta_t_s']
            self.obs_window.update_time(current_time)
            
            if info.get('ho_executed', False):
                self.obs_window.update_ho_time(current_time)
            
            self.obs_window.update_params(
                info.get('current_hys', 3.0),
                info.get('current_ttt', 160.0)
            )
            
            next_obs_extended = self.obs_window.build_observation(
                next_obs_raw, info,
                velocity_mps=self.env.velocity_mps,
                track_length_m=self.env.cfg['track_length_m']
            )
            next_window = self.obs_window.get_window()
            
            # 计算下一步的 ΔRSRP（用于辅助预测）
            delta_rsrp_next = info['rsrp_neig_dbm'] - info['rsrp_serv_dbm']
            delta_rsrp_norm = normalize_delta_rsrp(
                delta_rsrp_next,
                delta_min=self.config['normalization']['delta_rsrp_min'],
                delta_max=self.config['normalization']['delta_rsrp_max']
            )
            
            # 添加到 N-step 缓冲区
            n_step_exp = self.n_step_buffer.add(
                window, action, reward, next_window, done, delta_rsrp_norm
            )
            
            # 如果得到 N-step 经验，保存
            if n_step_exp is not None:
                experiences.append(n_step_exp)
            
            obs_raw = next_obs_raw
            step += 1
            
            # 防止无限循环
            if step >= 100000:
                break
        
        # Episode 结束时，清空 N-step 缓冲区
        remaining_exps = self.n_step_buffer.flush()
        experiences.extend(remaining_exps)
        
        return experiences
    
    def _select_action(self, policy_type: str, epsilon: float, window: np.ndarray) -> int:
        """选择动作"""
        num_actions = self.action_space.num_actions
        
        if policy_type == 'random':
            # 完全随机
            return np.random.randint(0, num_actions)
        
        elif policy_type == 'epsilon_greedy':
            # Epsilon-greedy（需要模型，这里简化使用随机）
            if np.random.random() < epsilon:
                return np.random.randint(0, num_actions)
            else:
                # 贪婪策略：随机选择一个动作（因为没有模型）
                # 实际应用中，这里应该使用训练好的模型
                return np.random.randint(0, num_actions)
        
        elif policy_type == 'fixed':
            # 固定策略：使用固定的 Hys/TTT
            # 例如：Hys=3.0, TTT=160
            hys = 3.0
            ttt = 160.0
            return self.action_space.hys_ttt_to_action(hys, ttt)
        
        elif policy_type == 'uniform_mix':
            # 均匀混合：随机选择不同的 Hys/TTT 组合
            hys_idx = np.random.randint(0, len(self.action_space.hys_set))
            ttt_idx = np.random.randint(0, len(self.action_space.ttt_set))
            action = hys_idx * len(self.action_space.ttt_set) + ttt_idx
            return action
        
        else:
            raise ValueError(f"未知的策略类型: {policy_type}")
    
    def collect_dataset(self, num_episodes: int, policy_type: str = 'random',
                       epsilon: float = 1.0, seed_start: int = 0) -> Dict:
        """
        收集数据集
        
        Args:
            num_episodes: episode数量
            policy_type: 策略类型
            epsilon: epsilon值
            seed_start: 起始随机种子
            
        Returns:
            dataset: 数据集字典
        """
        all_experiences = []
        
        print(f"开始收集数据...")
        print(f"策略类型: {policy_type}")
        print(f"Episode数量: {num_episodes}")
        
        for episode in tqdm(range(num_episodes), desc="收集数据"):
            self._apply_reward_phase_for_episode(episode, num_episodes)
            seed = seed_start + episode if seed_start is not None else None
            experiences = self.collect_episode(policy_type, epsilon, seed)
            all_experiences.extend(experiences)
        
        # 转换为numpy数组
        num_samples = len(all_experiences)
        print(f"\n收集完成！总样本数: {num_samples}")
        
        if num_samples == 0:
            raise ValueError("未收集到任何数据！")
        
        # 组织数据
        obs_windows = np.array([exp['obs_window'] for exp in all_experiences], dtype=np.float32)
        actions = np.array([exp['action'] for exp in all_experiences], dtype=np.int64)
        rewards = np.array([exp['reward'] for exp in all_experiences], dtype=np.float32)
        next_obs_windows = np.array([exp['next_obs_window'] for exp in all_experiences], dtype=np.float32)
        dones = np.array([exp['done'] for exp in all_experiences], dtype=bool)
        delta_targets = np.array([exp['delta_rsrp_target'] for exp in all_experiences], dtype=np.float32)
        
        dataset = {
            'obs': obs_windows,
            'actions': actions,
            'rewards': rewards,
            'next_obs': next_obs_windows,
            'dones': dones,
            'delta_targets': delta_targets,
            'num_samples': num_samples,
            'policy_type': policy_type,
            'config': self.config
        }
        
        return dataset


def save_dataset(dataset: Dict, save_path: str):
    """保存数据集"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # 保存为 .npz 格式
    np.savez_compressed(
        save_path,
        obs=dataset['obs'],
        actions=dataset['actions'],
        rewards=dataset['rewards'],
        next_obs=dataset['next_obs'],
        dones=dataset['dones'],
        delta_targets=dataset['delta_targets'],
        num_samples=dataset['num_samples'],
        policy_type=dataset['policy_type']
    )
    
    print(f"数据集已保存: {save_path}")
    print(f"  样本数: {dataset['num_samples']}")
    print(f"  观测形状: {dataset['obs'].shape}")
    print(f"  动作形状: {dataset['actions'].shape}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='数据收集脚本')
    parser.add_argument('--num_episodes', type=int, default=100, help='收集的episode数量')
    parser.add_argument('--policy_type', type=str, default='random', 
                       choices=['random', 'epsilon_greedy', 'fixed', 'uniform_mix'],
                       help='收集策略类型')
    parser.add_argument('--epsilon', type=float, default=1.0, help='Epsilon值（用于epsilon_greedy）')
    parser.add_argument('--output_path', type=str, default='data/offline_dataset.npz',
                       help='输出路径')
    parser.add_argument('--seed_start', type=int, default=0, help='起始随机种子')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("数据收集脚本")
    print("=" * 80)
    
    # 1. 加载配置
    base_dir = os.path.dirname(__file__)
    env_config_path = os.path.join(base_dir, "configs", "default_env_config.yaml")
    model_config_path = os.path.join(base_dir, "configs", "model_config.yaml")
    
    with open(model_config_path, 'r', encoding='utf-8') as f:
        model_config = yaml.safe_load(f)
    
    # 2. 创建环境和动作空间
    env = TrainHandoverEnv(config_path=env_config_path)
    action_space = ActionSpace()
    
    # 3. 创建观测窗口和N-step缓冲区
    obs_dim = model_config['observation']['obs_dim']
    window_size = model_config['observation']['window_size']
    
    obs_window = ObservationWindow(
        window_size=window_size,
        obs_dim=obs_dim
    )
    
    n_step_buffer = NStepBuffer(
        n_steps=model_config['training']['n_steps'],
        gamma=model_config['training']['gamma']
    )
    
    # 4. 创建数据收集器
    collector = DataCollector(env, action_space, obs_window, n_step_buffer, model_config)
    
    # 5. 收集数据
    dataset = collector.collect_dataset(
        num_episodes=args.num_episodes,
        policy_type=args.policy_type,
        epsilon=args.epsilon,
        seed_start=args.seed_start
    )
    
    # 6. 保存数据集
    output_path = os.path.join(base_dir, args.output_path)
    save_dataset(dataset, output_path)
    
    print("\n" + "=" * 80)
    print("数据收集完成！")
    print("=" * 80)


if __name__ == "__main__":
    main()

