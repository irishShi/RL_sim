"""
Rainbow DQN 训练脚本

实现完整的 Rainbow DQN + 辅助预测训练流程
"""
import os
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import yaml
from collections import deque
from typing import Dict, Optional
import time
import json

from envs.train_ho_env import TrainHandoverEnv
from models import RainbowWithForecast, ActionSpace, ObservationWindow
from utils import ReplayBuffer, NStepBuffer, project_distribution


def normalize_delta_rsrp(delta_rsrp_dbm: float, delta_min: float = -30.0, delta_max: float = 30.0) -> float:
    """归一化 ΔRSRP 到 [0, 1]"""
    return float(np.clip((delta_rsrp_dbm - delta_min) / (delta_max - delta_min + 1e-8), 0.0, 1.0))


def get_epsilon(step: int, epsilon_start: float = 1.0, epsilon_end: float = 0.01, 
                epsilon_decay: int = 10000) -> float:
    """计算当前 epsilon 值（用于 NoisyNet，实际上不使用）"""
    if step >= epsilon_decay:
        return epsilon_end
    return epsilon_start - (epsilon_start - epsilon_end) * (step / epsilon_decay)


def get_beta(step: int, beta_start: float = 0.4, beta_end: float = 1.0,
              beta_increment: float = 0.00001) -> float:
    """计算 PER beta 值"""
    return min(beta_end, beta_start + step * beta_increment)


def train_step(batch: Dict, online_net: RainbowWithForecast, target_net: RainbowWithForecast,
               optimizer: optim.Optimizer, config: Dict, device: str = 'cpu') -> Dict:
    """
    执行一次训练步骤
    
    Args:
        batch: 经验 batch
        online_net: 在线网络
        target_net: 目标网络
        optimizer: 优化器
        config: 配置字典
        device: 设备
        
    Returns:
        训练信息字典
    """
    # 转换为 tensor（使用 torch.from_numpy 更高效，直接在 GPU 上创建）
    obs = torch.from_numpy(batch['obs']).float().to(device)  # [B, N, F]
    action = torch.from_numpy(batch['action']).long().to(device)  # [B]
    reward_n = torch.from_numpy(batch['reward']).float().to(device)  # [B] - 注意：这是 N-step reward
    next_obs = torch.from_numpy(batch['next_obs']).float().to(device)  # [B, N, F]
    done = torch.from_numpy(batch['done']).float().to(device)  # [B]
    delta_target = torch.from_numpy(batch['delta_target']).float().to(device)  # [B]
    weights = torch.from_numpy(batch['weights']).float().to(device)  # [B]
    
    B = obs.size(0)
    num_atoms = config['network']['rainbow']['num_atoms']
    num_actions = config['action_space']['num_actions']
    v_min = config['network']['rainbow']['v_min']
    v_max = config['network']['rainbow']['v_max']
    gamma = config['training']['gamma']
    n_steps = config['training']['n_steps']
    lambda_aux = config['training']['lambda_aux']
    
    # 创建 support（直接在 GPU 上创建，避免数据传输）
    support = torch.linspace(v_min, v_max, num_atoms, device=device, dtype=torch.float32)
    
    # 重置噪声
    online_net.train()
    online_net.reset_noise()
    
    # 当前分布和预测
    dist, pred_delta = online_net(obs)
    action_idx = action.view(-1, 1, 1).expand(B, 1, num_atoms)
    dist_a = dist.gather(1, action_idx).squeeze(1)  # [B, num_atoms]
    
    # 目标分布（Double Q）
    with torch.no_grad():
        target_net.eval()
        # 用在线网络选动作
        next_dist_online, _ = online_net(next_obs)
        next_q = (next_dist_online * support.view(1, 1, -1)).sum(dim=-1)  # [B, num_actions]
        next_action = next_q.argmax(dim=1, keepdim=True)  # [B, 1]
        
        # 用目标网络得到分布
        next_dist_target, _ = target_net(next_obs)
        next_action_idx = next_action.view(-1, 1, 1).expand(B, 1, num_atoms)
        target_dist_a = next_dist_target.gather(1, next_action_idx).squeeze(1)  # [B, num_atoms]
        
        # C51 投影
        target_support = support.unsqueeze(0).expand(B, -1)  # [B, num_atoms]
        m = project_distribution(
            support, target_dist_a, target_support,
            v_min, v_max, gamma, reward_n, done, n_steps
        )
    
    # Rainbow 损失（KL 散度）
    log_p = torch.log(dist_a + 1e-6)
    rainbow_loss = -(m * log_p).sum(dim=1)  # [B]
    rainbow_loss = (rainbow_loss * weights).mean()
    
    # 预测损失
    forecast_loss = F.mse_loss(pred_delta, delta_target, reduction='none')  # [B]
    forecast_loss = (forecast_loss * weights).mean()
    
    # 总损失
    loss = rainbow_loss + lambda_aux * forecast_loss
    
    # 反向传播
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(online_net.parameters(), config['training']['grad_clip'])
    optimizer.step()
    
    # 计算 TD error（用于更新 PER 优先级）
    td_errors = (rainbow_loss.detach() / (weights + 1e-8)).cpu().numpy()
    
    return {
        'loss': loss.item(),
        'rainbow_loss': rainbow_loss.item(),
        'forecast_loss': forecast_loss.item(),
        'td_errors': td_errors
    }


def main():
    """主训练函数"""
    print("=" * 80)
    print("Rainbow DQN 训练脚本")
    print("=" * 80)
    
    # 1. 加载配置
    base_dir = os.path.dirname(__file__)
    env_config_path = os.path.join(base_dir, "configs", "default_env_config.yaml")
    model_config_path = os.path.join(base_dir, "configs", "model_config.yaml")
    
    with open(model_config_path, 'r', encoding='utf-8') as f:
        model_config = yaml.safe_load(f)
    
    # 2. 设置设备
    if torch.cuda.is_available():
        device = 'cuda'
        print(f"\n使用设备: {device}")
        print(f"GPU 名称: {torch.cuda.get_device_name(0)}")
        print(f"GPU 内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        # GPU 优化设置
        torch.cuda.empty_cache()  # 清空缓存
        # 启用 cuDNN 自动调优（如果可用）
        torch.backends.cudnn.benchmark = True
        # 设置默认 tensor 类型（可选，但可能影响其他代码）
        # torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        device = 'cpu'
        print(f"\n使用设备: {device} (未检测到 GPU，将使用 CPU)")
    
    # 3. 创建环境和动作空间
    env = TrainHandoverEnv(config_path=env_config_path)
    action_space = ActionSpace()
    
    # 从配置读取动作数量
    hys_set = model_config['action_space']['hys_set']
    ttt_set = model_config['action_space']['ttt_set']
    num_actions = len(hys_set) * len(ttt_set)
    
    print(f"动作空间: {num_actions} 个动作")
    print(f"  Hys 集合: {hys_set}")
    print(f"  TTT 集合: {ttt_set}")
    
    # 4. 创建模型
    obs_dim = model_config['observation']['obs_dim']
    window_size = model_config['observation']['window_size']
    
    online_net = RainbowWithForecast(
        obs_dim=obs_dim,
        num_actions=num_actions,
        n_steps=window_size,
        feature_hidden=model_config['network']['shared']['hidden_dim'],
        encoder_hidden=model_config['network']['encoder']['hidden_dim'],
        num_atoms=model_config['network']['rainbow']['num_atoms'],
        v_min=model_config['network']['rainbow']['v_min'],
        v_max=model_config['network']['rainbow']['v_max']
    ).to(device)
    
    target_net = RainbowWithForecast(
        obs_dim=obs_dim,
        num_actions=num_actions,
        n_steps=window_size,
        feature_hidden=model_config['network']['shared']['hidden_dim'],
        encoder_hidden=model_config['network']['encoder']['hidden_dim'],
        num_atoms=model_config['network']['rainbow']['num_atoms'],
        v_min=model_config['network']['rainbow']['v_min'],
        v_max=model_config['network']['rainbow']['v_max']
    ).to(device)
    
    target_net.load_state_dict(online_net.state_dict())
    
    # 确保目标网络也在正确的设备上
    target_net = target_net.to(device)
    
    print(f"\n模型参数总数: {sum(p.numel() for p in online_net.parameters()):,}")
    
    # 如果使用 GPU，显示内存使用情况
    if device == 'cuda':
        print(f"GPU 内存使用: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB / "
              f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    # 5. 创建优化器和缓冲区
    optimizer = optim.Adam(
        online_net.parameters(),
        lr=model_config['training']['learning_rate']
    )
    
    buffer = ReplayBuffer(
        capacity=model_config['training']['replay_buffer_size'],
        obs_window_size=window_size,
        obs_dim=obs_dim,
        per_alpha=model_config['training']['per_alpha']
    )
    
    n_step_buffer = NStepBuffer(
        n_steps=model_config['training']['n_steps'],
        gamma=model_config['training']['gamma']
    )
    
    # 6. 创建观测窗口
    obs_window = ObservationWindow(
        window_size=window_size,
        obs_dim=obs_dim
    )
    
    # 7. 训练参数
    batch_size = model_config['training']['batch_size']
    train_freq = model_config['training']['train_freq']
    target_update_freq = model_config['training']['target_update_freq']
    replay_start_size = model_config['training']['replay_start_size']
    update_freq = model_config['training']['update_freq']
    
    # 8. 训练统计
    training_stats = {
        'episode_rewards': [],
        'episode_lengths': [],
        'losses': [],
        'rainbow_losses': [],
        'forecast_losses': []
    }
    
    # 9. 训练循环
    print("\n开始训练...")
    print("-" * 80)
    
    step = 0
    episode = 0
    max_episodes = 500  # 可以根据需要调整
    
    # 保存上一步的状态（用于延迟存储）
    prev_window = None
    prev_action = None
    prev_reward = None
    prev_done = None
    
    start_time = time.time()
    
    for episode in range(max_episodes):
        obs_raw, info = env.reset()
        obs_window.reset()
        obs_window.update_time(0.0)
        
        # 初始化窗口（用第一个观测填充）
        for _ in range(window_size):
            obs_extended = obs_window.build_observation(
                obs_raw, info,
                velocity_mps=env.velocity_mps,
                track_length_m=env.cfg['track_length_m']
            )
        
        done = False
        episode_reward = 0.0
        episode_steps = 0
        
        # 重置状态
        prev_window = None
        prev_action = None
        prev_reward = None
        prev_done = None
        
        while not done:
            # 获取当前窗口
            window = obs_window.get_window()
            # 直接在 GPU 上创建 tensor，减少数据传输
            window_tensor = torch.from_numpy(window).float().unsqueeze(0).to(device)
            
            # 选择动作（NoisyNet 自动探索，不需要 epsilon-greedy）
            action = online_net.act(window_tensor, epsilon=0.0)
            
            # 执行动作
            next_obs_raw, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            episode_steps += 1
            
            # 更新观测窗口
            current_time = env.time_step * env.cfg['delta_t_s']
            obs_window.update_time(current_time)
            
            if info.get('ho_executed', False):
                obs_window.update_ho_time(current_time)
            
            obs_window.update_params(
                info.get('current_hys', 3.0),
                info.get('current_ttt', 160.0)
            )
            
            next_obs_extended = obs_window.build_observation(
                next_obs_raw, info,
                velocity_mps=env.velocity_mps,
                track_length_m=env.cfg['track_length_m']
            )
            next_window = obs_window.get_window()
            
            # 计算下一步的 ΔRSRP（用于辅助预测）
            delta_rsrp_next = info['rsrp_neig_dbm'] - info['rsrp_serv_dbm']
            delta_rsrp_norm = normalize_delta_rsrp(
                delta_rsrp_next,
                delta_min=model_config['normalization']['delta_rsrp_min'],
                delta_max=model_config['normalization']['delta_rsrp_max']
            )
            
            # 添加到 N-step 缓冲区
            n_step_exp = n_step_buffer.add(
                window, action, reward, next_window, done, delta_rsrp_norm
            )
            
            # 如果得到 N-step 经验，存储到回放缓冲区
            if n_step_exp is not None:
                buffer.store(**n_step_exp)
            
            # 延迟存储：存储上一步的经验（此时当前步的 ΔRSRP 已知）
            if prev_window is not None:
                # 如果 N-step 缓冲区还没满，使用单步经验
                # 这里简化处理，直接使用当前步的 delta_rsrp_norm
                pass  # N-step buffer 已经处理了
            
            # 保存当前状态（用于下一步的延迟存储）
            prev_window = window.copy()
            prev_action = action
            prev_reward = reward
            prev_done = done
            
            # 训练
            if step % train_freq == 0 and buffer.size >= replay_start_size:
                for _ in range(update_freq):
                    beta = get_beta(step, 
                                   beta_start=model_config['training']['per_beta'],
                                   beta_increment=model_config['training']['per_beta_increment'])
                    batch = buffer.sample(batch_size, beta=beta)
                    if batch is not None:
                        train_info = train_step(batch, online_net, target_net, optimizer,
                                               model_config, device)
                        
                        # 更新 PER 优先级
                        buffer.update_priorities(batch['indices'], train_info['td_errors'])
                        
                        # 记录损失
                        training_stats['losses'].append(train_info['loss'])
                        training_stats['rainbow_losses'].append(train_info['rainbow_loss'])
                        training_stats['forecast_losses'].append(train_info['forecast_loss'])
            
            # 更新目标网络
            if step % target_update_freq == 0 and step > 0:
                target_net.load_state_dict(online_net.state_dict())
                if device == 'cuda':
                    torch.cuda.empty_cache()  # 定期清空 GPU 缓存
            
            step += 1
        
        # Episode 结束时，清空 N-step 缓冲区
        remaining_exps = n_step_buffer.flush()
        for exp in remaining_exps:
            buffer.store(**exp)
        
        # 记录 episode 统计
        training_stats['episode_rewards'].append(episode_reward)
        training_stats['episode_lengths'].append(episode_steps)
        
        # 打印进度
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(training_stats['episode_rewards'][-10:])
            avg_length = np.mean(training_stats['episode_lengths'][-10:])
            avg_loss = np.mean(training_stats['losses'][-100:]) if training_stats['losses'] else 0.0
            
            elapsed = time.time() - start_time
            gpu_mem_info = ""
            if device == 'cuda':
                gpu_mem_used = torch.cuda.memory_allocated(0) / 1024**3
                gpu_mem_info = f" | GPU: {gpu_mem_used:.2f}GB"
            
            print(f"Episode {episode+1:4d}/{max_episodes} | "
                  f"Reward: {episode_reward:7.2f} (avg: {avg_reward:7.2f}) | "
                  f"Steps: {episode_steps:4d} (avg: {avg_length:5.1f}) | "
                  f"Loss: {avg_loss:.4f} | "
                  f"Buffer: {buffer.size:6d} | "
                  f"Time: {elapsed/60:.1f}m{gpu_mem_info}")
        
        # 定期保存模型
        if (episode + 1) % 50 == 0:
            save_dir = os.path.join(base_dir, "checkpoints")
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"rainbow_episode_{episode+1}.pth")
            torch.save({
                'episode': episode + 1,
                'step': step,
                'online_net_state_dict': online_net.state_dict(),
                'target_net_state_dict': target_net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'training_stats': training_stats,
                'config': model_config
            }, save_path)
            print(f"模型已保存: {save_path}")
    
    # 10. 保存最终模型
    final_save_path = os.path.join(base_dir, "checkpoints", "rainbow_final.pth")
    torch.save({
        'episode': episode + 1,
        'step': step,
        'online_net_state_dict': online_net.state_dict(),
        'target_net_state_dict': target_net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'training_stats': training_stats,
        'config': model_config
    }, final_save_path)
    
    print(f"\n训练完成！")
    print(f"最终模型已保存: {final_save_path}")
    print(f"总训练时间: {(time.time() - start_time)/60:.1f} 分钟")
    print("=" * 80)


if __name__ == "__main__":
    main()
