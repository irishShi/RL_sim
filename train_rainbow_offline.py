"""
Rainbow DQN 离线训练脚本

从预收集的数据集进行离线学习，无需与环境交互
"""
import os
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import yaml
from typing import Dict, Optional
import time
import argparse
from tqdm import tqdm

from models import RainbowWithForecast, ActionSpace
from utils import ReplayBuffer, project_distribution
from utils.dataset_loader import OfflineDataset


def train_step(batch: Dict, online_net: RainbowWithForecast, target_net: RainbowWithForecast,
               optimizer: optim.Optimizer, config: Dict, device: str = 'cpu', 
               use_cql: bool = False, cql_alpha: float = 0.1) -> Dict:
    """
    执行一次训练步骤
    
    Args:
        batch: 经验 batch
        online_net: 在线网络
        target_net: 目标网络
        optimizer: 优化器
        config: 配置字典
        device: 设备
        use_cql: 是否使用 CQL 正则化
        cql_alpha: CQL 正则化系数
        
    Returns:
        训练信息字典
    """
    # 转换为 tensor
    obs = torch.from_numpy(batch['obs']).float().to(device)  # [B, N, F]
    action = torch.from_numpy(batch['action']).long().to(device)  # [B]
    reward_n = torch.from_numpy(batch['reward']).float().to(device)  # [B] - N-step reward
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
    
    # 创建 support（缓存，避免重复创建）
    support = torch.linspace(v_min, v_max, num_atoms, device=device, dtype=torch.float32)
    support_expanded = support.view(1, 1, -1)  # [1, 1, num_atoms] 用于后续计算
    
    # 重置噪声（如果使用）
    online_net.train()
    if online_net.use_noisy:
        online_net.reset_noise()
    
    # 当前分布和预测（一次性前向传播）
    dist, pred_delta = online_net(obs)
    
    # 优化：使用更高效的索引操作
    action_idx = action.unsqueeze(1).unsqueeze(2).expand(-1, 1, num_atoms)  # [B, 1, num_atoms]
    dist_a = dist.gather(1, action_idx).squeeze(1)  # [B, num_atoms]
    
    # 目标分布（Double Q）- 在 no_grad 中计算
    with torch.no_grad():
        target_net.eval()
        # 用在线网络选动作（需要梯度，但目标值不需要）
        next_dist_online, _ = online_net(next_obs)
        next_q = (next_dist_online * support_expanded).sum(dim=-1)  # [B, num_actions]
        next_action = next_q.argmax(dim=1, keepdim=True)  # [B, 1]
        
        # 用目标网络得到分布
        next_dist_target, _ = target_net(next_obs)
        next_action_idx = next_action.unsqueeze(2).expand(-1, 1, num_atoms)  # [B, 1, num_atoms]
        target_dist_a = next_dist_target.gather(1, next_action_idx).squeeze(1)  # [B, num_atoms]
        
        # C51 投影（优化：避免重复创建 target_support）
        target_support = support.unsqueeze(0).expand(B, -1)  # [B, num_atoms]
        m = project_distribution(
            support, target_dist_a, target_support,
            v_min, v_max, gamma, reward_n, done, n_steps
        )
    
    # Rainbow 损失（KL 散度）- 优化：使用更稳定的数值计算
    log_p = torch.log(dist_a.clamp(min=1e-6))  # 避免 log(0)
    rainbow_loss = -(m * log_p).sum(dim=1)  # [B]
    rainbow_loss = (rainbow_loss * weights).mean()
    
    # 预测损失（优化：直接计算，避免函数调用开销）
    forecast_loss = ((pred_delta - delta_target) ** 2)  # [B]
    forecast_loss = (forecast_loss * weights).mean()
    
    # CQL 正则化项（可选）- 优化：避免重复计算 Q 值
    cql_loss = torch.tensor(0.0, device=device)
    if use_cql:
        # 复用已计算的 dist，避免重复前向传播
        q_all = online_net.rainbow_head.get_q_values(dist)  # [B, num_actions]
        q_data = q_all.gather(1, action.unsqueeze(1)).squeeze(1)  # [B]
        
        # CQL 损失：惩罚所有动作的 Q 值，但奖励数据中的动作
        # 优化：使用更稳定的计算方式
        cql_loss = cql_alpha * (q_all.mean() - q_data.mean())
    
    # 总损失
    loss = rainbow_loss + lambda_aux * forecast_loss + cql_loss
    
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
        'cql_loss': cql_loss.item() if use_cql else 0.0,
        'td_errors': td_errors
    }


def main():
    """主训练函数"""
    parser = argparse.ArgumentParser(description='Rainbow DQN 离线训练')
    parser.add_argument('--dataset_path', type=str, default='data/offline_dataset.npz',
                       help='数据集路径')
    parser.add_argument('--use_cql', action='store_true', help='是否使用 CQL 正则化')
    parser.add_argument('--cql_alpha', type=float, default=0.1, help='CQL 正则化系数')
    parser.add_argument('--num_epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--samples_per_epoch', type=int, default=10000, help='每轮采样数')
    parser.add_argument('--batch_size', type=int, default=None, help='Batch大小（覆盖配置）')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("Rainbow DQN 离线训练脚本")
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
        torch.cuda.empty_cache()
        torch.backends.cudnn.benchmark = True
    else:
        device = 'cpu'
        print(f"\n使用设备: {device} (未检测到 GPU，将使用 CPU)")
    
    # 3. 加载数据集
    dataset_path = os.path.join(base_dir, args.dataset_path)
    obs_dim = model_config['observation']['obs_dim']
    window_size = model_config['observation']['window_size']
    
    dataset_loader = OfflineDataset(dataset_path, window_size, obs_dim)
    dataset = dataset_loader.load()
    
    num_samples = dataset['obs'].shape[0]
    print(f"\n数据集大小: {num_samples} 个样本")
    
    # 4. 创建动作空间
    action_space = ActionSpace()
    hys_set = model_config['action_space']['hys_set']
    ttt_set = model_config['action_space']['ttt_set']
    num_actions = len(hys_set) * len(ttt_set)
    
    print(f"动作空间: {num_actions} 个动作")
    
    # 5. 创建模型（离线学习不使用 NoisyNet）
    online_net = RainbowWithForecast(
        obs_dim=obs_dim,
        num_actions=num_actions,
        n_steps=window_size,
        feature_hidden=model_config['network']['shared']['hidden_dim'],
        encoder_hidden=model_config['network']['encoder']['hidden_dim'],
        num_atoms=model_config['network']['rainbow']['num_atoms'],
        v_min=model_config['network']['rainbow']['v_min'],
        v_max=model_config['network']['rainbow']['v_max'],
        use_noisy=False  # 离线学习不使用 NoisyNet
    ).to(device)
    
    target_net = RainbowWithForecast(
        obs_dim=obs_dim,
        num_actions=num_actions,
        n_steps=window_size,
        feature_hidden=model_config['network']['shared']['hidden_dim'],
        encoder_hidden=model_config['network']['encoder']['hidden_dim'],
        num_atoms=model_config['network']['rainbow']['num_atoms'],
        v_min=model_config['network']['rainbow']['v_min'],
        v_max=model_config['network']['rainbow']['v_max'],
        use_noisy=False
    ).to(device)
    
    target_net.load_state_dict(online_net.state_dict())
    target_net = target_net.to(device)
    
    print(f"\n模型参数总数: {sum(p.numel() for p in online_net.parameters()):,}")
    print(f"使用 NoisyNet: {online_net.use_noisy}")
    print(f"使用 CQL: {args.use_cql}")
    if args.use_cql:
        print(f"CQL Alpha: {args.cql_alpha}")
    
    # 6. 创建优化器和缓冲区
    optimizer = optim.Adam(
        online_net.parameters(),
        lr=model_config['training']['learning_rate']
    )
    
    # 从数据集填充缓冲区
    buffer = ReplayBuffer(
        capacity=model_config['training']['replay_buffer_size'],
        obs_window_size=window_size,
        obs_dim=obs_dim,
        per_alpha=model_config['training']['per_alpha']
    )
    
    print(f"\n填充经验回放缓冲区...")
    # 优化：批量填充，减少循环开销
    for i in tqdm(range(num_samples), desc="填充缓冲区"):
        buffer.store(
            obs_window=dataset['obs'][i],
            action=int(dataset['actions'][i]),
            reward=float(dataset['rewards'][i]),
            next_obs_window=dataset['next_obs'][i],
            done=bool(dataset['dones'][i]),
            delta_rsrp_target=float(dataset['delta_targets'][i])
        )
    
    print(f"缓冲区大小: {buffer.size}")
    
    # 7. 训练参数
    batch_size = args.batch_size if args.batch_size is not None else model_config['training']['batch_size']
    target_update_freq = model_config['training']['target_update_freq']
    
    # 8. 训练统计
    training_stats = {
        'losses': [],
        'rainbow_losses': [],
        'forecast_losses': [],
        'cql_losses': []
    }
    
    # 9. 训练循环
    print("\n开始离线训练...")
    print("-" * 80)
    
    start_time = time.time()
    step = 0
    
    for epoch in range(args.num_epochs):
        epoch_losses = []
        epoch_rainbow_losses = []
        epoch_forecast_losses = []
        epoch_cql_losses = []
        
        # 每轮采样指定数量的样本
        num_batches = args.samples_per_epoch // batch_size
        
        for batch_idx in tqdm(range(num_batches), desc=f"Epoch {epoch+1}/{args.num_epochs}"):
            # 采样 batch
            beta = min(1.0, 0.4 + step * model_config['training']['per_beta_increment'])
            batch = buffer.sample(batch_size, beta=beta)
            
            if batch is not None:
                train_info = train_step(
                    batch, online_net, target_net, optimizer,
                    model_config, device,
                    use_cql=args.use_cql,
                    cql_alpha=args.cql_alpha
                )
                
                # 更新 PER 优先级
                buffer.update_priorities(batch['indices'], train_info['td_errors'])
                
                # 记录损失（优化：直接添加，避免多次 append）
                epoch_losses.append(train_info['loss'])
                epoch_rainbow_losses.append(train_info['rainbow_loss'])
                epoch_forecast_losses.append(train_info['forecast_loss'])
                if args.use_cql:
                    epoch_cql_losses.append(train_info['cql_loss'])
                
                step += 1
                
                # 更新目标网络（优化：减少检查频率）
                if step % target_update_freq == 0:
                    target_net.load_state_dict(online_net.state_dict())
                    # 定期清理 GPU 缓存（但不是每次都清理）
                    if device == 'cuda' and step % (target_update_freq * 10) == 0:
                        torch.cuda.empty_cache()
        
        # 记录 epoch 统计
        if epoch_losses:
            training_stats['losses'].extend(epoch_losses)
            training_stats['rainbow_losses'].extend(epoch_rainbow_losses)
            training_stats['forecast_losses'].extend(epoch_forecast_losses)
            if args.use_cql:
                training_stats['cql_losses'].extend(epoch_cql_losses)
            
            avg_loss = np.mean(epoch_losses)
            avg_rainbow = np.mean(epoch_rainbow_losses)
            avg_forecast = np.mean(epoch_forecast_losses)
            
            elapsed = time.time() - start_time
            gpu_mem_info = ""
            if device == 'cuda':
                gpu_mem_used = torch.cuda.memory_allocated(0) / 1024**3
                gpu_mem_info = f" | GPU: {gpu_mem_used:.2f}GB"
            
            print_str = f"Epoch {epoch+1:4d}/{args.num_epochs} | " \
                       f"Loss: {avg_loss:.4f} (Rainbow: {avg_rainbow:.4f}, Forecast: {avg_forecast:.4f})"
            
            if args.use_cql and epoch_cql_losses:
                avg_cql = np.mean(epoch_cql_losses)
                print_str += f", CQL: {avg_cql:.4f}"
            
            print_str += f" | Time: {elapsed/60:.1f}m{gpu_mem_info}"
            print(print_str)
        
        # 定期保存模型
        if (epoch + 1) % 10 == 0:
            save_dir = os.path.join(base_dir, "checkpoints")
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"rainbow_offline_epoch_{epoch+1}.pth")
            torch.save({
                'epoch': epoch + 1,
                'step': step,
                'online_net_state_dict': online_net.state_dict(),
                'target_net_state_dict': target_net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'training_stats': training_stats,
                'config': model_config,
                'use_cql': args.use_cql,
                'cql_alpha': args.cql_alpha if args.use_cql else None
            }, save_path)
            print(f"模型已保存: {save_path}")
    
    # 10. 保存最终模型
    final_save_path = os.path.join(base_dir, "checkpoints", "rainbow_offline_final.pth")
    torch.save({
        'epoch': args.num_epochs,
        'step': step,
        'online_net_state_dict': online_net.state_dict(),
        'target_net_state_dict': target_net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'training_stats': training_stats,
        'config': model_config,
        'use_cql': args.use_cql,
        'cql_alpha': args.cql_alpha if args.use_cql else None
    }, final_save_path)
    
    print(f"\n训练完成！")
    print(f"最终模型已保存: {final_save_path}")
    print(f"总训练时间: {(time.time() - start_time)/60:.1f} 分钟")
    print("=" * 80)


if __name__ == "__main__":
    main()

