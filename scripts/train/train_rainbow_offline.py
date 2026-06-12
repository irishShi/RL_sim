"""
Rainbow DQN 离线训练脚本

从预收集的数据集进行离线学习，无需与环境交互
"""
import os
import sys
import json
import math
import shutil
from contextlib import nullcontext
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import yaml
from typing import Dict, Optional
import time
import argparse
from datetime import datetime
from tqdm import tqdm

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models import RainbowWithForecast, RainbowWithPhysicsRisk, ActionSpace
from utils import ReplayBuffer, project_distribution
from utils.dataset_loader import OfflineDataset
from utils.risk_labels import FutureRiskLabelConfig, attach_future_risk_labels
from models.physics_features import parse_horizons_ms

def _set_global_seed(seed: int) -> None:
    """尽量保证离线训练可复现（含数据划分与 val batch）。"""
    if seed is None:
        return
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _split_indices(num_samples: int, val_ratio: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """返回 train_idx, val_idx（不打乱原数据内容，只打乱索引）。"""
    rng = np.random.default_rng(seed)
    idx = np.arange(num_samples, dtype=np.int64)
    rng.shuffle(idx)
    val_n = int(round(num_samples * float(val_ratio)))
    val_n = max(0, min(num_samples, val_n))
    val_idx = idx[:val_n]
    train_idx = idx[val_n:]
    return train_idx, val_idx


def _sample_val_batch(dataset: Dict, val_idx: np.ndarray, batch_size: int, rng: np.random.Generator) -> Dict:
    """从原始 dataset 的验证切分中均匀采样一个 batch（不使用 PER）。"""
    if val_idx.size == 0:
        return {}
    pick = rng.choice(val_idx, size=min(batch_size, val_idx.size), replace=False)
    # OfflineDataset.load() 的字段约定
    batch = {
        'obs': dataset['obs'][pick],
        # dataset 使用复数键名：actions/rewards/dones/delta_targets
        'action': dataset['actions'][pick],
        'reward': dataset['rewards'][pick],
        'next_obs': dataset['next_obs'][pick],
        'done': dataset['dones'][pick],
        'delta_target': dataset.get('delta_targets', np.zeros(len(pick), dtype=np.float32))[pick],
        # 验证不需要重要性权重/indices/td_errors
        'weights': np.ones(len(pick), dtype=np.float32),
    }
    if 'future_risk_labels' in dataset:
        batch['future_risk_label'] = dataset['future_risk_labels'][pick]
        batch['future_risk_weight'] = dataset.get(
            'future_risk_weights',
            np.ones_like(dataset['future_risk_labels'], dtype=np.float32),
        )[pick]
    return batch


def _torch_load_checkpoint(resume_path: str, map_location: str):
    """
    兼容 PyTorch 2.6+ 的 torch.load 默认 weights_only=True 行为变化。

    我们的 checkpoint 是本脚本用 torch.save(dict) 保存的训练状态（非纯权重），
    在 PyTorch 2.6+ 需要显式设置 weights_only=False 才能正常反序列化。
    """
    import inspect
    sig = None
    try:
        sig = inspect.signature(torch.load)
    except Exception:
        sig = None

    if sig is not None and 'weights_only' in sig.parameters:
        return torch.load(resume_path, map_location=map_location, weights_only=False)
    return torch.load(resume_path, map_location=map_location)


def _forward_outputs(net, obs: torch.Tensor):
    """兼容 RainbowWithForecast 与 RainbowWithPhysicsRisk 的前向输出。"""
    out = net(obs)
    if isinstance(out, tuple) and len(out) == 3:
        dist, pred_delta, risk_logits = out
        return dist, pred_delta, risk_logits
    dist, pred_delta = out
    return dist, pred_delta, None


@torch.no_grad()
def eval_rainbow_loss_on_batch(
    batch: Dict,
    online_net: RainbowWithForecast,
    target_net: RainbowWithForecast,
    config: Dict,
    device: str,
    support: torch.Tensor,
) -> float:
    """只评估 Rainbow loss（KL），用于 val 曲线与早停。"""
    if not batch:
        return float("nan")
    online_net.eval()
    target_net.eval()

    obs = torch.as_tensor(batch['obs'], dtype=torch.float32, device=device)
    action = torch.as_tensor(batch['action'], dtype=torch.long, device=device)
    reward_n = torch.as_tensor(batch['reward'], dtype=torch.float32, device=device)
    next_obs = torch.as_tensor(batch['next_obs'], dtype=torch.float32, device=device)
    done = torch.as_tensor(batch['done'], dtype=torch.float32, device=device)
    weights = torch.as_tensor(batch.get('weights', np.ones(len(action), dtype=np.float32)), dtype=torch.float32, device=device)

    num_atoms = config['network']['rainbow']['num_atoms']
    v_min = config['network']['rainbow']['v_min']
    v_max = config['network']['rainbow']['v_max']
    gamma = config['training']['gamma']
    n_steps = config['training']['n_steps']

    support_expanded = support.view(1, 1, -1)
    dist, _, _ = _forward_outputs(online_net, obs)
    action_idx = action.unsqueeze(1).unsqueeze(2).expand(-1, 1, num_atoms)
    dist_a = dist.gather(1, action_idx).squeeze(1)

    next_dist_online, _, _ = _forward_outputs(online_net, next_obs)
    next_q = (next_dist_online * support_expanded).sum(dim=-1)
    next_action = next_q.argmax(dim=1, keepdim=True)

    next_dist_target, _, _ = _forward_outputs(target_net, next_obs)
    next_action_idx = next_action.unsqueeze(2).expand(-1, 1, num_atoms)
    target_dist_a = next_dist_target.gather(1, next_action_idx).squeeze(1)

    target_support = support.unsqueeze(0).expand(obs.size(0), -1)
    m = project_distribution(
        support, target_dist_a, target_support,
        v_min, v_max, gamma, reward_n, done, n_steps
    )

    log_p = torch.log(dist_a.clamp(min=1e-6))
    rainbow_loss_per_sample = -(m * log_p).sum(dim=1)
    return float((rainbow_loss_per_sample * weights).mean().item())


def train_step(batch: Dict, online_net: RainbowWithForecast, target_net: RainbowWithForecast,
               optimizer: optim.Optimizer, config: Dict, device: str = 'cpu',
               use_cql: bool = True, cql_alpha: float = 0.1,
               support: Optional[torch.Tensor] = None,
               use_amp: bool = False,
               scaler: Optional[torch.cuda.amp.GradScaler] = None,
               enable_physics_risk: bool = False) -> Dict:
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
    obs = torch.as_tensor(batch['obs'], dtype=torch.float32, device=device)  # [B, N, F]
    action = torch.as_tensor(batch['action'], dtype=torch.long, device=device)  # [B]
    reward_n = torch.as_tensor(batch['reward'], dtype=torch.float32, device=device)  # [B] - N-step reward
    next_obs = torch.as_tensor(batch['next_obs'], dtype=torch.float32, device=device)  # [B, N, F]
    done = torch.as_tensor(batch['done'], dtype=torch.float32, device=device)  # [B]
    delta_target = torch.as_tensor(batch['delta_target'], dtype=torch.float32, device=device)  # [B]
    weights = torch.as_tensor(batch['weights'], dtype=torch.float32, device=device)  # [B]
    
    B = obs.size(0)
    num_atoms = config['network']['rainbow']['num_atoms']
    num_actions = config['action_space']['num_actions']
    v_min = config['network']['rainbow']['v_min']
    v_max = config['network']['rainbow']['v_max']
    gamma = config['training']['gamma']
    n_steps = config['training']['n_steps']
    lambda_aux = config['training']['lambda_aux']
    risk_cfg = config.get('physics_risk', {})
    lambda_risk = float(config['training'].get('lambda_risk', risk_cfg.get('lambda_risk', 0.0)))
    lambda_phys = float(config['training'].get('lambda_phys', risk_cfg.get('lambda_phys', 0.0)))
    lambda_action_phys = float(
        config['training'].get('lambda_action_phys', risk_cfg.get('lambda_action_phys', 0.0))
    )
    
    # 支持外部缓存 support，避免每步重复创建
    if support is None:
        support = torch.linspace(v_min, v_max, num_atoms, device=device, dtype=torch.float32)
    support_expanded = support.view(1, 1, -1)  # [1, 1, num_atoms] 用于后续计算
    
    # 重置噪声（如果使用）
    online_net.train()
    if online_net.use_noisy:
        online_net.reset_noise()
    
    autocast_ctx = torch.autocast(device_type='cuda', dtype=torch.float16) if use_amp else nullcontext()
    with autocast_ctx:
        # 当前分布和预测（一次性前向传播）
        dist, pred_delta, risk_logits = _forward_outputs(online_net, obs)

        # 优化：使用更高效的索引操作
        action_idx = action.unsqueeze(1).unsqueeze(2).expand(-1, 1, num_atoms)  # [B, 1, num_atoms]
        dist_a = dist.gather(1, action_idx).squeeze(1)  # [B, num_atoms]

        # 目标分布（Double Q）- 不需要梯度但结果需参与损失计算，用 no_grad 而非 inference_mode
        with torch.no_grad():
            target_net.eval()
            next_dist_online, _, _ = _forward_outputs(online_net, next_obs)
            next_q = (next_dist_online * support_expanded).sum(dim=-1)  # [B, num_actions]
            next_action = next_q.argmax(dim=1, keepdim=True)  # [B, 1]

            next_dist_target, _, _ = _forward_outputs(target_net, next_obs)
            next_action_idx = next_action.unsqueeze(2).expand(-1, 1, num_atoms)  # [B, 1, num_atoms]
            target_dist_a = next_dist_target.gather(1, next_action_idx).squeeze(1)  # [B, num_atoms]

            target_support = support.unsqueeze(0).expand(B, -1)  # [B, num_atoms]
            m = project_distribution(
                support, target_dist_a, target_support,
                v_min, v_max, gamma, reward_n, done, n_steps
            )

        # Rainbow 损失（KL 散度）
        log_p = torch.log(dist_a.clamp(min=1e-6))  # 避免 log(0)
        rainbow_loss_per_sample = -(m * log_p).sum(dim=1)  # [B]
        rainbow_loss = (rainbow_loss_per_sample * weights).mean()

        # 预测损失
        forecast_loss = ((pred_delta - delta_target) ** 2)  # [B]
        forecast_loss = (forecast_loss * weights).mean()

        # CQL 正则化项（自适应缩放：CQL 始终为 Rainbow loss 的 cql_alpha 倍）
        # 标准 logsumexp CQL 有固有下界 log(num_actions)≈3.87，绝对值远大于
        # 收敛后的 Rainbow loss（~0.6），直接用 alpha 控制会淹没学习信号。
        # 改为自适应：cql_alpha 表示"CQL 占 Rainbow loss 的比例"。
        cql_loss = torch.tensor(0.0, device=device)
        if use_cql:
            q_all = online_net.rainbow_head.get_q_values(dist)  # [B, num_actions]
            q_data = q_all.gather(1, action.unsqueeze(1)).squeeze(1)  # [B]
            logsumexp_q = torch.logsumexp(q_all, dim=1)  # [B]
            cql_loss_raw = (logsumexp_q - q_data).mean()
            # 自适应缩放：使 CQL 贡献 = cql_alpha * rainbow_loss
            with torch.no_grad():
                adaptive_scale = rainbow_loss.detach() / (cql_loss_raw.detach().abs() + 1e-6)
            cql_loss = cql_alpha * adaptive_scale * cql_loss_raw

        risk_loss = torch.tensor(0.0, device=device)
        physics_loss = torch.tensor(0.0, device=device)
        action_physics_loss = torch.tensor(0.0, device=device)
        if enable_physics_risk and risk_logits is not None and 'future_risk_label' in batch:
            risk_label = torch.as_tensor(batch['future_risk_label'], dtype=torch.float32, device=device)
            risk_weight = torch.as_tensor(
                batch.get('future_risk_weight', np.ones_like(batch['future_risk_label'])),
                dtype=torch.float32,
                device=device,
            )
            action_risk_logits = risk_logits.gather(
                1,
                action.view(-1, 1, 1).expand(-1, 1, risk_logits.size(-1)),
            ).squeeze(1)
            bce = F.binary_cross_entropy_with_logits(
                action_risk_logits,
                risk_label,
                reduction='none',
            )
            risk_sample_weight = weights.view(-1, 1) * risk_weight
            risk_loss = (bce * risk_sample_weight).sum() / risk_sample_weight.sum().clamp(min=1.0)

            risk_prob = torch.sigmoid(risk_logits)
            if risk_prob.size(-1) >= 2:
                # 预测窗口越长，累计风险不应越低。
                horizon_mono = F.relu(risk_prob[:, :, :-1] - risk_prob[:, :, 1:]).mean()
            else:
                horizon_mono = torch.tensor(0.0, device=device)

            action_params = torch.as_tensor(
                ActionSpace().action_to_params,
                dtype=torch.float32,
                device=device,
            )
            hys = action_params[:, 0].view(1, -1, 1)
            ttt = action_params[:, 1].view(1, -1, 1)
            long_action = (ttt >= float(risk_cfg.get('long_ttt_ms', 300.0))).float()
            high_hys_action = (hys >= float(risk_cfg.get('high_hys_db', 4.0))).float()
            risky_action_mask = torch.clamp(long_action + high_hys_action, 0.0, 1.0)
            safer_action_mask = ((ttt <= float(risk_cfg.get('safe_ttt_ms', 150.0))) &
                                 (hys <= float(risk_cfg.get('safe_hys_db', 3.0)))).float()

            current_sinr_norm = obs[:, -1, 3]
            current_delta_norm = obs[:, -1, 2]
            sinr_db = current_sinr_norm * 30.0 - 10.0
            delta_db = current_delta_norm * 60.0 - 30.0
            late_context = (
                (sinr_db <= float(risk_cfg.get('low_sinr_db', -3.0))) &
                (delta_db >= float(risk_cfg.get('delta_advantage_db', 1.5)))
            ).float().view(-1, 1, 1)
            severe_late_context = (
                (sinr_db <= float(risk_cfg.get('severe_low_sinr_db', -5.0))) &
                (delta_db >= float(risk_cfg.get('stress_delta_advantage_db', 2.0)))
            ).float().view(-1, 1, 1)
            if torch.any(late_context > 0):
                risky_mean = (risk_prob * risky_action_mask).sum(dim=1) / risky_action_mask.sum(dim=1).clamp(min=1.0)
                safe_mean = (risk_prob * safer_action_mask).sum(dim=1) / safer_action_mask.sum(dim=1).clamp(min=1.0)
                action_mono = (F.relu(safe_mean - risky_mean + 0.05) * late_context.squeeze(1)).sum()
                action_mono = action_mono / late_context.squeeze(1).sum().clamp(min=1.0)
            else:
                action_mono = torch.tensor(0.0, device=device)
            physics_loss = horizon_mono + action_mono

            # 全动作物理伪标签：在低 SINR 且邻区已占优时，长 TTT / 高 Hys 动作应被显式标为高风险。
            # 第一版只监督离线数据中实际执行的动作，容易低估未执行候选动作的风险。
            if torch.any(severe_late_context > 0):
                risky_target = torch.clamp(
                    risky_action_mask * float(risk_cfg.get('risky_action_target', 0.95)) +
                    (1.0 - risky_action_mask) * float(risk_cfg.get('non_risky_action_target', 0.25)),
                    0.0,
                    1.0,
                ).expand_as(risk_logits)
                action_weight = (
                    risky_action_mask * float(risk_cfg.get('risky_action_weight', 1.0)) +
                    safer_action_mask * float(risk_cfg.get('safer_action_weight', 0.5)) +
                    (1.0 - torch.clamp(risky_action_mask + safer_action_mask, 0.0, 1.0)) *
                    float(risk_cfg.get('neutral_action_weight', 0.2))
                ).expand_as(risk_logits)
                pseudo_bce = F.binary_cross_entropy_with_logits(
                    risk_logits,
                    risky_target,
                    reduction='none',
                )
                context_weight = severe_late_context * weights.view(-1, 1, 1)
                pseudo_weight = context_weight * action_weight
                action_physics_loss = (
                    (pseudo_bce * pseudo_weight).sum() /
                    pseudo_weight.sum().clamp(min=1.0)
                )

        # 总损失
        loss = rainbow_loss + lambda_aux * forecast_loss + cql_loss
        if enable_physics_risk:
            loss = (
                loss
                + lambda_risk * risk_loss
                + lambda_phys * physics_loss
                + lambda_action_phys * action_physics_loss
            )

    # 反向传播
    optimizer.zero_grad(set_to_none=True)
    if use_amp:
        assert scaler is not None, "AMP 模式需要 GradScaler"
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(online_net.parameters(), config['training']['grad_clip'])
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        torch.nn.utils.clip_grad_norm_(online_net.parameters(), config['training']['grad_clip'])
        optimizer.step()
    
    # 计算 TD error（用于更新 PER 优先级）
    # 使用标量 Q 值差替代 KL 散度，正确反映预测误差
    with torch.no_grad():
        q_current = (dist_a.detach() * support).sum(dim=1)  # [B]
        q_target = (m * support).sum(dim=1)  # [B]
        td_errors = torch.abs(q_current - q_target).cpu().numpy() + 1e-6
    
    return {
        'loss': loss.item(),
        'rainbow_loss': rainbow_loss.item(),
        'forecast_loss': forecast_loss.item(),
        'risk_loss': risk_loss.item(),
        'physics_loss': physics_loss.item(),
        'action_physics_loss': action_physics_loss.item(),
        'cql_loss': cql_loss.item() if use_cql else 0.0,
        'td_errors': td_errors
    }


@torch.no_grad()
def diagnose_policy(online_net: RainbowWithForecast, buffer: ReplayBuffer,
                    device: str, sample_size: int = 2000) -> Dict:
    """
    策略健康度诊断：检测策略塌缩和 Q 值质量。

    Args:
        online_net: 当前在线网络
        buffer: 经验缓冲区（用于采样观测）
        device: 设备
        sample_size: 诊断采样数量

    Returns:
        诊断指标字典
    """
    online_net.eval()
    batch = buffer.sample(min(sample_size, buffer.size), beta=1.0)
    if batch is None:
        return {}

    obs = torch.as_tensor(batch['obs'], dtype=torch.float32, device=device)
    q_values = online_net.get_q_values(obs)  # [B, num_actions]

    # 1. Q 值基本统计
    q_mean = q_values.mean().item()
    q_std = q_values.std().item()
    q_min = q_values.min().item()
    q_max = q_values.max().item()

    # 2. 动作区分度：每个样本的 Q 值跨动作方差
    q_var_per_sample = q_values.var(dim=1)  # [B]
    q_action_var_mean = q_var_per_sample.mean().item()
    # best-second gap
    q_sorted = q_values.sort(dim=1, descending=True)[0]
    best_second_gap = (q_sorted[:, 0] - q_sorted[:, 1]).mean().item()

    # 3. 最优动作分布
    best_actions = q_values.argmax(dim=1).cpu().numpy()
    action_counts = np.bincount(best_actions, minlength=q_values.size(1))
    num_samples = len(best_actions)

    # 动作分布熵
    probs = action_counts / num_samples
    probs_nonzero = probs[probs > 0]
    action_entropy = -float(np.sum(probs_nonzero * np.log(probs_nonzero)))
    max_entropy = float(np.log(q_values.size(1)))

    # top-1 集中度
    top1_ratio = float(action_counts.max()) / num_samples
    # top-3 覆盖
    top3_ratio = float(np.sort(action_counts)[-3:].sum()) / num_samples

    # top-5 动作详情
    action_space = ActionSpace()
    top5_idx = np.argsort(action_counts)[::-1][:5]
    top5_details = []
    for idx in top5_idx:
        hys, ttt = action_space.action_to_hys_ttt(int(idx))
        top5_details.append({
            'action': int(idx),
            'hys': float(hys), 'ttt': float(ttt),
            'count': int(action_counts[idx]),
            'ratio': float(action_counts[idx] / num_samples)
        })

    online_net.train()
    return {
        'q_mean': round(q_mean, 4),
        'q_std': round(q_std, 4),
        'q_min': round(q_min, 4),
        'q_max': round(q_max, 4),
        'q_action_var_mean': round(q_action_var_mean, 6),
        'best_second_gap': round(best_second_gap, 6),
        'action_entropy': round(action_entropy, 4),
        'max_entropy': round(max_entropy, 4),
        'entropy_ratio': round(action_entropy / max_entropy, 4),
        'top1_action_ratio': round(top1_ratio, 4),
        'top3_action_ratio': round(top3_ratio, 4),
        'top5_actions': top5_details,
    }


def main():
    """主训练函数"""
    parser = argparse.ArgumentParser(description='Rainbow DQN 离线训练')
    parser.add_argument('--dataset_path', type=str, default='data/datasets/offline_dataset.npz',
                       help='数据集路径')
    parser.add_argument('--use_cql', dest='use_cql', action='store_true', help='启用 CQL 正则化（默认关闭）')
    parser.add_argument('--no_cql', dest='use_cql', action='store_false', help='关闭 CQL 正则化')
    parser.add_argument('--cql_alpha', type=float, default=0.1, help='CQL 正则化系数')
    parser.add_argument('--num_epochs', type=int, default=300, help='最大训练轮数')
    parser.add_argument('--samples_per_epoch', type=int, default=10000, help='每轮采样数')
    parser.add_argument('--batch_size', type=int, default=None, help='Batch大小（覆盖配置）')
    parser.add_argument('--amp', dest='amp', action='store_true', help='启用混合精度训练（GPU建议开启）')
    parser.add_argument('--no_amp', dest='amp', action='store_false', help='关闭混合精度训练')
    parser.add_argument('--lr_min_ratio', type=float, default=0.05,
                       help='Cosine衰减最低学习率比例（相对初始lr）')
    parser.add_argument('--early_stop_patience', type=int, default=30,
                       help='早停耐心（连续 N 个 epoch Rainbow loss 未改善即停止；默认 30，0=禁用）')
    parser.add_argument('--early_stop_min_delta', type=float, default=1e-3,
                       help='早停最小改善阈值：相对模式下为比例（默认 1e-3 即 0.1%%），绝对模式下为绝对值')
    parser.add_argument('--early_stop_relative', dest='early_stop_relative', action='store_true',
                       help='使用相对阈值（min_delta 为 best loss 的比例；默认启用）')
    parser.add_argument('--early_stop_absolute', dest='early_stop_relative', action='store_false',
                       help='使用绝对阈值（min_delta 为绝对 loss 差值）')
    parser.add_argument('--val_ratio', type=float, default=0.05,
                       help='验证集比例（默认 0.05；0=不做验证）')
    parser.add_argument('--val_batch_size', type=int, default=2048,
                       help='每次验证用的 batch size（默认 2048，越大越平滑但更慢）')
    parser.add_argument('--seed', type=int, default=2026, help='随机种子（用于 val 划分与采样）')
    parser.add_argument('--resume', type=str, default=None,
                       help='断点续训 checkpoint 路径（例如 checkpoints/rainbow_offline_best.pth）')
    parser.add_argument('--early_stop_on', type=str, default='train',
                       choices=['train', 'val'],
                       help='早停/保存 best 依据：train 或 val（默认 train；val_ratio=0 时会自动退化为 train）')
    parser.add_argument('--train_log_interval', type=int, default=10,
                       help='控制台详细输出与诊断 JSON 写入间隔（epoch），建议 5 或 10')
    parser.add_argument('--run_dir', type=str, default=None,
                       help='实验输出目录；默认写入 experiments/runs/<timestamp>_rainbow_offline_seed<seed>')
    parser.add_argument('--enable_physics_risk', action='store_true',
                       help='启用物理规则约束的未来风险预测头')
    parser.add_argument('--risk_horizons_ms', type=str, default='100,200,300',
                       help='风险预测窗口，逗号分隔，单位 ms')
    parser.add_argument('--lambda_risk', type=float, default=0.2,
                       help='未来风险监督损失权重')
    parser.add_argument('--lambda_phys', type=float, default=0.05,
                       help='物理一致性约束损失权重')
    parser.add_argument('--lambda_action_phys', type=float, default=0.1,
                       help='全动作物理伪标签约束损失权重')
    parser.add_argument('--risk_short_window_steps', type=int, default=5,
                       help='物理派生特征短窗步数')
    
    parser.set_defaults(use_cql=False)
    parser.set_defaults(amp=True)
    parser.set_defaults(early_stop_relative=True)
    args = parser.parse_args()
    _set_global_seed(args.seed)
    
    print("=" * 80)
    print("Rainbow DQN 离线训练脚本")
    print("=" * 80)
    
    # 1. 加载配置
    base_dir = PROJECT_ROOT
    env_config_path = os.path.join(base_dir, "configs", "default_env_config.yaml")
    model_config_path = os.path.join(base_dir, "configs", "model_config.yaml")
    
    with open(model_config_path, 'r', encoding='utf-8') as f:
        model_config = yaml.safe_load(f)
    risk_horizons_ms = parse_horizons_ms(args.risk_horizons_ms)
    model_config.setdefault('physics_risk', {})
    model_config['physics_risk'].update({
        'enabled': bool(args.enable_physics_risk),
        'horizons_ms': risk_horizons_ms,
        'num_horizons': len(risk_horizons_ms),
        'short_window_steps': int(args.risk_short_window_steps),
        'lambda_risk': float(args.lambda_risk),
        'lambda_phys': float(args.lambda_phys),
        'lambda_action_phys': float(args.lambda_action_phys),
        'low_sinr_db': -3.0,
        'severe_low_sinr_db': -5.0,
        'delta_advantage_db': 1.5,
        'stress_delta_advantage_db': 2.0,
        'long_ttt_ms': 300.0,
        'high_hys_db': 4.0,
        'safe_ttt_ms': 150.0,
        'safe_hys_db': 3.0,
        'risky_action_target': 0.95,
        'non_risky_action_target': 0.25,
        'risky_action_weight': 1.0,
        'safer_action_weight': 0.5,
        'neutral_action_weight': 0.2,
    })
    model_config['training']['lambda_risk'] = float(args.lambda_risk)
    model_config['training']['lambda_phys'] = float(args.lambda_phys)
    model_config['training']['lambda_action_phys'] = float(args.lambda_action_phys)
    
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

    # 3.1 划分 train/val（val 只用于诊断/早停，不参与 PER）
    val_ratio = float(args.val_ratio)
    train_idx = None
    val_idx = None
    if val_ratio > 0.0:
        train_idx, val_idx = _split_indices(num_samples, val_ratio=val_ratio, seed=args.seed)
        print(f"数据划分: train={train_idx.size}, val={val_idx.size} (val_ratio={val_ratio})")
    else:
        print("数据划分: 未启用验证集（val_ratio=0）")
    
    # 4. 创建动作空间
    action_space = ActionSpace()
    hys_set = model_config['action_space']['hys_set']
    ttt_set = model_config['action_space']['ttt_set']
    num_actions = len(hys_set) * len(ttt_set)
    
    print(f"动作空间: {num_actions} 个动作")
    if args.enable_physics_risk:
        label_cfg = FutureRiskLabelConfig(horizons_ms=tuple(risk_horizons_ms))
        dataset = attach_future_risk_labels(dataset, action_space.action_to_params, label_cfg)
        print(f"已生成未来风险标签: horizons={risk_horizons_ms} ms, "
              f"shape={dataset['future_risk_labels'].shape}")
    
    # 5. 创建模型（离线学习不使用 NoisyNet）
    model_cls = RainbowWithPhysicsRisk if args.enable_physics_risk else RainbowWithForecast
    common_model_kwargs = dict(
        obs_dim=obs_dim,
        num_actions=num_actions,
        n_steps=window_size,
        feature_hidden=model_config['network']['shared']['hidden_dim'],
        encoder_hidden=model_config['network']['encoder']['hidden_dim'],
        num_atoms=model_config['network']['rainbow']['num_atoms'],
        v_min=model_config['network']['rainbow']['v_min'],
        v_max=model_config['network']['rainbow']['v_max'],
        use_noisy=False,
    )
    if args.enable_physics_risk:
        common_model_kwargs.update({
            'num_risk_horizons': len(risk_horizons_ms),
            'risk_hidden_dim': int(model_config.get('physics_risk', {}).get('risk_hidden_dim', 256)),
            'physics_short_window_steps': int(args.risk_short_window_steps),
            'physics_delta_t_s': 0.05,
            'physics_l3_alpha': 0.7,
        })

    online_net = model_cls(**common_model_kwargs).to(device)
    target_net = model_cls(**common_model_kwargs).to(device)
    
    target_net.load_state_dict(online_net.state_dict())
    target_net = target_net.to(device)
    
    print(f"\n模型参数总数: {sum(p.numel() for p in online_net.parameters()):,}")
    print(f"使用 NoisyNet: {online_net.use_noisy}")
    print(f"使用 CQL: {args.use_cql}")
    print(f"使用 AMP: {args.amp and device == 'cuda'}")
    print(f"使用 Physics Risk Head: {args.enable_physics_risk}")
    if args.use_cql:
        print(f"CQL Alpha: {args.cql_alpha}")
    
    # 6. 创建优化器和缓冲区
    base_lr = model_config['training']['learning_rate']
    optimizer = optim.Adam(online_net.parameters(), lr=base_lr)
    
    lr_min = base_lr * args.lr_min_ratio
    scheduler = CosineAnnealingLR(optimizer, T_max=args.num_epochs, eta_min=lr_min)
    
    use_amp = bool(args.amp and device == 'cuda')
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    support = torch.linspace(
        model_config['network']['rainbow']['v_min'],
        model_config['network']['rainbow']['v_max'],
        model_config['network']['rainbow']['num_atoms'],
        device=device,
        dtype=torch.float32
    )
    
    # 从数据集填充缓冲区
    buffer = ReplayBuffer(
        capacity=model_config['training']['replay_buffer_size'],
        obs_window_size=window_size,
        obs_dim=obs_dim,
        per_alpha=model_config['training']['per_alpha'],
        num_risk_horizons=len(risk_horizons_ms) if args.enable_physics_risk else 0,
    )
    
    print(f"\n填充经验回放缓冲区...")
    if train_idx is not None:
        # 仅将 train 子集放入回放，避免“训练看见验证集”
        train_dataset = {k: (v[train_idx] if isinstance(v, np.ndarray) and v.shape[0] == num_samples else v)
                         for k, v in dataset.items()}
        buffer.load_from_dataset(train_dataset)
    else:
        buffer.load_from_dataset(dataset)
    
    print(f"缓冲区大小: {buffer.size}")
    
    # 7. 训练参数
    batch_size = args.batch_size if args.batch_size is not None else model_config['training']['batch_size']
    target_update_freq = model_config['training']['target_update_freq']
    target_tau = model_config['training'].get('target_update_tau', 0.005)
    
    # 8. 训练统计
    training_stats = {
        'losses': [],
        'rainbow_losses': [],
        'forecast_losses': [],
        'risk_losses': [],
        'physics_losses': [],
        'action_physics_losses': [],
        'cql_losses': []
    }

    log_interval = max(1, int(args.train_log_interval))
    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    default_run_dir = os.path.join("experiments", "runs", f"{run_ts}_rainbow_offline_seed{args.seed}")
    run_dir = args.run_dir or default_run_dir
    run_dir = os.path.join(base_dir, run_dir) if not os.path.isabs(run_dir) else run_dir
    checkpoints_dir = os.path.join(run_dir, "checkpoints")
    diagnostics_dir = os.path.join(run_dir, "logs")
    metrics_dir = os.path.join(run_dir, "metrics")
    config_snapshot_dir = os.path.join(run_dir, "config")
    for path in (checkpoints_dir, diagnostics_dir, metrics_dir, config_snapshot_dir):
        os.makedirs(path, exist_ok=True)
    for src_path in (env_config_path, model_config_path):
        shutil.copy2(src_path, os.path.join(config_snapshot_dir, os.path.basename(src_path)))
    diagnostics_path = os.path.join(diagnostics_dir, f"training_diagnostics_{run_ts}.json")
    diagnostics_records: list = []

    # 单次运行元信息（超参 + 配置摘要，便于复现实验）
    run_meta = {
        'run_started_at': datetime.now().isoformat(timespec='seconds'),
        'run_dir': os.path.abspath(run_dir),
        'diagnostics_path': diagnostics_path,
        'cli': {
            'dataset_path': args.dataset_path,
            'dataset_path_resolved': os.path.abspath(dataset_path),
            'use_cql': args.use_cql,
            'cql_alpha': args.cql_alpha,
            'num_epochs': args.num_epochs,
            'samples_per_epoch': args.samples_per_epoch,
            'batch_size': batch_size,
            'amp': args.amp,
            'lr_min_ratio': args.lr_min_ratio,
            'early_stop_patience': args.early_stop_patience,
            'early_stop_min_delta': args.early_stop_min_delta,
            'early_stop_relative': args.early_stop_relative,
            'early_stop_on': args.early_stop_on,
            'val_ratio': val_ratio,
            'val_batch_size': args.val_batch_size,
            'seed': args.seed,
            'resume': args.resume,
            'train_log_interval': log_interval,
            'enable_physics_risk': bool(args.enable_physics_risk),
            'risk_horizons_ms': risk_horizons_ms,
            'lambda_risk': float(args.lambda_risk),
            'lambda_phys': float(args.lambda_phys),
            'lambda_action_phys': float(args.lambda_action_phys),
            'risk_short_window_steps': int(args.risk_short_window_steps),
        },
        'device': device,
        'dataset_num_samples': int(num_samples),
        'model': {
            'obs_dim': obs_dim,
            'window_size': window_size,
            'num_actions': num_actions,
            'param_count': int(sum(p.numel() for p in online_net.parameters())),
            'use_noisy': online_net.use_noisy,
            'model_class': online_net.__class__.__name__,
        },
        'training_yaml': {
            'learning_rate': model_config['training']['learning_rate'],
            'gamma': model_config['training']['gamma'],
            'n_steps': model_config['training']['n_steps'],
            'lambda_aux': model_config['training']['lambda_aux'],
            'lambda_risk': model_config['training'].get('lambda_risk', 0.0),
            'lambda_phys': model_config['training'].get('lambda_phys', 0.0),
            'lambda_action_phys': model_config['training'].get('lambda_action_phys', 0.0),
            'grad_clip': model_config['training']['grad_clip'],
            'replay_buffer_size': model_config['training']['replay_buffer_size'],
            'per_alpha': model_config['training']['per_alpha'],
            'per_beta_increment': model_config['training']['per_beta_increment'],
            'target_update_freq': model_config['training']['target_update_freq'],
            'target_update_tau': target_tau,
        },
        'physics_risk': model_config.get('physics_risk', {}),
        'rainbow': {
            'num_atoms': model_config['network']['rainbow']['num_atoms'],
            'v_min': model_config['network']['rainbow']['v_min'],
            'v_max': model_config['network']['rainbow']['v_max'],
        },
        'encoder': {
            'hidden_dim': model_config['network']['encoder']['hidden_dim'],
        },
        'shared': {
            'hidden_dim': model_config['network']['shared']['hidden_dim'],
        },
    }
    if device == 'cuda':
        run_meta['gpu_name'] = torch.cuda.get_device_name(0)
        run_meta['gpu_total_memory_gb'] = round(
            torch.cuda.get_device_properties(0).total_memory / 1024 ** 3, 2
        )

    def _write_diagnostics_file() -> None:
        payload = {'meta': run_meta, 'records': diagnostics_records}
        with open(diagnostics_path, 'w', encoding='utf-8') as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    # 9. 训练循环
    print("\n开始离线训练...")
    print(f"学习率: {base_lr} -> {lr_min} (cosine衰减)")
    delta_mode = "相对" if args.early_stop_relative else "绝对"
    print(f"最大Epoch: {args.num_epochs}, 早停耐心: {args.early_stop_patience}, "
          f"早停依据: {args.early_stop_on}, 阈值模式: {delta_mode}({args.early_stop_min_delta})")
    print(f"详细日志间隔: 每 {log_interval} 个 epoch（控制台 + 诊断 JSON）")
    print(f"诊断文件: {diagnostics_path}")
    print("-" * 80)
    
    start_time = time.time()
    step = 0
    best_rainbow_loss = float('inf')
    best_epoch = 0
    patience_counter = 0
    actual_epochs = 0
    stopped_by_early_stop = False
    best_metric_name = None

    # 断点续训
    if args.resume:
        resume_path = os.path.join(base_dir, args.resume) if not os.path.isabs(args.resume) else args.resume
        if os.path.exists(resume_path):
            ckpt = _torch_load_checkpoint(resume_path, map_location=device)
            ckpt_physics_risk = bool(ckpt.get('physics_risk_enabled', False))
            warm_start_only = bool(args.enable_physics_risk and not ckpt_physics_risk)
            if warm_start_only:
                missing, unexpected = online_net.load_state_dict(ckpt['online_net_state_dict'], strict=False)
                target_net.load_state_dict(online_net.state_dict())
                print(
                    f"\n已从旧 Rainbow checkpoint warm start 主干权重: {os.path.abspath(resume_path)}\n"
                    f"  新风险头随机初始化；跳过 optimizer/scheduler 恢复。\n"
                    f"  missing={len(missing)}, unexpected={len(unexpected)}"
                )
            else:
                online_net.load_state_dict(ckpt['online_net_state_dict'])
                target_net.load_state_dict(ckpt.get('target_net_state_dict', ckpt['online_net_state_dict']))
                optimizer.load_state_dict(ckpt['optimizer_state_dict'])
                step = int(ckpt.get('step', 0))
                resume_epoch = int(ckpt.get('epoch', 0))
                best_rainbow_loss = float(ckpt.get('best_rainbow_loss', best_rainbow_loss))
                best_epoch = int(ckpt.get('best_epoch', best_epoch)) if 'best_epoch' in ckpt else best_epoch
                # scheduler：简单处理为“按已完成 epoch 先 step 到位”，避免 lr 直接回到初始
                for _ in range(max(0, resume_epoch)):
                    scheduler.step()
                print(f"\n已从 checkpoint 恢复: {os.path.abspath(resume_path)} (epoch={resume_epoch}, step={step})")
        else:
            print(f"\n警告：resume 路径不存在，将从头训练: {resume_path}")

    val_rng = np.random.default_rng(args.seed + 17)

    num_batches = args.samples_per_epoch // batch_size

    for block_start in range(0, args.num_epochs, log_interval):
        block_end = min(block_start + log_interval, args.num_epochs)
        block_desc = f"Epoch {block_start + 1}-{block_end}/{args.num_epochs}"
        for epoch in tqdm(
            range(block_start, block_end),
            desc=block_desc,
            leave=True,
            unit="epoch",
        ):
            epoch_losses = []
            epoch_rainbow_losses = []
            epoch_forecast_losses = []
            epoch_risk_losses = []
            epoch_physics_losses = []
            epoch_action_physics_losses = []
            epoch_cql_losses = []

            for batch_idx in range(num_batches):
                beta = min(1.0, 0.4 + step * model_config['training']['per_beta_increment'])
                batch = buffer.sample(batch_size, beta=beta)

                if batch is not None:
                    train_info = train_step(
                        batch, online_net, target_net, optimizer,
                        model_config, device,
                        use_cql=args.use_cql,
                        cql_alpha=args.cql_alpha,
                        support=support,
                        use_amp=use_amp,
                        scaler=scaler,
                        enable_physics_risk=args.enable_physics_risk,
                    )

                    buffer.update_priorities(batch['indices'], train_info['td_errors'])

                    epoch_losses.append(train_info['loss'])
                    epoch_rainbow_losses.append(train_info['rainbow_loss'])
                    epoch_forecast_losses.append(train_info['forecast_loss'])
                    epoch_risk_losses.append(train_info.get('risk_loss', 0.0))
                    epoch_physics_losses.append(train_info.get('physics_loss', 0.0))
                    epoch_action_physics_losses.append(train_info.get('action_physics_loss', 0.0))
                    if args.use_cql:
                        epoch_cql_losses.append(train_info['cql_loss'])

                    step += 1

                    if step % target_update_freq == 0:
                        with torch.no_grad():
                            for p_online, p_target in zip(online_net.parameters(), target_net.parameters()):
                                p_target.data.mul_(1.0 - target_tau).add_(p_online.data, alpha=target_tau)
                        if device == 'cuda' and step % (target_update_freq * 10) == 0:
                            torch.cuda.empty_cache()

            scheduler.step()
            actual_epochs = epoch + 1

            if epoch_losses:
                training_stats['losses'].extend(epoch_losses)
                training_stats['rainbow_losses'].extend(epoch_rainbow_losses)
                training_stats['forecast_losses'].extend(epoch_forecast_losses)
                training_stats['risk_losses'].extend(epoch_risk_losses)
                training_stats['physics_losses'].extend(epoch_physics_losses)
                training_stats['action_physics_losses'].extend(epoch_action_physics_losses)
                if args.use_cql:
                    training_stats['cql_losses'].extend(epoch_cql_losses)

                avg_loss = np.mean(epoch_losses)
                avg_rainbow = np.mean(epoch_rainbow_losses)
                avg_forecast = np.mean(epoch_forecast_losses)
                avg_risk = float(np.mean(epoch_risk_losses)) if epoch_risk_losses else 0.0
                avg_physics = float(np.mean(epoch_physics_losses)) if epoch_physics_losses else 0.0
                avg_action_physics = (
                    float(np.mean(epoch_action_physics_losses)) if epoch_action_physics_losses else 0.0
                )
                avg_cql = float(np.mean(epoch_cql_losses)) if (args.use_cql and epoch_cql_losses) else None
                current_lr = optimizer.param_groups[0]['lr']

                elapsed = time.time() - start_time
                gpu_mem_gb = None
                gpu_mem_info = ""
                if device == 'cuda':
                    gpu_mem_gb = round(torch.cuda.memory_allocated(0) / 1024**3, 4)
                    gpu_mem_info = f" | GPU: {gpu_mem_gb:.2f}GB"

                # 计算 val_rainbow_loss（可选）
                val_rainbow = None
                if val_idx is not None and val_idx.size > 0:
                    val_bs = int(args.val_batch_size)
                    val_batch = _sample_val_batch(dataset, val_idx, batch_size=val_bs, rng=val_rng)
                    val_rainbow = eval_rainbow_loss_on_batch(
                        val_batch, online_net, target_net, model_config, device, support
                    )

                # 早停判断 + 保存最佳模型：默认以 train 为准；若选 val 但未启用，则退化为 train
                metric_name = args.early_stop_on
                if metric_name == 'val' and (val_rainbow is None or not np.isfinite(val_rainbow)):
                    metric_name = 'train'
                metric_value = float(val_rainbow) if metric_name == 'val' else float(avg_rainbow)

                # 相对阈值：min_delta 为 best loss 的比例；绝对阈值：min_delta 为绝对差值
                min_delta_raw = float(args.early_stop_min_delta)
                if args.early_stop_relative and best_rainbow_loss < float('inf'):
                    min_delta = abs(best_rainbow_loss) * min_delta_raw
                else:
                    min_delta = min_delta_raw
                if metric_value < best_rainbow_loss - min_delta:
                    best_rainbow_loss = metric_value
                    patience_counter = 0
                    best_epoch = epoch + 1
                    best_metric_name = metric_name
                    best_save_dir = checkpoints_dir
                    os.makedirs(best_save_dir, exist_ok=True)
                    best_save_path = os.path.join(best_save_dir, "rainbow_offline_best.pth")
                    torch.save({
                        'epoch': epoch + 1,
                        'step': step,
                        'online_net_state_dict': online_net.state_dict(),
                        'target_net_state_dict': target_net.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'training_stats': training_stats,
                        'config': model_config,
                        'model_class': online_net.__class__.__name__,
                        'physics_risk_enabled': bool(args.enable_physics_risk),
                        'use_cql': args.use_cql,
                        'cql_alpha': args.cql_alpha if args.use_cql else None,
                        'best_rainbow_loss': best_rainbow_loss,
                        'best_epoch': best_epoch,
                        'best_metric_name': best_metric_name,
                    }, best_save_path)
                else:
                    patience_counter += 1

                trigger_early_stop = (
                    args.early_stop_patience > 0 and patience_counter >= args.early_stop_patience
                )
                is_last_epoch = (epoch + 1) == args.num_epochs
                should_log = (
                    (epoch + 1) % log_interval == 0 or trigger_early_stop or is_last_epoch
                )

                if should_log:
                    diag = diagnose_policy(online_net, buffer, device)
                    record = {
                        'epoch': epoch + 1,
                        'step': step,
                        'avg_loss': round(float(avg_loss), 6),
                        'avg_rainbow_loss': round(float(avg_rainbow), 6),
                        'val_rainbow_loss': round(float(val_rainbow), 6) if (val_rainbow is not None and np.isfinite(val_rainbow)) else None,
                        'avg_forecast_loss': round(float(avg_forecast), 6),
                        'avg_risk_loss': round(float(avg_risk), 6) if args.enable_physics_risk else None,
                        'avg_physics_loss': round(float(avg_physics), 6) if args.enable_physics_risk else None,
                        'avg_action_physics_loss': (
                            round(float(avg_action_physics), 6) if args.enable_physics_risk else None
                        ),
                        'avg_cql_loss': round(avg_cql, 6) if avg_cql is not None else None,
                        'lr': float(current_lr),
                        'elapsed_sec': round(elapsed, 2),
                        'elapsed_min': round(elapsed / 60.0, 4),
                        'gpu_mem_gb': gpu_mem_gb,
                        'best_rainbow_loss': round(float(best_rainbow_loss), 6) if best_rainbow_loss < float('inf') else None,
                        'best_epoch': int(best_epoch) if best_epoch > 0 else None,
                        'best_metric_name': best_metric_name,
                        'patience_counter': int(patience_counter),
                        'batches_per_epoch': int(num_batches),
                        'policy_diagnostics': diag if diag else {},
                    }
                    diagnostics_records.append(record)
                    _write_diagnostics_file()

                    print_str = (
                        f"Epoch {epoch+1:4d}/{args.num_epochs} | "
                        f"Loss: {avg_loss:.4f} (Rainbow: {avg_rainbow:.4f}, Forecast: {avg_forecast:.4f})"
                    )
                    if args.enable_physics_risk:
                        print_str += (
                            f" | Risk: {avg_risk:.4f}, Phys: {avg_physics:.4f}, "
                            f"ActionPhys: {avg_action_physics:.4f}"
                        )
                    if val_rainbow is not None and np.isfinite(val_rainbow):
                        print_str += f" | ValRainbow: {val_rainbow:.4f}"
                    if avg_cql is not None:
                        print_str += f", CQL: {avg_cql:.4f}"
                    print_str += f" | lr: {current_lr:.2e} | Time: {elapsed/60:.1f}m{gpu_mem_info}"
                    print(print_str)
                    if diag:
                        print(
                            f"  诊断: Q=[{diag['q_min']:.2f}, {diag['q_max']:.2f}] "
                            f"action_var={diag['q_action_var_mean']:.4f} "
                            f"gap={diag['best_second_gap']:.4f} "
                            f"entropy={diag['action_entropy']:.2f}/{diag['max_entropy']:.2f} "
                            f"top1={diag['top1_action_ratio']:.1%}"
                        )

                if trigger_early_stop:
                    stopped_by_early_stop = True
                    metric_disp = best_metric_name or ('val' if val_ratio > 0 else 'train')
                    print(f"\n早停触发：{metric_disp} Rainbow loss 连续 {args.early_stop_patience} 个 epoch 未改善 "
                          f"(best={best_rainbow_loss:.4f}, epoch {best_epoch})")
                    break

            # 定期保存模型
            if (epoch + 1) % 10 == 0:
                save_dir = checkpoints_dir
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
                    'model_class': online_net.__class__.__name__,
                    'physics_risk_enabled': bool(args.enable_physics_risk),
                    'use_cql': args.use_cql,
                    'cql_alpha': args.cql_alpha if args.use_cql else None
                }, save_path)
                print(f"模型已保存: {save_path}")

        if stopped_by_early_stop:
            break

    # 10. 最终模型 = 最佳模型（而非最后一个 epoch）
    final_save_path = os.path.join(checkpoints_dir, "rainbow_offline_final.pth")
    best_ckpt_path = os.path.join(checkpoints_dir, "rainbow_offline_best.pth")
    if os.path.exists(best_ckpt_path):
        shutil.copy2(best_ckpt_path, final_save_path)
    else:
        torch.save({
            'epoch': actual_epochs,
            'step': step,
            'online_net_state_dict': online_net.state_dict(),
            'target_net_state_dict': target_net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'training_stats': training_stats,
            'config': model_config,
            'model_class': online_net.__class__.__name__,
            'physics_risk_enabled': bool(args.enable_physics_risk),
            'use_cql': args.use_cql,
            'cql_alpha': args.cql_alpha if args.use_cql else None
        }, final_save_path)

    print(f"\n训练完成！实际训练 {actual_epochs}/{args.num_epochs} 个 epoch")
    if os.path.exists(best_ckpt_path) and best_rainbow_loss < float('inf') and best_epoch > 0:
        metric_disp = best_metric_name or ('val' if float(args.val_ratio) > 0 else 'train')
        print(
            f"★ 最佳模型: epoch {best_epoch}, {metric_disp}_rainbow_loss={best_rainbow_loss:.4f} | "
            f"{os.path.abspath(best_ckpt_path)}"
        )
    else:
        print("（未产生 best checkpoint，最终模型为最后一轮权重）")
    print(f"最终模型已保存: {os.path.abspath(final_save_path)}")
    print(f"总训练时间: {(time.time() - start_time)/60:.1f} 分钟")

    run_meta['run_finished_at'] = datetime.now().isoformat(timespec='seconds')
    run_meta['actual_epochs'] = int(actual_epochs)
    run_meta['stopped_by_early_stop'] = stopped_by_early_stop
    run_meta['best_rainbow_loss_final'] = (
        round(float(best_rainbow_loss), 6) if best_rainbow_loss < float('inf') else None
    )
    run_meta['best_epoch_final'] = int(best_epoch) if best_epoch > 0 else None
    run_meta['best_metric_name_final'] = best_metric_name
    run_meta['total_train_sec'] = round(time.time() - start_time, 2)
    if diagnostics_records:
        _write_diagnostics_file()
        print(f"诊断日志已保存: {diagnostics_path}（共 {len(diagnostics_records)} 条记录）")
    print("=" * 80)


if __name__ == "__main__":
    main()
