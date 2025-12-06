"""测试 Rainbow DQN 模型是否正常工作"""
try:
    import torch
except ImportError:
    raise ImportError(
        "PyTorch 未安装。请运行以下命令安装：\n"
        "  pip install torch>=2.0.0\n"
        "或者安装所有依赖：\n"
        "  pip install -r requirements.txt"
    )

import numpy as np
import yaml
import os
from models import RainbowWithForecast, ActionSpace, ObservationWindow

# 配置 matplotlib 支持中文显示
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端


def test_model():
    """测试模型前向传播"""
    print("=" * 60)
    print("测试 Rainbow DQN 模型")
    print("=" * 60)
    
    # 1. 加载配置
    base_dir = os.path.dirname(__file__)
    config_path = os.path.join(base_dir, "configs", "model_config.yaml")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 2. 创建动作空间
    action_space = ActionSpace()
    print(f"\n【动作空间】")
    print(f"  Hys 集合: {action_space.hys_set}")
    print(f"  TTT 集合: {action_space.ttt_set}")
    print(f"  总动作数: {action_space.num_actions}")
    
    # 测试动作转换
    test_action = 10
    hys, ttt = action_space.action_to_hys_ttt(test_action)
    print(f"\n  测试动作 {test_action} -> Hys={hys:.1f} dB, TTT={ttt:.0f} ms")
    
    # 3. 创建模型
    obs_dim = config['observation']['obs_dim']
    window_size = config['observation']['window_size']
    num_actions = config['action_space']['num_actions']
    
    model = RainbowWithForecast(
        obs_dim=obs_dim,
        num_actions=num_actions,
        n_steps=window_size,
        feature_hidden=config['network']['shared']['hidden_dim'],
        encoder_hidden=config['network']['encoder']['hidden_dim'],
        num_atoms=config['network']['rainbow']['num_atoms'],
        v_min=config['network']['rainbow']['v_min'],
        v_max=config['network']['rainbow']['v_max']
    )
    
    print(f"\n【模型结构】")
    print(f"  观测维度: {obs_dim}")
    print(f"  时间窗口: {window_size}")
    print(f"  动作数量: {num_actions}")
    print(f"  模型参数总数: {sum(p.numel() for p in model.parameters()):,}")
    
    # 4. 测试前向传播
    batch_size = 4
    x = torch.randn(batch_size, window_size, obs_dim)
    
    print(f"\n【前向传播测试】")
    print(f"  输入形状: {x.shape}")
    
    # 重置噪声（训练时）
    model.train()
    model.reset_noise()
    
    dist, pred_delta = model(x)
    print(f"  输出分布形状: {dist.shape} (应该是 [{batch_size}, {num_actions}, {config['network']['rainbow']['num_atoms']}])")
    print(f"  预测 ΔRSRP 形状: {pred_delta.shape} (应该是 [{batch_size}])")
    
    # 检查分布是否有效（概率和为1）
    prob_sum = dist.sum(dim=-1)  # [B, A]
    print(f"  分布概率和（每动作）: min={prob_sum.min().item():.4f}, max={prob_sum.max().item():.4f}")
    
    # 5. 测试 Q 值计算
    q_values = model.get_q_values(x)
    print(f"  Q 值形状: {q_values.shape} (应该是 [{batch_size}, {num_actions}])")
    print(f"  Q 值范围: min={q_values.min().item():.2f}, max={q_values.max().item():.2f}")
    
    # 6. 测试动作选择
    model.eval()
    x_single = torch.randn(1, window_size, obs_dim)
    action = model.act(x_single, epsilon=0.0)
    print(f"\n【动作选择测试】")
    print(f"  贪婪动作: {action}")
    hys, ttt = action_space.action_to_hys_ttt(action)
    print(f"  对应参数: Hys={hys:.1f} dB, TTT={ttt:.0f} ms")
    
    # 7. 测试观测窗口构建
    print(f"\n【观测窗口测试】")
    obs_window = ObservationWindow(
        window_size=window_size,
        obs_dim=obs_dim
    )
    
    # 模拟添加观测
    for i in range(5):
        obs_raw = np.random.rand(7).astype(np.float32)  # 原始7维观测
        info = {
            'rsrp_serv_dbm': -90.0 + np.random.randn() * 5,
            'rsrp_neig_dbm': -90.0 + np.random.randn() * 5,
        }
        obs_extended = obs_window.build_observation(
            obs_raw, info,
            velocity_mps=80.0,
            track_length_m=3000.0
        )
        obs_window.update_time(i * 0.1)
        print(f"  步骤 {i+1}: 扩展观测形状 {obs_extended.shape}, 窗口长度 {len(obs_window.window)}")
    
    # 获取完整窗口
    window = obs_window.get_window()
    print(f"  完整窗口形状: {window.shape} (应该是 [{window_size}, {obs_dim}])")
    
    # 8. 测试模型推理
    window_tensor = torch.from_numpy(window).unsqueeze(0)  # [1, N, F]
    with torch.no_grad():
        dist_test, pred_delta_test = model(window_tensor)
        q_test = model.get_q_values(window_tensor)
        action_test = model.act(window_tensor, epsilon=0.0)
    
    print(f"\n【完整推理测试】")
    print(f"  输入窗口形状: {window_tensor.shape}")
    print(f"  输出分布形状: {dist_test.shape}")
    print(f"  预测 ΔRSRP: {pred_delta_test.item():.4f}")
    print(f"  Q 值形状: {q_test.shape}")
    print(f"  选择动作: {action_test}")
    hys_test, ttt_test = action_space.action_to_hys_ttt(action_test)
    print(f"  对应参数: Hys={hys_test:.1f} dB, TTT={ttt_test:.0f} ms")
    
    print("\n" + "=" * 60)
    print("✅ 所有测试通过！模型搭建成功！")
    print("=" * 60)


if __name__ == "__main__":
    test_model()
