"""简单测试脚本：测试环境是否正常工作"""
import os
import numpy as np
import matplotlib.pyplot as plt
from envs.train_ho_env import TrainHandoverEnv

# 配置 matplotlib 支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']  # 使用黑体或微软雅黑
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


def traditional_handover_algorithm(obs, info, hysteresis_db=3.0):
    """
    传统切换算法：基于A3事件（相对门限）
    
    当邻区RSRP - 服务小区RSRP > 迟滞值(hysteresis)时，触发切换
    
    Args:
        obs: 当前观测（归一化后的值）
        info: 信息字典，包含rsrp_serv_dbm和rsrp_neig_dbm
        hysteresis_db: 迟滞值（dB），用于避免乒乓切换，默认3.0 dB
        
    Returns:
        action: 0=不切换, 1=切换
    """
    rsrp_serv = info.get("rsrp_serv_dbm", 0.0)
    rsrp_neig = info.get("rsrp_neig_dbm", 0.0)
    
    # A3事件：RSRP_neig - RSRP_serv > hysteresis
    if rsrp_neig - rsrp_serv > hysteresis_db:
        return 1  # 切换
    else:
        return 0  # 不切换


def run_episode(env, policy_func, policy_name="策略", verbose=True, seed=42, policy_obj=None):
    """
    运行一个episode并返回统计信息
    
    Args:
        env: 环境实例
        policy_func: 策略函数，输入(obs, info)，输出action
        policy_name: 策略名称
        verbose: 是否打印详细信息
        seed: 随机种子
        policy_obj: 策略对象（如果有reset方法，会在episode开始时调用）
        
    Returns:
        stats: 统计字典
    """
    # 如果策略对象有reset方法，在episode开始时重置状态
    if policy_obj is not None and hasattr(policy_obj, 'reset'):
        policy_obj.reset()
    
    obs, info = env.reset(seed=seed)
    
    # 收集环境参数
    env_params = {
        "seed": seed,
        "temperature_c": env.weather_model.temperature,
        "humidity_percent": env.weather_model.humidity,
        "pm25": env.weather_model.pm25,
        "velocity_kmh": env.velocity_mps * 3.6,
        "velocity_mps": env.velocity_mps,
        "track_length_m": env.cfg["track_length_m"],
        "delta_t_s": env.cfg["delta_t_s"],
        "Ptx_A_dbm": env.cfg["Ptx_A_dbm"],
        "Ptx_B_dbm": env.cfg["Ptx_B_dbm"],
        "shadow_sigma_A": env.cfg["shadow_sigma_A"],
        "shadow_sigma_B": env.cfg["shadow_sigma_B"],
        "shadow_corr_distance_m": env.cfg.get("shadow_corr_distance_m", 80.0),
        "l3_alpha": env.cfg["l3_alpha"],
        "sinr_outage_db": env.cfg["sinr_outage_db"],
        "T_guard_s": env.cfg["T_guard_s"],
    }
    
    # 计算天气损耗
    weather_loss_db = env.weather_model.compute_weather_loss_db()
    env_params["weather_loss_db"] = weather_loss_db
    
    done = False
    total_reward = 0.0
    step_count = 0
    ho_count = 0
    outage_count = 0
    min_sinr = float('inf')
    max_sinr = float('-inf')
    sinr_list = []
    trajectory = []  # 记录轨迹用于可视化（位置、RSRP_A/B、切换点等）
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"运行策略: {policy_name}")
        print(f"{'='*60}")
        print(f"\n【环境参数】")
        print(f"  随机种子: {env_params['seed']}")
        print(f"  轨道长度: {env_params['track_length_m']:.1f} m")
        print(f"  时间步长: {env_params['delta_t_s']*1000:.1f} ms")
        print(f"  列车速度: {env_params['velocity_kmh']:.2f} km/h ({env_params['velocity_mps']:.2f} m/s)")
        print(f"\n【天气条件】")
        print(f"  温度: {env_params['temperature_c']:.2f} °C")
        print(f"  湿度: {env_params['humidity_percent']:.2f} %")
        print(f"  PM2.5: {env_params['pm25']:.2f}")
        print(f"  天气额外损耗: {env_params['weather_loss_db']:.3f} dB")
        print(f"\n【基站参数】")
        print(f"  基站A发射功率: {env_params['Ptx_A_dbm']:.1f} dBm")
        print(f"  基站B发射功率: {env_params['Ptx_B_dbm']:.1f} dBm")
        print(f"  阴影衰落标准差 - A: {env_params['shadow_sigma_A']:.1f} dB")
        print(f"  阴影衰落标准差 - B: {env_params['shadow_sigma_B']:.1f} dB")
        print(f"  阴影相关距离: {env_params['shadow_corr_distance_m']:.1f} m")
        print(f"\n【系统参数】")
        print(f"  L3滤波系数: {env_params['l3_alpha']:.2f}")
        print(f"  SINR Outage阈值: {env_params['sinr_outage_db']:.1f} dB")
        print(f"  切换保护时间: {env_params['T_guard_s']:.1f} s")
        print(f"\n【初始状态】")
        print(f"  初始位置: {env.position_m:.2f} m")
        print(f"  初始服务小区: {'A' if env.serving_cell == 0 else 'B'}")
        print(f"  初始RSRP - 服务: {info.get('rsrp_serv_dbm', 0):.2f} dBm, "
              f"邻区: {info.get('rsrp_neig_dbm', 0):.2f} dBm")
        print(f"  初始SINR: {info.get('sinr_serv_db', 0):.2f} dB")
        print("-" * 60)
    
    while not done:
        # 使用策略函数选择动作
        action = policy_func(obs, info)
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        step_count += 1
        
        sinr = info.get("sinr_serv_db", 0.0)
        sinr_list.append(sinr)
        min_sinr = min(min_sinr, sinr)
        max_sinr = max(max_sinr, sinr)
        
        ho_executed = info.get("ho_executed", False)
        if ho_executed:
            ho_count += 1

        # 直接使用环境返回的原始 RSRP_A / RSRP_B（未经过 L3 滤波）
        # 这样画出的曲线与 plot_rsrp.py 一致（都是原始信道模型的输出）
        rsrp_A = info.get("rsrp_A_dbm", 0.0)
        rsrp_B = info.get("rsrp_B_dbm", 0.0)
        rsrp_serv = info.get("rsrp_serv_dbm", 0.0)
        rsrp_neig = info.get("rsrp_neig_dbm", 0.0)
        serving_cell = info.get("serving_cell", 0)

        trajectory.append({
            "step": step_count,
            "x": info.get("position_m", 0.0),
            "serving_cell": serving_cell,
            "rsrp_A_dbm": rsrp_A,  # 原始 A 基站 RSRP
            "rsrp_B_dbm": rsrp_B,  # 原始 B 基站 RSRP
            "rsrp_serv_dbm": rsrp_serv,  # L3 滤波后的服务小区 RSRP
            "rsrp_neig_dbm": rsrp_neig,  # L3 滤波后的邻区 RSRP
            "sinr_serv_db": sinr,
            "ho_executed": ho_executed,
        })
        
        if info.get("outage", False):
            outage_count += 1
        
        # 每50步打印一次信息
        if verbose and step_count % 50 == 0:
            print(f"Step {step_count:4d} | "
                  f"位置: {info['position_m']:7.2f}m | "
                  f"服务小区: {'A' if info['serving_cell'] == 0 else 'B'} | "
                  f"RSRP: 服务={info['rsrp_serv_dbm']:6.2f}dBm, "
                  f"邻区={info['rsrp_neig_dbm']:6.2f}dBm | "
                  f"SINR: {sinr:6.2f}dB | "
                  f"奖励: {reward:6.3f}")
        
        done = terminated or truncated
        
        # 防止无限循环
        if step_count >= 10000:
            if verbose:
                print("\n警告：达到最大步数限制！")
            break
    
    avg_sinr = np.mean(sinr_list) if sinr_list else 0.0
    
    stats = {
        "policy_name": policy_name,
        "total_steps": step_count,
        "total_reward": total_reward,
        "ho_count": ho_count,
        "outage_count": outage_count,
        "min_sinr": min_sinr,
        "max_sinr": max_sinr,
        "avg_sinr": avg_sinr,
        "final_position": info.get("position_m", 0.0),
        "arrived": info.get("arrived", False),
        "outage_occurred": info.get("outage", False),
        "trajectory": trajectory,
        "env_params": env_params,  # 添加环境参数到统计信息
    }
    
    if verbose:
        print("-" * 60)
        print(f"\nEpisode 结束！")
        print(f"总步数: {step_count}")
        print(f"总切换次数: {ho_count}")
        print(f"Outage次数: {outage_count}")
        print(f"累计奖励: {total_reward:.2f}")
        print(f"平均SINR: {avg_sinr:.2f}dB")
        print(f"最小SINR: {min_sinr:.2f}dB")
        print(f"最大SINR: {max_sinr:.2f}dB")
        print(f"终止原因: ", end="")
        if info.get("outage", False):
            print("Outage（信号中断）")
        elif info.get("arrived", False):
            print("到达终点")
        else:
            print("其他原因")
        print(f"最终位置: {info['position_m']:.2f}m")
        print(f"最终SINR: {info['sinr_serv_db']:.2f}dB")
    
    return stats


def main():
    """主测试函数：对比随机策略和传统切换算法"""
    print("=" * 60)
    print("铁路切换算法环境测试 - 策略对比")
    print("=" * 60)

    # 创建环境（显式从 configs/default_env_config.yaml 读取配置）
    # 注意：测试代码使用兼容模式（2动作），因为策略函数返回0/1
    base_dir = os.path.dirname(__file__)
    cfg_path = os.path.join(base_dir, "configs", "default_env_config.yaml")
    # 使用兼容模式（2动作：0=不切换, 1=切换）
    env = TrainHandoverEnv(config_path=cfg_path, config={"use_hys_ttt": False})
    print("\n环境创建成功！")
    print(f"观测空间: {env.observation_space}")
    print(f"动作空间: {env.action_space} (兼容模式：2动作)")
    
    # 传统算法 + TTT：需要在策略内部维护一个计时器
    # 使用类来封装状态，避免 nonlocal 的复杂性
    delta_t = env.cfg["delta_t_s"]
    
    class TraditionalHandoverPolicy:
        """传统切换算法策略类，封装TTT计时器状态
        
        实现标准的5G/LTE A3事件切换算法：
        - A3事件：邻区RSRP - 服务小区RSRP > 迟滞门限(hysteresis)
        - TTT (Time-To-Trigger)：条件需持续满足一定时间才触发切换
        """
        def __init__(self, hysteresis_db, ttt_s):
            """
            Args:
                hysteresis_db: 迟滞门限（dB），用于避免乒乓切换
                ttt_s: Time-To-Trigger（秒），条件需持续满足的时间
            """
            self.hysteresis_db = hysteresis_db
            self.ttt_s = ttt_s
            self.ttt_timer = 0.0
        
        def reset(self):
            """重置TTT计时器（每次episode开始时调用）"""
            self.ttt_timer = 0.0
        
        def decide(self, obs, info):
            """决策函数
            
            Args:
                obs: 观测向量（未使用，但保留接口一致性）
                info: 信息字典，包含rsrp_serv_dbm和rsrp_neig_dbm
                
            Returns:
                action: 0=不切换, 1=切换
            """
            rsrp_serv = info.get("rsrp_serv_dbm", 0.0)
            rsrp_neig = info.get("rsrp_neig_dbm", 0.0)
            
            # A3事件条件：邻区RSRP - 服务小区RSRP > 迟滞门限
            if rsrp_neig - rsrp_serv > self.hysteresis_db:
                # 条件满足，累积TTT计时器
                self.ttt_timer += delta_t
                if self.ttt_timer >= self.ttt_s:
                    # 条件持续超过TTT -> 触发切换，然后清零计时器
                    self.ttt_timer = 0.0
                    return 1
                else:
                    # 条件满足但未达到TTT，继续等待
                    return 0
            else:
                # 条件不再满足，清零计时器（重置TTT）
                self.ttt_timer = 0.0
                return 0
    
    # 创建策略实例
    policy_3db = TraditionalHandoverPolicy(hysteresis_db=3.0, ttt_s=0.1)
    policy_5db = TraditionalHandoverPolicy(hysteresis_db=5.0, ttt_s=0.1)
    
    def traditional_policy_3db(obs, info):
        """传统切换算法（A3事件，迟滞3dB + TTT）"""
        return policy_3db.decide(obs, info)
    
    def traditional_policy_5db(obs, info):
        """传统切换算法（A3事件，迟滞5dB + TTT）"""
        return policy_5db.decide(obs, info)
    
    # 运行不同策略
    results = []
    
    # 生成随机seed，确保所有策略使用相同的环境条件进行公平对比
    random_seed = np.random.randint(0, 10000)
    print(f"\n使用随机种子: {random_seed}（所有策略共享相同环境条件）")
    print("-" * 60)
    
    # 1. 传统算法（3dB迟滞）
    stats_traditional_3db = run_episode(
        env,
        traditional_policy_3db,
        policy_name="传统切换算法（A3事件，迟滞3dB）",
        verbose=True,
        seed=random_seed,  # 使用相同种子保证公平对比
        policy_obj=policy_3db  # 传递策略对象以便重置TTT计时器
    )
    results.append(stats_traditional_3db)
    
    # 3. 传统算法（5dB迟滞）
    stats_traditional_5db = run_episode(
        env,
        traditional_policy_5db,
        policy_name="传统切换算法（A3事件，迟滞5dB）",
        verbose=True,
        seed=random_seed,  # 使用相同种子保证公平对比
        policy_obj=policy_5db  # 传递策略对象以便重置TTT计时器
    )
    results.append(stats_traditional_5db)
    
    # 打印环境参数（从第一个策略的结果中获取，因为所有策略使用相同环境）
    if results:
        env_params = results[0]["env_params"]
        print("\n" + "=" * 80)
        print("【环境参数总结】")
        print("=" * 80)
        print(f"随机种子: {env_params['seed']}")
        print(f"轨道长度: {env_params['track_length_m']:.1f} m")
        print(f"时间步长: {env_params['delta_t_s']*1000:.1f} ms")
        print(f"列车速度: {env_params['velocity_kmh']:.2f} km/h ({env_params['velocity_mps']:.2f} m/s)")
        print(f"温度: {env_params['temperature_c']:.2f} °C | "
              f"湿度: {env_params['humidity_percent']:.2f} % | "
              f"PM2.5: {env_params['pm25']:.2f}")
        print(f"天气额外损耗: {env_params['weather_loss_db']:.3f} dB")
        print(f"基站发射功率: A={env_params['Ptx_A_dbm']:.1f} dBm, B={env_params['Ptx_B_dbm']:.1f} dBm")
        print(f"阴影衰落: σ_A={env_params['shadow_sigma_A']:.1f} dB, "
              f"σ_B={env_params['shadow_sigma_B']:.1f} dB, "
              f"相关距离={env_params['shadow_corr_distance_m']:.1f} m")
        print(f"L3滤波系数: {env_params['l3_alpha']:.2f} | "
              f"Outage阈值: {env_params['sinr_outage_db']:.1f} dB | "
              f"保护时间: {env_params['T_guard_s']:.1f} s")
    
    # 打印对比结果
    print("\n" + "=" * 80)
    print("策略性能对比总结")
    print("=" * 80)
    print(f"{'策略名称':<30} | {'总步数':<8} | {'切换次数':<8} | {'Outage次数':<10} | "
          f"{'累计奖励':<10} | {'平均SINR':<10} | {'最小SINR':<10}")
    print("-" * 80)
    
    for stats in results:
        print(f"{stats['policy_name']:<30} | "
              f"{stats['total_steps']:<8} | "
              f"{stats['ho_count']:<8} | "
              f"{stats['outage_count']:<10} | "
              f"{stats['total_reward']:<10.2f} | "
              f"{stats['avg_sinr']:<10.2f} | "
              f"{stats['min_sinr']:<10.2f}")
    
    print("=" * 80)

    # -----------------------------
    # 距离-信号强度 可视化（包含 HO 点）
    # -----------------------------
    print("\n生成距离-信号强度曲线图（含各策略切换点标记）...")

    # 使用第一条轨迹（随机策略）的 RSRP_A/B 曲线作为背景参考
    base_traj = results[0]["trajectory"]
    xs = np.array([p["x"] for p in base_traj])
    rsrp_A = np.array([p["rsrp_A_dbm"] for p in base_traj])
    rsrp_B = np.array([p["rsrp_B_dbm"] for p in base_traj])

    plt.figure(figsize=(9, 5))
    plt.plot(xs, rsrp_A, label="RSRP A小区", color="tab:blue")
    plt.plot(xs, rsrp_B, label="RSRP B小区", color="tab:orange")

    # 为每种策略标记发生切换的位置（在切换后新服务小区的原始RSRP曲线上打点）
    markers = ["o", "s", "D", "^", "v", "<", ">", "p", "*", "h"]
    colors = ["red", "green", "purple", "blue", "orange", "brown", "pink", "gray", "olive", "cyan"]
    
    print("\n检查各策略的切换点数量：")
    for idx, stats in enumerate(results):
        traj = stats["trajectory"]
        ho_count = sum(1 for p in traj if p["ho_executed"])
        print(f"  {stats['policy_name']}: {ho_count} 个切换点")
    
    for idx, stats in enumerate(results):
        traj = stats["trajectory"]
        ho_x = []
        ho_y = []
        for p in traj:
            if p["ho_executed"]:
                ho_x.append(p["x"])
                # 切换后，serving_cell 已经是新小区了
                # 使用原始 RSRP（rsrp_A_dbm 或 rsrp_B_dbm）让点落在背景曲线上
                if p["serving_cell"] == 0:  # 切换后服务小区是 A
                    ho_y.append(p["rsrp_A_dbm"])
                else:  # 切换后服务小区是 B
                    ho_y.append(p["rsrp_B_dbm"])
        
        if ho_x:
            plt.scatter(
                ho_x,
                ho_y,
                marker=markers[idx % len(markers)],
                color=colors[idx % len(colors)],
                s=80,  # 增大标记尺寸，更容易看到
                alpha=0.8,
                edgecolors='black',
                linewidths=1.0,
                label=f"切换点 - {stats['policy_name']}",
                zorder=5 + idx,  # 不同策略使用不同的zorder，避免完全重叠
            )
        else:
            print(f"  警告：{stats['policy_name']} 没有产生任何切换点！")

    plt.xlabel("距离 x / m")
    plt.ylabel("RSRP / dBm")
    plt.title("沿轨道的 RSRP 曲线与不同策略的切换位置")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # 找出最佳策略
    best_policy = max(results, key=lambda x: x['total_reward'])
    print(f"\n最佳策略（按累计奖励）: {best_policy['policy_name']}")
    print(f"  累计奖励: {best_policy['total_reward']:.2f}")
    print(f"  切换次数: {best_policy['ho_count']}")
    print(f"  平均SINR: {best_policy['avg_sinr']:.2f}dB")
    
    print("\n" + "=" * 60)
    print("测试完成！")


if __name__ == "__main__":
    main()

