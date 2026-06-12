# Simu5G 与 Rainbow DQN 切换策略对接方案

本文记录当前项目将 Rainbow DQN 切换算法接入 Simu5G 的第一阶段方案。核心目标不是让 Simu5G 直接训练 RL，而是先把已经训练好的策略离线导出为 `policy_table.csv`，让 Simu5G 在运行时根据实时无线观测查表选择当前 A3 切换参数 `Hys/TTT`。

## 1. 为什么可以用 policy table

项目中的 Rainbow DQN 本质上学习的是一个策略函数：

```text
policy(observation_window) -> action_id -> (Hys, TTT)
```

其中 `observation_window` 是最近 15 个时刻的 7 维观测，`action_id` 对应 48 个离散动作之一。Simu5G 运行时并不需要重新训练模型，只需要在每个决策周期拿到当前观测量，然后得到一个 `(Hys, TTT)` 参数组合。

`policy_table.csv` 的作用是把神经网络策略预先蒸馏成离散查表形式：

```text
Simu5G 当前观测量
  -> 量化到最近表格 bin
  -> 查询 action_id
  -> 得到 Hys/TTT
  -> 交给 Simu5G 中的 A3 + TTT 切换逻辑执行
```

所以这里不是让 RL “脱离 Simu5G 参数凭空预测”，而是让 Simu5G 继续提供实际观测量。区别只在于策略推理从“运行时调用 Python/PyTorch 模型”变成了“运行时查 CSV 表”。

## 2. 当前观测量映射

项目模型使用的单步观测维度为 7：

| 索引 | 特征 | 含义 | Simu5G 对应量 |
| --- | --- | --- | --- |
| 0 | `RSRP_serv` | 服务小区 RSRP，归一化 | `servingRsrp` |
| 1 | `RSRP_neig` | 最强邻区 RSRP，归一化 | `neighborRsrp` |
| 2 | `delta_rsrp` | 邻区 RSRP - 服务区 RSRP，归一化 | `deltaRsrp` |
| 3 | `SINR_serv` | 服务链路 SINR，归一化 | `servingSinr` 或等效 SINR |
| 4 | `v_norm` | 速度归一化 | `ueSpeed` |
| 5 | `pos_norm` | 轨道位置归一化 | `uePositionX` |
| 6 | `time_since_last_ho` | 距离上次切换时间，归一化 | `timeSinceLastHandover` |

当前 Simu5G 已经改造出大部分观测信号。后续接表时，C++ 侧需要维护这些量的最近值，并按与 Python 项目一致的归一化范围进行离散化。

## 3. 第一阶段表格形式

导出脚本生成的 CSV 每一行是一组离散观测状态及其最优动作，主要字段包括：

```text
position_m
speed_kmh
rsrp_serv_dbm
rsrp_neig_dbm
delta_rsrp_db
sinr_db
time_since_ho_s
action_id
hys_db
ttt_ms
q_value
q_margin
```

其中：

- `action_id` 是 Rainbow DQN 选出的动作编号。
- `hys_db/ttt_ms` 是 Simu5G 最终需要使用的 A3 参数。
- `q_value` 是该动作对应的 Q 值。
- `q_margin` 是最优动作与第二优动作的 Q 值差，越小表示策略越不确定。

## 4. 与项目真实场景的关系

当前项目的典型场景是 3 km 铁路区间，两端各有一个基站，基站间距约 3 km。因此导出表默认按以下思想建表：

- `position_m` 覆盖 `0 ~ 3000 m`。
- `speed_kmh` 默认包含 `300 km/h`，也可扩展为多个速度档位。
- `RSRP/SINR/deltaRSRP` 不强行由几何模型推导，而是作为 Simu5G 实时观测量查表。

这样做的好处是：即使后续调整 Simu5G 的路径损耗、阴影衰落、发射功率、基站高度或干扰模型，只要 Simu5G 输出的观测量仍然落在表格覆盖范围内，就可以继续查表。

## 5. 当前方案的局限

第一阶段导出表默认使用 `repeat_current` 历史窗口，即把当前单步观测重复 15 次输入 GRU。这是为了先把完整链路跑通，属于近似蒸馏：

```text
当前观测 x_t -> [x_t, x_t, ..., x_t] -> Rainbow DQN -> action
```

它不能完整表达模型训练时利用的历史变化趋势。因此后续建议升级为两种更真实的方式之一：

1. `trace_window`：先从 Simu5G 运行若干 A3 策略，导出真实观测轨迹，再用真实 15 步窗口批量生成策略表。
2. `online_inference`：Simu5G 运行时通过 Python/C++ 推理服务直接调用 Rainbow DQN，保留完整历史窗口。

第一阶段仍然有价值，因为它能先验证：Simu5G 能否实时获得 RL 所需观测量，A3+TTT 参数是否能被外部策略动态控制，以及最终 KPI 是否有变化。

## 6. 推荐实验流程

1. 在 Python 项目中训练或选择一个 checkpoint。
2. 使用 `scripts/export/export_policy_table.py` 导出 `policy_table.csv`。
3. 在 Simu5G 中每 `50 ms` 或 `100 ms` 更新一次观测量。
4. Simu5G 将观测量量化到最近表格 bin，查出 `Hys/TTT`。
5. Simu5G 的 A3+TTT 逻辑使用该参数执行切换。
6. 对比固定 A3、A3 参数扫描、RL 表策略三类结果：
   - 切换位置
   - 切换次数
   - ping-pong 次数
   - SINR/CQI 在切换点附近的下降
   - delay 峰值
   - packet loss / outage ratio

## 7. 后续要做

- 在 Simu5G C++ 侧实现 `policy_table.csv` 读取和最近邻查表。
- 将查表得到的 `Hys/TTT` 接入当前已新增的 A3+TTT 控制参数。
- 用 Simu5G 的 `.vec/.sca` 自动分析脚本对比固定 A3 与 RL 表策略。
- 如果表策略效果不稳定，优先改进为基于 Simu5G trace 的 `trace_window` 导出，而不是立刻改模型。

## 8. 当前已生成的文件

当前导出脚本：

```bash
python scripts/export/export_policy_table.py
```

当前正式表：

```text
results/policy_tables/policy_table.csv
results/policy_tables/policy_table.metadata.json
```

导出时自动选中的 checkpoint：

```text
experiments/runs/20260602_rainbow_offline_obs7_lowho_nocql_seed40000/checkpoints/rainbow_offline_best.pth
```

默认网格规模：

```text
position_m: 0~3000 m, step=100 m
speed_kmh: 300 km/h
rsrp_serv_dbm: -115~-65 dBm, step=5 dB
delta_rsrp_db: -20~20 dB 的重点离散档位
sinr_db: -10~20 dB 的重点离散档位
time_since_ho_s: 0, 0.5, 1, 2, 5, 10 s
```

如果只想快速生成小表验证链路，可以使用：

```bash
python scripts/export/export_policy_table.py --output_path=results/policy_tables/policy_table_probe.csv --max_rows=1000
```
