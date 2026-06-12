# AGENTS.md

本文档给 Codex / Claude 等代码代理说明本仓库的当前状态、工作约定和常用入口。仓库内容、文档、注释与提交信息以 **中文** 为主，后续修改也优先保持中文表达。

## 项目现状

RL_sim 是一个面向高速铁路蜂窝/5G-R 场景的强化学习切换优化项目。核心问题是：列车沿一维轨道从小区 A 行驶到小区 B，智能体根据无线观测动态选择 A3 切换参数 `(Hys, TTT)`，在降低 outage、晚切、乒乓和切换中断之间取得平衡。

当前主线已从早期“单一 3 km 场景 + Rainbow 离线训练”扩展为：

- Python Gymnasium 环境中训练/评估 `Obs7 + GRU + Rainbow DQN`。
- 通过 `configs/scenario_profiles.yaml` 做多 profile domain randomization 训练与 holdout 泛化测试。
- 将训练好的策略导出为 Simu5G 可查表的 `policy_table.csv`，由 Simu5G 运行时动态控制 A3 的 `Hys/TTT`。
- 在导出阶段可叠加 late handover guard，限制低 SINR/邻区明显更强时过长 TTT 和过大 Hys，缓解晚切风险。

截至 2026-06-04，当前推荐实验基线是：

- 数据集：`data/datasets/offline_dataset_v1_domain_random_obs7_seed20260604.npz`
- 训练 run：`experiments/runs/20260604_v1_domain_random_obs7_nocql_seed20260604`
- checkpoint：`experiments/runs/20260604_v1_domain_random_obs7_nocql_seed20260604/checkpoints/rainbow_offline_best.pth`
- 推荐默认策略表：`results/policy_tables/policy_table.csv`
- 推荐 guard 预设：`delta3_low5`
- 详细记录：`docs/experiments/v1_domain_random_obs7_full_pipeline_20260604.md`、`docs/experiments/late_handover_guard_measured_sinr_20260604.md`

不要把当前结果表述为“全面优于固定 A3”。现状更准确的判断是：RL/guard 方案明显降低保守策略晚切的一部分风险，在 WeakCoverage、NlosFading、HeavyTraffic 等场景有改善或持平，但 Stress / HighInterference 仍存在尾部时延、SINR 低分位或业务负载耦合问题。

## 主要目标与算法边界

本项目的核心目标是高速铁路相邻小区重叠区内的 A3 `Hys/TTT` 自适应控制：减少晚切、过早切、乒乓、切换中断和 outage。弱覆盖、强干扰、NLOS/fading 等中等无线退化应纳入鲁棒性验证，因为合理切换仍可缓解部分退化；但当前算法不应被表述为覆盖增强、干扰抑制、资源调度或业务拥塞控制方案。

切换算法能够控制的是切换参数和切换时机；当邻区仍有可利用质量优势时，它可以缓解弱覆盖、干扰或 fading 下的晚切风险。真正不能单独解决的是两个候选小区都不可用、目标小区质量同样差、深衰落/极强干扰下无可用切换目标、资源不足和 App/RLC 排队尾时延等问题。Stress / HighInterference 中的剩余问题要先做归因：若属于晚切/早切/乒乓，继续优化切换；若属于覆盖或资源极限，则归入算法边界或跨层优化方向。

新增模型模块、观测维度或 reward 项前，必须先判断它是否解决“切换可控”的问题，并用核心 KPI 证明收益超过复杂度代价。核心 KPI 包括：`ho_per_km`、`ping_pong_count`、`outage_time_ratio`、`sinr_p5_db`、`overlap_zone_sinr_mean_db`、`comm_interruption_ratio_in_overlap_zone`，以及可实现时的 `handover_failure/RLF proxy`。详细说明见 `docs/design/problem_scope_and_boundaries.md`。

## 目录结构

```text
configs/                  # 环境、模型、场景 profile 配置
envs/                     # Gymnasium 环境、信道模型、A3/TTT 切换逻辑
models/                   # RainbowWithForecast、动作空间、15步观测窗口
utils/                    # PER、C51 投影、数据加载、场景 profile、action hold、late guard
scripts/data/             # 离线数据采集、真实轨迹预处理
scripts/train/            # 在线/离线训练入口
scripts/eval/             # Python 侧单场景、批量、profile、late guard 评估
scripts/export/           # policy_table 导出
tools/                    # Simu5G patch、矩阵运行/分析、guard 预设导出
docs/                     # 设计说明、使用说明、实验记录
experiments/runs/         # 每次训练/评估的完整 run 产物
results/                  # 可复用结果、policy table、Simu5G 解析结果
paper/                    # 论文材料
```

新的训练、评估和分析输出应写入 `experiments/runs/<run_id>/` 或 `results/<topic>/`。不要把 PNG、JSON、checkpoint 或临时数据直接输出到项目根目录。

## 常用命令

安装依赖：

```powershell
pip install -r requirements.txt
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

采集当前 V1 domain random Obs7 数据：

```powershell
.\.venv\Scripts\python.exe scripts\data\collect_data.py `
  --num_episodes 500 `
  --output_path data/datasets/offline_dataset_v1_domain_random_obs7_seed20260604.npz `
  --policy_type stratified `
  --scenario_profiles_path configs/scenario_profiles.yaml `
  --profile_split train `
  --save_profile_metadata `
  --seed_start 2026060400 `
  --action_hold_steps 0
```

离线训练：

```powershell
.\.venv\Scripts\python.exe scripts\train\train_rainbow_offline.py `
  --dataset_path data/datasets/offline_dataset_v1_domain_random_obs7_seed20260604.npz `
  --no_cql `
  --num_epochs 120 `
  --samples_per_epoch 12000 `
  --batch_size 64 `
  --amp `
  --val_ratio 0.05 `
  --early_stop_on val `
  --early_stop_patience 18 `
  --seed 20260604 `
  --run_dir experiments/runs/20260604_v1_domain_random_obs7_nocql_seed20260604
```

Python 侧评估：

```powershell
# 单场景对比
.\.venv\Scripts\python.exe scripts\eval\test_simple.py --checkpoint_path <checkpoint.pth>

# 100 场景批量对比
.\.venv\Scripts\python.exe scripts\eval\test_batch_comparison.py `
  --checkpoint_path <checkpoint.pth> --num_scenarios 100 --base_seed 10000

# profile holdout 泛化评估
.\.venv\Scripts\python.exe scripts\eval\test_profile_generalization.py `
  --checkpoint_path <checkpoint.pth> `
  --scenario_profiles_path configs/scenario_profiles.yaml `
  --profile_split test `
  --num_seeds 20

# late guard 泛化评估
.\.venv\Scripts\python.exe scripts\eval\test_late_guard_generalization.py `
  --checkpoint_path <checkpoint.pth> `
  --scenario_profiles_path configs/scenario_profiles.yaml `
  --profile_split test `
  --num_seeds 20
```

导出 Simu5G 策略表：

```powershell
# 通用导出
.\.venv\Scripts\python.exe scripts\export\export_policy_table.py `
  --checkpoint_path <checkpoint.pth> `
  --output_path results/policy_tables/policy_table.csv

# 当前推荐：late guard delta3_low5，并复制为默认 policy_table.csv
.\.venv\Scripts\python.exe tools\export_late_guard_policy_preset.py delta3_low5 --copy_to_default
```

Simu5G 复杂矩阵运行与解析：

```powershell
wsl -d Ubuntu -- bash -lc "cd /mnt/c/Users/qichengshi/Desktop/RL_sim && bash tools/run_simu5g_complex_matrix.sh <RunName>"

.\.venv\Scripts\python.exe tools\analyze_simu5g_complex_matrix.py `
  --raw_root results/simu5g/<RunName>/raw `
  --parsed_root results/simu5g/<RunName>/parsed `
  --window_s 2.0
```

## 核心设计

### 环境与信道

- `envs/train_ho_env.py`：Gymnasium 环境。每步执行位置推进、应用 `(Hys, TTT)`、A3 事件检测、TTT 计时、切换中断、outage/RLF 检测、KPI 更新和奖励计算。
- `envs/channel_model.py`：路径损耗 + 距离相关 AR(1) 阴影衰落 + 动态同频干扰；支持额外干扰和 Rayleigh/Rician 快衰落。
- `envs/ho_logic.py`：A3 条件、TTT timer、切换保护时间/距离逻辑。
- 动态切换重叠区 KPI 基于 `ho_overlap_delta_rsrp_db` 在线确定单一连续重叠区，重点统计区内 SINR、中断和乒乓。

### 模型与观测

- 当前模型输入固定为 `[15, 7]`：15 步历史窗口，每步 7 个特征。
- 7 个特征为：`RSRP_serv`、`RSRP_neig`、`Delta RSRP`、`SINR_serv`、速度、位置、距上次切换时间。
- 当前 Hys/TTT **不进入策略输入**，避免网络学习“复制当前动作”的捷径。
- `models/rainbow_model.py` 中 `RainbowWithForecast` 使用 GRU encoder、shared FC、C51 distributional dueling head、NoisyLinear 支持，以及辅助 `Delta RSRP` 预测头。
- 离线训练 checkpoint 通常不使用 NoisyNet 推理噪声；导出脚本会根据 checkpoint 自动推断 `use_noisy`。

### 动作与 Action Hold

- 动作空间固定为 48 个离散动作：8 个 Hys `{1.5, 2.0, ..., 5.0}` × 6 个 TTT `{0, 50, 100, 150, 300, 650}`。
- 数据采集、在线训练、评估统一使用 `utils/action_hold.py`。`--action_hold_steps 0` 表示按 TTT 自适应保持，默认 `max(6, ceil(TTT / delta_t) + 2)`，上限 20 步。
- 不要在新评估脚本里每步强制重新采样动作，否则 TTT 计时器会被反复重置，结果会偏离真实切换逻辑。

### 奖励函数

- `configs/default_env_config.yaml` 当前默认 `reward_type: "R5"`。
- R5 是切换区聚焦型奖励：切换区外奖励近零，仅保留 outage/中断安全网；切换区内惩罚长期服务次优小区、outage、中断、切换和乒乓。
- R3/R4 仍在代码中保留用于对比或历史实验，但 AGENTS 中不要再把 R3 当作当前默认方案。
- `reward_training_phases` 可在数据采集阶段动态调整 outage/interruption/HO 权重，用于前后期训练目标调度。

### 场景 Profile

- `configs/scenario_profiles.yaml` 是当前鲁棒泛化训练的核心配置。
- train split：`normal`、`weak_coverage`、`high_noise`、`nlos_fading`、`high_interference`、`mixed_mild`。
- test split：`stress_radio_holdout`、`speed_400`、`distance_2km`、`distance_4km`。
- `utils/scenario_profiles.py` 只负责按 seed 生成环境参数覆盖，不改变观测维度和模型结构。

### Policy Table 与 Simu5G

- `scripts/export/export_policy_table.py` 将 Rainbow checkpoint 蒸馏成 CSV 查表策略。默认 `history_mode` 是 `repeat_current`：把当前单步观测重复 15 次喂给 GRU。
- `repeat_current` 是打通 Simu5G 链路的近似方案，不能完整表达真实历史趋势。若后续策略不稳，优先考虑 `trace_window` 或在线推理服务，而不是直接扩大网络。
- Simu5G C++ 侧已支持读取 `policy_table.csv` 并记录 `railwayPolicyAction / Hys / TTT / Q` 等指标。当前 late guard 相关验证中，查表 SINR 优先使用服务小区 measured SINR，初始化失败时才回退到 RSRP proxy。
- Simu5G 工作区在 WSL：`/home/qcshi/simu5g-workspace/simu5g-1.4.4`。项目内 `tools/*patch*` 和 `tools/simu5g_*` 保存了补丁、工作副本和分析脚本。

## 当前实验结论要点

- V1 domain randomization 后，RL 策略比激进 A3 明显减少切换和乒乓，但相比 Oracle A3 仍有差距。
- 原始 V1 policy table 偏向长 TTT，尤其 `650 ms`，总体风格是少切换、偏保守，因此在低 SINR/快速退化时有晚切风险。
- late guard `delta3_low5` 在 Python holdout profile 中降低 outage、改善 SINR p5；在 Simu5G 中对 Stress、WeakCoverage、NlosFading、HeavyTraffic 等有不同程度改善或持平。
- Stress 剩余问题不完全是切换时机问题，还与低 SINR、NLOS/fading、资源减少和重业务排队有关。后续应先做误差归因和轻量消融；若低 SINR 主要由晚切造成，可以继续优化 guard/R6 reward；若相邻基站都处于不可用质量或主要受资源排队限制，则应作为算法边界或跨层优化方向。

## 工作约定

- 修改代码前先确认是否已有未提交改动；不要回滚用户或其他工具留下的变更。
- 读取中文文件时在 PowerShell 中优先显式设置 UTF-8，例如：

```powershell
[Console]::OutputEncoding=[System.Text.Encoding]::UTF8; Get-Content -Raw -Encoding UTF8 <file>
```

- 新脚本应使用项目根目录相对路径解析，输出目录默认落到 `experiments/runs/` 或 `results/`。
- 实验结论要引用具体 run、seed、profile、checkpoint 和输出路径，避免只写“最新结果”。
- 更新 Simu5G 相关能力时，同步维护 `docs/design/simu5g_rl_policy_table.md`、相关 `docs/experiments/*.md` 和 `tools/*patch*`。
- 如需新增观测维度，必须同步更新 `model_config.yaml`、`ObservationWindow`、数据集采集、训练、评估、policy table 导出和 Simu5G 查表映射；旧 `[15, 7]` checkpoint 不能直接复用。
