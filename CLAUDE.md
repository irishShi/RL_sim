# CLAUDE.md

本文件给 Claude Code 说明当前仓库状态和常用入口。仓库内容、文档、注释和提交信息以中文为主；后续修改也优先使用中文。

## 项目概览

RL_sim 是面向高速铁路蜂窝/5G-R 场景的强化学习切换优化项目。核心任务是：列车沿一维轨道穿越相邻小区重叠区时，智能体根据无线观测动态选择 A3 切换参数 `(Hys, TTT)`，在降低晚切、过早切、乒乓、切换中断和 outage 之间取得平衡。

当前主线不是早期单一 3 km 场景，而是：

- `Obs7 + GRU + Rainbow DQN` 离线训练。
- `configs/scenario_profiles.yaml` 多 profile domain randomization。
- Python holdout profile / baseline 对比。
- policy table 导出到 Simu5G 查表控制 A3。
- late handover guard `delta3_low5` 缓解低 SINR 或邻区明显更强时的晚切风险。

论文和实验表述必须遵守 [docs/design/problem_scope_and_boundaries.md](docs/design/problem_scope_and_boundaries.md)：本项目优化切换参数和切换时机，不应被写成覆盖增强、干扰抑制、资源调度或业务拥塞控制方案。

## 当前推荐基线

- 数据集：`data/datasets/offline_dataset_v1_domain_random_obs7_seed20260604.npz`
- 训练 run：`experiments/runs/20260604_v1_domain_random_obs7_nocql_seed20260604`
- checkpoint：`experiments/runs/20260604_v1_domain_random_obs7_nocql_seed20260604/checkpoints/rainbow_offline_best.pth`
- 默认 policy table：`results/policy_tables/policy_table.csv`
- 推荐 guard：`delta3_low5`
- 主实验记录：`docs/experiments/v1_domain_random_obs7_full_pipeline_20260604.md`
- guard 记录：`docs/experiments/late_handover_guard_measured_sinr_20260604.md`
- baseline 主表记录：`docs/experiments/handover_success_proxy_comparison_20260611.md`

不要把当前结果表述为“全面优于固定 A3”。更准确的说法是：`RL + LateGuard` 在低切换、低乒乓和低重叠区中断方面有优势，并显著缓解原始 RL 的晚切失败；但固定 A3 在部分 outage / SINR 尾部指标上仍是强基线，Stress / HighInterference 还包含覆盖、干扰、资源或队列边界。

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

Python 评估：

```powershell
.\.venv\Scripts\python.exe scripts\eval\test_profile_generalization.py `
  --checkpoint_path experiments/runs/20260604_v1_domain_random_obs7_nocql_seed20260604/checkpoints/rainbow_offline_best.pth `
  --scenario_profiles_path configs/scenario_profiles.yaml `
  --profile_split test `
  --num_seeds 20

.\.venv\Scripts\python.exe scripts\eval\test_late_guard_generalization.py `
  --checkpoint_path experiments/runs/20260604_v1_domain_random_obs7_nocql_seed20260604/checkpoints/rainbow_offline_best.pth `
  --scenario_profiles_path configs/scenario_profiles.yaml `
  --profile_split test `
  --num_seeds 20
```

论文 baseline 对比：

```powershell
.\.venv\Scripts\python.exe comparison_algorithms\scripts\run_comparison.py `
  --include_rainbow `
  --include_rainbow_late_guard `
  --q_table_path experiments\runs\20260604_comparison_algorithms_test20_v2\tables\tabular_q_policy.json `
  --profile_split test `
  --num_seeds 20 `
  --base_seed 25000 `
  --include_oracle_a3 `
  --output_dir experiments\runs\20260611_handover_success_proxy_comparison_test20
```

导出 Simu5G 策略表：

```powershell
.\.venv\Scripts\python.exe tools\export_late_guard_policy_preset.py delta3_low5 --copy_to_default
```

Simu5G 复杂矩阵：

```powershell
wsl -d Ubuntu -- bash -lc "cd /mnt/c/Users/qichengshi/Desktop/RL_sim && bash tools/run_simu5g_complex_matrix.sh <RunName>"

.\.venv\Scripts\python.exe tools\analyze_simu5g_complex_matrix.py `
  --raw_root results/simu5g/<RunName>/raw `
  --parsed_root results/simu5g/<RunName>/parsed `
  --window_s 2.0
```

## 核心结构

- `envs/train_ho_env.py`：Gymnasium 环境，执行位置推进、A3 条件、TTT 计时、切换中断、outage/RLF proxy、KPI 和奖励。
- `envs/channel_model.py`：路径损耗、距离相关 AR(1) 阴影衰落、动态同频干扰、可选快衰落和校准参数。
- `models/rainbow_model.py`：GRU encoder、C51 distributional dueling Rainbow head、NoisyLinear、辅助 `Delta RSRP` 预测头。
- `models/action_space.py`：48 个离散动作，8 个 Hys × 6 个 TTT。
- `utils/action_hold.py`：统一 action hold 逻辑，避免每步重选动作导致 TTT 计时被反复重置。
- `utils/scenario_profiles.py`：按 seed 生成 domain randomization profile 覆盖。
- `utils/late_handover_guard.py`：导出/评估阶段的轻量晚切保护。
- `comparison_algorithms/`：论文 baseline runner。
- `scripts/export/export_policy_table.py`：checkpoint 蒸馏为 Simu5G 查表 CSV。
- `tools/`：Simu5G patch、矩阵运行、结果解析、policy preset 导出和论文表格生成。

## 模型与奖励

- 输入固定为 `[15, 7]`：`RSRP_serv`、`RSRP_neig`、`Delta RSRP`、`SINR_serv`、速度、位置、距上次切换时间。
- 当前 Hys/TTT 不进入策略输入，避免网络学习“复制当前动作”的捷径。
- 动作空间固定为 48 个 `(Hys, TTT)` 组合。
- 默认奖励为 R5：切换区聚焦型奖励，区外奖励近零，仅保留 outage/中断安全网。
- R3/R4 保留为历史对比或消融，不是当前默认方案。

## 文档入口

- [docs/README.md](docs/README.md)：文档总索引。
- [docs/experiments/README.md](docs/experiments/README.md)：实验记录时间线。
- [paper/README.md](paper/README.md)：论文材料和 PaperSpine 准备状态。
- [comparison_algorithms/README.md](comparison_algorithms/README.md)：baseline 策略与 runner。

## Git 与输出约定

- 修改前先看 `git status --short`，不要回滚用户或其他工具留下的改动。
- 新训练/评估输出写入 `experiments/runs/<run_id>/`。
- 精选结果写入 `results/<topic>/`。
- 论文最终图表写入 `paper/assets/`。
- 不要把 PNG、JSON、checkpoint、临时数据直接输出到根目录。
- 大型数据、checkpoint、实验完整输出、venv 和缓存由 `.gitignore` 排除。
