# 训练、评估与导出指南

本文档记录当前主线命令。早期包装器方案和 R3 奖励说明已经归档在 `docs/archive/` 或 `docs/usage/TRAINING_GUIDE.md`，不代表当前默认实现。

## 当前主线

- 模型：`Obs7 + GRU + Rainbow DQN`
- 输入：`[15, 7]` 历史观测窗口
- 动作：48 个 A3 `(Hys, TTT)` 离散组合
- 奖励：默认 R5，聚焦切换重叠区
- 训练：离线 Rainbow，当前推荐 `--no_cql`
- 鲁棒性：`configs/scenario_profiles.yaml` 多 profile domain randomization
- 部署：导出 `policy_table.csv` 给 Simu5G 查表
- 安全层：推荐 late guard 预设 `delta3_low5`

## 安装依赖

```powershell
pip install -r requirements.txt
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

## 数据采集

当前推荐 V1 domain random Obs7 数据采集命令：

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

其中 `--action_hold_steps 0` 表示按 TTT 自适应保持动作，默认 `max(6, ceil(TTT / delta_t) + 2)`，上限 20 步。

## 离线训练

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
  --early_stop_min_delta 0.001 `
  --seed 20260604 `
  --run_dir experiments/runs/20260604_v1_domain_random_obs7_nocql_seed20260604
```

当前推荐 checkpoint：

```text
experiments/runs/20260604_v1_domain_random_obs7_nocql_seed20260604/checkpoints/rainbow_offline_best.pth
```

## Python 评估

单场景对比：

```powershell
.\.venv\Scripts\python.exe scripts\eval\test_simple.py `
  --checkpoint_path experiments/runs/20260604_v1_domain_random_obs7_nocql_seed20260604/checkpoints/rainbow_offline_best.pth
```

100 场景批量对比：

```powershell
.\.venv\Scripts\python.exe scripts\eval\test_batch_comparison.py `
  --checkpoint_path experiments/runs/20260604_v1_domain_random_obs7_nocql_seed20260604/checkpoints/rainbow_offline_best.pth `
  --num_scenarios 100 `
  --base_seed 10000
```

profile holdout 泛化：

```powershell
.\.venv\Scripts\python.exe scripts\eval\test_profile_generalization.py `
  --checkpoint_path experiments/runs/20260604_v1_domain_random_obs7_nocql_seed20260604/checkpoints/rainbow_offline_best.pth `
  --scenario_profiles_path configs/scenario_profiles.yaml `
  --profile_split test `
  --num_seeds 20
```

late guard 泛化：

```powershell
.\.venv\Scripts\python.exe scripts\eval\test_late_guard_generalization.py `
  --checkpoint_path experiments/runs/20260604_v1_domain_random_obs7_nocql_seed20260604/checkpoints/rainbow_offline_best.pth `
  --scenario_profiles_path configs/scenario_profiles.yaml `
  --profile_split test `
  --num_seeds 20
```

## Baseline 对比

论文主表使用 `comparison_algorithms` runner。当前主结果来自：

```text
experiments/runs/20260611_handover_success_proxy_comparison_test20
```

复现实验命令：

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

论文表格生成：

```powershell
.\.venv\Scripts\python.exe tools\build_paper_metric_tables.py
```

输出：

```text
paper/assets/tables/final/table_main_python_comparison_metrics.md
paper/assets/tables/final/table_profile_python_comparison_metrics.md
paper/assets/tables/source/table_weighted_handover_success_proxy.md
```

## Policy Table 导出

通用导出：

```powershell
.\.venv\Scripts\python.exe scripts\export\export_policy_table.py `
  --checkpoint_path experiments/runs/20260604_v1_domain_random_obs7_nocql_seed20260604/checkpoints/rainbow_offline_best.pth `
  --output_path results/policy_tables/policy_table.csv
```

当前推荐导出：

```powershell
.\.venv\Scripts\python.exe tools\export_late_guard_policy_preset.py delta3_low5 --copy_to_default
```

## Simu5G 验证

```powershell
wsl -d Ubuntu -- bash -lc "cd /mnt/c/Users/qichengshi/Desktop/RL_sim && bash tools/run_simu5g_complex_matrix.sh <RunName>"

.\.venv\Scripts\python.exe tools\analyze_simu5g_complex_matrix.py `
  --raw_root results/simu5g/<RunName>/raw `
  --parsed_root results/simu5g/<RunName>/parsed `
  --window_s 2.0
```

Simu5G 相关设计见 [../design/simu5g_rl_policy_table.md](../design/simu5g_rl_policy_table.md)。

## 指标口径

主指标：

- `ho_per_km`
- `ping_pong_count`
- `outage_time_ratio`
- `sinr_p5_db`
- `overlap_zone_sinr_mean_db`
- `comm_interruption_ratio_in_overlap_zone`
- `ho_attempt_success_rate`
- `late_ho_failure_count`
- `mobility_success_rate_proxy`

论文中不要只汇报 `HO attempt success`。少切换策略可能尝试切换很少，因此已触发切换成功率很高，但 late-HO failure 很多。

## 结论口径

当前推荐写法：

> `RL + LateGuard` 在低切换、低乒乓和低重叠区中断方面较强，并显著降低原始 RL 的 late-HO failure；但固定 A3 在部分 outage / SINR 低分位上仍是强基线，Stress / HighInterference 的剩余问题包含覆盖、干扰、资源或队列边界。

不要写成：

> RL 全面优于固定 A3。
