# Physics-informed Risk Head 第一阶段 Smoke 记录（2026-06-06）

本文记录“基于物理规则的神经网络未来切换风险预测”第一批实现与 smoke 验证。

## 已完成内容

1. 固化任务要求：
   - `docs/design/physics_informed_risk_prediction_tasks.md`

2. 新增物理派生特征：
   - 文件：`models/physics_features.py`
   - 输入仍为 `[B, 15, 7]` Obs7 历史窗口。
   - 输出 12 维物理特征，包括 `SINR` 波动、`Delta RSRP` 斜率、邻区优势比例、effective fading margin 和 pseudo L3 residual。

3. 新增未来风险标签：
   - 文件：`utils/risk_labels.py`
   - 默认预测窗口：`100 ms / 200 ms / 300 ms`。
   - 第一版标签为实际执行动作下的 `future_late_outage_risk`。

4. 新增模型：
   - 文件：`models/rainbow_model.py`
   - 新类：`RainbowWithPhysicsRisk`
   - 输出：

```text
dist: [B, 48, 51]
pred_delta: [B]
risk_logits: [B, 48, 3]
```

5. 改造离线训练：
   - 文件：`scripts/train/train_rainbow_offline.py`
   - 新增开关：`--enable_physics_risk`
   - 新增损失：

```text
L = L_Rainbow
  + lambda_aux * L_delta
  + lambda_risk * L_future_risk
  + lambda_phys * L_physics_constraint
  + L_CQL
```

## 验证命令

静态编译：

```powershell
.\.venv\Scripts\python.exe -m py_compile `
  models\physics_features.py `
  models\rainbow_model.py `
  utils\risk_labels.py `
  utils\replay_buffer.py `
  scripts\train\train_rainbow_offline.py
```

前向与标签 smoke：

```text
features (12, 12) 12
forward (4, 48, 51) (4,) (4, 48, 3)
labels (12, 3) 30.0
```

训练 smoke：

```powershell
.\.venv\Scripts\python.exe scripts\train\train_rainbow_offline.py `
  --dataset_path data/datasets/offline_dataset_v1_domain_random_obs7_seed20260604.npz `
  --enable_physics_risk `
  --no_cql `
  --num_epochs 1 `
  --samples_per_epoch 128 `
  --batch_size 32 `
  --no_amp `
  --val_ratio 0 `
  --early_stop_patience 0 `
  --train_log_interval 1 `
  --run_dir experiments/runs/smoke_physics_risk_train
```

Smoke 结果：

```text
模型参数总数: 1,131,476
使用 Physics Risk Head: True
Risk: 0.6977, Phys: 0.1711
best checkpoint: experiments/runs/smoke_physics_risk_train/checkpoints/rainbow_offline_best.pth
```

## 下一步正式实验

策略入口 smoke：

```powershell
.\.venv\Scripts\python.exe comparison_algorithms\scripts\run_comparison.py `
  --include_rainbow `
  --include_rainbow_physics_risk `
  --checkpoint_path experiments/runs/smoke_physics_risk_train/checkpoints/rainbow_offline_best.pth `
  --profile_split test `
  --profiles speed_400 `
  --num_seeds 1 `
  --base_seed 27000 `
  --output_dir experiments/runs/smoke_physics_risk_policy_eval
```

输出：

```text
experiments/runs/smoke_physics_risk_policy_eval/metrics/comparison_summary_overall.csv
```

建议从当前最佳 Rainbow checkpoint warm start：

```powershell
.\.venv\Scripts\python.exe scripts\train\train_rainbow_offline.py `
  --dataset_path data/datasets/offline_dataset_v1_domain_random_obs7_seed20260604.npz `
  --enable_physics_risk `
  --resume experiments/runs/20260604_v1_domain_random_obs7_nocql_seed20260604/checkpoints/rainbow_offline_best.pth `
  --no_cql `
  --num_epochs 80 `
  --samples_per_epoch 12000 `
  --batch_size 64 `
  --amp `
  --val_ratio 0.05 `
  --early_stop_on val `
  --early_stop_patience 12 `
  --seed 20260606 `
  --run_dir experiments/runs/20260606_physics_risk_obs7_seed20260606
```

完成正式训练后，还需要新增 `RL_Rainbow_GRU_PhysicsRisk` 策略，用 `Q - lambda * Risk` 选择动作，并与 `RL_Rainbow_GRU`、`RL_Rainbow_GRU_LateGuard` 做同一 test split 对比。
