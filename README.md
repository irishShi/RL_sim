# RL_sim：高速铁路场景下的强化学习切换优化

本项目用于研究高速铁路 5G/蜂窝网络场景中的切换参数优化问题。系统将列车沿一维轨道从小区 A 移动到小区 B 的过程建模为强化学习环境，使用 Rainbow DQN 学习自适应的切换迟滞 `Hys` 与触发时间 `TTT` 参数，以降低中断、过晚切换和不必要切换。

当前仓库已经按“源码、数据、实验、结果、论文材料”重新整理。完整分类规则见 [docs/项目文件分类与整理规范.md](docs/项目文件分类与整理规范.md)。

文档入口：

- [docs/README.md](docs/README.md)：文档总索引。
- [docs/experiments/README.md](docs/experiments/README.md)：实验记录时间线和主线证据。
- [paper/README.md](paper/README.md)：论文材料、表格和 PaperSpine 准备状态。
- [comparison_algorithms/README.md](comparison_algorithms/README.md)：baseline 对比算法说明。

## 主要目标与边界

本项目的主要目标是优化高速铁路相邻小区重叠区内的 A3 参数选择：让智能体根据 `RSRP/SINR/速度/位置/距上次切换时间` 等观测动态选择 `(Hys, TTT)`，在减少晚切、过早切、乒乓、切换中断和 outage 之间取得平衡。

需要明确的是，弱覆盖、强干扰、NLOS/fading 等无线退化并不是完全排除在算法能力之外。在一定程度内，若相邻小区之间仍存在可利用的质量差异，切换算法应通过更合适的 `Hys/TTT` 缓解晚切、过早切或乒乓带来的损失。真正的边界是极端情况下两个候选小区都质量很差、目标小区同样不可用，或业务/资源不足导致排队和时延尾部恶化；这时单靠调整 `Hys/TTT` 不能替代网络规划、覆盖增强、资源调度或多链路可靠性设计。

后续新增观测维度、reward 项或模型模块时，应先证明它确实改善“切换可控”的核心指标，例如 `ho_per_km`、`ping_pong_count`、`outage_time_ratio`、`sinr_p5_db`、`overlap_zone_sinr_mean_db` 和 `comm_interruption_ratio_in_overlap_zone`，且收益足以抵消复杂度、训练成本和推理效率损失。详细边界说明见 [docs/design/problem_scope_and_boundaries.md](docs/design/problem_scope_and_boundaries.md)。

## 目录结构

```text
RL_sim/
├─ configs/                  # 环境、模型、训练配置
├─ envs/                     # 仿真环境、信道模型、切换逻辑
├─ models/                   # Rainbow 网络、动作空间、观测窗口
├─ utils/                    # 经验回放、C51 投影、数据加载、场景生成
├─ scripts/                  # 可执行脚本
│  ├─ train/                 # 在线/离线训练
│  ├─ eval/                  # 单场景与批量评估
│  ├─ data/                  # 数据收集、真实轨迹处理
│  └─ plot/                  # 绘图脚本
├─ data/                     # 数据文件
│  ├─ raw/                   # 原始数据
│  ├─ processed/             # 清洗后的数据
│  ├─ datasets/              # 离线训练数据集
│  └─ scenarios/             # 固定随机种子的评估场景
├─ experiments/runs/         # 每次实验的完整输出
├─ results/                  # 精选结果，用于分析和论文
├─ paper/                    # 论文正文、图表、参考文献和投稿材料
├─ docs/                     # 技术文档、使用说明和历史文档
└─ _inbox/                   # 临时收纳箱
```

## 安装依赖

```bash
pip install -r requirements.txt
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

## 兼容性提示

当前环境已移除天气损耗、温度观测特征以及策略输入中的当前 Hys/TTT 特征，模型输入由 `[15, 10]` 调整为 `[15, 7]`。整理前的历史数据集、legacy checkpoint 以及旧 9 特征版本 checkpoint 仍保留在归档目录中，但不能直接用于当前代码；继续实验前需要重新采集离线数据并重新训练模型。

## 当前论文准备状态

截至 2026-06-12，项目已经具备启动 PaperSpine `build_from_materials` 的材料基础：主模型流水线、domain randomization、late guard、baseline 对比、移动性成功率 proxy、Simu5G 查表验证、实测轨迹校准和文献综述均已有记录。当前仍不应写成“RL 全面优于固定 A3”；更稳妥的论文主线是强调 `RL + LateGuard` 在低切换、低乒乓和低重叠区中断方面的折中优势，以及对原始 RL 晚切失败的缓解。

论文候选表格位于：

```text
paper/assets/tables/final/
```

论文材料索引见 [paper/README.md](paper/README.md)。

## Action Hold 规则

数据采集、在线训练、评估和真实轨迹验证统一使用自适应 action hold：智能体选定 `(Hys, TTT)` 后会保持若干步，默认保持步数为 `max(6, ceil(TTT / delta_t) + 2)`，并限制在 20 步以内。这样可以让 TTT 计时器完整生效，避免每步重选参数导致切换逻辑被反复重置。

## 常用命令

完整命令和更多评估入口见 [docs/usage/README_TRAINING.md](docs/usage/README_TRAINING.md)。下面保留当前推荐主线的核心命令。

收集 V1 domain random Obs7 数据：

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

单场景评估：

```powershell
.\.venv\Scripts\python.exe scripts\eval\test_simple.py `
  --checkpoint_path experiments/runs/20260604_v1_domain_random_obs7_nocql_seed20260604/checkpoints/rainbow_offline_best.pth
```

profile holdout 泛化评估：

```powershell
.\.venv\Scripts\python.exe scripts\eval\test_profile_generalization.py `
  --checkpoint_path experiments/runs/20260604_v1_domain_random_obs7_nocql_seed20260604/checkpoints/rainbow_offline_best.pth `
  --scenario_profiles_path configs/scenario_profiles.yaml `
  --profile_split test `
  --num_seeds 20
```

导出当前推荐 Simu5G 策略表：

```powershell
.\.venv\Scripts\python.exe tools\export_late_guard_policy_preset.py delta3_low5 --copy_to_default
```

真实轨迹预处理：

```bash
python scripts/data/validate_gnb_trace.py --preprocess_only --save_csv
```

真实轨迹绘图：

```bash
python scripts/plot/plot_gnb_trace.py
```

## 输出约定

新的训练和评估结果应写入 `experiments/runs/<run_id>/`。历史整理前的实验产物已经收纳到：

```text
experiments/runs/legacy_20260416_offline_rainbow/
```

论文最终使用的图表放入：

```text
paper/assets/figures/final/
paper/assets/tables/final/
```

不要再把 PNG、JSON、checkpoint 或临时数据直接输出到项目根目录。
