# RL_sim：高速铁路场景下的强化学习切换优化

本项目用于研究高速铁路 5G/蜂窝网络场景中的切换参数优化问题。系统将列车沿一维轨道从小区 A 移动到小区 B 的过程建模为强化学习环境，使用 Rainbow DQN 学习自适应的切换迟滞 `Hys` 与触发时间 `TTT` 参数，以降低中断、过晚切换和不必要切换。

当前仓库已经按“源码、数据、实验、结果、论文材料”重新整理。完整分类规则见 [docs/项目文件分类与整理规范.md](docs/项目文件分类与整理规范.md)。

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

## 常用命令

收集离线训练数据：

```bash
python scripts/data/collect_data.py --num_episodes 100 --output_path data/datasets/offline_dataset.npz
```

离线训练：

```bash
python scripts/train/train_rainbow_offline.py --dataset_path data/datasets/offline_dataset.npz --use_cql
```

在线训练：

```bash
python scripts/train/train_rainbow.py
```

单场景评估：

```bash
python scripts/eval/test_simple.py
```

批量评估：

```bash
python scripts/eval/test_batch_comparison.py
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
