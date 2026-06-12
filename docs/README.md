# RL_sim 文档索引

本文档是仓库文档入口，用于说明哪些文件代表当前主线，哪些文件只是历史记录或探索支线。项目当前以中文文档为主，实验结论应优先引用具体日期、run、seed、checkpoint 和输出路径。

## 推荐阅读顺序

1. [README.md](../README.md)：项目目标、目录结构和常用入口。
2. [AGENTS.md](../AGENTS.md)：给代码代理的最新工作约定和当前推荐实验基线。
3. [design/problem_scope_and_boundaries.md](design/problem_scope_and_boundaries.md)：论文和实验表述的边界，避免把 A3 切换优化夸大成覆盖增强或资源调度。
4. [experiments/README.md](experiments/README.md)：当前实验时间线、主线证据和探索支线。
5. [paper/README.md](../paper/README.md)：论文材料、已生成表格和 PaperSpine 启动状态。

## 设计文档

| 文件 | 作用 | 当前状态 |
| --- | --- | --- |
| [design/problem_scope_and_boundaries.md](design/problem_scope_and_boundaries.md) | 定义核心问题、算法可控范围和不可控边界 | 当前主线，论文表述优先遵守 |
| [design/README_MODEL.md](design/README_MODEL.md) | 说明 `Obs7 + GRU + Rainbow DQN` 模型结构、动作空间和训练/导出入口 | 当前主线 |
| [design/simu5g_rl_policy_table.md](design/simu5g_rl_policy_table.md) | 说明 policy table 导出和 Simu5G 查表控制 A3 的接口 | 当前主线 |
| [design/risk_aware_model_improvement_plan.md](design/risk_aware_model_improvement_plan.md) | 风险感知 guard / risk head 的改进路线 | 探索支线，不能替代默认方案 |
| [design/physics_informed_risk_prediction_tasks.md](design/physics_informed_risk_prediction_tasks.md) | 物理风险预测头任务说明 | 探索支线 |

## 使用文档

| 文件 | 作用 | 当前状态 |
| --- | --- | --- |
| [usage/QUICKSTART_OFFLINE.md](usage/QUICKSTART_OFFLINE.md) | 快速采集数据、离线训练和评估 | 可用，适合快速回忆命令 |
| [usage/README_TRAINING.md](usage/README_TRAINING.md) | 当前训练、评估、导出命令总览 | 当前主线 |
| [usage/install_pytorch_cuda.md](usage/install_pytorch_cuda.md) | CUDA 版 PyTorch 安装说明 | 可用 |
| [usage/TRAINING_GUIDE.md](usage/TRAINING_GUIDE.md) | 早期训练架构草稿 | 历史参考，不作为当前实现说明 |

## 实验记录

正式实验记录放在 [experiments/](experiments/)；该目录中的 [README.md](experiments/README.md) 按时间线标出主线证据、论文可用证据和探索支线。新的实验记录应包含：

- 实验目标和算法边界。
- 数据集、checkpoint、run 目录、随机种子和命令。
- 指标表格和明确结论。
- 哪些结论可用于论文，哪些只是工程诊断。

## 对比算法与论文材料

| 位置 | 作用 |
| --- | --- |
| [../comparison_algorithms/README.md](../comparison_algorithms/README.md) | Python 侧 baseline 策略、runner 和输出说明 |
| [../comparison_algorithms/docs/literature_to_baselines.md](../comparison_algorithms/docs/literature_to_baselines.md) | 文献思想到 baseline 的映射 |
| [../paper/README.md](../paper/README.md) | 论文表格、文献综述和 PaperSpine 准备状态 |
| [../paper/references/literature_review_handover_rl_5gr_20260611.md](../paper/references/literature_review_handover_rl_5gr_20260611.md) | 高铁 5G-R/LTE-R 切换优化文献调研 |

## 历史材料

[archive/](archive/) 保存早期动作空间、环境改造和 GPU 优化记录。这些材料可帮助理解项目演进，但其中的旧接口、旧奖励或旧观测维度不代表当前主线。当前模型固定为 `[15, 7]` 观测窗口，默认奖励为 R5，训练/评估统一使用 action hold。

## 输出与 Git 约定

- 新训练和评估输出写入 `experiments/runs/<run_id>/`。
- 可复用结果写入 `results/<topic>/`。
- 论文最终表格和图放入 `paper/assets/`。
- 临时文件放 `_inbox/`，阶段整理时迁出或删除。
- 大型数据、checkpoint、实验完整输出、venv 和缓存由 `.gitignore` 排除，不应进入普通代码提交。
