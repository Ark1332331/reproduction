# NSR 复现项目

本目录用于复现论文 *Neural Scene Representation for Locomotion on
Structured Terrain*（结构化地形上的运动场景表示）。论文 PDF 是判断方法是否
符合原文的主要依据。仓库中没有作者的原始代码；论文没有明确说明的参数，都会
在代码或文档中标记为“实现假设”。

## 从这里开始

1. 阅读 [`PAPER_ALIGNMENT.md`](PAPER_ALIGNMENT.md)，了解论文方法和当前代码的对应关系。
2. 阅读 [`AGENTS.md`](AGENTS.md)，了解实验历史、已完成工作和当前状态。
3. 使用 `python -m training.run_r7_paper_train` 和 `python -m evaluation.run_r7_paper_eval` 进行主线训练和评估。
4. 阅读 [`MAINLINE_ARCHITECTURE.md`](MAINLINE_ARCHITECTURE.md)，了解文件职责和执行顺序。

论文相关的默认参数集中在 [`paper_config.py`](paper_config.py) 中，包括：

- `64×64×64` 体素网格；
- `3.2×3.2×3.2 m` 的空间范围；
- `0.05 m` 体素大小；
- 12 步时间滚动预测；
- 剪枝阈值 `alpha=0.5`；
- Adam 学习率从 `0.01` 指数衰减到 `0.0001`。

主线训练默认关闭 `target_guard`，以保持论文方法的候选生成逻辑。只有在做工程对比
时，才使用 `--target-guard`，并在实验记录中明确标注。

## 项目目录和文件分工

| 文件或目录 | 作用 |
|---|---|
| `configs/` | 论文参数和明确标注的实现假设 |
| `simulation/` | 地形生成、IsaacLab 采集、采集调度和数据溯源 |
| `data_pipeline/` | 训练/验证划分和数据接口工具 |
| `representation/` | 位姿对齐、点云体素化和体素质心偏移表示 |
| `models/` | MinkowskiEngine 稀疏输入和四层稀疏网络 |
| `losses/` | 占用、likelihood 和位置偏移损失 |
| `rollout/` | 12 步自回归滚动预测 |
| `training/`、`evaluation/` | 训练、评估、数据增强和评估指标 |
| `tests/` | 单元测试和方法契约测试 |
| `data/` | 本地轨迹、manifest、日志和诊断结果 |
| `results/` | 本地 checkpoint 和评估结果 |

`data/` 和 `results/` 中的内容是实验证据，不是源代码。它们保留在本地，并被 Git
忽略，不会进入源码仓库。

新采集的 NPZ 文件包含 `capture_schema_version` 和 `provenance` 信息。使用历史 NPZ
文件时，需要检查地形配置、运动随机化方式和 provenance，不能直接与新数据混用。

## Conda 环境

本机默认 Python 环境没有 `numpy`、`torch` 或 `MinkowskiEngine`。请在正确的 Conda
环境中运行代码：

- `isaaclab`：运行 IsaacLab/Isaac Sim、ANYmal、相机和地形采集；
- `nsr-me-cu130-t291`：运行 PyTorch、MinkowskiEngine、模型训练、评估和稀疏网络测试。

在仓库目录运行测试：

```bash
python -m unittest discover -s tests -p 'test_*.py'
```

如果缺少依赖，这是环境配置问题，不代表模型或复现方法本身失败。
