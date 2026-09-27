# NSR Retry

`retry/` 是论文 *Neural Scene Representation for Locomotion on Structured Terrain* 的**审计式重新复现路径**。

仓库根目录现有代码保留为旧实现参考，不自动视为论文真实实现。

## 工作方式

后续统一按照以下顺序推进：

**论文要求 → 本阶段精读 → 经确认的实现契约 → 最小实现 → 验证 → 阶段报告 → commit + push**

不允许把旧仓库整体复制进 `retry/`。`retry/` 的代码结构应该随着复现理解逐阶段增长。

## 文档说明

- `docs/PAPER_REQUIREMENTS_MATRIX.md`：全文复现要求总索引。
- `docs/DECISIONS.md`：已经确认的偏差、实现假设和未决选择。
- `docs/PROGRESS.md`：简洁的当前进度和阶段状态。
- `docs/stages/`：每个阶段一份可独立审计的交付文档。

## 计划顺序

- Stage -1：全文复现要求扫描
- Stage 0：任务定义、输入输出、坐标系、成功标准
- Stage 1：仿真与数据生成
- Stage 2：点云预处理与稀疏表示
- Stage 3：4D 稀疏网络与剪枝
- Stage 4：损失函数与训练流程
- Stage 5：自回归时序反馈
- Stage 6：数据增强与鲁棒性设置
- Stage 7：评估指标与论文实验
- Stage 8：论文—retry 最终对齐审计
