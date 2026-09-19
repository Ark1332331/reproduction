# 论文方法与代码对应关系

本文件是当前仓库的论文对齐清单。这里的“论文”指目录中的 NSR PDF（2022 年接收
版本）。“实现假设”表示论文没有公开足够信息，无法确定该参数，不能把它当作作者
明确给出的超参数。

## 方法对应关系

| 论文中的方法 | 对应代码 | 当前状态 |
|---|---|---|
| 当前输出和上一时刻输出转换到当前坐标系，再沿时间维拼接 | `r7_autoregressive_rollout.py`、`r1_data_representation.py` | 已实现；上一帧输入是模型停止梯度后的预测，标记为 `k=1` |
| 使用 `64×64×64` 网格表示 `3.2×3.2×3.2 m` 空间 | `paper_config.py`、`r1_data_representation.py` | 默认配置一致 |
| 体素特征是点相对于体素角点的质心偏移 | `r1_data_representation.py` | 已实现 |
| 四次空间下采样卷积，时间维保持不变 | `r5_sparse_model.py` | 结构一致；论文没有公开精确 kernel 和 channel 数 |
| U-Net 跳跃连接和稀疏生成式解码器上采样 | `r5_sparse_model.py` | 结构一致；具体 block 布局属于实现假设 |
| 解码阶段使用 likelihood 剪枝，`α=0.5` | `r5_sparse_model.py` | 机制和默认阈值已实现；target guard 是额外的工程保护选项 |
| 最终输出是 3D 子体素点估计 | `r5_sparse_model.py`、`r7_autoregressive_rollout.py` | 已实现；最终候选限制为 `k=0` 是根据方法契约作出的明确推断 |
| 占用 BCE 加平均欧氏位置偏移损失 | `r5_sparse_loss.py` | 已实现；论文没有说明两项损失的精确权重 |
| 12 步滚动预测，时间之间不传播梯度 | `r7_autoregressive_rollout.py` | 已实现 |
| Adam，初始学习率 `0.01`，指数衰减到 `0.0001` | `paper_config.py`、`run_r7_paper_train.py` | 默认配置一致 |
| 位置、倾角、高度 patch、剪枝、离群点、位姿噪声和 x/y 镜像增强 | `r7_data_augmentation.py`、`r7_measurement_augmentation.py` | 主线训练已实现；具体 patch 采样方式属于实现假设 |
| 前后左右四个深度相机，向下倾斜 30° | `collect_isaaclab_anymal_vectorized.py` | 方向、数量和倾角意图一致；安装位置、分辨率和点数上限属于实现假设 |

## 数据生成对应关系

| 论文描述 | 当前实现 | 影响 |
|---|---|---|
| IsaacGym 随机生成 stairs、boxes、walls、roadblocks/结构化障碍物和 corridors | IsaacLab ANYmal-C 采集器和五类命名地形 | 仿真器和运动控制栈不同，因此不是完全相同的数据源；新的 vectorized 采集默认使用随机速度和初始 yaw |
| stairs 宽度 `[0.2, 0.5] m`，高度 `[0.08, 0.25] m` | `paper_terrains.py` | 已按论文范围设置；修改前采集的 NPZ 仍属于历史数据 |
| boxes 长宽 `[0.2, 2.0] m`，高度 `[0.08, 0.25] m` | `paper_terrains.py` | 是当前最接近论文的实现；box 数量和布局属于实现假设 |
| walls 产生宽度 `[2, 6] m` 的 corridors | `paper_terrains.py` | 已使用该范围；墙体尺寸、高度和布局属于实现假设 |
| pole 尺寸和完整场景采样过程 | `paper_terrains.py`、采集器 | 论文没有完整说明；当前 pole 几何是实现假设 |
| 超过 200,000 个时间步观测，平均可见率 43% | `data/` 中的采集数据和 manifest | 当前数据量必须以 manifest 为准，不能仅凭运行评估声称达到论文规模；新数据包含代码、checkpoint 和环境溯源，历史 NPZ 不一定包含 |

## 评估对应关系

论文报告机器人中心 `64³` 体素网格上的平均 Precision、Recall、F1 和平均绝对高度
误差。当前评估器同时输出：

- 逐帧平均的 macro 指标，这是主要的论文风格报告；
- 按总计数计算的 micro 指标；
- 高度误差和共享 XY 单元覆盖率。

高度误差属于“论文启发的诊断指标”：当前实现对每个 XY 单元取最高 z，并只在预测和
目标共享 XY 单元上计算。论文没有明确高度匹配和归约规则，因此不能在不说明限制的
情况下直接与论文表格比较。

当前测量合并 baseline 是针对本地采集数据的工程 baseline。论文中的 baseline 数值
来自真实机器人实验，不能仅通过运行当前评估器得到。

## 主线和诊断配置

下面的选项属于诊断配置，使用时必须写入 checkpoint 或结果 JSON：

- `--disable-data-augmentation`；
- 不等于 `0.5` 的 `--pruning-alpha`；
- 不等于 pruning alpha 的 `--feedback-alpha`；
- 外部 `--likelihood-logit-offset` 校准；
- 正类权重、warmup、修改后的 channel 或 generative kernel；
- `--target-guard` 和 no-history ablation。

当前主线训练默认关闭 target guard，保持严格论文模式；target guard 只能作为工程
保护或对比实验使用。

本地 `data/` 和 `results/` 中包含多个诊断实验结果。它们适合排查问题，但不能和
论文对齐的主线结果混为一谈。

## 当前复现状态

当前项目仍处于**部分完成**状态：体素表示、稀疏网络契约、采集契约和多项诊断已经
完成，但还没有稳定的论文规模训练/评估结果。详细证据、失败 gate、checkpoint 和
下一步计划见 `AGENTS.md`。
