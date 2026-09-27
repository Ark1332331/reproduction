# Fig.2 网络结构精细审计

论文：*Neural Scene Representation for Locomotion on Structured Terrain*

本文件只审计 Fig.2 及其紧邻正文中能够直接支持网络结构复现的信息。目的不是替作者补全未公开细节，而是把图里已经给出的信息逐层展开，避免被一句“4D U-Net-like encoder-decoder”吞掉。

## 1. 审计结论

Fig.2 不是一个泛化示意图，它已经提供了相当具体的结构信息：

- 1 条输入主线：当前测量 `M_t` 与变换到当前坐标系的上一时刻预测 `P̂_{t-1}` 先沿时间维拼接；
- 9 个连续的视觉尺度块（这是本审计为便于追踪所作的块划分，论文正文未给这些块命名）；
- 22 个可按图例识别的卷积类操作；
- 4 种卷积/剪枝操作类型；
- 4 条 encoder→decoder skip connection；
- encoder 中 4 次 stride=2 空间下采样；
- decoder 中 4 次 stride=2 transposed convolution 空间上采样；
- decoder 中 4 次 “convolution followed by pruning”；
- 最后一个普通 convolution 输出 3 维 feature，对应 sub-voxel position。

图中没有给出：各层 channel 数、kernel size、padding、具体激活函数、normalization、skip 的 concat/add 方式、每个 block 内除图示操作外是否还有未画出的辅助层。

## 2. 四种图例操作

| 图例颜色 | 论文图例名称 | 本审计简称 | 图中计数 |
|---|---|---|---:|
| 绿色 | Convolution | `Conv` | 10 |
| 蓝色 | Convolution with stride 2 | `Conv-S2` | 4 |
| 灰色 | Transposed convolution with stride 2 | `TConv-S2` | 4 |
| 红色 | Convolution followed by pruning | `Conv+Prune` | 4 |
|  | **合计** |  | **22** |

这里的“22”是按 Fig.2 中每个颜色操作块逐个计数得到，不是由正文句子推导出来的。

## 3. 9 个视觉块与 22 个操作逐层编号

> 注意：`E1/E2/.../D1` 是 retry 审计标签，不是论文原命名。它们的作用只是让后续代码逐层对应。

| 序号 | 审计层 ID | 所属视觉块 | 图中类型 | 作用/位置 | 论文支持状态 |
|---:|---|---|---|---|---|
| 1 | E1-C1 | Encoder-1 | Conv | 输入分辨率上的普通卷积 | 图中明确 |
| 2 | E1-DOWN | Encoder-1 | Conv-S2 | 第 1 次空间下采样 | 图中明确；正文明确 stride=2 |
| 3 | E2-C1 | Encoder-2 | Conv | 第 2 个尺度普通卷积 | 图中明确 |
| 4 | E2-DOWN | Encoder-2 | Conv-S2 | 第 2 次空间下采样 | 图中明确 |
| 5 | E3-C1 | Encoder-3 | Conv | 第 3 个尺度普通卷积 | 图中明确 |
| 6 | E3-DOWN | Encoder-3 | Conv-S2 | 第 3 次空间下采样 | 图中明确 |
| 7 | E4-C1 | Encoder-4 | Conv | 第 4 个尺度普通卷积 | 图中明确 |
| 8 | E4-DOWN | Encoder-4 | Conv-S2 | 第 4 次空间下采样 | 图中明确 |
| 9 | B-C1 | Bottleneck | Conv | 最低空间分辨率普通卷积 | 图中明确 |
| 10 | B-UP | Bottleneck | TConv-S2 | decoder 第 1 次上采样 | 图中明确 |
| 11 | D4-C1 | Decoder-4 | Conv | 接收对应 skip 后的普通卷积 | 图中明确，具体 skip 合并方式未说明 |
| 12 | D4-PRUNE | Decoder-4 | Conv+Prune | 生成 likelihood 并执行 pruning 的图示阶段 | 图中明确；正文说明 pruning 机制 |
| 13 | D4-UP | Decoder-4 | TConv-S2 | decoder 第 2 次上采样 | 图中明确 |
| 14 | D3-C1 | Decoder-3 | Conv | 普通卷积 | 图中明确 |
| 15 | D3-PRUNE | Decoder-3 | Conv+Prune | pruning | 图中明确 |
| 16 | D3-UP | Decoder-3 | TConv-S2 | decoder 第 3 次上采样 | 图中明确 |
| 17 | D2-C1 | Decoder-2 | Conv | 普通卷积 | 图中明确 |
| 18 | D2-PRUNE | Decoder-2 | Conv+Prune | pruning | 图中明确 |
| 19 | D2-UP | Decoder-2 | TConv-S2 | decoder 第 4 次上采样 | 图中明确 |
| 20 | D1-C1 | Decoder-1 | Conv | 输入空间尺度附近的普通卷积 | 图中明确 |
| 21 | D1-PRUNE | Decoder-1 | Conv+Prune | 最后一次图示 pruning | 图中明确 |
| 22 | OUT-C1 | Output | Conv | 最终普通卷积；正文说明 feature 维度映射到 3 | 图中明确 + 正文明确输出 3 维 |

## 4. 为什么是 9 个“大块”而不是 22 个“大块”

按图的视觉分组，可以把连续操作组合为：

1. Encoder-1：`Conv → Conv-S2`
2. Encoder-2：`Conv → Conv-S2`
3. Encoder-3：`Conv → Conv-S2`
4. Encoder-4：`Conv → Conv-S2`
5. Bottleneck：`Conv → TConv-S2`
6. Decoder-4：`Conv → Conv+Prune → TConv-S2`
7. Decoder-3：`Conv → Conv+Prune → TConv-S2`
8. Decoder-2：`Conv → Conv+Prune → TConv-S2`
9. Decoder-1 / Output：`Conv → Conv+Prune → Conv`

因此：

- 4 个 encoder 块 × 2 层 = 8 层；
- bottleneck = 2 层；
- 3 个中间 decoder 块 × 3 层 = 9 层；
- 最终 decoder/output 块 = 3 层；
- 总计 `8 + 2 + 9 + 3 = 22`。

## 5. Skip connections

Fig.2 明确画出 4 条 encoder→decoder 跨层连接，分别连接四个对应空间尺度。

可直接从图确认的是“存在 4 条对应尺度 skip path”；不能从图或正文确认的是：

- skip 是 `concatenate`、`add` 还是其他融合方式；
- 融合发生在 decoder 普通卷积之前还是通过具体框架的 coordinate manager 做其他处理；
- 融合后 channel 如何变化。

这些继续保留为待审批实现假设。

## 6. 空间/时间分辨率信息

正文明确：

- encoder 共 4 个 strided convolution；
- 每次仅在三个 spatial axis 上缩小 2 倍；
- temporal dimension 不被下采样；
- 因此空间总缩小倍率为 16；
- 两个时间层在不同空间尺度上保持分离，同时卷积跨空间和时间执行。

结合输入 `64×64×64`，空间尺寸序列可确定为：

`64³ → 32³ → 16³ → 8³ → 4³`

时间坐标在 encoder 下采样过程中保持两层，因此 latent 的讨论尺寸为 `4×4×4×2`。

论文随后说明，如果对这个近乎满占据的 latent 做普通上采样，会趋向产生 `64×64×64×1` 的满网格，因此 decoder 使用 generative upsampling + pruning 来避免稠密爆炸。

“2 → 1 的时间维如何在 decoder 中具体实现”没有被论文完全展开，仍是实现审批项。

## 7. Pruning 机制与图中红色层的关系

正文给出的机制是：

1. 当前 sparse feature tensor 为 `T_f`；
2. 用一个单独 convolution 产生 likelihood；
3. 经过 sigmoid 得到 sparse tensor `T_p`；
4. 删除 `T_p < alpha` 对应的 `T_f` 元素；
5. `T_p` 自身不传到后续层；
6. 默认 `alpha=0.5`。

Fig.2 中红色图例名称是 **“Convolution followed by pruning”**。因此图中 4 个红色操作应当逐一对应 decoder 的 4 个 pruning stage，而不能只在代码里实现一次末端剪枝。

## 8. 对后续代码审计的硬要求

进入网络实现阶段时，代码必须能提供一张逐层对照表，至少包含：

| Fig.2 层 ID | retry 类/模块 | 具体代码位置 | 操作类型 | stride | kernel | in/out channels | temporal stride | pruning? | 验证结果 |
|---|---|---|---|---|---|---|---|---|---|

其中论文没给出的 `kernel`、`channel`、skip merge 等，不允许空白后偷偷采用旧仓库值；必须指向 `DECISIONS.md` 中已批准的实现假设。

## 9. 与旧 Stage -1 的差异

旧扫描只记录“U-Net-like 4D encoder-decoder + skip connections”和“4 次 stride-2 下采样”等摘要，粒度不足。

本次修订后，Fig.2 至少被拆成：

- 4 类操作；
- 9 个视觉块；
- 22 个卷积类操作；
- 4 条 skip；
- 4 次 downsample；
- 4 次 upsample；
- 4 次 decoder pruning；
- 1 个最终 3 维输出 convolution；
- 若干明确的未公开实现细节。

后续 `PAPER_REQUIREMENTS_MATRIX.md` 必须按这一粒度追踪，而不能退回一句话概括。
