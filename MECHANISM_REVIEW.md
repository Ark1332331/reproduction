# 论文复现：训练机制修正与初始化不稳定性的审查材料

> 目的：给外部 AI 审查用。本文件自包含，不需要项目上下文。
> 项目：复现论文《Neural Scene Representation for Locomotion on Structured Terrain》（4D 稀疏 U-Net 场景补全，MinkowskiEngine）。
> 日期：2026-08-27。当前阶段：论文式逐层 likelihood/pruning 的坐标合同刚完成修复与 smoke 验证；尚未具备 validation 条件。

---

## 1. 论文方法（我们要复现的核心）

- 输入：当前帧 + 上一帧的**体素化点云**（4D 坐标 [batch, x, y, z, k]，k=0/1 表示当前/上一帧）。
- 网络：4D 稀疏 U-Net（MinkowskiEngine），encoder 4 次 stride=[2,2,2,1] 下采样（时间维 k 不下采样），decoder 用**生成式转置卷积**（MinkowskiGenerativeConvolutionTranspose）在每层**生成候选体素**，生成式卷积本身能产生输入坐标附近的新坐标（这是"补全"的机制来源）。
- 每层 decoder 输出 likelihood（occupancy 概率），按 alpha 阈值剪枝；最终输出：occupancy logits + sub-voxel offset（3D 位移，把体素中心移到真实表面点）。
- 训练：per-layer likelihood BCE 监督 + 位置损失（仅对正样本）。推理：12 步 detached autoregressive rollout（上一帧模型估计变换后作为 k=1 输入）。

论文未公开代码；作者未在正文给出 channels/kernel/剪枝细节。**以下实现均为我们的推断**。

但以下三项是原文明确陈述、而不是推断：每个 decoder layer 都生成 likelihood 并按 alpha pruning；每层的
likelihood 都有 BCE target；作者实验上发现 alpha=0.5 是 precision/recall 的良好折中。原文还说明训练数据
有超过 200,000 个 point-cloud observations（III-B/III-C/III-D，https://arxiv.org/abs/2206.08077）。

## 2. 我们的实现（与论文的偏差点，均为自行选择）

| 组件 | 我们的实现 |
|---|---|
| 网络 | FourLevel4DCompletionModel，channels=(4,8,12,16)，四层 4D U-Net |
| 生成候选 | 每层 GenerativeConvTranspose 产生候选，`align_skip_to_candidates` 对齐 skip |
| 逐层监督 | `multiscale_likelihood_loss`：把 target 逐层 downsample（stride 8/4/2/1），对 decoder 中间层候选做 BCE（无加权） |
| 训练剪枝 | forward 中 `prune_candidates`（alpha + target-guard：保护坐标恰好等于 target 坐标的候选） |
| 最终 loss | `completion_loss`：occupancy BCE（**最近加了 pos_weight**）+ 正样本 offset 均方距离；+ likelihood 项 |
| 数据 | Isaac Lab 真实 ANYmal-C 物理轨迹（官方 rsl_rl checkpoint 驱动）在 stairs/boxes/walls/poles/corridors 地形上行走；4 深度相机 3.2m 局部窗口；体素化 0.05m/64 grid；40 条轨迹（train 20 + validation 20，每条 2-12 帧） |

## 3. 已确认的实现错误与失败链

1. **多尺度 target 坐标错位（2026-08-27 修复）**：旧 `downsample_occupancy_target` 直接将 target x/y/z 除以 stride，例如 fine `(17,9,3)` 变成 `(2,1,0)`。但 MinkowskiEngine 的 stride-8 decoder tensor 仍使用原始坐标单位，坐标为 `(16,8,0)`，stride 记录在 metadata。因此 l3/l2 的正样本 coverage 被错误审计为 0，per-layer BCE 也实际收不到正样本。现已改为对 stride grid 对齐：`(coord // factor) * factor`。43 帧审计修复后 l3/l2/l1/l0 coverage 分别为 0.599/0.606/0.615/0.623（旧算法为 0/0/0.125/0.623）。
2. **时间标签污染（后续发现）**：temporal encoder 让 k=1 输入传播到 decoder；旧 decoder 还会生成 k>0 final candidates，而 target 仅有 k=0。于是所有 k>0 candidate 被错误标成负样本，正样本率被从约 6% 拉低到约 1%。现已改为 final output 只保留 k=0，loss 同样只监督 k=0。
3. **过强 pos_weight（后续发现）**：在正确标签合同下，单帧 200-step 控制实验中：pos_weight=1 达到 F1=0.552（P=0.494，R=0.627），auto (~16) 仅 F1=0.112，20 为 F1=0.362。auto/20 使大量空候选卡在高 sigmoid，主要失败模式是 false positive，而非全空。
4. **likelihood/pruning 已完成合同 smoke，但未完成泛化验证**：修复坐标后，随机初始化的 alpha=0.5 target-guard probe 在 43 帧上每层都能得到正 BCE target（平均 l3=40、l2=164、l1=673），且 target guard 保留全部“生成候选本来可达”的 target。单 boxes 轨迹 20 step smoke 中，启用 likelihood、alpha=0.5 后 likelihood loss 从 0.6929 降至 0.6459，final loss 从 1.4634 降至 1.4503；这只证明计算链能训练，不能证明论文方法或稳定性。

## 4. 当前有效的训练合同与证据

当前受到控制验证的 ablation：

```text
final output candidate: k=0 only
completion supervision: k=0 only
occupancy pos_weight: 1 (unweighted BCE)
likelihood_weight: 0
training pruning_alpha: 0
training feedback_alpha: 0
evaluation: internal pruning disabled; output_alpha 和 feedback_alpha 分离
```

以上仍是唯一有多-seed F1 证据的 ablation。论文式 likelihood/pruning 现在已经通过坐标、target guard 与单轨迹 smoke
验证，但尚未在 43 帧五轨迹设置做多 seed F1 对照，因此不能将它写成“已验证有效的最终训练配置”。

单 trajectory（boxes_s10）300-step 结果：output_alpha=0.3 时 autoregressive F1=0.7613、
baseline F1=0.3594、MAE=0.0006 vs baseline 0.0680。无历史 F1=0.7525，时间收益在单轨迹上很小。

五 trajectory（seed 0 的五种 terrain）300-step 结果，预先固定 output_alpha=0.1：

| 初始化 seed | history F1 | 无 history F1 | baseline F1 | 结论 |
|---|---:|---:|---:|---|
| 0 | 0.448 | 0.305 | 0.338 | 训练集通过；有历史收益 |
| 1 | 约 0.475 | 约 0.394 | 0.338 | 训练集通过；有历史收益 |
| 2 | 0.314（阈值扫描最佳） | 未跑 | 0.338 | 训练集失败 |

因此当前实验不能进入 validation；它显示的是**有效解存在但初始化敏感**，不是稳定复现成功。

## 5. 当前状态与下一步

- 已完成三次 5-trajectory / 300-step 训练；其中 2 次超过训练集 baseline、1 次失败。
- 当前 P0 不是 validation，而是解释 seed 2 为什么保留了高 recall（约 0.60）却有极低 precision（约 0.21）。
- 三 checkpoint 只读审计已完成：三者正样本 sigmoid 均值接近 0.20；seed 2 的负样本 sigmoid p90=0.50，seed 0/1
  约 0.11。所有模型在 `feedback_alpha=0` 下每帧反馈约 40,579 个点（即 final candidates 全部反馈），并形成
  平均 24k-28k 个 k=1 输入体素；每帧 target 约 1,899 个。它说明 seed 2 的主要表现差异是 false-positive tail，
  并使全候选 feedback 成为可检验的高风险机制，但尚不构成因果证明。
- 已完成最小干预：只用失败的 seed 2 重跑 `feedback_alpha=0.1`，训练和评估保持相同阈值，其余配置完全固定。
  结果 F1 约 0.296，低于 baseline 且不如 feedback_alpha=0 的 seed 2；它只使每步计算从约 16 秒降至约 11 秒，
  不能解释或修复初始化不稳定性。下一步应先核查论文的 pruning / likelihood / 训练假设，而不是继续调 feedback 阈值。
- 该核查改变了当前路线：本地五条训练轨迹只有 43 time steps，而论文使用超过 200,000 observations。虽然 per-layer
  likelihood/pruning 的实现合同现已修正并做过短 smoke，小数据仍不能替代论文级训练。下一阶段应先扩展 randomized
  IsaacLab training distribution，再以预先固定的多-seed 训练集指标验证 paper-style loss/pruning；在此之前停止继续搜索
  seed、feedback alpha 或 learning rate。
- 数据目录盘点补充：目前有 40 条 IsaacLab ANYmal trajectory、共 319 帧；seeds 0--3 的完整训练划分为 20 条/164 帧，
  seeds 10--14 的验证划分为 20 条/155 帧。164 帧上的单 optimizer-step preflight 已完成（总 loss=2.1931、final=1.4999、
  likelihood=0.6932），证明扩展已有数据与 alpha=.5 可共同运行，但 319 仍与论文的 200k+ observations 相差三个数量级。
  当前 collector 是单机器人、单 trajectory 进程，不能靠手工多跑几条达到论文量级；下一项工程任务应是设计可恢复的批量/并行采集
  或有明确独立性边界的数据增强，而非直接开多 seed 长训。
- 采集流程基础设施已补齐并做了真实 pilot：`collect_r7_dataset.py` 以 manifest 管理 terrain/seed 工作项、检查 NPZ 的 R7
  metadata/array 合同、跳过有效样本、为每条采集保留日志，并显式进入 `isaaclab` 环境。它还修复了 Isaac Sim headless
  `close()` 完成后卡住 GPU 的问题。`boxes_s4` 采到 9 帧、`stairs_s4` 采到 12 帧，均通过 R7 loader；这验证的是采集链路，
  并不改变“数据规模不足”的论文级判断。
- 只有稳定化机制经多 seed 训练集验证后，才在独立 validation 轨迹上比较 baseline。

## 6. 请外部 AI 重点审查的问题

1. **初始化不稳定性**：在相同五条训练轨迹、相同 300-step budget 下，seed 0/1 超过 baseline，seed 2 的最优 F1 仍低于 baseline。最可能的失稳点是初始化、Adam/学习率日程、全候选 feedback，还是三者交互？应先做哪一种最低成本的受控诊断？
2. **feedback_alpha=0 的合理性**：它使每个 final candidate 都进入下一帧 k=1 history（包括低 likelihood 的大量点）。这与论文 autoregressive feedback 的 intended contract 是否相符？怎样避免重新引入训练/推理不一致，同时验证更稀疏 feedback 是否改善稳定性？
3. **逐层 likelihood target 的解释是否站得住**：当前已确认 target 必须保持 MinkowskiEngine 原始坐标单位、只对齐 stride grid。除这一坐标合同外，论文未给作者代码；每层 occupied target 的构造是否还可能与论文不同？
4. **生成式候选的覆盖问题**：k2 对 target 的 candidate coverage 约 60%，因此限制 recall。论文如何保证生成式候选覆盖 target？在当前稳定性问题未解决前，是否应避免再切 k3？
5. **alpha=0.5 的下一项有效验证**：短 smoke 已证明 alpha=0.5 可运行；数据规模扩展后，应如何预先注册一个最低成本的多-seed、训练集对照，来判断 pruning 是否提高而非损害 F1/MAE？
6. **指标与 baseline**：模型在 occupancy F1 上可超过 current-measurement baseline，但 offset height MAE 常落后。当前 height MAE 的比较是否存在“只对 matched occupancy 计算”的选择偏差？论文使用什么更公平的完整几何指标？
7. **整体判断**：当前应优先稳定训练（例如 feedback/优化日程的受控 ablation），还是暂停实现调研作者细节/论文补充材料？哪些证据才足以允许进入独立 validation？

## 7. 2026-09-06 论文实现对齐冻结（当前目标）

本节是后续实验的唯一 canonical 入口；在它通过小规模验收前，不启动 5k/10k/20k 长时间训练。

### 必须与论文一致或明确标注的项目

1. **输入与时序**：当前帧与上一帧体素化点云使用 `k=0/1`；训练和推理都验证 12-step detached autoregressive feedback，且只把合约规定的预测送入下一帧。
2. **生成与逐层监督**：四个 decoder 层均生成候选并输出 likelihood；每层使用 stride-grid 对齐后的 occupancy target 做 BCE，target 坐标不再除以 stride 而改变 MinkowskiEngine 坐标单位。
3. **最终输出与损失**：最终候选只保留 `k=0`；最终 occupancy BCE、正样本 sub-voxel position loss、逐层 likelihood BCE 的权重固定并记录，禁止为追求单次 F1 临时关闭其中一项。
4. **剪枝与阈值**：主结果使用内部 pruning、论文指定的 `alpha=0.5`；任何外部 logit offset 只能作为单独诊断，不得混入主结果。
5. **随机性与预算**：固定 Python/NumPy/PyTorch seeds、轨迹顺序和 optimizer/scheduler；记录真实 12-step 展开与 optimizer step 定义，避免把当前全数据循环误称为论文训练预算。
6. **数据与评估**：明确记录当前约 20k train/5k validation 与论文 200k+ observations 的差距、terrain 参数偏差、validation 轨迹数量和 baseline 口径；不能把受限复现写成论文级复现。

### 对齐验收顺序

- A1：现有代码/单元测试证明坐标、`k=0` 输出、逐层 target 和 pruning 合约。
- A2：用 1--2 条轨迹做过拟合；若不能显著降低训练 loss 或提高候选覆盖，先修实现而非扩数据。
- A3：对 seed0/1/2 现有 checkpoint 做只读 logits/候选覆盖审计，阈值只允许在 train split 校准，再固定到 validation。
- A4：canonical 配置下做 1k、10-step、三 seed 小实验；要求没有近零 recall 的 seed，再考虑恢复 5k。

当前 5k 探针只提供吞吐证据（约 1200 s/step），不构成模型质量证据；已有 A/B checkpoint 均保留，不覆盖、不重命名。
