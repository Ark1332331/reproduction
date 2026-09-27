# 论文复现要求矩阵（Stage -1 V2 精细扫描版）

论文：*Neural Scene Representation for Locomotion on Structured Terrain*。

本版修订原因：第一次扫描对 Fig.2 等图的粒度过粗，把大量可复现结构信息压缩成了“4D U-Net-like encoder-decoder”。V2 改为 **逐页 → 逐段 → 逐图 → 逐图例 → 逐 caption → 逐表 → 逐公式** 扫描。

## 分类规则

- `TEXT_EXPLICIT`：正文或 caption 直接明确。
- `FIGURE_EXPLICIT`：图中可直接读取的结构/操作/连线/坐标信息。
- `TABLE_EXPLICIT`：表格直接给出。
- `EQUATION_EXPLICIT`：公式直接定义。
- `PAPER_INFERRED`：可由论文强约束推导，但论文没有逐字写出。
- `UNSPECIFIED`：实现需要，但论文没有提供足够信息。
- `OPEN_APPROVAL`：不能由 Agent 自行定口径，进入对应阶段时必须请用户确认。

> 旧仓库映射只在后续阶段作为候选参考；本表不把旧仓库选择反写成论文事实。

---

## A. 任务定义、输入输出与系统边界

| ID | 原文位置 | 要求 / 行为 | 分类 | 后续阶段 |
|---|---|---|---|---|
| TASK-001 | p.1 Abstract | 从 onboard cameras 的 depth measurements 和 robot trajectory 重建机器人附近 local terrain。 | TEXT_EXPLICIT | Stage 0 |
| TASK-002 | p.1 Abstract | 输入 measurements noisy、partial、occluded，并存在 camera blind spots。 | TEXT_EXPLICIT | Stage 0/1 |
| TASK-003 | p.1 Abstract | 模型是 point clouds 上的 4D fully convolutional network。 | TEXT_EXPLICIT | Stage 3 |
| TASK-004 | p.1 Abstract | 模型学习 geometric priors，用 context completion missing scene。 | TEXT_EXPLICIT | Stage 0/7 |
| TASK-005 | p.1 Abstract | 使用 auto-regressive feedback 利用 spatio-temporal consistency 和 past evidence。 | TEXT_EXPLICIT | Stage 5 |
| TASK-006 | p.1 Abstract | 只用 synthetic data 训练，并依靠 extensive augmentation 实现 real-world robustness。 | TEXT_EXPLICIT | Stage 1/6/8 |
| TASK-007 | Fig.1 | noisy point cloud + previous output 进入网络，输出 surrounding scene point cloud estimate。 | FIGURE_EXPLICIT | Stage 0 |
| TASK-008 | Fig.1 caption | reconstruction 随时间 refined，进入 blind spot 的物体仍被记住。 | TEXT_EXPLICIT | Stage 5/7 |
| TASK-009 | p.2 Intro | pipeline 输入 robot pose、camera partial observations、latest map estimate。 | TEXT_EXPLICIT | Stage 0 |
| TASK-010 | p.2 Intro | output 可作为 standalone module 被不同 model-based / learning-based policy 使用，无需为每个 policy 重训 vision model。 | TEXT_EXPLICIT | Stage 8 |
| TASK-011 | p.2 Intro | 论文明确限定为较简单的 structured urban setting，不声称解决完整 3D rough terrain。 | TEXT_EXPLICIT | Stage 8 |
| TASK-012 | §III / Fig.2 | 推理是逐时刻的局部重建：`(M_t, P̂_{t-1}, pose difference) -> P̂_t`；不是预测机器人控制动作，也不是单次生成全局地图。 | TEXT_EXPLICIT / 任务形式化 | Stage 0 |
| TASK-013 | §IV / Table I | 仿真验证和真实机器人 Stairs/Box 结果属于不同实验域；Table I 不构成仿真 F1 的同口径数值门槛。 | TEXT_EXPLICIT / 比较边界 | Stage 0/7 |

---

## B. 时序关系与坐标变换

| ID | 原文位置 | 要求 / 行为 | 分类 | 后续阶段 |
|---|---|---|---|---|
| TEMP-001 | Fig.2 / §III | 使用 previous/current pose difference。 | TEXT_EXPLICIT | Stage 0/5 |
| TEMP-002 | Fig.2 / §III | `P̂_{t-1}` 先变换到 current measurement frame。 | TEXT_EXPLICIT | Stage 5 |
| TEMP-003 | Fig.2 caption | transformed previous output 与 current measurement `M_t` 沿 temporal 维 concatenated。 | TEXT_EXPLICIT | Stage 5 |
| TEMP-004 | §III | 网络输出 current estimate `P̂_t`。 | TEXT_EXPLICIT | Stage 0 |
| TEMP-005 | p.7 | local method 可隐式补偿 drift；map frame 可随 robot drift，不必和 GT global frame 一致。 | TEXT_EXPLICIT | Stage 0/7 |
| TEMP-006 | p.7 | 对 locomotion 来说 robot reference frame 中 map 正确即可。 | TEXT_EXPLICIT | Stage 0/7 |

---

## C. 输入预处理与稀疏表示

| ID | 原文位置 | 要求 / 行为 | 分类 | 后续阶段 |
|---|---|---|---|---|
| REP-001 | §III-A | point cloud 转为 voxel grid data structure。 | TEXT_EXPLICIT | Stage 2 |
| REP-002 | §III-A | 使用 sparse formulation，避免 dense cubic memory cost。 | TEXT_EXPLICIT | Stage 2 |
| REP-003 | §III-A | grid=`64×64×64`。 | TEXT_EXPLICIT | Stage 2 |
| REP-004 | §III-A | local map=`3.2×3.2×3.2 m`。 | TEXT_EXPLICIT | Stage 2 |
| REP-005 | §IV | evaluation cell=`0.05×0.05×0.05 m`；与 3.2/64 一致。 | TEXT_EXPLICIT | Stage 2/7 |
| REP-006 | Eq.(1) 前 | 每个 occupied voxel 使用其中 points 的 centroid `p_i=[x_i,y_i,z_i]`。 | TEXT_EXPLICIT | Stage 2 |
| REP-007 | Eq.(1) 前 | `p_i` 在 grid reference frame 中表示，one unit = one cell。 | TEXT_EXPLICIT | Stage 2 |
| REP-008 | Eq.(1) | `c_i=[floor(x_i),floor(y_i),floor(z_i),k_i]`。 | EQUATION_EXPLICIT | Stage 2 |
| REP-009 | Eq.(1) | `f_i=p_i mod 1`。 | EQUATION_EXPLICIT | Stage 2 |
| REP-010 | §III-A | `f_i` 是 centroid 相对 voxel bottom-left-rear corner 的 offset。 | TEXT_EXPLICIT | Stage 2 |
| REP-011 | §III-A | `f_i∈[0,1]`。 | TEXT_EXPLICIT | Stage 2 |
| REP-012 | §III-A | continuous centroid = cell index + feature。 | TEXT_EXPLICIT | Stage 2 |
| REP-013 | §III-A | current measurement `k=0`。 | TEXT_EXPLICIT | Stage 2/5 |
| REP-014 | §III-A | previous output `k=1`。 | TEXT_EXPLICIT | Stage 2/5 |
| REP-015 | §III-A | 该 time coordinate 使网络可以做 4D convolution across space and time。 | TEXT_EXPLICIT | Stage 3 |

---

## D. Fig.2 网络拓扑（精细展开）

> 详细 22 层编号见 `audits/FIG2_ARCHITECTURE_AUDIT.md`。以下“9 个视觉块”是审计标签，不是作者命名。

| ID | 原文位置 | 要求 / 行为 | 分类 | 后续阶段 |
|---|---|---|---|---|
| ARCH-F2-001 | Fig.2 | 可按连续尺度组织划分为 9 个视觉块：4 encoder + bottleneck + 4 decoder/output。 | FIGURE_EXPLICIT / 审计分组 | Stage 3 |
| ARCH-F2-002 | Fig.2 legend | 图例类型1：Convolution（绿色）。 | FIGURE_EXPLICIT | Stage 3 |
| ARCH-F2-003 | Fig.2 legend | 图例类型2：Convolution with stride 2（蓝色）。 | FIGURE_EXPLICIT | Stage 3 |
| ARCH-F2-004 | Fig.2 legend | 图例类型3：Transposed convolution with stride 2（灰色）。 | FIGURE_EXPLICIT | Stage 3 |
| ARCH-F2-005 | Fig.2 legend | 图例类型4：Convolution followed by pruning（红色）。 | FIGURE_EXPLICIT | Stage 3 |
| ARCH-F2-006 | Fig.2 | 按图逐项计数，共 22 个卷积类操作。 | FIGURE_EXPLICIT | Stage 3 |
| ARCH-F2-007 | Fig.2 | 其中普通 Conv 共 10 个。 | FIGURE_EXPLICIT | Stage 3 |
| ARCH-F2-008 | Fig.2 | stride-2 Conv 共 4 个。 | FIGURE_EXPLICIT | Stage 3 |
| ARCH-F2-009 | Fig.2 | stride-2 Transposed Conv 共 4 个。 | FIGURE_EXPLICIT | Stage 3 |
| ARCH-F2-010 | Fig.2 | Conv+Prune 共 4 个。 | FIGURE_EXPLICIT | Stage 3 |
| ARCH-F2-011 | Fig.2 | 4 条 encoder→decoder skip connection。 | FIGURE_EXPLICIT | Stage 3 |
| ARCH-F2-012 | Fig.2 | 每个 encoder 视觉块是 `Conv→Conv-S2`，共 4 块。 | FIGURE_EXPLICIT | Stage 3 |
| ARCH-F2-013 | Fig.2 | bottleneck 视觉块是 `Conv→TConv-S2`。 | FIGURE_EXPLICIT | Stage 3 |
| ARCH-F2-014 | Fig.2 | 三个中间 decoder 块是 `Conv→Conv+Prune→TConv-S2`。 | FIGURE_EXPLICIT | Stage 3 |
| ARCH-F2-015 | Fig.2 | 最终 decoder/output 块是 `Conv→Conv+Prune→Conv`。 | FIGURE_EXPLICIT | Stage 3 |
| ARCH-001 | §III-B | 网络为 U-Net-like fully convolutional 4D encoder-decoder with skip connections。 | TEXT_EXPLICIT | Stage 3 |
| ARCH-002 | §III-B | encoder 共 4 个 strided convolutions。 | TEXT_EXPLICIT | Stage 3 |
| ARCH-003 | §III-B | 每次空间维 downsample ×2，总计 ×16。 | TEXT_EXPLICIT | Stage 3 |
| ARCH-004 | §III-B | temporal dimension 在 encoder stride conv 中保持。 | TEXT_EXPLICIT | Stage 3 |
| ARCH-005 | §III-B | 两个 temporal channels 在不同空间 resolution 保持分离，同时 convolution across space/time。 | TEXT_EXPLICIT | Stage 3 |
| ARCH-006 | §III-B | corresponding encoder feature maps 通过 skip connections 送到 decoder block。 | TEXT_EXPLICIT | Stage 3 |
| ARCH-007 | §III-B | decoder upsample 回 input stride。 | TEXT_EXPLICIT | Stage 3 |
| ARCH-008 | p.4 | latent 很可能成为 fully occupied `4×4×4×2` block。 | TEXT_EXPLICIT | Stage 3 |
| ARCH-009 | p.4 | naive upsampling 会产生 fully occupied `64×64×64×1`，过慢且破坏 sparse training。 | TEXT_EXPLICIT | Stage 3 |
| ARCH-010 | p.4 | decoder 因此使用 pruning。 | TEXT_EXPLICIT | Stage 3 |
| ARCH-011 | p.4 | 每层由当前 sparse `T_f` 通过 separate convolution + sigmoid 产生 likelihood `T_p`。 | TEXT_EXPLICIT | Stage 3/4 |
| ARCH-012 | p.4 | pruning 删除 `T_p` likelihood < `alpha` 对应的 `T_f` elements。 | TEXT_EXPLICIT | Stage 3/4 |
| ARCH-013 | p.4 | `T_p` 不 forwarded 到 next layers。 | TEXT_EXPLICIT | Stage 3 |
| ARCH-014 | p.4 | final convolution 映射到 feature dimension 3。 | TEXT_EXPLICIT | Stage 3 |
| ARCH-015 | 未说明 | exact channel counts。 | UNSPECIFIED / OPEN_APPROVAL | Stage 3 |
| ARCH-016 | 未说明 | exact kernel sizes / padding。 | UNSPECIFIED / OPEN_APPROVAL | Stage 3 |
| ARCH-017 | 未说明 | skip merge 是 concat / add / 其他。 | UNSPECIFIED / OPEN_APPROVAL | Stage 3 |
| ARCH-018 | 未说明 | decoder 中 temporal `2→1` 的精确实现。 | UNSPECIFIED / OPEN_APPROVAL | Stage 3 |
| ARCH-019 | 未说明 | activation / normalization / block 内是否存在未画出的额外层。 | UNSPECIFIED / OPEN_APPROVAL | Stage 3 |

---

## E. Loss、pruning threshold 与训练流程

| ID | 原文位置 | 要求 / 行为 | 分类 | 后续阶段 |
|---|---|---|---|---|
| TRAIN-001 | §III-C | target 是 ground-truth point cloud 按 Eq.(1) 转换。 | TEXT_EXPLICIT | Stage 4 |
| TRAIN-002 | §III-C | loss 由两部分组成。 | TEXT_EXPLICIT | Stage 4 |
| TRAIN-003 | §III-C | 第一部分：每个 `T_p` layer 与该尺度 target occupancy 0/1 做 BCE。 | TEXT_EXPLICIT | Stage 4 |
| TRAIN-004 | §III-C | 第二部分：output features 与 target features 的 mean Euclidean distance，估计 sub-voxel position。 | TEXT_EXPLICIT | Stage 4 |
| TRAIN-005 | §III-C | `T_p` feature 表示 voxel occupied confidence。 | TEXT_EXPLICIT | Stage 4 |
| TRAIN-006 | §III-C | `alpha` 是 key hyper-parameter。 | TEXT_EXPLICIT | Stage 4/7 |
| TRAIN-007 | §III-C | alpha 高：保守、只保留更有信心 points、output density 降低、holes 增加。 | TEXT_EXPLICIT | Stage 4/7 |
| TRAIN-008 | §III-C | alpha 低：output 更密，但可能生成 target 中不存在的 points。 | TEXT_EXPLICIT | Stage 4/7 |
| TRAIN-009 | §III-C | 作者实验选择 `alpha=0.5` 作为 recall/precision trade-off。 | TEXT_EXPLICIT | Stage 4 |
| TRAIN-010 | §III-C | rollout sequence length=12 time steps。 | TEXT_EXPLICIT | Stage 4/5 |
| TRAIN-011 | §III-C | 每个 step 都计算 loss。 | TEXT_EXPLICIT | Stage 4 |
| TRAIN-012 | §III-C | authors found propagating gradients over time no benefit。 | TEXT_EXPLICIT | Stage 4/5 |
| TRAIN-013 | §III-C | Adam optimizer。 | TEXT_EXPLICIT | Stage 4 |
| TRAIN-014 | §III-C | initial LR=0.01。 | TEXT_EXPLICIT | Stage 4 |
| TRAIN-015 | §III-C | LR exponentially decayed until 0.0001。 | TEXT_EXPLICIT | Stage 4 |
| TRAIN-016 | 未说明 | 两部分 loss 的相对权重。 | UNSPECIFIED / OPEN_APPROVAL | Stage 4 |
| TRAIN-017 | 未说明 | exact exponential decay schedule / training horizon。 | UNSPECIFIED / OPEN_APPROVAL | Stage 4 |
| TRAIN-018 | 未说明 | batch size、weight decay、gradient clipping、initialization 等。 | UNSPECIFIED / OPEN_APPROVAL | Stage 4 |

---

## F. 数据生成与 Fig.3

| ID | 原文位置 | 要求 / 行为 | 分类 | 后续阶段 |
|---|---|---|---|---|
| DATA-001 | p.2 / §III-D | 使用 NVIDIA IsaacGym 生成 structured randomized scenes。 | TEXT_EXPLICIT | Stage 1 |
| DATA-002 | §III-D | terrain types：stairs、boxes on ground、walls、poles、narrow corridors。 | TEXT_EXPLICIT | Stage 1 |
| DATA-003 | p.2 | Intro 另写 walls、roadblocks、flights of stairs、boxes；roadblocks 与后文 taxonomy 的对应未说明。 | TEXT_EXPLICIT / 歧义 | Stage 1 |
| DATA-004 | Fig.3 caption | stairs width uniform `[0.2,0.5]m`。 | TEXT_EXPLICIT | Stage 1 |
| DATA-005 | Fig.3 caption | stairs height uniform `[0.08,0.25]m`。 | TEXT_EXPLICIT | Stage 1 |
| DATA-006 | Fig.3 caption | boxes width/length uniform `[0.2,2.0]m`。 | TEXT_EXPLICIT | Stage 1 |
| DATA-007 | Fig.3 caption | boxes height uniform `[0.08,0.25]m`。 | TEXT_EXPLICIT | Stage 1 |
| DATA-008 | Fig.3 caption | walls sampled to produce corridor width `[2,6]m`。 | TEXT_EXPLICIT | Stage 1 |
| DATA-009 | §III-D | scene element dimensions/locations randomized。 | TEXT_EXPLICIT | Stage 1 |
| DATA-010 | §III-D | robot 被 rough-terrain locomotion policy 控制 toward reachable position。 | TEXT_EXPLICIT | Stage 1 |
| DATA-011 | §III-D | randomized speed。 | TEXT_EXPLICIT | Stage 1 |
| DATA-012 | §III-D | randomized base orientation。 | TEXT_EXPLICIT | Stage 1 |
| DATA-013 | §III-D | 4 simulated Intel RealSense depth cameras：front/back/left/right。 | TEXT_EXPLICIT | Stage 1 |
| DATA-014 | §III-D | cameras downward tilt=30°。 | TEXT_EXPLICIT | Stage 1 |
| DATA-015 | §III-D | 称为 standard ANYmal C configuration。 | TEXT_EXPLICIT | Stage 1 |
| DATA-016 | §III-D | GT 每 time step 从 robot 周围 terrain mesh 采 dense point cloud。 | TEXT_EXPLICIT | Stage 1 |
| DATA-017 | §III-D | dataset 平均 43% GT points visible in measurements。 | TEXT_EXPLICIT | Stage 1/7 |
| DATA-018 | §III-D | dataset comprises 200000 time steps。 | TEXT_EXPLICIT | Stage 1 |
| DATA-019 | p.2 | Intro 说 more than 200,000 point cloud observations；需保留 wording 差异。 | TEXT_EXPLICIT | Stage 1 |
| DATA-020 | §III-D | network intentionally overfits structured urban environments，尝试隐式识别 stairs length/height 等参数。 | TEXT_EXPLICIT | Stage 1/8 |
| DATA-021 | §IV-A | **所有 training walls 使用相同高度**。 | TEXT_EXPLICIT | Stage 1（从实验段反写数据契约） |
| DATA-022 | §IV-A | 固定 wall height 是为了只保留 locomotion relevant features。 | TEXT_EXPLICIT | Stage 1/8 |
| DATA-023 | 未说明 | poles 尺寸/几何。 | UNSPECIFIED / OPEN_APPROVAL | Stage 1 |
| DATA-024 | 未说明 | wall height/thickness、boxes 数量、layout/count distributions。 | UNSPECIFIED / OPEN_APPROVAL | Stage 1 |
| DATA-025 | 未说明 | exact locomotion policy/checkpoint、speed/orientation distributions。 | UNSPECIFIED / OPEN_APPROVAL | Stage 1 |
| DATA-026 | 未说明 | dense GT sampling density / exact algorithm。 | UNSPECIFIED / OPEN_APPROVAL | Stage 1 |
| DATA-027 | 未说明 | train/validation split / seed policy。 | UNSPECIFIED / OPEN_APPROVAL | Stage 1/7 |
| DATA-028 | 未说明 | 平均可见率 43% 的点匹配容差、匹配方式和统计分母未给出，不可直接把本地体素 recall 称为论文同口径可见率。 | UNSPECIFIED / OPEN_APPROVAL | Stage 0/1/7 |

---

## G. Camera / sensor 未公开细节

| ID | 原文位置 | 要求 / 行为 | 分类 | 后续阶段 |
|---|---|---|---|---|
| SENSOR-001 | §III-D | 4×RealSense + 四方向 + 30° tilt。 | TEXT_EXPLICIT | Stage 1 |
| SENSOR-002 | 未说明 | resolution / FOV / intrinsics / depth range。 | UNSPECIFIED / OPEN_APPROVAL | Stage 1 |
| SENSOR-003 | 未说明 | exact camera translations / full rotations。 | UNSPECIFIED / OPEN_APPROVAL | Stage 1 |
| SENSOR-004 | 未说明 | point-cloud conversion、downsampling、per-camera point cap。 | UNSPECIFIED / OPEN_APPROVAL | Stage 1 |
| SENSOR-005 | 未说明 | synthetic collection frame rate。 | UNSPECIFIED / OPEN_APPROVAL | Stage 1 |

---

## H. Data augmentation

| ID | 原文位置 | 要求 / 行为 | 分类 | 后续阶段 |
|---|---|---|---|---|
| AUG-001 | §III-E | Position：每个 point 均匀扰动 `[-0.05,0.05]m`。 | TEXT_EXPLICIT | Stage 6 |
| AUG-002 | §III-E | Tilt：random direction，angle uniform `[-1°,1°]`。 | TEXT_EXPLICIT | Stage 6 |
| AUG-003 | §III-E | Height：random patches height uniform `[-0.05,0.05]m`。 | TEXT_EXPLICIT | Stage 6 |
| AUG-004 | §III-E | Pruning：random patches removed。 | TEXT_EXPLICIT | Stage 6 |
| AUG-005 | §III-E | Outliers：random clusters of points added。 | TEXT_EXPLICIT | Stage 6 |
| AUG-006 | §III-E | Robot Pose：position uniform disturbed `[-0.05,0.05]m`。 | TEXT_EXPLICIT | Stage 6 |
| AUG-007 | §III-E | whole trajectory 随机沿 x / y axes mirror。 | TEXT_EXPLICIT | Stage 6 |
| AUG-008 | §III-E | augmentation 目标：denoising + completion + 利用 previous evidence 估计 partially observable world state。 | TEXT_EXPLICIT | Stage 6 |
| AUG-009 | 未说明 | patch size/count/probability。 | UNSPECIFIED / OPEN_APPROVAL | Stage 6 |
| AUG-010 | 未说明 | outlier cluster size/count/distribution。 | UNSPECIFIED / OPEN_APPROVAL | Stage 6 |
| AUG-011 | 未说明 | augmentations 的采样概率、顺序、组合策略。 | UNSPECIFIED / OPEN_APPROVAL | Stage 6 |

---

## I. Sparse backbone

| ID | 原文位置 | 要求 / 行为 | 分类 | 后续阶段 |
|---|---|---|---|---|
| IMPL-001 | §III-E | backbone relies on MinkowskiEngine。 | TEXT_EXPLICIT | Stage 2/3 |
| IMPL-002 | §III-E | MinkowskiEngine 提供 spatially sparse tensors 的 CPU/GPU acceleration。 | TEXT_EXPLICIT | Stage 2/3 |
| IMPL-003 | Related Work | 作者明确说其 upsampling/pruning 思路 similar to Gwak et al. [29]，也提及 Sun et al. [28]。 | TEXT_EXPLICIT（参考锚点） | Stage 3 外部交叉阅读 |
| IMPL-004 | §III-B | 网络 U-Net-like，引用 U-Net [30]。 | TEXT_EXPLICIT（参考锚点） | Stage 3 |
| IMPL-005 | §III-E | sparse backbone 引用 Minkowski 4D spatio-temporal convnets [31]。 | TEXT_EXPLICIT（参考锚点） | Stage 2/3 |

---

## J. Evaluation 总口径

| ID | 原文位置 | 要求 / 行为 | 分类 | 后续阶段 |
|---|---|---|---|---|
| EVAL-001 | §IV | baselines：Elevation Mapping [3]、Voxblox [7]。 | TEXT_EXPLICIT | Stage 7 |
| EVAL-002 | §IV | report mean Precision。 | TEXT_EXPLICIT | Stage 7 |
| EVAL-003 | §IV | report mean Recall。 | TEXT_EXPLICIT | Stage 7 |
| EVAL-004 | §IV | report mean F1。 | TEXT_EXPLICIT | Stage 7 |
| EVAL-005 | §IV | evaluation grid：robot-centric `64×64×64`。 | TEXT_EXPLICIT | Stage 7 |
| EVAL-006 | §IV | cell=`0.05×0.05×0.05m`。 | TEXT_EXPLICIT | Stage 7 |
| EVAL-007 | §IV | report mean absolute height difference reconstruction vs GT。 | TEXT_EXPLICIT | Stage 7 |
| EVAL-008 | §IV | height metric 对 locomotion 重要，因为 policy 直接使用 height info。 | TEXT_EXPLICIT | Stage 7 |
| EVAL-009 | 未说明 | height MAE 的 exact XY matching/reduction。 | UNSPECIFIED / OPEN_APPROVAL | Stage 7 |
| EVAL-010 | 未说明 | baseline exact hyperparameters/tuning。 | UNSPECIFIED / OPEN_APPROVAL | Stage 7 |
| EVAL-011 | 未说明 | mean Precision/Recall/F1 对 frame、trajectory、scene 的聚合顺序以及空预测/空目标的处理未完全给出。 | UNSPECIFIED / OPEN_APPROVAL | Stage 0/7 |
| EVAL-012 | §IV-B / Table I | 真实机器人 Ground Truth 来自 BLK2GO + ICP，不能把 IsaacLab terrain mesh GT 静默视为同一评估域。 | TEXT_EXPLICIT / 比较边界 | Stage 0/7/8 |

---

## K. Fig.4 Temporal memory 实验

| ID | 原文位置 | 要求 / 行为 | 分类 | 后续阶段 |
|---|---|---|---|---|
| EXP-F4-001 | Fig.4 | 两列时间：trajectory beginning 与 `1.5s` later。 | FIGURE/TEXT_EXPLICIT | Stage 7 |
| EXP-F4-002 | Fig.4 | 三行：Measurement / Reconstruction / Ground truth。 | FIGURE_EXPLICIT | Stage 7 |
| EXP-F4-003 | caption | right measurement 中 two diagonal boxes 已不可见。 | TEXT_EXPLICIT | Stage 7 |
| EXP-F4-004 | caption | boxes 仍被正确 reconstruction，利用 evidence from past。 | TEXT_EXPLICIT | Stage 7 |
| EXP-F4-005 | §IV-A | wall 即使只看到 bottom 也被完整 recreate。 | TEXT_EXPLICIT | Stage 7 |
| EXP-F4-006 | §IV-A | 原因之一：all walls same height in training data。 | TEXT_EXPLICIT | Stage 1/7 |
| EXP-F4-007 | §IV-A / Fig.4 caption | 正文末句称 temporal info 对 “left column” 必要，但此前正文与 caption 均将历史记忆例子放在 right column；原文存在左右指代冲突，复现实验以可观测时间/遮挡条件描述，不擅自改原文。 | TEXT_EXPLICIT / 原文冲突 | Stage 0/7 |

---

## L. Fig.5 Zero-shot spatial completion

| ID | 原文位置 | 要求 / 行为 | 分类 | 后续阶段 |
|---|---|---|---|---|
| EXP-F5-001 | Fig.5 | 两面板：Measurement / Reconstruction。 | FIGURE_EXPLICIT | Stage 7 |
| EXP-F5-002 | caption | zero-shot = only current time-step measurement。 | TEXT_EXPLICIT | Stage 7 |
| EXP-F5-003 | §IV-A | 用于验证 spatial context，不依赖 temporal info。 | TEXT_EXPLICIT | Stage 7 |
| EXP-F5-004 | caption | reconstruction 可恢复 vertical surfaces（原 caption wording: walls）。 | TEXT_EXPLICIT | Stage 7 |

---

## M. Fig.6 Measurement removal robustness

| ID | 原文位置 | 要求 / 行为 | 分类 | 后续阶段 |
|---|---|---|---|---|
| EXP-F6-001 | Fig.6 | x=`Removed data [%]`，范围显示 0–80。 | FIGURE_EXPLICIT | Stage 7 |
| EXP-F6-002 | Fig.6 | y=`Performance [%]`，约 60–100。 | FIGURE_EXPLICIT | Stage 7 |
| EXP-F6-003 | Fig.6 | 三曲线：F1 / Recall / Precision。 | FIGURE_EXPLICIT | Stage 7 |
| EXP-F6-004 | §IV-A | 对 same validation trajectories 做不同 measurement data removal。 | TEXT_EXPLICIT | Stage 7 |
| EXP-F6-005 | §IV-A | removal rate 不包含 already missing blind-spot points。 | TEXT_EXPLICIT | Stage 7 |
| EXP-F6-006 | caption | performance stays constant up to 50% removal，之后 recall rapidly decreases。 | TEXT_EXPLICIT | Stage 7 |
| EXP-F6-007 | §IV-A | 50% omission 时 F1≈88%。 | TEXT_EXPLICIT（结果） | Stage 7 |
| EXP-F6-008 | §IV-A | 超过约 50% 后 output density 急降、holes 出现；更稀疏区域 Recall 文中降至 62%，不是 50% 时的 Recall。 | TEXT_EXPLICIT（结果，横轴位置未精确给出） | Stage 7 |
| EXP-F6-009 | §IV-A | 更稀疏区域 Precision 文中达到 82%，不是 50% 时的 Precision；精确横轴位置未给出。 | TEXT_EXPLICIT（结果，横轴位置未精确给出） | Stage 7 |
| EXP-F6-010 | §IV-A | >80% omission 时 pruning 可能产生 empty tensors。 | TEXT_EXPLICIT（失败区间） | Stage 7 |

---

## N. Table I Real-robot quantitative results

| ID | Dataset/Method | Precision | Recall | F1 | MAE | 分类 |
|---|---|---:|---:|---:|---:|---|
| TAB1-01 | Stairs Measurement | 87.7 | 50.7 | 64.0 | 0.64cm | TABLE_EXPLICIT |
| TAB1-02 | Stairs Elevation Mapping | 73.2 | 79.8 | 76.3 | 2.1cm | TABLE_EXPLICIT |
| TAB1-03 | Stairs Voxblox | 76.3 | 72.3 | 73.5 | 1.6cm | TABLE_EXPLICIT |
| TAB1-04 | Stairs Ours | 86.0 | 89.9 | 88.9 | 0.8cm | TABLE_EXPLICIT |
| TAB1-05 | Box Measurement | 80.8 | 61.6 | 69.8 | 1.2cm | TABLE_EXPLICIT |
| TAB1-06 | Box Elevation Mapping | 72.2 | 80.0 | 75.9 | 1.7cm | TABLE_EXPLICIT |
| TAB1-07 | Box Voxblox | 74.0 | 77.9 | 75.8 | 1.4cm | TABLE_EXPLICIT |
| TAB1-08 | Box Ours | 84.8 | 84.9 | 84.8 | 1.0cm | TABLE_EXPLICIT |

注意：这些是 real-robot Stairs / Box dataset 的论文结果，不应静默作为 IsaacLab retry 仿真结果的同口径目标。

---

## O. Robot deployment / real-world evaluation protocol

| ID | 原文位置 | 要求 / 行为 | 分类 | 后续阶段 |
|---|---|---|---|---|
| ROBOT-001 | §IV-B | deploy on NVIDIA Jetson Xavier。 | TEXT_EXPLICIT | Stage 8 / optional |
| ROBOT-002 | §IV-B | ROS node separate thread 处理 front/left/right/back RealSense point clouds。 | TEXT_EXPLICIT | Stage 8 |
| ROBOT-003 | §IV-B | data mapped to GPU for inference。 | TEXT_EXPLICIT | Stage 8 |
| ROBOT-004 | §IV-B | pose 来自 state estimator。 | TEXT_EXPLICIT | Stage 8 |
| ROBOT-005 | §IV-B | publish estimated point cloud，controller 查询 heights。 | TEXT_EXPLICIT | Stage 8 |
| ROBOT-006 | §IV-B | inference average 70ms。 | TEXT_EXPLICIT（结果） | Stage 8 |
| ROBOT-007 | §IV-B | whole node runs 6Hz。 | TEXT_EXPLICIT（结果） | Stage 8 |
| ROBOT-008 | §IV-B | map update 间 measurements discarded。 | TEXT_EXPLICIT | Stage 8 |
| ROBOT-009 | §IV-B | reconstructed map=`3.2×3.2×3.2m`。 | TEXT_EXPLICIT | Stage 8 |
| ROBOT-010 | §IV-B | policy terrain input=`1.6×1.0m`。 | TEXT_EXPLICIT | Stage 8 |
| ROBOT-011 | §IV-B | controller=50Hz。 | TEXT_EXPLICIT | Stage 8 |
| ROBOT-012 | §IV-B | policy perceptive inputs 按 relative pose to latest map 计算。 | TEXT_EXPLICIT | Stage 8 |
| ROBOT-013 | §IV-B | map extent 推得 speed limit 4.8m/s；实际 policy max=1m/s。 | TEXT_EXPLICIT | Stage 8 |
| ROBOT-014 | §IV-B | classical baselines process incoming measurements at 15Hz。 | TEXT_EXPLICIT | Stage 8 |
| ROBOT-015 | §IV-B | baseline 输出 mesh，因此从 terrain estimates sample dense point clouds 比较。 | TEXT_EXPLICIT | Stage 7/8 |
| ROBOT-016 | §IV-B | real GT 用 BLK2GO LiDAR scanner。 | TEXT_EXPLICIT | Stage 8 |
| ROBOT-017 | §IV-B | measurement frame 与 BLK2GO map 做 ICP registration。 | TEXT_EXPLICIT | Stage 8 |
| ROBOT-018 | §IV-B | Stairs experiment：various sizes/textures/illumination。 | TEXT_EXPLICIT | Stage 8 |
| ROBOT-019 | §IV-B | Box experiment：large reflective wooden box，bright conditions，部分相机不可见。 | TEXT_EXPLICIT | Stage 8 |

---

## P. Fig.7 Heavy state-estimator drift

| ID | 原文位置 | 要求 / 行为 | 分类 | 后续阶段 |
|---|---|---|---|---|
| EXP-F7-001 | Fig.7 | 两列 `t=2.76s`、`t=3.30s`。 | FIGURE_EXPLICIT | Stage 7/8 |
| EXP-F7-002 | Fig.7 | 五行：Measurement / Elevation Mapping / Voxblox / Our approach / Ground truth。 | FIGURE_EXPLICIT | Stage 7/8 |
| EXP-F7-003 | p.7 | state estimator 在 `t=2.76s` 后向下 drift 7cm。 | TEXT_EXPLICIT | Stage 7/8 |
| EXP-F7-004 | caption/p.7 | Ours 检测 previous map/current measurement mismatch，并立即对齐 shifted measurement。 | TEXT_EXPLICIT | Stage 7/8 |
| EXP-F7-005 | p.7 | Elevation Mapping 不能处理该 drift，形成 uneven map。 | TEXT_EXPLICIT | Stage 7/8 |
| EXP-F7-006 | p.7 | Voxblox 较能纠正 drift，但 robot 下方仍有错误。 | TEXT_EXPLICIT | Stage 7/8 |
| EXP-F7-007 | p.6-7 | stairs map border 略弯，作者归因于 edge noise 在 border 更强。 | TEXT_EXPLICIT | Stage 8 |
| EXP-F7-008 | p.7 | Ours 可能有 small holes，归因于 pruning + 未训练 watertight output。 | TEXT_EXPLICIT | Stage 3/8 |
| EXP-F7-009 | p.7 | holes 很局部，不影响 controller performance。 | TEXT_EXPLICIT（作者观察） | Stage 8 |

---

## Q. Fig.8 Map drift 对 locomotion 的影响

| ID | 原文位置 | 要求 / 行为 | 分类 | 后续阶段 |
|---|---|---|---|---|
| LOCO-001 | Fig.8 | x=`Drift[m]`，0–0.20。 | FIGURE_EXPLICIT | Stage 8 |
| LOCO-002 | Fig.8 | y=`Performance[%]`。 | FIGURE_EXPLICIT | Stage 8 |
| LOCO-003 | Fig.8 | curves：Success rate / Survival rate。 | FIGURE_EXPLICIT | Stage 8 |
| LOCO-004 | caption | average over 4000 trajectories。 | TEXT_EXPLICIT | Stage 8 |
| LOCO-005 | §IV-C | target-reaching task in simulation。 | TEXT_EXPLICIT | Stage 8 |
| LOCO-006 | §IV-C | robot 必须在 7s 内 cross challenging terrain。 | TEXT_EXPLICIT | Stage 8 |
| LOCO-007 | §IV-C | drift varied 0–20cm。 | TEXT_EXPLICIT | Stage 8 |
| LOCO-008 | §IV-C | 用 measured terrain 中随机放置对应 height bumps 模拟 drift。 | TEXT_EXPLICIT | Stage 8 |
| LOCO-009 | §IV-C | survival at high drift 约降至 70%。 | TEXT_EXPLICIT（结果） | Stage 8 |
| LOCO-010 | §IV-C | 20cm drift 时 fewer than 20% reach target on time。 | TEXT_EXPLICIT（结果） | Stage 8 |
| LOCO-011 | §IV-C | drift>10cm 时 knee-ground collision ≈3× drift-free。 | TEXT_EXPLICIT（结果） | Stage 8 |
| LOCO-012 | §IV-C | 另将 map 接到 perceptive rough-terrain model-based controller [32]，由其计算 feasible footholds + MPC tracking。 | TEXT_EXPLICIT | Stage 8 |
| LOCO-013 | §IV-C | supplementary video 用于展示该 controller 可跨越 structured terrain。 | TEXT_EXPLICIT | Stage 8 |

---

## R. 论文局限与结果解释边界

| ID | 原文位置 | 内容 | 分类 |
|---|---|---|---|
| LIMIT-001 | Conclusion | dynamic obstacles 的部分 points 会残留并缓慢消失。 | TEXT_EXPLICIT |
| LIMIT-002 | Conclusion | 作者建议 dynamic-obstacle dataset 或 better noise model。 | TEXT_EXPLICIT |
| LIMIT-003 | Conclusion | depth camera object-edge outliers 难以在 simulation 复现。 | TEXT_EXPLICIT |
| LIMIT-004 | Conclusion | 作者建议增加 real robot data。 | TEXT_EXPLICIT |
| LIMIT-005 | p.8 | unstructured setting 更难，可能需要真实任务数据。 | TEXT_EXPLICIT |
| LIMIT-006 | p.8 | future：tables / higher obstacles。 | TEXT_EXPLICIT |
| LIMIT-007 | p.8 | future：point cloud + proprioceptive foot contact。 | TEXT_EXPLICIT |
| LIMIT-008 | p.8 | future：policy 直接使用 latent representation，避免 decoding。 | TEXT_EXPLICIT |

---

## Stage -1 V2 结论

本次扫描确认：原论文对“宏观方法”之外的许多工程行为其实给得比第一版矩阵更具体，尤其是 Fig.2 的 22 个卷积类操作、4 种操作类型、4 条 skip、4 次 downsample / upsample / prune，以及 Fig.4–8 的具体实验协议。

但论文仍不足以 bit-identical reconstruction。关键未公开项仍包括：channel、kernel、skip merge、decoder temporal 2→1 实现、loss 权重、精确 LR schedule、sensor intrinsics/extrinsics、scene micro-distributions、augmentation micro-parameters、split policy、height-MAE reduction。

因此 Stage -1 的交付标准不是“所有实现细节都已决定”，而是：**论文已经说了什么不能漏；论文没说什么不能假装它说过。**
