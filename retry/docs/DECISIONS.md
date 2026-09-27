# Retry 决策与待审批项

审批规则（用户于 2026-09-27 明确）：论文明确的要求必须一致，无法做到须单独批准；论文未写且明显影响结果的选择提出一个合理方案及影响，用户仅需接受/不接受；论文未写的普通工程细节自行选合理默认值，记为 `IMPLEMENTATION_ASSUMPTION`，不逐项请批。本文件中的 `OPEN_APPROVAL` 仅指结果敏感事项，不把普通工程选择加入审批队列。

## 已批准的项目级规则

1. **论文是第一规范**：旧仓库与第三方实现只作为参考。
2. **建立独立 `retry/` 路径**：按认知与阶段增量迁移，不整仓复制。
3. **全文扫描 + 阶段精读**：Stage -1 防遗漏；真正实现前再精读对应原文。
4. **重要口径差异必须用户批准**：Agent 不得静默决定。
5. **每阶段文档化并 commit/push 后再推进**。
6. **Figure/Table/Equation 与正文同等审计**：不能只读正文概括图。
7. **Stage -1 V2 与补充审计已通过**：用户于 2026-09-27 确认；后续每阶段仍须回原文精读。
8. **Stage 0 的任务定义、系统边界、输入输出、分层成功定义已通过**：用户于 2026-09-27 确认进入 Stage 1；指标算法、数据/位姿/仿真器细节仍按对应决策逐项审批。

## Fig.2 V2 审计后已澄清、无需再作为“未知”的事项

以下此前过度粗化的信息，现在已从 Fig.2 直接展开：

- 图中可识别 4 种卷积类操作；
- 22 个卷积类操作；
- 10 个普通 Conv；
- 4 个 stride-2 Conv；
- 4 个 stride-2 Transposed Conv；
- 4 个 Conv+Prune；
- 4 条 skip connection；
- 可按尺度组织为 9 个连续视觉块。

这些不再允许在 Stage 3 被缩写成一句“U-Net-like”。

## 仍需后续审批的关键项

### D-ARCH-001 — exact channels
状态：`OPEN_APPROVAL`

论文没有给出各层 channel 数。

### D-ARCH-002 — exact kernels / padding
状态：`OPEN_APPROVAL`

论文没有给出 kernel size、padding 等。

### D-ARCH-003 — skip merge
状态：`OPEN_APPROVAL`

图和正文只说明 skip connection / feature forwarding，没有说明 concat / add / 其他融合。

### D-ARCH-004 — decoder temporal 2→1
状态：`OPEN_APPROVAL`

论文正文讨论 latent `4×4×4×2` 和 naive upsampling 到 `64×64×64×1`，但没有完整说明 decoder 中 temporal dimension 如何从 2 变 1。

### D-LOSS-001 — loss weighting
状态：`OPEN_APPROVAL`

论文说明 BCE + mean Euclidean distance 两部分，但没有给相对权重。

### D-TRAIN-001 — exact LR decay schedule
状态：`OPEN_APPROVAL`

只给 0.01 指数衰减到 0.0001，没有给精确 decay rate / step schedule / total horizon。

### D-DATA-001 — terrain micro-distributions
状态：`OPEN_APPROVAL`

未给 poles 尺寸、wall 厚度、对象数量和完整布局分布等；wall 训练高度单独见已批准的 `D-WALL-001`。

### D-WALL-001 — 训练墙固定高度
状态：`APPROVED`（Stage 1；用户于 2026-09-27 批准）

论文明确要求所有训练墙同高，但未公布具体数值。用户批准 retry 的训练墙统一固定为 `1.2 m`，分类为 `IMPLEMENTATION_ASSUMPTION`，并在数据 manifest 与结果中记录。这个批准只覆盖训练墙高度，不批准墙体厚度/数量/位置，也不代表旧 `walls_terrain` 的随机墙高数据合格。采集前须测试每个训练场景所有 wall 的高度均为 `1.2 m`。

### D-SIM-001 — IsaacGym → IsaacLab
状态：`APPROVED`（Stage 1；用户于 2026-09-27 批准）

原文明确使用 IsaacGym；用户批准 retry 在 Stage 1 使用 IsaacLab 作为仿真器替代，须在数据 manifest、阶段报告和最终结果中标记 `SIMULATOR_DEVIATION: IsaacGym -> IsaacLab`。此批准不等于旧地形/相机/策略/GT/位姿参数自动获批，也不表示 IsaacLab 的观测分布与 IsaacGym 等价。后续对照应记录渲染、深度、运动策略和可见性差异。

### D-SENSOR-001 — camera micro-configuration
状态：`OPEN_APPROVAL`

未给 resolution/FOV/intrinsics/depth range/exact mounting translation/full rotations。

### D-AUG-001 — augmentation micro-parameters
状态：`OPEN_APPROVAL`

未给 patch size/count/probability、outlier cluster statistics、各增强组合概率和顺序。

### D-SPLIT-001 — dataset split
状态：`OPEN_APPROVAL`

论文未给 train/validation split 与 seed policy。

### D-EVAL-001 — height MAE exact reduction
状态：`OPEN_APPROVAL`

论文只说 mean absolute height difference，未完整给从 3D reconstruction 到 height matching 的算法。

### D-BASELINE-001 — baseline tuning
状态：`OPEN_APPROVAL`

论文未提供 Elevation Mapping / Voxblox 的完整调参口径。

### D-SCOPE-001 — retry 阶段性成功边界
状态：`APPROVED`（Stage 0；用户于 2026-09-27 确认候选 A）

已确认第一阶段目标：完成可审计的仿真方法复现，包含论文方法机制与独立仿真验证；不把真实机器人数据、BLK2GO/ICP、Table I 数值、真实部署作为本阶段通过条件。真实机器人复现可作为后续独立目标，不能因此宣称已复现 Table I。此项 Stage 0 批准本身仅确定范围；仿真器替代后来由 `D-SIM-001` 单独批准，指标计算、位姿及任何数值达标阈值仍未批准；旧实验结果不得写作 retry 实测。

### D-METRIC-001 — 仿真主指标协议
状态：`OPEN_APPROVAL`（Stage 0 定义边界，Stage 7 落地）

论文未给跨帧/轨迹的 mean 聚合顺序、空输出规则及高度 matching。候选使用逐帧 macro P/R/F1 并附 micro、覆盖率和完整空输出计数；也可选择全局 micro 为主。两者对长轨迹与空预测权重不同，不能静默选择或宣称与论文精确一致。

### D-POSE-001 — 输入位姿与真值位姿边界
状态：`OPEN_APPROVAL`（Stage 0 定义契约，Stage 1/5 落地）

论文使用相邻 pose difference 对齐上一预测，机器人部署 pose 来自 state estimator；仿真采集可用无噪声真位姿或模拟估计误差。两者对时序补全和 drift 测试影响显著。需分别标识输入 pose、真值 pose 及坐标系，正式口径待批准。
