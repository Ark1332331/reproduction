# Retry 当前进度

## 当前里程碑

**Stage 1 — 数据生成契约（2026-09-27 启动）**

状态：`IN_PROGRESS / PLAN_DRAFT_AWAITING_REVIEW`。用户已批准 Stage -1、Stage 0，以及 Stage 1 的 IsaacLab 仿真器替代与固定 `1.2 m` 训练墙高；其余数据口径已集中整理为 Stage 1 实现计划草案，待用户审核后才迁移采集代码或正式采集。

### Stage -1 V2 背景

用户在审查 Fig.2 时发现第一版矩阵只记录了“4D U-Net-like encoder-decoder + skip connections”，遗漏图中可直接读取的 9 个视觉块、22 个卷积类操作、4 种卷积类型等信息。因此 Stage -1 从“完成”撤回，按 Figure/Table/Equation 粒度重扫。

### Stage -1 V2 扫描记录

- 8 页全文重新逐页检查；
- Fig.1–Fig.8 全部重新审计；
- Eq.(1) 单独审计；
- Table I 单独审计；
- Fig.2 形成独立逐层文档：`docs/audits/FIG2_ARCHITECTURE_AUDIT.md`；
- 图/表/公式综合文档：`docs/audits/FIGURE_TABLE_EQUATION_AUDIT.md`；
- 更新 `PAPER_REQUIREMENTS_MATRIX.md`；
- 更新 `DECISIONS.md`。

### 当前禁止事项

在用户确认 Stage 1 数据口径之前：

- Stage 1 仅开展数据契约与差异审计，不迁移 simulation/model 代码；
- 不启动新的训练；
- 不把旧仓库口径当成 retry 正式口径。

### Stage -1 补充审计与 Stage 0 草案

- 发现 Fig.4 左/右指代冲突；明确 Fig.6 的 F1≈88% 与 Recall 62% / Precision 82% 不是同一删除率工作点。
- 登记 43% 可见率、指标 mean 聚合与真实机器人/仿真真值域的未定义边界。
- 更新矩阵、图表审计与 `DECISIONS.md`；建立 `docs/stages/S00_TASK_DEFINITION.md`。
- 用户已确认 `D-SCOPE-001`：首阶段完成可审计的仿真方法复现，真实机器人 Table I 不属于首阶段验收。
- Stage -1 V2 已于 2026-09-27 获用户通过；Stage 0 草案现列出目标、阶段任务、证据、对照指标与验收分档。
- 指标计算细节和位姿边界仍待后续审批；仿真器替代已在 Stage 1 单独批准。本轮无 retry 代码、无新训练指标。

### Stage 0 通过与 Stage 1 启动

- 用户批准 Stage 0 任务定义、系统边界、输入输出及分层成功定义，可进入 Stage 1；指标计算细节留给 Stage 7。
- Stage -1 独立提交 `1636eca`、Stage 0 独立提交 `fa0aa82` 已推送至 `origin/main`；本段为 Stage 1 进行中的本地进度更新。
- Stage 1 精读 §III-D、Fig.3、§IV-A 数据条件，核对旧采集与外部参考；草案见 `docs/stages/S01_DATA_GENERATION.md`。
- `D-SIM-001` 已批准 IsaacLab 并要求明确记录偏差；正式采集前仍需确认数据微观分布、相机配置、GT 采样、位姿输入与场景级数据划分。
- `D-WALL-001` 已批准训练墙统一 `1.2 m`（论文未给数值的实现假设）；旧 `walls_terrain` 仍随机墙高，不能将旧数据直接纳入 retry 正式训练集。
- 按更新的 `AGENTS.md`，已在 `docs/stages/S01_DATA_GENERATION.md` 第 7 节提交实施顺序、候选口径、旧数据复用资格和停止条件供用户集中审核；候选未获批，暂无 retry 采集器或新实验。
- 用户明确三级审批规则：论文明确条件必须遵守（做不到单独批准）；论文未写但明显影响结果的候选由用户接受/不接受；普通工程默认值记录即可。已按此规则修订 Stage 1 草案，待审核项仅保留结果敏感选择。
- Stage 1 草案改为文首页 A–E 五项接受/不接受审核页，详细代码线索和工程记录移至正文/技术附录；用户尚未批准这些候选，不将计划推送视为通过。

## 阶段状态

| Stage | 状态 | 核心交付 |
|---|---|---|
| -1 论文要求审计 | **用户已通过；`1636eca` 已推送** | matrix + Fig.2 audit + figure/table/equation audit |
| 0 验收规定 | **用户已通过；`fa0aa82` 已推送** | S00_TASK_DEFINITION.md |
| 1 数据生成 | **进行中：计划草案待审核；IsaacLab/训练墙高已批** | S01_DATA_GENERATION.md |
| 2 稀疏表示 | 未开始 | S02_REPRESENTATION.md |
| 3 网络结构 | 未开始 | S03_NETWORK.md |
| 4 Loss/训练 | 未开始 | S04_TRAINING.md |
| 5 时序反馈 | 未开始 | S05_TEMPORAL_FEEDBACK.md |
| 6 数据增强 | 未开始 | S06_AUGMENTATION.md |
| 7 Evaluation/论文实验 | 未开始 | S07_EVALUATION.md |
| 8 最终复现审计 | 未开始 | S08_FINAL_AUDIT.md |
