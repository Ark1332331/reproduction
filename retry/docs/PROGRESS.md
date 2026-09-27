# Retry 当前进度

## 当前里程碑

**Stage 1 — 数据生成契约（2026-09-27 启动）**

状态：`IN_PROGRESS / AWAITING_DATA_DECISIONS`。用户已批准 Stage -1 和 Stage 0 的任务、边界、输入输出及成功分层定义；Stage 1 首先完成原文精读和数据契约审计，未获数据口径批准前不迁移采集代码或正式采集。

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
- 指标计算细节、位姿边界和仿真器替代仍待后续审批；本轮无 retry 代码、无新训练指标。

### Stage 0 通过与 Stage 1 启动

- 用户批准 Stage 0 任务定义、系统边界、输入输出及分层成功定义，可进入 Stage 1；指标计算细节留给 Stage 7。
- Stage 1 精读 §III-D、Fig.3、§IV-A 数据条件，核对旧采集与外部参考；草案见 `docs/stages/S01_DATA_GENERATION.md`。
- 正式采集前必须确认 `D-SIM-001`、数据微观分布、相机配置、GT 采样、位姿输入与场景级数据划分。

## 阶段状态

| Stage | 状态 | 核心交付 |
|---|---|---|
| -1 论文要求审计 | **用户已通过；提交状态待记录** | matrix + Fig.2 audit + figure/table/equation audit |
| 0 验收规定 | **用户已通过；提交状态待记录** | S00_TASK_DEFINITION.md |
| 1 数据生成 | **进行中：契约草案及数据决策待审** | S01_DATA_GENERATION.md |
| 2 稀疏表示 | 未开始 | S02_REPRESENTATION.md |
| 3 网络结构 | 未开始 | S03_NETWORK.md |
| 4 Loss/训练 | 未开始 | S04_TRAINING.md |
| 5 时序反馈 | 未开始 | S05_TEMPORAL_FEEDBACK.md |
| 6 数据增强 | 未开始 | S06_AUGMENTATION.md |
| 7 Evaluation/论文实验 | 未开始 | S07_EVALUATION.md |
| 8 最终复现审计 | 未开始 | S08_FINAL_AUDIT.md |
