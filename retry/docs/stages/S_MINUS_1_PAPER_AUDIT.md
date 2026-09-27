# Stage -1 — 论文复现要求审计（V2）

## 1. 目标

在写 `retry/` 实现之前，把论文中会影响复现的实现行为、超参数、网络拓扑、数据生成条件、小实验、指标、运行协议和结果口径全部建立索引。

V2 特别强调：**图、图例、caption、表、公式与正文具有同等审计优先级。**

## 2. V1 失败在哪里

V1 虽然扫描了全文正文，但对结构图采用了摘要式记录。例如 Fig.2 被压成“4D U-Net-like encoder-decoder + skip connections”，没有逐层记录图里已经画出的网络操作。

用户审查后指出 Fig.2 实际包含：

- 9 个明显视觉块；
- 22 个卷积类操作；
- 4 种卷积类型。

这个反馈成立。因此 V1 不能作为 Stage -1 完成交付。

## 3. V2 扫描方法

逐页检查：

1. 正文段落；
2. Figure 本体；
3. Figure legend；
4. Figure caption；
5. Table；
6. Equation；
7. 紧邻正文对图/表的解释；
8. 后文章节对前面实现条件的补充。

特别检查“实现条件藏在 Experiments 中”的情况，例如 training walls 同高。

## 4. Fig.2 修订结果

独立文档：`../audits/FIG2_ARCHITECTURE_AUDIT.md`

当前逐图确认：

- 4 种图例操作；
- 22 个卷积类操作；
- 10 普通 Conv；
- 4 stride-2 Conv；
- 4 stride-2 Transposed Conv；
- 4 Conv+Prune；
- 4 skip connections；
- 9 个按视觉尺度组织的连续块；
- encoder 4 次空间下采样；
- decoder 4 次空间上采样；
- decoder 4 个 pruning stage；
- final Conv 输出 3D sub-voxel feature。

仍未公开：channels、kernels、skip merge、temporal 2→1 细节等。

## 5. 其他图表重新扫描结果

独立文档：`../audits/FIGURE_TABLE_EQUATION_AUDIT.md`

包括：

- Fig.1 autoregressive overall flow；
- Eq.(1) sparse COO + sub-voxel feature；
- Fig.3 terrain parameter distributions；
- Fig.4 temporal blind-spot memory；
- Fig.5 current-only zero-shot spatial completion；
- Fig.6 measurement-removal robustness；
- Table I Stairs/Box real-robot P/R/F1/MAE；
- Fig.7 heavy state-estimator drift；
- Fig.8 4000 trajectories target-reaching drift experiment。

## 6. 本轮新发现的关键“不能漏”信息

- Fig.2 精确层级比 V1 记录得丰富得多；
- all training walls same height 出现在 Experiment 而非 Data Generation；
- Fig.6 的 removal rate 不包含原本 blind-spot 缺点；
- >80% removal 会让 pruning 产生 empty tensors；
- Fig.5 zero-shot 明确是 current measurement only；
- Fig.7 具体给出约 7cm state-estimator drift；
- Fig.8 不只看 survival，还看 7s deadline 下 success，且平均 4000 trajectories；
- real-robot baseline 输出 mesh 后会重新 sample dense point cloud，再按同一评估方式比较。

## 7. 本阶段不做什么

- 不决定未公开 channels/kernels；
- 不决定 IsaacLab 是否可作为 canonical simulator replacement；
- 不决定 height MAE 自己的计算方式；
- 不迁移任何模型代码；
- 不运行训练。

这些都需要在对应阶段重新精读并审批。

## 8. V2 交付物

- `PAPER_REQUIREMENTS_MATRIX.md`
- `DECISIONS.md`
- `PROGRESS.md`
- `audits/FIG2_ARCHITECTURE_AUDIT.md`
- `audits/FIGURE_TABLE_EQUATION_AUDIT.md`
- 本报告

## 9. 当前状态

**COMPLETE / USER_APPROVED（2026-09-27）**

用户已明确确认 Stage -1 通过。本阶段通过表示全文要求索引可作为后续逐阶段精读的起点，不表示论文未公开的实现细节已得到解决。

## 10. Stage 0 启动前的补充审计（2026-09-27）

- Fig.4 正文末句的 "left column" 与前文及 caption 的 right-column memory example 不一致；矩阵增加 `EXP-F4-007`，不私自修正文义。
- Fig.6 的 F1≈88% 属于约 50% omission；Recall 62%、Precision 82% 属于之后更稀疏的区域，不能写成同一个混淆矩阵工作点。矩阵与图审计已勘误。
- 43% GT-point visibility 没有公开具体点匹配协议；不能与体素 recall 混用。矩阵增加 `DATA-028`。
- 论文的 "mean" 指标没有完整给出跨帧/轨迹聚合与空预测规则；真实机器人 BLK2GO/ICP 和仿真 mesh GT 也不是同一个目标域。矩阵增加 `EVAL-011/012`。
- 上述补充已纳入用户通过的 Stage -1 审计范围；正式实验口径仍须在对应阶段批准。
