# 图、表、公式逐项审计

本文件重新扫描论文所有 Figure、Table 和 Equation。原则：图本身、图例、caption、正文对图的解释都视为复现信息来源，不再只扫描正文段落。

## Fig.1 — Overall approach

### 图中/图注明确信息
- 输入来自 depth sensors 的 noisy point cloud。
- previous output 会反馈到下一时刻网络。
- 图中体现 Transform，再进入下一时刻网络。
- 输出是 surrounding scene 的 point-cloud reconstruction。
- 这是 auto-regressive 流程，reconstruction 会随时间 refined。
- 网络能保留已经进入机器人 blind spots 的对象。

### 后续对应
- Stage 0：任务输入输出与时序关系。
- Stage 5：autoregressive feedback / transform。
- Stage 7：blind-spot temporal memory 小实验。

---

## Fig.2 — Architecture

完整逐层审计见 `FIG2_ARCHITECTURE_AUDIT.md`。

关键结论：
- 9 个视觉尺度块；
- 22 个卷积类操作；
- 10×普通 Conv；
- 4×stride-2 Conv；
- 4×stride-2 Transposed Conv；
- 4×Conv+Prune；
- 4 条 skip connection；
- encoder 空间尺度：`64³→32³→16³→8³→4³`；
- temporal dimension 在 encoder 下采样时保持；
- 最终 Conv 输出 3 维 sub-voxel feature。

---

## Eq.(1) — Sparse COO representation

论文定义：

`c_i = [floor(x_i), floor(y_i), floor(z_i), k_i]`

`f_i = p_i mod 1`

逐项含义：
- `p_i=[x_i,y_i,z_i]` 是第 i 个 occupied voxel 内点集 centroid，在 grid reference frame 表示；
- “one unit is equal to one cell”；
- `c_i` 是 sparse COO coordinate；
- `f_i` 是 centroid 相对 voxel bottom-left-rear corner 的归一化 sub-voxel offset；
- `f_i∈[0,1]`；
- `k=0` 表示 current measurement；
- `k=1` 表示 previous output；
- 连续 centroid 可通过 cell index + feature 恢复；
- 这就是网络进行 4D space-time convolution 的输入坐标设计。

后续必须做：5 个手算点的 voxel/offset/round-trip 验证。

---

## Fig.3 — Randomized simulation environments

### 图及 caption 明确信息
- 图展示 randomized simulation environments。
- stairs width：均匀采样 `[0.2,0.5] m`。
- stairs height：均匀采样 `[0.08,0.25] m`。
- boxes width/length：均匀采样 `[0.2,2.0] m`。
- boxes height：均匀采样 `[0.08,0.25] m`。
- walls 用于产生 corridor width `[2,6] m`。

### 紧邻正文补充
- 结构化环境类别：stairs、boxes on the ground、walls、poles、narrow corridors。
- simulator：NVIDIA IsaacGym。
- scene element dimensions/locations randomized。
- robot 向 reachable position 行走，speed 与 base orientation randomized，控制器为 rough-terrain locomotion policy。
- 4 个 simulated Intel RealSense depth cameras：front/back/left/right，向下倾斜 30°；称为 standard ANYmal C configuration。
- GT：每 timestep 从机器人周围 terrain mesh 密集采点。
- 平均只有 43% GT points 在 measurements 可见。
- dataset：200000 time steps（Introduction 另写 “more than 200,000 point cloud observations”）。

### 仍未说明
pole 尺寸、wall 高/厚、每类对象数量、布局分布、camera intrinsics/FOV/resolution、GT 采点密度等。

---

## Fig.4 — Temporal reconstruction on validation trajectory

### 版式
- 两列：trajectory 开始时 vs `1.5 s` 后。
- 三行：`(a) Measurement`、`(b) Reconstruction`、`(c) Ground truth`。

### 明确实验行为
- 右列中两个对角方向的 boxes 当前已经无法在 measurement 中看到；
- reconstruction 仍能恢复它们，证据来自 previous measurements；
- 用于验证 temporal memory / auto-regressive feedback；
- wall 即便只看见底部也能被补全；
- 正文说明这是因为 training data 中所有 walls 使用相同高度；
- 固定 wall 高度是作者主动设计：只保留 locomotion 相关特征，因为机器人无法越过更高障碍，精确表示其高度收益有限。
- 正文末句说 temporal information 对 "left column" 必要，但正文前文和 caption 把已进入盲区的两个 boxes 指向 right column。保留这一原文指代冲突，后续按实际时间与可见状态构造测试，不据此选择左/右列作为算法条件。

这条“所有训练墙高度相同”出现在 Experiments，不在 Data Generation 主段落，因此必须回写到数据生成契约。

---

## Fig.5 — Zero-shot stairs

### 图及 caption
- 只有 `(a) Measurement` 与 `(b) Reconstruction` 两个面板；没有 GT 面板。
- “zero-shot” 在这里明确指 **只使用 current time-step measurement**。
- 不使用 previous reconstruction / temporal evidence。
- 用于证明仅靠 spatial context 也能补全结构。
- caption 说明可正确重建 vertical surfaces（原文称 “vertical surfaces of the walls”）。

### 后续实验要求
必须保留一个 no-history / current-only 模式，不能只有完整 autoregressive 模式。

---

## Fig.6 — Measurement removal robustness

### 坐标与曲线
- 横轴：`Removed data [%]`，显示范围 0–80%。
- 纵轴：`Performance [%]`，显示范围约 60–100%。
- 红：F1；绿：Recall；蓝：Precision。
- caption：validation trajectories 上，性能随 measurement 中被额外删除的数据量变化。

### 协议细节
- 使用同一批 validation trajectories；
- removal rate **不包含本来就因 blind spots 缺失的点**；
- մինչև 50% 额外删除时性能基本稳定；
- 正文报告 50% omission 时 F1≈88%；
- 更高删除率后 output density 急剧下降、holes 增加；
- 文中报告 sparse regime 下 Recall 到 62%、Precision 到 82%；
- 62% / 82% 是更高删除率之后的叙述，不能与 50% 时的 F1≈88% 当成同一工作点；原文没有为这两个数字精确标注横轴位置。
- 删除超过 80% 时，decoder 可能因 sparsity + pruning 产生 empty tensors。

### 后续要求
这不是普通 validation，而是独立 robustness sweep；需要固定 validation trajectories 后做 controlled removal。

---

## Table I — Box / Stairs real-robot quantitative comparison

列：Method / Precision / Recall / F1 / MAE[cm]

### Stairs
| Method | Precision | Recall | F1 | MAE |
|---|---:|---:|---:|---:|
| Measurement | 87.7 | 50.7 | 64.0 | 0.64 cm |
| Elevation Mapping | 73.2 | 79.8 | 76.3 | 2.1 cm |
| Voxblox | 76.3 | 72.3 | 73.5 | 1.6 cm |
| Ours | 86.0 | 89.9 | 88.9 | 0.8 cm |

### Box
| Method | Precision | Recall | F1 | MAE |
|---|---:|---:|---:|---:|
| Measurement | 80.8 | 61.6 | 69.8 | 1.2 cm |
| Elevation Mapping | 72.2 | 80.0 | 75.9 | 1.7 cm |
| Voxblox | 74.0 | 77.9 | 75.8 | 1.4 cm |
| Ours | 84.8 | 84.9 | 84.8 | 1.0 cm |

### 口径
- 这些结果属于 real-robot Stairs / Box datasets，不是当前 IsaacLab 仿真数据直接应达到的数值。
- metrics 在 robot-centric `64³` voxel grid、cell size `0.05 m` 下计算。
- 还报告 mean absolute height difference。
- 精确 height matching/reduction 规则论文未完整说明。

---

## Fig.7 — Heavy state-estimator drift comparison

### 版式
- 两列时间：`t=2.76 s`、`t=3.30 s`。
- 五行：Measurement / Elevation Mapping / Voxblox / Our approach / Ground truth。

### 实验与现象
- trajectory 含 heavy state-estimator drift；
- 正文说明机器人上楼梯时，在 `t=2.76 s` 刚过后 state estimator 向下漂移约 `7 cm`；
- Our approach 检测 previous map 与 current measurement 的 height discrepancy，并整体向下对齐 map estimate；
- Elevation Mapping 仍在旧高度更新，形成 uneven map；
- Voxblox 可较合理纠正 drift，但机器人正下方仍有错误；
- stairs 在 map border 附近略弯，作者归因于 step edge measurement noise 在边缘更强；
- Our approach 可能出现小 holes，作者归因于 pruning 且没有显式训练 watertight output；这些 holes 很局部，对 controller 影响不大；
- local method 隐式对齐后，map frame 会随机器人漂移，不再与 ground-truth global frame 一致，但 robot reference frame 中仍可正确服务 locomotion。

---

## Fig.8 — Locomotion impact under map drift

### 坐标和曲线
- 横轴：`Drift [m]`，0–0.20 m。
- 纵轴：`Performance [%]`，0–100%。
- 绿：Success rate。
- 红：Survival rate。

### 协议
- simulation target-reaching task；
- 每个 drift 条件统计总计 4000 trajectories 的平均结果（caption）；
- robot 必须在 `7 s` 内穿过 challenging terrain 并到达目标；
- drift 从 0 到 20 cm；
- drift 通过在 measured terrain 中随机放置对应高度的 bumps 来模拟；
- survival 在大 drift 下仍明显高于 success，说明“没摔倒”不等于“按时完成任务”；
- 20 cm drift 时 fewer than 20% reach target on time；
- 文中说明 survival 降到约 70%；
- drift >10 cm 时 knee-ground collisions 相对 drift-free 约增加 3 倍。

---

## 论文中没有第二个编号公式

正文唯一显式编号公式是 Eq.(1)。其余训练目标、pruning 规则、指标均以文字描述，没有给单独编号公式。因此后续实现时不能把 Agent 自己写出的数学式误标成“论文公式”。
