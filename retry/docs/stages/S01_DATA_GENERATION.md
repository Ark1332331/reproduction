# Stage 1 - 仿真数据生成（契约草案）

状态：`IN_PROGRESS / AWAITING_REMAINING_DATA_DECISIONS`。`D-SIM-001` 已批准 IsaacLab 替代 IsaacGym，`D-WALL-001` 已批准训练墙固定 `1.2 m`；其他数据口径获批之前不迁移采集代码，不把旧 NPZ 标为 canonical retry 数据。

## 1. 原文与矩阵

本阶段精读：论文 p.2 Introduction（>200,000 observations）、§III-D Data Generation、Fig.3 图与图注、§IV-A 对同高 training walls 的补充、§IV Evaluation 的局部 64³/0.05 m 指标区域。矩阵直接条目 `DATA-001..028`、`SENSOR-001..005`；跨阶段 `TASK-001/002/006`、`TEMP-001/002`、`REP-003/004`、`AUG-001..011`、`EVAL-005/006/011`。Stage 6 的测量增强与 Stage 1 的**干净原始采集**分开记录。

论文明确：IsaacGym 随机场景有 stairs、ground boxes、walls、poles、narrow corridors；Fig.3 的楼梯 step width `[0.2,0.5] m`、step height `[0.08,0.25] m`，boxes width/length `[0.2,2.0] m`、height `[0.08,0.25] m`，corridor width `[2,6] m`，均匀采样；training walls 同高。ANYmal C 由 rough-terrain locomotion policy 驱动，向可达位置运动，随机 speed/base orientation；前后左右四路模拟 RealSense 向下倾斜 30°；每帧从机器人周围 terrain mesh 取密集 GT。作者报 200000 time steps，平均 43% GT points 对测量可见；后者匹配定义缺失。上述是论文条件，不代表旧仓库已经满足。

## 2. 数据从哪里来，如何流动

```text
场景生成器(seed, 类型, 参数) -> terrain mesh + 元数据
    -> 运动策略 + 仿真时钟 -> 每帧 robot pose
    -> 四路相机深度 -> 各相机外参投影 -> 当前局部测量 M_t
    -> 同一 mesh 在同一时刻/局部区域采样 -> 仅监督用 GT_t
    -> 一条轨迹文件 + provenance -> 按场景身份划分 train/val/test
```

关键不变量：同一帧的深度、pose、GT 必须对应同一仿真时刻；四相机坐标正确并可回投；局部地图围绕机器人；GT 不得进入推理输入；场景身份在数据划分间不重叠；原始仿真真位姿与模型可用的位姿分开存放。Stage 5 再决定相对位姿的算法和噪声注入。

## 3. 拟议数据契约（未批准的字段语义不可当作论文原设）

| 层 | 至少保存 | 验证方式 |
|---|---|---|
| 场景 | terrain 类型、scene identity、layout/motion seeds、生成器版本、所有抽样参数及 mesh 标识 | 同 seed 可复现，五类场景可视化，论文公开分布做范围测试；固定 training wall height 可检查 |
| 相机 | 每路方向、分辨率、FOV/内参、完整外参、深度范围、时间戳、有效深度/有效点统计 | 投影/逆投影测试、相机顺序和方向图；观察遮挡与盲区，不能只靠四个非空数组判定正确 |
| 帧 | frame index、仿真时间、原始 `T_world_base` 4×4、输入用 pose 来源/噪声版本、`M_t`、`GT_t`、坐标系说明、每路点数 | 首帧及连续帧可重投影；测量与 GT 在同一区域；缺帧和提前结束显式标记 |
| 轨迹/划分 | episode id、scene identity、结束原因、captured/requested frames、schema version、collector/策略/仿真器版本、代码 commit、split manifest | split 交集为空、文件可完整读取、随机抽样逐帧可视化和重复采集比对 |

这只是读写接口草案：使用 NPZ/JSON 或其他容器及实际键名待批准；不得通过对旧 NPZ 重命名绕过数据有效性审计。采集按 **小型冒烟集 → 五类独立场景的验证集 → 规模扩充** 递进；200000 时步是作者数据规模对照，未达到则报告差距。可见率须先定义点匹配和分母，本地结果标 `local visibility`，不能直接宣称达到论文 43%。

## 4. 旧实现/外部参考：可以借什么，不能借什么

| 来源 | 已能看见的实现 | 本阶段差异/风险 |
|---|---|---|
| `simulation/paper_terrains.py` | 五类地形、Fig.3 范围、几何结构缓存 | 8 m tile、四级台阶、数量/布局/杆尺寸是实现假设；**`walls_terrain` 每段墙的高度随机采样 `[0.4,1.0] m`，违反论文所有训练墙同高的条件**；retry 采集前须改为已批准的固定 `1.2 m` 并测试，旧文件不能直接复用 |
| `simulation/collect_isaaclab_anymal_trajectory.py` | IsaacLab、四相机、局部点云与 primitive-derived GT、NPZ 元数据 | 不是原文 IsaacGym；记录的 `translation.z=0` 且仅有 yaw 差，不是完整 3D pose；相机尺寸与点数上限是本地选择 |
| `simulation/collect_isaaclab_anymal_vectorized.py` | 多环境批采与逐环境相机对齐、visible camera counts | 基于同一平面 pose/GT 思路；向量化不得引入跨环境相机/mesh 对错 |
| `data_pipeline/freeze_r7_split_manifest.py` | 按 terrain/scene seed 检查 train/val 重叠 | 需确认 seed 是否足以代表 mesh identity，多个种子组合和 test 独立性尚须验证 |
| xjtu-wang/3D_Reconstruction 的 `isaaclab_datacollect_anymal_sequential.py` / `prepare_scene_split.py` | 完整 `[T,4,4]` poses、打包点云、scene-level split | 非作者官方数据契约；不能直接与旧仓库的 `translation_XX/yaw_XX` 同读，且其他模拟参数仍需单独审核 |

## 5. 小实验与阶段验收（尚未运行）

1. 五类地形各固定 seed 生成图与参数直方图；核对论文已知范围，并列出未知参数的审批结果。训练场景的每一段墙必须等于 `1.2 m`，且 manifest 标识此本地假设。
2. 四路相机分别显示 depth、世界点云与机器人局部投影；构造已知位置物体做几何检查。
3. 对单条连续轨迹抽查首/中/末帧，复算相对位姿并将上一帧 GT/测量变换到当前局部帧做几何诊断（**只用于数据测试，不把 GT 送给网络**）；核查帧同步与地形 GT。
4. 做可重复采集、跨 split 场景 identity 交集为空、坏帧和缺相机显式失败、旧数据迁移资格审计；输出各地形的帧/轨迹/点数、遮挡和空帧统计。
5. 完成论文 ↔ 规范 ↔ 代码 ↔ 样本/图/统计的四向核查后，记录偏差、采集命令及数据 manifest；由用户审计通过再进入 Stage 2。

当前证据：已重新精读上述段落与图注、核对旧采集代码和第三方参考；**尚无 retry 采集器、测试、样本或实测指标**。验收前不运行大规模采集，避免把未经审核的分布固化。

## 6. 需要审批的具体口径

| 决策 | 论文已知 / 可选方案 | 推荐先核查的影响 |
|---|---|---|
| `D-SIM-001` | **已批准 IsaacLab 替代论文 IsaacGym** | 数据 manifest 与结果均标为仿真器偏差；不得声称渲染/运动/深度分布同论文一致 |
| `D-WALL-001` | **已批准训练墙高统一 `1.2 m`，论文未给具体高度** | 逐墙断言并写入 manifest；旧随机墙高数据不得冒充正式训练数据 |
| `D-DATA-001`、`DATA-023..026` | poles/墙厚/box 布局、策略/速度分布、GT 密度未知；候选旧生成器参数或重新设定 | 固定墙高之外的参数及可见性统计仍需审核 |
| `D-SENSOR-001`、`SENSOR-002..005` | 四路/30°已知；FOV、内外参、深度范围、采样 cap 与帧率未知 | 可用旧设置作候选，但相机覆盖会改变遮挡和 43% 可见性 |
| `D-POSE-001` | 旧 planar delta+yaw、另存完整真位姿+明确模型输入估计位姿 | 缺 z/roll/pitch 会影响阶梯与漂移；建议先保留原始完整 pose，输入版本在 Stage 5 批准 |
| `D-SPLIT-001`、`DATA-027/028` | 作者未给 split/可见率算法；建议 scene identity 隔离与本地可见率协议 | 防止同布局跨 split 泄漏；本地 visibility 与论文 43% 只能作描述性比较 |

`D-SIM-001` 与 `D-WALL-001` 已分别获批；其余选择仍为 `IMPLEMENTATION_ASSUMPTION / OPEN_APPROVAL`，不是 Stage 0 的隐含批准。后续逐项冻结参数与测试；任何被拒绝的候选可替换，不影响本阶段要求清单。

## 7. Stage 1 实现计划草案（提交用户审核，非批准记录）

本节落实 `AGENTS.md` 的“先计划、后实现”门槛。以下所有候选仅用于讨论；**用户审核计划及其中的重要口径之前，不迁移采集代码、不开始正式采集**。论文明确的范围和已批准的 `D-SIM-001`、`D-WALL-001` 不在此重复征求批准。参数不能因旧仓库已使用就自动升级为论文事实。

### 7.1 拟议实施顺序与可检查产物

| 顺序 | 工作模块 | 产物与通过条件 |
|---|---|---|
| A | 数据契约 | `retry` 独立数据 schema 与 manifest：每帧时间戳、四相机深度/标定、局部测量与 GT、完整原始真位姿；约定坐标系和版本；坏帧不可静默丢弃 |
| B | 地形生成 | 只迁移五类必要几何并保存逐场景抽样参数/mesh identity；已知 Fig.3 范围与所有训练墙高 `1.2 m` 自动断言，生成每类可视化 |
| C | 仿真与传感器 | IsaacLab ANYmal C + 可核验来源的 rough-terrain policy；四路相机分别回投、检查 30° 下倾和帧同步；GT 直接来自对应地形几何，不从深度拼接 |
| D | 小样本采集 | 五类各至少一条固定 seed 轨迹，首/中/末帧几何图、点数/缺测/终止原因统计和同 seed 重跑对照；零有效帧为失败 |
| E | 数据复用/划分 | 逐个旧 NPZ 做 schema、逐场景参数与证据审计；能证明满足正式口径的才可复用，否则标 `legacy_only`；冻结 scene-disjoint train/val/test manifest |
| F | 四向审计 | 论文条目 → 契约 → 实际代码 → 样本和指标；记录偏差、最小复现命令与证据；用户验收后才 commit/push Stage 1 |

建议物理存储优先采用**每轨迹 NPZ + 每次采集 JSON manifest**，以减少与现有诊断脚本的格式摩擦；但是新 schema 必须独立版本化，不能直接把旧字段改名。先保存原始深度与标定，再保存回投点云及其生成版本，以便追溯。具体字段名/压缩和容量预算在实施时小样本验证，格式选择本身仍待本计划审核。

### 7.2 一次审核需决定的本地口径

| 编号 | 候选方案（全部为 `IMPLEMENTATION_ASSUMPTION`） | 另一可行选择 / 后果 |
|---|---|---|
| `D-DATA-001` 几何 | 以旧生成器的 `8 m` tile、4 级楼梯（depth `[1.5,2.5] m`）、8 boxes、3 walls（厚 `0.1 m`、长 `[1,3] m`）、10 poles（半径 `[0.05,0.15] m`、高 `[0.4,1.0] m`）、2 corridor walls（厚 `0.08 m`、长 `7 m`）为**起始候选**；按论文公开范围采样，其余布局参数逐项写 manifest；训练墙高执行已批准的 `1.2 m`，corridor 墙是否同属此类在审核时明确 | 另设更广的数量/布局分布。旧值便于代码复用，但可能覆盖不足或造成动力学偏差；必须用可视化/统计检查实际分布 |
| `D-DATA-001` 运动 | 优先核验旧 ANYmal C rough-terrain policy 的 checkpoint、任务定义和许可/可运行性；固定 seed 的随机速度与基座朝向候选以现有采集器设置为起点，完整保存每轨迹实际命令、范围、初始姿态与终止原因 | 另选兼容策略或运动分布；会改变可到达位置、遮挡及轨迹长度。**在拿到旧脚本真实默认值和运行证据前不冻结数值范围** |
| `D-DATA-001` GT | 从本场景实际 mesh 采样局部 dense GT，记录采样间距/算法、网格哈希与局部窗口；独立复核旧 primitive-derived GT 是否与渲染 mesh 相符 | 沿用旧 primitive 参数的解析采样更快，但几何不一致时可能泄漏错误监督；采样密度会改变点数、可见率 |
| `D-SENSOR-001` | 优先核验旧采集器 `160×120`、深度裁剪 `[0.1,20] m`、四路偏移 `0.35 m`、逐相机 `1500` 点上限；FOV、完整安装姿态、采样时序从实际 CameraCfg/运行信息导出后再提具体值 | 选不同配置或无点数上限；遮挡率、可见性、开销都会改变。旧参数只能算候选，不能声称与原 RealSense 等价 |
| `D-POSE-001` 存储 | 每帧保存真实 `T_world_base`（4×4）和坐标系、仿真时间；额外标注“给模型的 pose 来源/版本”，不把 GT 偷渡给网络；模型输入相对位姿和噪声在 Stage 5 单独批准 | 仅旧 `xy+yaw` 会丢弃 `z/roll/pitch`，难以检查上下台阶；保存完整真位姿并**不等于**批准用真位姿作正式模型输入 |
| `D-SPLIT-001` | 使用生成器版本+地形类型+布局 seed+所有几何抽样参数/mesh hash 组成 scene identity；按 identity 做 train/val/test 互斥、按轨迹汇总指标；先小规模冻结分割再扩容 | 只按文件名/轨迹 seed 会有同场景跨 split 泄漏；具体比例、seed 列表仍需单独登记并审查 |
| `DATA-028` 可见率 | 在相同局部窗口中，以 `0.05 m` 体素定义 GT voxel 有相机测量落入为“可见”；报告逐帧 GT 可见体素比例及平均值，并显式命名 `local_voxel_visibility` | 最近邻距离容差/表面点投影会产生不同的比例；论文 43% 的原算法未知，**不可直接宣称本地值与之同口径** |

上述候选包含两个尚需先做事实核查的项目（策略命令默认值、相机 FOV/采样频率）：在未读出并展示实际值前不得把含糊的“沿用旧值”当成已经批准的具体参数。审核可以先确认方向，具体值随后以补充决策再批准。

### 7.3 数据复用判定与停止条件

旧文件必须证明：地形 mesh 和参数可追溯；训练墙逐墙为 `1.2 m`；四路相机配置和同帧 pose/深度/GT 对齐；完整 3D 真位姿可恢复或明确限用于非正式诊断；场景身份可隔离。缺任何必要证据时保留原件并标注 `legacy_only`，不纳入 canonical 集。旧数据**不是一概作废**，但只靠文件可读取或字段名一致不算通过。

如果点云与 mesh 不一致、四相机几何无法回投、pose 与深度不同步、训练墙高不合规、scene identity 无法判定，停止扩容并保存失败样本与原因；不通过“让模型多训”掩盖数据错误。本阶段只交付可审计数据和初步数据统计，不报告网络 F1 为 Stage 1 成功指标。
