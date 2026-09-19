# NSR 复现任务说明

## 2026-09-19 主线整理

仓库当前只保留论文复现主线、必要的采集/切分工具、机制测试和审计文档。
R2/R3/R3b toy pipeline、R6 控制器原型、旧 R5 合成训练入口、旧深度采集器、
一次性诊断/过拟合/剪枝探针及 GPU 等待脚本已经从源码树移除；历史证据仍保留
在本机 `data/` 和 `results/`，但这些运行产物被 `.gitignore` 排除，不随源码提交。
具体文件职责和执行顺序见 `MAINLINE_ARCHITECTURE.md`，删除类别见
`ARCHIVED_EXPERIMENTS.md`。主线测试在 `nsr-me-cu130-t291` 环境中已通过。
当前训练、评估和采集调度使用包入口（例如 `python -m training.run_r7_paper_train`）；
根目录转发脚本已删除，避免同一功能存在多个入口。

## 目标

复现 NSR 论文的核心实验，并形成可验证的复现结论。

成功不等于“代码能运行”。必须记录：

- 使用的数据集、数据划分与预处理；
- 实际运行的配置、随机种子和环境；
- 训练过程与关键指标；
- 最终评估结果与论文目标的对比；
- 与论文不一致之处、原因假设和未完成项。

## 工作原则

1. 先阅读本仓库现有 README、配置文件、训练脚本和已有实验记录，再行动。
2. 不要编造已运行的结果、文件、指标或结论。
3. 每次执行命令后，根据真实输出更新进展。
4. 遇到报错，优先定位根因；不要反复执行同一条已知失败命令。
5. 不要删除数据、模型、日志或用户已有改动。修改代码前说明目的。
6. 除非明确要求，不要替换依赖版本、重构大量代码或下载超大数据集。

## 当前复现状态

- 论文：NSR
- 当前阶段：`数据采集扩容与训练前验证`
- 已完成：
  - 串行与 4 环境并行 ANYmal 采集器、输出契约及可恢复批处理启动器已实现并经小规模数据验证。
  - 已有 44 条轨迹、356 个有效采集帧；其中训练划分 201 帧、验证划分 155 帧，仅可用于流程与小规模诊断。
  - 论文数据量锚点（20 万时间步、五类结构化地形、四向深度相机）及当前分布偏差已完成审计。
  - 已确认宿主机 GPU（RTX 5070、驱动 580.173.02）及 IsaacLab 环境内 PyTorch CUDA 可用；终端恢复阻塞已解除。
  - 四环境向量化恢复试采集完成，4 个 boxes 输出的帧数为 12、12、12、8，逐帧 R7 数据契约已独立检查通过。
  - 已修复并验证向量化采集器：单方向相机在机器人局部地图中无点时作为缺测观测记录，而不再中止整个批次；四路均无点才会报错。启动器与串行采集契约测试共 5 项通过。
  - 首个 train 向量化批次（seed 60--63）已产生 boxes 3 条（12、12、11 帧）和 walls 3 条（12、12、5 帧）有效轨迹；原始及重试 manifest 均已保留。
  - 替代批次（seed 64--67）完成：boxes 为 12、12、4、8 帧，walls 为 12、3、3、7 帧。`vectorized_plan` 现有 14 条轨迹、125 帧，全部独立通过 R7 字段与形状契约；其中 8 条轨迹达到建议的 >=10 帧训练门槛。
  - 向量化启动器已增加显式 `--min-trajectory-frames` 门槛，并将 `min_trajectory_frames` 与逐轨迹 `training_eligible` 写入 manifest；默认值保持旧契约兼容，本轮 scale1 使用 10 帧。
  - scale1 五类 terrain（stairs、boxes、walls、poles、corridors）的 seed 80--83 采集已完成：15 条文件、146 帧，其中 7 条达到 >=10 帧门槛；执行 manifest 为 `data/r7_vectorized_train_scale1_manifest.json`。
  - 吞吐探针：8 环境 boxes、3 帧为 8/8 条、24 帧、46.7 秒；同配置 4 环境为 3/4 条、9 帧、42.6 秒。按有效帧/秒，8 环境约为 4 环境的 2.4 倍；尚未用 12 帧正式轨迹验证该比例。
  - 8 环境、12 帧 boxes 长轨迹验证完成：8 条文件、68 帧、83.3 秒，无 OOM；其中 4 条达到 >=10 帧。按全部采集帧约 0.82 帧/秒，按训练合格帧约 0.58 帧/秒。
  - 提前终止根因已定位为高可信的 episode 计数初始化问题：ANYmal `_reset_idx` 在整批 reset 时将 `episode_length_buf` 随机化；采集器随后只读取合并的 `dones`，未区分 timeout 与 base contact。seed 80 下 env index 2 在五类 terrain 均于首帧前消失，符合同一随机 episode 起点而非 terrain-specific 摔倒；`_get_dones` 的另一可能原因是基座接触力 >1.0。
  - 已在向量化采集器中修复 capture reset 的 `episode_length_buf`，并记录 `reset_terminated`/`reset_time_outs` 对应的 `termination_causes`；新增 `--random-motion`（每轨迹独立速度、侧向速度、偏航和初始 yaw）及元数据记录。
  - 启动器已支持 `--resample-failures --max-resample-batches N`：失败批次使用不重叠 seed 自动补采，并在 manifest 中记录父批次和尝试次数；默认 collector 路径改为脚本同目录，支持从仓库根目录或 reproduction 目录启动。
  - random-motion boxes pilot（seed 300、8 env、12 steps、>=10 帧）完成：8/8 合格、95 帧、83.3 秒；1 条在第 11 帧发生 base_contact，无 timeout，速度/yaw 元数据与四相机可见性均已写入 NPZ。
  - 五类 terrain scale2 probe（seed 320、8 env、每类 12 steps、>=10 帧、失败最多补采 1 批）完成：stairs 8/8 合格、96 帧；boxes 8/8、96 帧；walls 原批 6/8、补采批 4/8（合计 10/16、150 帧）；poles 原批 5/8、补采批 7/8（合计 12/16、163 帧）；corridors 原批 3/8、补采批 3/8（合计 6/16、130 帧）。失败样本均以 base_contact 为主，无 timeout。
  - 新增 `collect_r7_quota.py` 配额调度器：逐 terrain 统计合格帧，按不重叠 seed 调用现有启动器，保存每类 next seed 和历史，可从中断处恢复；`--max-batches-per-terrain` 支持有限时间的分段运行。
  - 配额调度器真实 smoke（stairs、seed 1100、8 env、目标 90 合格帧、随机运动）完成：单批 84.6 秒得到 94 合格帧，状态保存于 `data/r7_quota_smoke_state.json`，证明按配额停止与恢复状态有效。
  - 正式扩容 tranche（split `train_quota_scale3`，seed 1200 起，8 env，随机运动，最多每类 2 主批/3 次补采）已完成：10 个主批次产生 210 个 NPZ；合格帧为 stairs 186、boxes 272、walls 506、poles 470、corridors 249。状态保存于 `data/r7_quota_scale3_state.json`，未声称达到 4,000/terrain 目标。
  - 配额调度器已改为轮转：每轮对每个未达标 terrain 至多启动一个主批，避免任务中断时只积累易地形。持续任务在 2026-09-03 15:55 左右遭遇 corridors seed 1552 的 Vulkan `ERROR_DEVICE_LOST`；当前目录实际已有 stairs 1,631、boxes 2,020、walls 3,081、poles 2,901、corridors 1,245 合格帧（约 10,878），状态和已写入 NPZ 均保留，可从中断批次恢复。宿主机 `nvidia-smi` 正常，故此前沙盒里的驱动错误判断已更正。
  - 新增 `audit_r7_collection_distribution.py`，从实际合格 NPZ 汇总每类的帧数、唯一 scene seed、随机 command/yaw 范围、终止原因和相机缺测率。持续采集首批后审计显示 stairs 278、boxes 365、walls 611、poles 470、corridors 249 合格帧；尚不可据小样本断言接受后的运动分布已匹配论文。
  - 2026-09-04 清理了四个已确认无效或逃逸的遗留 collector（boxes seed 1600、corridors seed 1664、stairs seed 1728、walls seed 2112）；`SIGTERM` 无响应或日志停滞后仅对这些精确 PID 使用 `SIGKILL`。显存由约 7,676/8,151 MiB 降至单批次占用约 5.5--5.9 GiB，配额 scheduler 未受影响并继续轮转。宿主机 inotify 上限仍为 128/65536；无交互 sudo 无法修改，`tail` 会回退到轮询但不影响日志写入。
  - 修正采集编排：launcher 将 CUDA OOM、device-lost、GPU crash、segmentation fault 分类为 simulator failure，不再对其盲目重采样；quota scheduler 在每批结束和下一批开始前清理独立进程组及同一输出目录的逃逸 collector，并支持 simulator GPU failure 后按 terrain 自动从 8 env 降至 4 env。单元测试 3 项和 quota dry-run 均通过。修正版 scheduler 已从保存状态恢复（PID 909469），stairs seed 2448 的 8-env 批次成功增加 94 合格帧，当前只保留一个 collector。
  - 进一步修正超时语义：launcher 对每个 IsaacLab attempt 单独设置 300 秒超时并在自己的进程组内清理；quota scheduler 的外层等待覆盖有限的补采队列，timeout 也会触发 terrain 级降至 4 env。一次重启中发现并修复了错误传给 `subprocess.Popen` 的 `check=False` 参数；修复后 3 项单元测试通过。持续配额采集随后完成：stairs 4,135、boxes 4,017、walls 4,027、poles 4,089、corridors 4,092 合格帧，日志写出 `quota_complete=True`。
  - 已预注册训练规模曲线前的数据 gate（`DATA_VALIDATION_GATE.md`），并将其编码到审计器：每类 4,000 合格帧、>=250 unique scene seed、被接受 trajectory 无 timeout、部分相机帧 <=5%、base-contact <=25%、随机运动完整记录、四个运动边际的五个 bin 各 >=20 trajectory。最近审计为总 5,154 合格帧（stairs 730、boxes 976、walls 1,543、poles 1,308、corridors 597）；各类尚未通过规模/seed/覆盖 gate，训练规模曲线尚未启动。
- 当前阻塞：
  - 配额采集及 corridors 定向补采已完成，最终严格分布审计通过；五类合计约 20,436 合格帧，corridors yaw-rate 五个 bin 为 20/90/147/82/20。seed-disjoint validation 已完成：stairs 1,024、boxes 1,032、walls 1,101、poles 1,021、corridors 1,006 合格帧；冻结 manifest 为 `data/r7_train_validation_frozen_manifest.json`。训练规模曲线尚未启动。
  - 2026-09-04 16:41--16:47 又出现逃逸的 walls seed 2176 collector（PID 838045，约 4.1 GiB，日志停在 syntheticdata frame 9508），导致 poles/corridors/stairs/boxes 的连续重试全部报 CUDA OOM；该 PID 已停止。说明当前策略的主要系统性问题是失败后仍会残留 IsaacLab 子进程，seed 重采样无法修复 OOM。单 collector 清理后 walls seed 2360 已正常跑到 frame 11 并写出 12 帧轨迹，证明此前整批失败主要是资源竞争而非所有 terrain 同时失效。
- 下一步：
  - validation 已冻结；独立审计的核心质量项通过，但 validation 的 20-trajectory-per-motion-bin 统计门槛因每类仅 86–93 条轨迹未满足，需在报告中作为验证集规模限制记录。训练脚本已修复空 seed 过滤、增加按 terrain 的完整轨迹帧上限，评估脚本同步过滤短轨迹并支持每类轨迹上限。
  - 已完成优化基准：静态 current/target voxel 缓存 + 跨轨迹 microbatch=8 将小探针单步从 31.3 s 降至 27.8 s（约 11%）；均衡 1k/seed0、5-step 训练完成，单步 313.9–317.4 s，checkpoint 为 `results/r7_scale1_seed0_optimized5.pt`，loss 2.1995→2.1937。
  - 首个快速验证（冻结 validation 每类 10 条、alpha=0.1）已完成，但模型 mean F1=0.1173、precision=0.0646、recall=0.6692、height MAE=0.7590，明显差于 merge baseline F1=0.3665、MAE=0.0415；这只是 5-step/1k smoke，不能外推 20k 或论文指标。当前应先修复训练/评估数值与架构对齐问题，再考虑多 seed 或扩大规模。
  - 体素质心聚合已改为 NumPy 向量化实现（保持首次出现顺序），15 个表示/自回归单元测试通过；200k 点 CPU 微基准由 0.613s 降至 0.246s（约 2.5 倍），但尚未证明可将 314s/step 降到论文级吞吐。完整 validation 仍未运行，仅有每类 10 条的快速评估。
  - 端到端 1k/1-step 基准（同一 84 条轨迹、microbatch=8）在该优化后为 234.5s/step，相比此前 313.9–317.4s 约降低 25%；新 checkpoint 为 `results/r7_scale1_seed0_vectorized1.pt`。仍不足以支撑 20k×3 seeds：线性估算约 78 分钟/step。
  - 原计划的 1k/200-step/seed0 长跑（PID 1315278）在预处理阶段、尚未产生 checkpoint 时已安全停止，避免无效消耗；未产生指标。
  - 当前改为短时筛选：20 帧/terrain、20 steps、seed0，配置为 `pruning-alpha=0.5`、`occupancy-pos-weight=1`、`likelihood-pos-weight=1`、microbatch=8，后台 PID 1316320；完成后每类 5 条 validation 快速评估。训练与评估日志为 `data/r7_proxy20x20_seed0_{master,train,eval25}.log`，目标 checkpoint `results/r7_proxy20x20_seed0.pt`。
  - 上述短时筛选已完成：20-step loss 2.1885→约 2.140（约 2.2%），25 条 validation 的 mean F1=0.0372、precision=0.0458、recall=0.0316、height MAE=0.1972，baseline F1=0.3601、MAE=0.0413；因此未通过进入门槛 A 的趋势筛选，不启动 50-step/1k 长跑。checkpoint 为 `results/r7_proxy20x20_seed0.pt`，评估为 `data/r7_proxy20x20_seed0_eval25.json`。
  - alpha sweep（同一 proxy checkpoint、关闭内部 pruning、每类 2 条）显示 alpha=0.1–0.3 的 mean F1≈0.1223、recall≈0.6643，alpha=0.5 时 F1≈0.0187、recall≈0.0465；alpha=0.5 确有严重阈值/覆盖风险，但低 alpha 仍远低于 baseline。
  - no-pruning 20-step proxy 已完成：loss 2.1860→2.1365（约 2.3%），25 条 validation、alpha=0.1 的 mean F1=0.1180、precision=0.0650、recall=0.6605、height MAE=0.7641，仍未超过 baseline F1=0.3601；因此单独关闭训练 pruning 不是充分修复，暂不启动 50-step/1k。
  - 单帧数值诊断已完成（5 类各取 1 条，checkpoint `r7_proxy20x20_nopruning_seed0.pt`）：训练 target guard 下最终候选 26,704、target coverage 69.7%；无 guard 推理 alpha=0.5 后候选 coverage 仅 6.5%；no-pruning 候选 coverage 仍 69.7%。occupancy logits 均值约 0、标准差约 1e-2，尚未形成判别边界；结果写入 `data/r7_proxy20x20_numeric_diagnostic.json`。这表明候选生成/剪枝覆盖和占用 loss 信号是当前优先诊断点，不应直接进入 50-step 长训。
  - 新增 `--pruning-warmup-steps` 并完成 5-step warm-up + 15-step alpha=0.5 代理：loss 2.1860→2.1407，alpha=0.1 的 mean F1=0.1180、alpha=0.5 的 F1=0.0325，均未改善；warm-up 不是充分修复。结果为 `data/r7_proxy20x20_warmup5_seed0_eval25.json`，因此仍不进入 50-step 门槛 A。
  - 复用历史 validated 配置（`pruning=0`、`likelihood_weight=0`、`occupancy_pos_weight=1`、固定 LR=0.003）的 20-step 代理已完成：loss 1.4928→1.4607（约 2.1%），25 条 validation alpha=0.1 的 mean F1=0.1180、baseline=0.3601，仍未改善。结果为 `data/r7_proxy20x20_validatedcfg_seed0_eval25.json`，当前启动 50-step 代理前需先通过短预算趋势检查。
  - 当前 50-step 代理（同 validated 配置、每类 20 帧、PID 1336333）已启动，目标 checkpoint `results/r7_proxy20x20_validatedcfg50_seed0.pt`，完成后每类 5 条 validation 评估；在其结束前不启动 1k/50-step 或更大规模。
  - 上述 50-step 代理已完成：固定 LR=0.003、pruning=0、likelihood_weight=0、position_weight=1；loss 1.4928→0.6947（约 53.5% 下降），但 25 条 validation alpha=0.1 的 mean F1=0.1954、precision=0.1190、recall=0.5603、height MAE=0.4888，baseline F1=0.3601、MAE=0.0413，仍未通过目标。数值诊断显示首帧 logits 均值 -0.098、std 0.081，0.5 概率正率为 0，候选 target coverage 约 69.7%；训练集同样仅 F1=0.2012，故更像 occupancy 分数未形成判别边界/位置目标干扰，而非单纯跨 seed 泛化。
  - 训练脚本新增可配置 `--position-weight`（默认 1.0，保持旧行为）；首次 `position_weight=0` 代理在第 0--5 步 loss 仅 0.6944→0.6942 后中止，说明单独去除位置损失不是充分修复。该中止 run 未生成 checkpoint/eval，不得当作结果。
  - 为 decoder likelihood heads 与 position head 增加显式 bias（默认初始化行为不变，旧无 bias checkpoint 不兼容新结构）；bias 20-step proxy loss 1.6873→1.5517（8%），恢复至 50 steps 后 loss 1.6873→1.1709（30.6%）。同一 checkpoint 在 25 条 validation 上 alpha=0.5（关闭内部 pruning）F1=0.6434、precision=0.8586、recall=0.5184、MAE=0.0407，明显高于 baseline F1=0.3601。
  - 该 bias checkpoint 已完成冻结 validation 全量 438 trajectories 的 alpha=0.5、关闭内部 pruning 评估：F1=0.6546、precision=0.8694、recall=0.5308、height MAE=0.0356；baseline F1=0.3653、MAE=0.0409。alpha=0.1 仍因阈值过低保留几乎全部候选（F1≈0.118），因此 alpha=0.5 才是当前 paper-shaped pruning 的有效口径；尚不能声称内部 likelihood/pruning ablation 已验证。
  - 进入 1k 规模前的下一步为全量约 1k 帧/terrain、10-step 预检（bias heads、pruning=0、likelihood_weight=0、position_weight=1、LR=.003、microbatch=8），通过后再从 checkpoint 恢复至 50 steps；不得直接启动 20k 或多 seed。
  - 1k/10-step 预检曾启动且只有一个 GPU 训练进程：84 条轨迹，stairs/boxes/walls/poles/corridors 分别选取 198/200/199/199/192 帧（约 988 帧），首步耗时 239.1s、loss=1.7036；因后续 7 分钟仍未完成 step 1、且 alpha=.1 目标尚未通过，已安全停止，未产生 checkpoint/eval。
  - 当前回退到低成本代理：从 50-step bias checkpoint `results/r7_proxy20x10_bias_seed0.pt` 恢复至 100 steps（5 条轨迹、每类 10 帧），目标是检验 alpha=.1 的负候选校准；该任务未通过前不再启动 1k/50、20k 或多 seed。
  - 100-step bias 代理完成：loss `1.6873→0.9881`；raw alpha=.1 仍 F1=`0.1180`，证明未校准阈值会保留全部候选。新增评估参数 `--likelihood-logit-offset` 后，以 offset=`-2.197224577`（使 alpha=.1 决策边界对齐 raw alpha=.5）重新评估。
  - 独立 20-step bias、LR=.005 代理完成：loss `1.6873→1.4537`（下降 13.8%），checkpoint `results/r7_proxy20x10_bias_lr005_seed0.pt`；25 条 validation 校准 alpha=.1 F1=`0.3704`、baseline=`0.3601`。
  - 同一 20-step checkpoint 的完整 438 条冻结 validation 校准 alpha=.1 评估已完成：F1=`0.3758`、precision=`0.8683`、recall=`0.2420`、height MAE=`0.0222`；baseline F1=`0.3653`、MAE=`0.0409`。这满足当前目标的 20-step loss/F1 硬门槛，但仍需完成后续 1k/50-step 和完整 validation 作为目标终验。
  - 目标门槛通过后，已启动正式 1k/50-step（每类最多 200 帧、约 988 帧，bias heads、pruning=0、likelihood_weight=0、LR=.003、microbatch=8），目标 checkpoint `results/r7_scale1_bias50_seed0.pt`，训练日志 `data/r7_scale1_bias50_seed0_train.log`；训练完成后将以 offset=`-2.197224577`、alpha=.1、关闭内部 pruning 对 438 条 validation 做最终评估。该长跑预计约 3--4 小时，期间不启动其它 GPU 任务。
  - 该正式 1k/50-step 当前会话已正常完成 step 0（约 237.0s，loss=1.7036）和 step 1（约 236.1s，loss=1.6968）；step 2 正在运行，GPU 显存约 1.8 GiB、无 OOM/残留，尚未生成 save-every=10 的 checkpoint，也尚未开始最终评估。按实测约 4 分钟/step，预计总耗时约 3.5 小时。

开始工作时，先检查仓库与已有日志，并把以上状态改成真实情况。

## 推荐复现链

按以下顺序推进，并在每个阶段留下证据：

1. **环境确认**
   - 确认 Python、CUDA、PyTorch、关键依赖和 GPU 可用性。
   - 记录实际环境，而不是只记录配置文件中的要求。

2. **数据确认**
   - 找到数据目录、下载方式、预处理脚本和数据划分。
   - 验证样本数量、输入形状、标签或任务定义是否合理。

3. **最小可运行验证**
   - 用少量数据或短训练确认训练、验证、保存、恢复、推理链路均可运行。
   - 此阶段只证明流程通，不代表复现成功。

4. **正式训练**
   - 使用明确的配置、随机种子和输出目录。
   - 保存训练命令、配置副本、日志和 checkpoint 路径。

5. **评估与对比**
   - 使用论文对应的评估协议。
   - 将实际结果与论文报告值并列比较。
   - 清楚标注“达到”“接近”“未达到”或“无法判断”。

6. **结论与交接**
   - 总结当前复现阶段、证据、限制和下一步。
   - 让下一位执行者无需重新阅读全部上下文即可继续。

## 每次完成一轮工作后的输出格式

请使用下面格式汇报，语言简洁、基于证据：

### 本轮目标

一句话说明本轮要验证什么。

### 已执行

- 命令或操作：
- 使用的配置：
- 关键输出或日志位置：

### 结果

- 成功/失败：
- 关键数值：
- 结论：

### 证据

- 文件：
- 日志：
- checkpoint：
- 图表或评估输出：

### 限制与异常

- 哪些结论尚不能成立：
- 报错或偏差：
- 原因假设：

### 下一步

给出一个可立即执行、可验证的下一步。

## 复现完成判定

只有同时满足以下条件，才可称为“已完成复现”：

- [ ] 环境和数据来源可追溯；
- [ ] 训练与评估流程实际跑通；
- [ ] 最终指标有真实日志或结果文件支持；
- [ ] 已与论文的任务、数据划分和指标口径进行比较；
- [ ] 偏差与局限已明确记录；
- [ ] 已输出可交接的复现结论。

如果只完成了部分，请明确写“部分完成”，不要写“复现完成”。

## 2026-09-05 1k/50-step 正式诊断进度

### 本轮目标

在代理门槛通过后，执行 1k 帧、单 seed、50-step 正式诊断，并在同一冻结 validation 上完成完整评估；未完成前不启动更大规模或多 seed 训练。

### 已执行

- 启动 `run_r7_paper_train.py`，使用五类地形、每类最多 200 帧、`microbatch-size=8`、bias likelihood heads、`pruning-alpha=0`、固定学习率 0.003。
- 当前训练进程 PID 1371627，日志为 `data/r7_scale1_bias50_seed0_train.log`，目标 checkpoint 为 `results/r7_scale1_bias50_seed0.pt`。

### 结果

- 截至记录时完成 step 0–5；loss 从 1.7036 降至 1.6696（约 2.0%），单步约 236–237 秒。
- 进程仍在运行，尚未到 `save-every=10` 的首个 checkpoint 保存点。

### 限制与异常

- 该正式诊断预计约 3.3 小时；当前没有 OOM 或进程退出证据。
- 训练使用 `likelihood-weight=0` 且评估需显式 logit offset 的校准配置，结果只能作为当前实现的诊断，不等同于论文最终配置。

## 2026-09-06 1k/50-step 门槛 A 完成

### 本轮目标

在不覆盖 seed 0 checkpoint 的前提下完成 seed 1/2 的 1k、50-step 训练，并在冻结 validation 的全部 438 条轨迹上以 alpha=0.1 完整评估，汇总三 seed 稳定性。

### 已执行

- seed 1/2 使用与 seed 0 相同的五类 terrain、每类最多 200 帧、`microbatch-size=8`、bias heads、`pruning-alpha=0`、`feedback-alpha=0`、`likelihood-weight=0`、固定学习率 0.003 配置。
- 分别生成 `results/r7_scale1_bias50_seed1.pt` 与 `results/r7_scale1_bias50_seed2.pt`，并在冻结 validation 上运行完整 438 条轨迹评估。

### 结果

- seed 1 训练 loss `1.6747 -> 1.0867`；完整 validation F1 `0.000398`、precision `0.532849`、recall `0.000199`、height MAE `inf`。
- seed 2 训练 loss `1.8206 -> 0.9917`；完整 validation F1 `0.101879`、precision `0.154267`、recall `0.077773`、height MAE `0.444795`。
- seed 0/1/2 的 F1 分别为 `0.510279/0.000398/0.101879`，均值 `0.204186`、样本标准差 `0.269897`；仅 seed 0 达到 F1 `>=0.36`，门槛 A 的稳定性判定失败。
- precision 均值 `0.413127`、样本标准差 `0.224389`；recall 均值 `0.187059`、样本标准差 `0.259386`。height MAE 因 seed 1 为 `inf`，三 seed 均值为 `inf`；有限两 seed 均值 `0.262545`、样本标准差 `0.257741`。

### 证据

- 汇总：`data/r7_scale1_bias50_seeds012_summary.json`
- seed 1 训练/评估：`data/r7_scale1_bias50_seed1_train.log`、`data/r7_scale1_bias50_seed1_eval_full_calibrated.json`
- seed 2 训练/评估：`data/r7_scale1_bias50_seed2_train.log`、`data/r7_scale1_bias50_seed2_eval_full_calibrated.json`
- seed 0 基准：`data/r7_scale1_bias50_seed0_train.log`、`data/r7_scale1_bias50_seed0_eval_full_calibrated.json`

### 限制与异常

- 三次评估均使用显式 `likelihood_logit_offset=-2.197224577` 并关闭内部 pruning；因此不等同论文原始 alpha=0.5 的内部 pruning 口径。
- seed 间方差极大，seed 1 几乎没有正召回且 height MAE 不可定义；不能据此声称 1k 配置稳定，也不能把 seed 0 的高 F1 外推到更大规模。

### 下一步

- 按用户已确认的顺序，在记录 A 失败门槛后进入门槛 B：先做 5k 单 seed 的 1--5 step 计时探针，确认预算后再运行 50-step 与完整 validation；不得覆盖上述 checkpoint。

## 2026-09-06 门槛 B 探针启动状态

### 已执行

- 已准备 5k（每类最多 1,000 帧）、seed 0、3-step 探针，目标 checkpoint `results/r7_scale5_bias_probe_seed0.pt`，日志 `data/r7_scale5_bias_probe_seed0_train.log`。

### 结果与限制

- 当前 Codex sandbox 启动即报 `RuntimeError: No CUDA GPUs are available`，未执行任何训练 step，也未生成 checkpoint；这不是模型或数据失败。
- 需要在能访问 RTX 5070 的宿主终端执行同一命令后，才能取得 B 的真实耗时；在此之前不应以 CPU 运行替代探针。
- 已将可复用启动器保存为 `run_r7_scale5_probe.sh`；它会先检查 CUDA 可见性，再运行 3-step 探针并写入同一日志。

### 探针结果

- 宿主 GPU 已完成 5k、seed 0、3-step 探针：单步 `1201.0/1198.6/1201.6 s`，均值约 `1200.4 s`（20.0 分钟）；loss `1.7025 -> 1.6888`。
- checkpoint 为 `results/r7_scale5_bias_probe_seed0.pt`。按实测，50-step 纯训练约 `16.7 h`，尚未包含完整 validation。
- 正式 B 启动器已保存为 `run_r7_scale5_train.sh`，输出 checkpoint `results/r7_scale5_bias50_seed0.pt`，不会覆盖 1k checkpoint。
- 正式启动器现会在 50-step 训练成功后自动运行冻结 validation 的 438 条轨迹评估，输出 `data/r7_scale5_bias50_seed0_eval_full_calibrated.json`；训练失败时不会进入评估。

### 宿主 GPU 资源排队（2026-09-06）

- 宿主机 `nvidia-smi` 正常；当前唯一 CUDA 计算进程为 Unitree G1 训练 PID `51522`，占用 `3190 MiB`，GPU 利用率约 `34%`。没有 NSR 正式 5k 进程，因此先前“终端无反应”不是 NSR 卡死，而是正式脚本尚未进入运行。
- 已启动保持会话的等待器 `run_r7_scale5_when_gpu_free.sh`（会话 PID 41410）：每分钟检查 `nvidia-smi --query-compute-apps`，待其他 CUDA 进程退出后自动启动 `run_r7_scale5_train.sh`；不终止、不干扰 G1 任务。

## 2026-09-05 1k/50-step 与完整 validation 结果

### 结果

- 1k/50-step 单 seed 训练已完成，日志末行为 `[DONE] trained 50 steps; loss 1.7036 -> 1.1959`，相对下降约 29.8%。
- 在冻结 validation 的全部 438 条轨迹上完成 alpha=0.1 评估：model F1 `0.510279`，baseline F1 `0.365330`；model precision `0.552265`、recall `0.483205`、height MAE `0.080295`。

### 证据

- checkpoint：`results/r7_scale1_bias50_seed0.pt`
- 训练日志：`data/r7_scale1_bias50_seed0_train.log`
- 完整评估 JSON：`data/r7_scale1_bias50_seed0_eval_full_calibrated.json`
- 完整评估日志：`data/r7_scale1_bias50_seed0_eval_full_calibrated.log`

### 限制与异常

- 评估使用显式 `likelihood_logit_offset=-2.197224577` 将 alpha=0.1 阈值校准到当前输出分布，并关闭内部 pruning；这与论文原始 alpha=0.5/pruning 配置并不完全等价。
- 本轮仅完成单 seed、约 1k 帧规模；不能据此声称已完成论文规模的多 seed 复现。

## 2026-09-06 1k 多 seed 稳定性门槛启动

### 本轮目标

按审计门槛 A，在不覆盖 seed 0 结果的前提下运行 1k/50-step seed 1，并随后在同一冻结 validation 上完成完整评估。

### 已执行

- 已确认 GPU 空闲且无残留 NSR 进程。
- 已启动 seed 1：`results/r7_scale1_bias50_seed1.pt`，训练日志为 `data/r7_scale1_bias50_seed1_train.log`。

### 结果

- seed 1 已完成 step 0，loss `1.6747`，单步耗时约 241.5 秒；训练进程仍在运行。

### 限制与异常

- seed 1 尚未完成 50 steps，也尚未生成 validation 指标；不能据此判断多 seed 稳定性。

## 后续目标变更

用户已明确要求：门槛 A（seed 0/1/2）完成并验收后，立即继续门槛 B（5k 单 seed + 完整 validation）；在 A 未完成前仍不得启动 5k。

## 当前目标变更（2026-09-06）

用户将当前目标改为“对齐论文实现”。在论文实现对齐契约与小规模验收（坐标/时序、逐层 likelihood、内部 pruning、loss 权重、评估口径、1--2 条轨迹过拟合）通过前，不再启动新的 5k/10k/20k 长时间训练。已有 A 结果和 5k 探针仅作审计证据保留。

### 2026-09-06 对齐目标执行记录

- 已冻结后续 canonical 入口：论文式 12-step detached autoregressive feedback、四层 decoder/逐层 likelihood target、最终 k=0 输出、occupancy/position/likelihood 三项损失、内部 pruning alpha=0.5，以及固定随机性和评估口径。
- 已执行 A1 合约测试：`test_r5_sparse_loss.py`、`test_r5_sparse_model.py`、`test_r7_autoregressive_rollout.py`、`test_r5_multiscale_target.py`、`test_r5_four_level_training.py`、`test_r5_training.py`，共 21 项，全部通过。
- A1 通过不代表论文实现已经被证明；下一步必须完成 1--2 条轨迹过拟合（A2），再做 checkpoint 只读数值审计（A3），最后才运行 1k/10-step/3-seed 小实验（A4）。
- A2 的宿主等待会话 `30473` 已在 CPU A2 通过后停止；不再等待或重复占用 GPU。原计划的 GPU A2 产物未生成，CPU canonical 结果以 `results/r7_alignment_a2_boxes_seed0_cpu_200.pt` 为准。

### A3 只读审计预分析（尚未作为最终 A3 验收）

- 现有数值诊断显示最终候选对 target 的几何 coverage 约 `69.66%`；无 target guard 的 alpha=0.5 推理在早期 proxy 上曾降至约 `6.53%`，说明 pruning 阈值会直接造成候选支持损失。
- `validatedcfg50` proxy 的 final logits 均值约 `-0.098`、标准差约 `0.081`，occupancy head 梯度约 `0.075`，而 decoder likelihood heads 梯度约 `2.115`；这支持“逐层 likelihood 信号与最终 occupancy 分数不平衡”的诊断假设，但不能代替 canonical checkpoint 审计。
- A 的三 seed 完整 validation 仍为 F1 `0.5103/0.0004/0.1019`，均值 `0.2042`、样本标准差 `0.2699`；且三次均使用外部 offset/关闭内部 pruning，因此只作为失败稳定性的背景证据。
- 已对现有 `results/r7_scale1_bias50_seed0.pt` 做 CPU 只读诊断，输出为 `data/r7_alignment_a3_seed0_cpu_alpha01.json`：alpha=0.1 时最终候选 coverage `0.696639`、occupancy 正率约 `0.02675`，三类梯度均非零；同一 checkpoint 直接用 alpha=0.5 会触发 `EmptyPruningError`（所有候选均被剪掉）。因此 A3 暂记为“发现关键阈值/空候选问题”，尚未通过 canonical alpha=0.5 验收。
- alpha=0.5 的异常已单独固化为 `data/r7_alignment_a3_seed0_pruning_error.json`，明确记录为中间 decoder scale 的空候选异常，避免把异常吞掉或误写成模型指标。
- 已对 `r7_scale1_bias50_seed1.pt` 与 `r7_scale1_bias50_seed2.pt` 完成同口径 CPU 只读诊断：输出分别为 `data/r7_alignment_a3_seed1_cpu_alpha01.json`、`data/r7_alignment_a3_seed2_cpu_alpha01.json`。三 seed 在 alpha=0.1 诊断下的 final logits 均值/标准差、0.5 概率正率、target coverage 为：seed0 `-0.07276/0.02476/0.02675/0.69664`，seed1 `-0.30892/0.01025/0/0.69664`，seed2 `-0.02665/0.05330/0.44429/0.69664`。这与 validation 中 seed1 近零 recall、seed2 高方差一致，A3 已有三 seed 背景证据，但 canonical alpha=0.5 的可运行性仍待 A2 后重新验证。

### A2 论文式单轨迹过拟合结果

- 首次 canonical A2 在 alpha=0.5 feedback 下因初始空候选中止；已在 `run_r7_paper_train.py` 中将空 detached prediction 明确处理为空 history，不注入 ground truth，并通过相关训练测试。
- 修正后使用 CPU 完成一条 `isaac_anymal_boxes_s1200_e0`（10 帧）200-step 训练：loss `2.3633 -> 1.0258`（下降约 `56.6%`）。checkpoint 为 `results/r7_alignment_a2_boxes_seed0_cpu_200.pt`，训练日志为 `data/r7_alignment_a2_boxes_seed0_cpu_200_train.log`。
- 同一训练轨迹以内部 alpha=0.5、feedback alpha=0.5 评估：model F1 `0.6470`、height MAE `0.0078`；baseline F1 `0.3851`、MAE `0.0346`。评估文件为 `data/r7_alignment_a2_boxes_seed0_cpu_200_eval.json`。A2 通过，但仅证明单轨迹可学习，不能外推 validation 或论文级泛化。

### A4 启动状态

- 已创建并启动串行宿主会话 `24886`（`run_r7_alignment_a4_when_gpu_free.sh`）：等待 G1 GPU 进程结束后，依次运行 seed0/1/2 的 1k（每类最多 200 帧）10-step canonical 训练；每个 seed 独立保存 checkpoint，并在冻结 validation 上每类 5 条轨迹、alpha=0.5/internal pruning 开启的条件下做快速 gate。
- A4 不使用 external logit offset；任一 seed 出现近零 recall、空候选或 CUDA OOM 将停止后续 seed，并保留日志作为对齐验收证据。
- `r7_autoregressive_rollout.train_detached_rollout` 也已加入空 detached feedback 处理，与主训练器保持一致；相关 12 项回归测试通过。
- 因 G1 长时间占用 GPU，原 A4 GPU 等待会话已停止，改为 CPU 串行执行同一 canonical 1k/10-step/3-seed 小实验。seed0 当前运行会话由 `run_r7_paper_train.py` 产生 `data/r7_alignment_a4_1k_seed0_cpu_train.log`，目标 checkpoint `results/r7_alignment_a4_1k_seed0_cpu.pt`；截至记录已完成 step 0--2，单步约 85 秒，loss `2.3820 -> 2.3699`，尚未完成 seed0。
- 上述沙盒 CPU 会话在 step 2 后被会话回收且无 checkpoint；随后已用宿主持久会话 `77300` 重新启动同一 seed0 配置，日志和 checkpoint 路径不变。截至最新记录完成 step 0--1，耗时 `109.6/112.9s`，loss `2.3820 -> 2.3759`，训练仍在运行。
- 宿主持久 seed0 A4 已完成 10 steps：loss `2.3820 -> 2.3281`（约下降 `2.26%`），checkpoint `results/r7_alignment_a4_1k_seed0_cpu.pt`，训练日志 `data/r7_alignment_a4_1k_seed0_cpu_train.log`。每类 5 条、共 25 条冻结 validation 的 raw alpha=0.5 评估全部 model F1=`0`、MAE=`inf`，结果 `data/r7_alignment_a4_1k_seed0_cpu_eval25.json`；seed0 未通过 A4 稳定性 gate。
- 即使 seed0 gate 失败，仍继续运行 seed1/seed2 以完成预注册的三 seed 证据；不据单个失败 seed 提前宣称结论。
- seed0 训练日志已写 `[DONE]` 但 Python 进程未自行退出，已用精确 PID 结束该残留进程；checkpoint 与日志均保留。seed1 CPU 训练已启动（PID `223105`），当前处于预处理/首步阶段。
- 最新 seed1 进度：已完成 step 0（loss `2.5137`，`99.5s`）和 step 1（loss `2.5141`，`94.0s`），训练仍在运行；seed1 checkpoint 尚未生成。
- 上述 PTY seed1 进程在约 5 分钟后被会话回收，未生成 checkpoint；已改用脱离终端的宿主后台进程重新运行，当前 PID `244069`，日志沿用 `data/r7_alignment_a4_1k_seed1_cpu_train.log`，截至最新记录已写入 step 2（loss `2.5061`，`94.4s`），进程仍存活。
- 后续核查发现 PID `244069` 虽保持 `Rsl`、CPU 约 600%，但启动后约 6.5 小时仅推进到 step 4，未生成 checkpoint；判断为未启用 MinkowskiEngine cache 清理导致的严重退化。已向该精确 PID 发送 `SIGTERM`，未触碰 GPU 上的 Unitree 任务。
- seed1 已用脱离终端的宿主后台进程恢复，PID `276389`，设置 `OMP_NUM_THREADS=12`、`MKL_NUM_THREADS=12`、`--clear-me-cache`、`--save-every 1`，日志改写入 `data/r7_alignment_a4_1k_seed1_cpu_train_retry.log`，checkpoint 仍为 `results/r7_alignment_a4_1k_seed1_cpu.pt`。重启后进程处于 `Rsl`，内存约 `1.2 GB`，正在进行首步；待完成后再执行快速 validation。
- 重启后的 seed1 已完成 step 0--3，耗时 `97.4/110.6/108.8/105.3s`，loss `2.5137 -> 2.4699`，每步 checkpoint 均已落盘；PID `276389` 仍在运行，未启动 seed2。
- seed1 重启运行已完成 10 steps：最终 loss `2.5137 -> 2.4562`，日志 `data/r7_alignment_a4_1k_seed1_cpu_train_retry.log`，checkpoint `results/r7_alignment_a4_1k_seed1_cpu.pt`；随后在 25 条冻结 validation 轨迹上完成 raw alpha=0.5 评估，模型 F1 全部为 `0`、MAE 全部为 `inf`，结果 `data/r7_alignment_a4_1k_seed1_cpu_eval25.json`，因此 seed1 未通过 A4 gate。
- 按预注册三 seed 证据计划，已启动 seed2 canonical 1k/10-step CPU 训练，PID `299647`，设置 `OMP_NUM_THREADS=12`、`MKL_NUM_THREADS=12`、`--clear-me-cache`、`--save-every 1`；日志 `data/r7_alignment_a4_1k_seed2_cpu_train.log`，checkpoint `results/r7_alignment_a4_1k_seed2_cpu.pt`。启动后首步尚在计算，未启动更大规模训练。
- seed2 已完成 step 0，耗时 `87.7s`，loss `2.4442`，checkpoint 已落盘；PID `299647` 继续运行后续 9 步。
- seed2 已完成 10 steps：最终 loss `2.4442 -> 2.3937`，日志 `data/r7_alignment_a4_1k_seed2_cpu_train.log`，checkpoint `results/r7_alignment_a4_1k_seed2_cpu.pt`。
- seed2 快速 validation 已完成，结果 `data/r7_alignment_a4_1k_seed2_cpu_eval25.json`；与 seed0/seed1 相同，25 条冻结 validation 轨迹在 raw alpha=0.5/internal pruning 下模型 F1、precision、recall 全为 `0`，height MAE 全为 `inf`。
- A4 三 seed 汇总：seed0/1/2 均完成 canonical 1k/10-step 训练和每类 5 条（共 25 条）validation；三 seed 模型 F1 均值均为 `0.0`，因此 A4 stability gate 失败。该结果证明当前 canonical alpha=0.5 配置在 10-step 小预算下无法形成可评估候选，不能进入 5k/10k/20k 扩展；应回到论文实现/初始化/pruning 耦合的诊断，而不是扩大数据规模。
- A4 汇总文件为 `data/r7_alignment_a4_summary.json`，记录了固定配置、三 seed 的 checkpoint/log/eval 路径及 gate 失败结论。最终审计尝试用当前 conda 环境重跑 A1 pytest，但该环境未安装 `pytest`（`No module named pytest`）；此前已保存的 A1 证据仍为 21 项全部通过，未将依赖缺失误报为测试失败。

## 2026-09-07 canonical 空候选诊断续记

- `run_r7_paper_pruning_probe.py` 已支持加载 checkpoint、报告每层 raw likelihood keep fraction/logit 均值，并修正 IsaacLab scene seed 文件匹配。
- A4 seed0 在单轨迹 probe 中：l3/l2/l1 raw logits 均为负（均值约 `-0.134/-0.093/-0.381`），alpha=.5 raw keep fraction 均为 `0`；相同轨迹的 A2 200-step checkpoint l3/l2/l1 均值约 `16.75/24.98/0.60`，keep fraction 约 `0.50/0.50/0.496`。这证明逐层 likelihood 需要足够 optimizer steps 才形成判别边界。
- 固定 LR `.003 -> .003` 的 1k/10-step 对照仍得到与原 A4 相同的 loss `2.3820 -> 2.3281`，且 alpha=.5 validation 25 条轨迹仍全部 F1=`0`；学习率日程不是根因。证据文件：`results/r7_diag_a4_lr003_seed0_cpu.pt`、`data/r7_diag_a4_lr003_seed0_cpu_eval25.json`。
- 低成本多地形对照（每类 50 帧、250 帧、固定 LR=.003、canonical alpha=.5）30-step 完成后，25 条 validation 得到 F1=`0.5851`、precision=`0.8275`、recall=`0.4587`、height MAE=`0.0292`，超过 baseline F1=`0.3601`；60-step probe 的 l3/l2/l1 keep fraction 为 `0.525/0.506/0.332`，不再系统性空输出。证据：`results/r7_diag_250f_60step_seed0_cpu.pt`、`data/r7_diag_250f_60step_seed0_eval25.json`、`data/r7_diag_250f_60step_seed0_probe.json`。
- 为检验 1k 小预算，已在持久终端会话 `82471` 启动 1k/20-step、seed0、固定 LR=.003、canonical alpha=.5 训练，checkpoint `results/r7_diag_1k_20step_seed0_cpu.pt`，日志 `data/r7_diag_1k_20step_seed0_cpu.log`；截至记录完成 step 3，loss `2.3639`，逐步 checkpoint 已保存，尚未运行 validation。

## 2026-09-07 canonical 空候选诊断

- 对 A4 seed0 checkpoint 进行 alpha 扫描：关闭中间 pruning、固定 `feedback_alpha=0.1` 后，alpha `0.05/0.1/0.2/0.3/0.5` 的结果完全相同（F1 `0.117996`、precision `0.064972`、recall `0.660523`、height MAE `0.769876`）。因此 A4 的 F1=0 不是最终输出阈值单独造成的；关闭中间 pruning 后模型虽有输出，但几何/offset 仍未学好。
- 新增可加载 checkpoint 的 `run_r7_paper_pruning_probe.py`，并修正 IsaacLab 文件 scene seed 匹配；对 A4 seed0 的单轨迹逐层 probe 显示 alpha=.5 下 l3/l2/l1 的 raw likelihood keep fraction 均为 `0.0`，而 target-guard 仅在训练时保留可达 target。这直接解释了 inference 时中间层先清空候选的路径。
- 发现 A4 实际命令未传 `--learning-rate`/`--final-learning-rate`：checkpoint config 显示默认 `0.01 -> 0.0001`，10 步后 LR 约 `6e-5`；此前口头记录的 `.003` 并未用于 A4。已启动单个 1k/10-step seed0 诊断（PID `335830`），显式固定 LR `.003 -> .003`，日志 `data/r7_diag_a4_lr003_seed0_cpu_train.log`，checkpoint `results/r7_diag_a4_lr003_seed0_cpu.pt`；截至记录完成 step 3，loss `2.3820 -> 2.3639`，未启动更大规模训练。

### 2026-09-07 诊断收口：1k/60-step 已摆脱系统性空输出

- 固定学习率 `.003 -> .003` 的 1k/60-step、seed0 canonical 训练已完成（索引 step 0--59；checkpoint `results/r7_diag_1k_60step_seed0_cpu.pt`）。训练 loss 从 `2.3820` 降至 `1.4318`，未发生 CUDA OOM；该运行通过持久会话/逐步 checkpoint 保证了长任务不会因终端回收而丢失。
- 同一 checkpoint 在冻结 validation 的每类 5 条、共 25 条轨迹上使用 alpha=`0.5`、内部 pruning 开启、feedback alpha=`0.5` 完成评估：model F1=`0.456299`、precision=`0.455473`、recall=`0.466053`、height MAE=`0.071194`；baseline F1=`0.360119`、MAE=`0.041278`。25 条轨迹均产生有限预测，没有 A4 中的全零 F1/Inf MAE。评估文件为 `data/r7_diag_1k_60step_seed0_eval25.json`。
- 最终单轨迹逐层 probe `data/r7_diag_1k_60step_seed0_probe.json` 显示 l3/l2/l1 的 raw likelihood keep fraction 约为 `0.525/0.506/0.333`，逐层 surviving coverage 与 unpruned coverage 相同（约 `0.590/0.599/0.611`）。因此 alpha=`0.5` 的 pruning 路径本身可运行；A4 的空候选是短训练未形成 likelihood 判别边界，而非最终阈值单独错误。
- 与 1k/10-step 固定 LR 对照（`data/r7_diag_a4_lr003_seed0_cpu_eval25.json`，仍 F1=`0`）及 1k/20-step（`data/r7_diag_1k_20step_seed0_cpu_eval25.json`，仍 F1=`0`）联合看，学习率日程不是根因，主要因素是 optimizer budget 不足；250 帧/60-step 对照也得到 F1=`0.585127`，支持“需要足够步数才能激活逐层 likelihood/pruning”的结论。
- 诊断目标已达到：A2 单轨迹 200-step 和 1k/60-step 短实验均不再系统性空输出。没有理由为此修改论文的 alpha/pruning 语义或注入 ground truth；当前最小修正是使用足够的训练步数、固定 LR 配置并保留逐步 checkpoint。尚未启动 5k/10k/20k；下一阶段应先做同配置的 1k 多 seed/完整 validation，再依据真实耗时与离散度决定是否恢复大规模曲线。

### 2026-09-08 1k/60-step 多 seed 稳定性续跑

- 继续前先复核了 seed0 checkpoint 的真实配置：84 条约 1k 帧轨迹、60 steps、CPU、固定 LR `.003 -> .003`、内部/feedback alpha=`.5`、四层 likelihood loss、无数据增强、每 step 清理 ME cache。宿主 GPU 在当前 sandbox 中不可见，故未占用或干扰另一 GPU 任务。
- seed1 首次按当前脚本默认 `target_guard=False` 启动时，未进入任何 optimizer step 即在初始 l1 pruning 抛出 `EmptyPruningError`。这暴露出当前默认值与 seed0 所用历史训练器契约不一致：HEAD 原实现训练 forward 始终传递各尺度 target coordinates；guard 仅用于训练候选可达性，不应用于 feedback 或 validation。
- 已保留该失败日志（`data/r7_diag_1k_60step_seed1_cpu_train.log`），并以显式 `--target-guard` 重启同一 seed1；这使配置可审计而不改变 raw alpha=.5 推理评估。重启后 step0 成功完成，loss=`2.5017`（final=`1.8120`，likelihood=`.6897`，约 `65.2s`），每 step checkpoint 为 `results/r7_diag_1k_60step_seed1_cpu.pt`，训练日志为 `data/r7_diag_1k_60step_seed1_cpu_train_guarded.log`。训练仍在进行；未启动 seed2 或 5k。
