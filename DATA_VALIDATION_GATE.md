# 训练规模曲线前的数据最小验证门槛

本文件用于预先注册数据 gate。它的目标是保证“训练规模增加”这一变量可解释；通过 gate 不等同于论文数据已被完全复刻。

## 采集单位和划分

- 训练样本只计入 `captured_frames >= 10` 的 trajectory。
- 每个 terrain 的训练配额：4,000 合格帧，合计至少 20,000 帧。
- 同一 trajectory 的所有帧必须进入同一个 split。
- validation 使用从未用于 train 的独立 seed 区间；在开始训练规模曲线前单独采集，不能从训练 quotas 中随机切帧。
- validation 每类至少 1,000 个合格帧（或与目标训练规模保持固定比例），并单独记录 validation manifest。
- 每类至少 250 个唯一 scene seed，防止 4,000 帧主要来自少数相关轨迹。

## 采集质量 gate

每个 terrain 必须同时满足：

| 项目 | 门槛 | 原因 |
|---|---:|---|
| 合格帧 | >= 4,000 | 类别均衡的最低规模目标 |
| 唯一 scene seed | >= 250 | 限制轨迹/布局相关性 |
| timeout | 0 个被接受 trajectory | 捕获 reset 计数错误的回归 |
| 部分相机帧比例 | <= 5% | 维持四方向观察质量 |
| base-contact 结束率 | <= 25% | 防止只保留偶然短暂存活的运动条件 |
| 随机运动 | 所有接受 trajectory 均有记录 | 保证采样可审计 |
| 运动边际覆盖 | 速度、侧向速度、偏航率、初始 yaw 的五个 bin 各 >= 20 trajectory | 发现失败筛选造成的条件偏差 |

当前 gate 由 collection manifests、`capture_summary` 和人工审计共同检查；
旧的 `audit_r7_collection_distribution.py` 一次性脚本已移除，不再作为主线
运行时依赖。

## 与论文的关系

- 已覆盖五类结构化地形、四方向深度相机、机器人中心 3.2 m 地图和 randomized locomotion；这些是当前可追溯的 paper-aligned 条件。
- 论文明确的数据量锚点是 200k+ observations；20k 是用于验证训练缩放趋势的中间里程碑，不是“达到论文规模”的声明。
- 论文没有完整公开运动 command 分布、pole 尺寸和所有 terrain 生成细节。因此 motion coverage gate 是本复现的可审计工程标准，不应表述为论文原始参数。
- 已知偏差必须保留：stairs rise 当前限于 0.15 m（论文范围到 0.25 m），corridor width 因 8 m tile 限于 3.4 m（论文范围到 6 m），poles 的几何范围为实现假设。

## 训练规模曲线的启动条件

只有在训练数据通过本 gate、独立 validation seed 区间已冻结、并写入训练/验证 manifest 后，才启动 1k、5k、10k、20k 的固定配置规模曲线。每个规模使用相同的模型、优化器、训练预算和至少三个训练随机 seed；报告 validation F1、precision、recall、height/offset MAE 及其均值和离散度。
