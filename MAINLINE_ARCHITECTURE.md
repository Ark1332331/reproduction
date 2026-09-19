# NSR 复现主线架构

本文件描述仓库当前维护的论文复现路径。历史实验代码已经移出源码目录；如果本机
仍有旧实验产生的数值、checkpoint 或日志，它们只作为本地证据保留在 `data/` 和
`results/` 中。

## 整体执行流程

```text
paper_config.py
       │
       ├── IsaacLab 数据采集（isaaclab 环境）
       │     paper_terrains.py                 地形生成
       │     collect_isaaclab_anymal_vectorized.py  仿真采集
       │     collect_r7_vectorized_dataset.py  批量采集
       │     collect_r7_quota.py               按地形配额采集
       │     freeze_r7_split_manifest.py      固定训练/验证划分
       │
       └── 模型训练和评估（nsr-me-cu130-t291 环境）
             r1_data_representation.py         点云体素表示
             r5_sparse_input.py                稀疏张量输入
             r5_sparse_model.py                 稀疏补全网络
             r5_sparse_loss.py                  损失函数
             r7_autoregressive_rollout.py      自回归滚动预测
             r7_data_augmentation.py           训练数据增强
                    │
                    ├── run_r7_paper_train.py  训练入口
                    └── run_r7_paper_eval.py   评估入口
```

## 文件职责

| 模块 | 文件 | 负责内容 |
|---|---|---|
| 参数和契约 | `paper_config.py` | 论文中的空间范围、体素大小、滚动步数、剪枝阈值和优化器参数；同时记录实现假设 |
| 地形生成 | `paper_terrains.py` | stairs、boxes、walls、poles、corridors 五类结构化地形 |
| IsaacLab 采集 | `collect_isaaclab_anymal_vectorized.py`、`collect_isaaclab_anymal_trajectory.py` | 使用四个深度相机采集 ANYmal 数据；vectorized 版本是主路径，trajectory 版本是串行备用路径 |
| 采集调度 | `collect_r7_dataset.py`、`collect_r7_vectorized_dataset.py`、`collect_r7_quota.py` | 分配 seed、重试失败任务、清理超时进程、生成 manifest、按地形配额采集 |
| 数据划分 | `freeze_r7_split_manifest.py` | 固定训练集和验证集的轨迹成员，拒绝 scene seed 重叠 |
| 采集溯源 | `r7_vectorized_capture_contract.py`、`r7_capture_provenance.py` | 固定文件格式、环境信息、代码版本、checkpoint 和采集配置 |
| R1 表示 | `r1_data_representation.py` | 将点云变成机器人中心坐标系下的 3D 体素和质心偏移 |
| 稀疏输入 | `r5_sparse_input.py` | 将 NumPy 体素数据转换为 MinkowskiEngine 稀疏张量 |
| R5 网络 | `r5_sparse_model.py` | 四层 4D 稀疏 U-Net、生成式解码器、候选剪枝和最终 `k=0` 输出 |
| 损失函数 | `r5_sparse_loss.py` | 占用损失、位置偏移损失、多尺度 likelihood 目标和 BCE |
| 数据增强 | `r7_measurement_augmentation.py`、`r7_data_augmentation.py` | 模拟测量误差、遮挡、离群点、位姿噪声和轨迹镜像；目标数据保持干净 |
| R7 方法 | `r7_autoregressive_rollout.py` | 将上一帧预测作为下一帧历史输入，执行 12 步滚动训练和评估 |
| 评估指标 | `r5_sparse_evaluation.py` | Precision、Recall、F1、macro/micro 汇总、高度误差和覆盖率 |
| 运行入口 | `run_r7_paper_train.py`、`run_r7_paper_eval.py` | 主线 checkpoint 训练和验证 |

## 两个必要环境

- `isaaclab`：负责 IsaacLab/Isaac Sim、ANYmal 控制、相机、地形和数据采集。
- `nsr-me-cu130-t291`：负责 PyTorch、MinkowskiEngine、模型训练、评估和测试。

两个环境不能简单合并，因为 IsaacLab 和 MinkowskiEngine 在当前机器上的 Python、
CUDA 和依赖版本不同。

## 不属于主线的内容

以下内容已经从源码目录删除：

- R2/R3/R3b toy 流程；
- R6 控制器、height scan 和 policy play 原型；
- urban-depth 实验；
- 一次性剪枝、特征、目标、seed、时间和过拟合诊断脚本；
- 旧版非 checkpoint 训练包装器；
- GPU 等待和旧版 scale 启动脚本。

这些实验产生的证据不会被重新解释为论文结果。
