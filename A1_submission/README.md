# A1：NumPy 全连接图像分类

本项目使用 NumPy 实现全连接神经网络图像分类器。
数据集选用 MNIST。
网络计算、反向传播、优化器和指标均由 NumPy 实现。
Matplotlib 仅用于绘图。
梯度计算由手写的求导公式完成。

## 成果入口

- [实验报告](REPORT.md)：方法、实测结果、完整图表和复现说明。
- [PDF 报告](REPORT.pdf)：用于阅读和打印。
- [汇总结果](results/aggregate.csv)：3 个随机种子的均值与标准差。
- [逐次结果](results/summary.csv)：每次实验的最佳轮次和指标。
- [最终模型](results/best_model.npz)：按验证结果选择的模型。
- [实验配置](configs/experiments.json)：网络、种子和学习率候选。

## 运行环境

前置条件：已安装 uv，首次安装依赖需要联网。
项目附带 MNIST 压缩数据，支持本地读取。
本项目使用 uv 管理的 Python 3.11。
项目环境路径为 `.venv`。
依赖固定在 `uv.lock`，训练使用中央处理器（CPU）。

1. 进入项目目录。

   ```bash
   cd A1_submission
   ```

2. 同步依赖。

   ```bash
   uv sync --managed-python --locked
   ```

3. 运行正确性测试。

   ```bash
   uv run --managed-python pytest -q
   ```

4. 运行实验流程。

   ```bash
   OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OMP_NUM_THREADS=1 uv run --managed-python a1-experiment --resume
   ```

`results/` 保存完整实验结果。
如果结果已完成，`--resume` 会直接返回。
如果运行中断，程序复用已完成的单次实验。
中断的单次实验会从初始化开始重跑。
`--resume` 以完整的单次实验为复用单位。

程序通过源码和配置摘要验证结果是否匹配。
已有结果使用 `--resume` 读取，新实验通过 `--output` 指定输出路径。

全量复跑支持将输出放入临时目录。
临时输出用于复现验证。

```bash
OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OMP_NUM_THREADS=1 uv run --managed-python a1-experiment --output tmp/reproduction
```

如果需要更新正式结果，先验证临时输出。
正式结果的固定路径为 `results/`。
正式报告的固定路径为 `REPORT.md`。

## 实验设计

原始训练数据包含 60,000 张图像。
程序分层划分 50,000 张训练图像和 10,000 张验证图像。
官方测试集包含 10,000 张图像。
像素值除以 255，再展平为 784 维向量。
数据说明见 [MNIST 文档](https://www.tensorflow.org/datasets/catalog/mnist)。

多层感知机（MLP）的结构为 `784 → 128 → 64 → 10`。
隐藏层使用修正线性单元（ReLU）。
输出层产生分数，Softmax 将分数转换为概率。

| 对照组 | 配置 |
| --- | --- |
| 优化器 | SGD、Momentum、Adagrad、Adam、Adamax |
| 损失函数 | 交叉熵、概率上的均方误差（MSE） |
| 正则化 | 无正则化、标签平滑、L2、Dropout |
| 分类器 | MLP、无隐藏层的线性 Softmax 分类器 |

随机梯度下降（SGD）在本项目中使用小批次。
Momentum 的动量系数为 0.9。
Adam 和 Adamax 的衰减系数为 0.9、0.999。
标签平滑系数为 0.1，L2 系数为 0.0001。
Dropout 丢弃概率为 0.2。

每种优化器搜索 2 个学习率，每个候选训练 3 轮。
搜索只使用验证集，搜索种子为 42。
正式实验共 10 种配置，每种配置运行 3 个种子。
正式实验每次训练 12 轮，批次大小为 128。
种子为 42、43、44，划分种子为 2026。

程序根据验证准确率保存最佳轮次，并根据平均验证准确率选择最终配置。
最终模型固定使用种子 42。
选择结束后，程序才加载测试集。

标签平滑采用 `q = (1 - alpha) * one_hot + alpha / K`。
MSE 定义为每个样本的平方误差和乘以 0.5。
L2 定义为权重平方和乘以 `lambda / 2`。
L2 正则化作用于权重，偏置梯度来自数据损失。
各类原始训练损失采用各自的定义和尺度。
报告另用硬标签交叉熵作共同指标。

## 文件说明

| 路径 | 用途 |
| --- | --- |
| `src/a1_numpy/data.py` | 下载、摘要验证、读取、分层划分 |
| `src/a1_numpy/network.py` | 前向传播、手写梯度、模型保存恢复 |
| `src/a1_numpy/optim.py` | 5 种优化器 |
| `src/a1_numpy/metrics.py` | 样本加权损失、准确率、混淆矩阵、F1 |
| `src/a1_numpy/experiment.py` | 学习率搜索、训练、验证、最终测试 |
| `src/a1_numpy/predict.py` | 加载最终模型并预测 |
| `tests/test_core.py` | 梯度、优化器、评估和训练测试 |
| `scripts/export_pdf.sh` | 将报告导出为 PDF，另需 Pandoc 和 XeLaTeX |
| `results/manifest.json` | 配置、版本、源码摘要、运行状态 |
| `results/selection.json` | 测试前确定的模型选择结果 |
| `results/verification.json` | 结果核对与确定性复现记录 |
| `results/split_indices.npz` | 训练集与验证集的固定索引 |
| `results/runs/<配置>/seed_<种子>/` | 配置、模型、训练历史、测试指标 |
| `results/tuning/` | 学习率搜索的实验记录 |
| `results/figures/` | 报告中的图表 |
| `data/` | 自动下载的数据，Git 忽略此目录 |
| `tmp/` | 临时输出，Git 忽略此目录 |

不同配置与种子的文件用于保存实验依据。
每次实验保存验证准确率最高的模型。
主报告和最终模型使用固定路径。

`data/` 中的压缩文件和 `results/` 中的实验记录随目录提供。
PDF 报告支持直接阅读和打印。
重新导出 PDF 时，字体设置见 `scripts/export_pdf.sh`。
脚本先生成 `tmp/pdfs/REPORT.pdf`，检查排版后再替换正式 PDF。

## 模型推理

输入文件采用 NumPy 的 `.npy` 格式。
图像形状为 `(N, 28, 28)` 或 `(N, 784)`。
`uint8` 输入取值为 0 至 255。
浮点输入必须已经归一化到 0 至 1。
图像约定为黑色背景、白色笔画。

```bash
uv run --managed-python a1-predict path/to/images.npy
```

程序逐行输出样本索引、预测数字和最大类别概率。
模型推理直接加载保存的模型和本地输入图像。

## 核心验证

测试使用中心差分核对反向传播。
梯度测试覆盖交叉熵、MSE、标签平滑、L2 和 Dropout。
Dropout 的数值梯度测试固定随机掩码。
优化器测试验证 5 种算法的两步更新。
评估测试覆盖末尾短批次和预测计数为零的类别。

测试还验证数据划分互斥、模型恢复和合成数据训练。
完整实验结果另作为真实数据上的运行证据。

## 实验设置

本项目使用一个固定数据划分。
3 个种子用于评估初始化与批次顺序带来的波动。
每种优化器搜索 2 个学习率候选，每个候选训练 3 轮。
每种正则化方法采用一个预设强度。
损失函数与正则化对照沿用 Adam 基线学习率。
实验结果对应配置文件列出的参数设置。
