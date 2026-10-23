# A2：卷积网络

本目录使用 CIFAR-10，完成图像分类作业与 3 项附加任务。

## 成果入口

- `REPORT.md`：实验报告与结果分析。
- `REPORT.pdf`：按参考报告排版的 PDF 报告。
- `main.ipynb`：数据检查、结果查看与可视化入口。
- `configs/experiment.json`：正式实验配置。
- `src/cifar10_lab/`：数据、模型、训练和报告代码。
- `results/`：交叉验证、逐轮指标、最佳权重和测试预测。
- `figures/`：卷积核、特征图、训练曲线和性能图。
- `requirements.lock`：本次运行的依赖版本。
- `tests/`：模型与评估协议检查。

## 环境要求

- Python 3.11。
- 中央处理器（CPU）或图形处理器（GPU）。
- GPU 后端支持 Apple Metal 和 NVIDIA CUDA。
- 首次下载需要网络连接。
- 本次环境与实际设备记录在 `results/manifest.json`。

以下命令从本目录运行。

1. 创建虚拟环境。

   ```bash
   uv venv .venv --python 3.11
   ```

2. 安装固定版本的依赖。

   ```bash
   uv pip install --python .venv/bin/python -r requirements.lock
   ```

3. 安装本项目。

   ```bash
   uv pip install --python .venv/bin/python -e . --no-deps
   ```

4. 运行协议检查。

   ```bash
   .venv/bin/python -m pytest -q
   ```

5. 运行完整实验。

   ```bash
   .venv/bin/python -u -m cifar10_lab.experiment --resume
   ```

6. 生成报告与图像。

   ```bash
   .venv/bin/python -m cifar10_lab.report
   ```

7. 核验成果之间的一致性。

   ```bash
   .venv/bin/python -m cifar10_lab.verify
   ```

打开 `main.ipynb` 后，选择 `.venv/bin/python` 解释器。
Notebook 读取已有结果，不会自动启动训练。

程序按固定路径保存成果。
`--resume` 跳过协议相同且已完成的训练。
程序从第 1 轮重跑尚未完成的单个训练。

如果配置发生变化，程序拒绝复用已有结果。
不带 `--resume` 时，程序重新训练并覆盖正式结果。
修改配置后，完整重跑实验并生成报告。

## 分阶段运行

交叉验证使用训练子集。
正式训练使用固定训练集和验证集。
最终评估使用提前保存的模型清单。

```bash
.venv/bin/python -m cifar10_lab.experiment --stage cv --resume
.venv/bin/python -m cifar10_lab.experiment --stage train --resume
.venv/bin/python -m cifar10_lab.experiment --stage evaluate --resume
.venv/bin/python -m cifar10_lab.report
```

仅重建报告时，程序读取已有结果和最佳权重。
程序不会启动训练。

## PDF 导出

导出环境需要 Pandoc、XeLaTeX 和本项目的 Python 依赖。
默认字体为 Times New Roman、Songti SC 和 Menlo。
PDF 使用 A4 页面、22 mm 页边距和 1.15 倍行距。
训练曲线按列分为 3 组，每组单独占 1 页。

1. 生成临时 PDF。

   ```bash
   bash scripts/export_pdf.sh
   ```

2. 打开 `tmp/pdfs/REPORT.pdf`，检查图表和分页。

3. 检查通过后，替换正式 PDF。

   ```bash
   mv tmp/pdfs/REPORT.pdf REPORT.pdf
   ```

导出脚本读取现有报告和训练记录。
导出脚本不启动训练。
字体覆盖变量见 `scripts/export_pdf.sh`。

## 数据与选择规则

数据程序下载 fast.ai 的 CIFAR-10 图像归档。
程序核对归档 SHA-256、图像尺寸和每类数量。
归档来源保存在 `results/data_provenance.json`。

程序按类别分层划分 40,000 张训练图像和 10,000 张验证图像。
交叉验证从训练子集抽取 9,000 张图像。
程序对 15 组配置执行 3 折交叉验证。
每折训练 3 轮。

程序只按交叉验证平均准确率选择配置。
正式训练固定为 20 轮。
程序只按验证准确率保存最佳权重。
程序在全部训练完成后计算测试指标。

集成模型对 3 组类别概率取等权平均。
集成成员为 `selected`、`deep`、`residual_cutout`。
程序在测试评估前固定集成成员。

## 任务覆盖

| 要求 | 实现 | 结果位置 |
| --- | --- | --- |
| 卷积神经网络（CNN）分类 | 3 块浅层网络与交叉熵 | `results/runs/selected/` |
| Dropout | 丢弃概率 0、0.2、0.5 | `results/cross_validation.json` |
| 归一化 | 无归一化、BatchNorm、InstanceNorm、LayerNorm、GroupNorm | `results/cross_validation.json` |
| 超参数交叉验证 | 训练子集内的分层 3 折验证 | `results/selection.json` |
| 性能改进 | 深层网络、残差、Cutout、模型集成 | `results/model_comparison.csv` |
| 卷积核分析 | 3×3、5×5、7×7 的权重与特征图 | `figures/` |
| 缺陷与验证 | 泛化差距、误分类、遮挡测试 | `REPORT.md` |

`layer` 使用真正的 `nn.LayerNorm((C,H,W))`。
验证预处理没有随机增强。
所有模型均从随机权重开始训练。

## 逐项验收

基础任务和 3 项附加题均已完成。
验收同时核对实现、训练记录、指标和报告分析。
核验结果保存在 `results/verification.json` 的 `requirement_coverage` 字段。

| 题目要求 | 状态 | 已完成证据 |
| --- | --- | --- |
| CIFAR-10 图像分类 | 已完成 | 卷积神经网络、训练过程、最佳权重、10,000 张测试图像的预测 |
| Dropout 与多种归一化 | 已完成 | 3 个丢弃概率、4 种归一化及无归一化对照、正式消融实验 |
| 交叉验证选择超参数 | 已完成 | 15 组配置、45 次分折训练、选参记录，测试集未参与选参 |
| 附加题：提升性能 | 已完成 | 深层网络、残差连接、Cutout、模型集成及性能对比 |
| 附加题：卷积核与可视化 | 已完成 | 3×3、5×5、7×7 的性能、参数量、权重、特征图与输入梯度分析 |
| 附加题：模型缺陷与解决方案 | 已完成 | 泛化差距、类别混淆、遮挡敏感性、解决方案和 Cutout 实验验证 |

45 次分折训练均完成 3 轮。
9 组正式训练均完成 20 轮。
17 项测试和 Notebook 的 9 个代码单元已通过。

集成模型的干净测试准确率为 88.08%。
Cutout 将残差模型的中心遮挡准确率从 76.81% 提升到 83.19%。
基础任务和附加题的完成不要求每种方法都提高准确率。
报告同时记录正向结果、负向结果和结论限制。

## 文件约定

数据缓存位于 `data/`，虚拟环境位于 `.venv/`。
临时脚本、日志和预览位于 `tmp/`。
本目录的 `.gitignore` 排除这 3 类文件。

每种成果格式使用固定路径。
报告生成器先验证临时图像，再替换正式图像。
程序只保留每次训练的最佳权重。
程序同时保存每轮训练记录。

## 结论边界

短轮次交叉验证只支持本次预算内的参数选择。
正式实验只使用 1 个训练种子。
配对重采样区间只描述测试样本抽样变化。
卷积核对比没有匹配参数量。
遮挡测试只覆盖固定中心区域。

完整结果和解释见 `REPORT.md`。
