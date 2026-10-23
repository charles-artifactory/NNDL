# A2：卷积网络

本目录包含提交所需的报告、代码、实验结果和数据。

## 阅读顺序

1. 阅读 `REPORT.pdf`。
2. 查看报告第 1 节的题目与代码对应说明。
3. 打开 `main.ipynb` 查看结果和图表。
4. 阅读 `src/cifar10_lab/` 中的代码。

## 文件说明

| 文件或目录 | 内容 |
| --- | --- |
| `REPORT.pdf` | PDF 实验报告 |
| `REPORT.md` | 报告文字与图表引用 |
| `main.ipynb` | 题目对应说明、结果表和图表 |
| `src/cifar10_lab/` | 数据处理、模型、训练、交叉验证与绘图代码 |
| `configs/experiment.json` | 实验参数 |
| `results/` | 45 次交叉验证、9 组正式训练、模型权重及测试结果 |
| `figures/` | 报告使用的 9 幅图表 |
| `data/` | CIFAR-10 图像归档、数组文件及来源记录 |
| `tests/` | 模型和指标计算检查 |
| `scripts/` | PDF 导出与排版脚本 |
| `requirements.lock` | 实验使用的依赖版本 |
| `SHA256SUMS.txt` | 文件校验值 |

## 安装与运行

使用 Python 3.11 和 uv。安装依赖需要网络连接。
以下命令在本目录执行。

1. 创建 Python 环境。

   ```bash
   uv venv .venv --python 3.11
   ```

2. 安装依赖。

   ```bash
   uv pip install --python .venv/bin/python -r requirements.lock
   ```

3. 安装项目。

   ```bash
   uv pip install --python .venv/bin/python -e . --no-deps
   ```

4. 检查代码。

   ```bash
   .venv/bin/python -m pytest -q
   ```

打开 Notebook 后，选择本目录的 `.venv/bin/python`。
Notebook 读取保存的实验结果。
`clean` 表示完整图像准确率。
`occluded` 表示中心遮挡图像准确率。
中心遮挡使用中灰色覆盖 8×8 像素。

## 重新运行实验

```bash
.venv/bin/python -m cifar10_lab.experiment --resume
```

`--resume` 读取设置相同且训练完成的结果。
删除 `--resume` 后运行会重新训练并覆盖结果。

生成报告和图表使用以下命令。

```bash
mkdir -p tmp
.venv/bin/python -m cifar10_lab.report
```

导出 PDF 需要 Pandoc 和 XeLaTeX。
默认字体为 Times New Roman、Songti SC 和 Menlo。

```bash
bash scripts/export_pdf.sh
```

脚本生成 `tmp/pdfs/REPORT.pdf`。
检查分页和图表后，用该文件替换 `REPORT.pdf`。

## 检查提交文件

```bash
shasum -a 256 -c SHA256SUMS.txt
```

校验值对应本次提交时的文件内容。
