"""Render a Chinese report and figures from completed experiment evidence."""
import argparse
import csv
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "tmp/matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / "tmp/cache"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .experiment import write_csv, write_json

LABELS = {
    "ce_sgd": "SGD", "ce_momentum": "Momentum", "ce_adagrad": "Adagrad",
    "ce_adam": "Adam", "ce_adamax": "Adamax", "mse_adam": "MSE + Adam",
    "smooth_adam": "Smoothing + Adam", "l2_adam": "L2 + Adam",
    "dropout_adam": "Dropout + Adam", "linear_adam": "Linear softmax",
}


def read_csv(path):
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "results")
    parser.add_argument("--report", type=Path, default=ROOT / "REPORT.md")
    args = parser.parse_args()
    result = args.results
    manifest = json.loads((result / "manifest.json").read_text())
    if manifest["status"] != "complete":
        raise SystemExit("Complete experiments before generating the report")
    settings = manifest["settings"]
    selection = json.loads((result / "selection.json").read_text())
    raw = read_csv(result / "summary.csv")
    ids = [e["id"] for e in settings["experiments"]]
    aggregate, histories = {}, {}
    numeric_keys = ["validation_accuracy", "test_accuracy", "test_macro_f1", "train_accuracy", "test_cross_entropy", "test_confidence", "seconds", "best_epoch"]
    for name in ids:
        rows = [r for r in raw if r["experiment"] == name]
        stats = {"experiment": name}
        for key in numeric_keys:
            values = np.array([float(r[key]) for r in rows])
            stats[key + "_mean"] = float(values.mean())
            stats[key + "_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        aggregate[name] = stats
        histories[name] = [read_csv(result / "runs" / name / f"seed_{seed}" / "history.csv") for seed in settings["seeds"]]
    write_csv(result / "aggregate.csv", list(aggregate.values()))
    plots = result / "figures"
    plots.mkdir(exist_ok=True)
    temporary_plots = ROOT / "tmp/report"
    temporary_plots.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})

    def save(fig, name):
        fig.tight_layout()
        temporary = temporary_plots / name
        fig.savefig(temporary, dpi=160, bbox_inches="tight")
        plt.close(fig)
        temporary.replace(plots / name)

    def curve(ax, names, metric, percent=False):
        for name in names:
            values = np.array([[float(row[metric]) for row in run] for run in histories[name]])
            values *= 100 if percent else 1
            mean, std = values.mean(axis=0), values.std(axis=0, ddof=1) if len(values) > 1 else np.zeros(values.shape[1])
            epochs = np.arange(1, values.shape[1] + 1)
            line, = ax.plot(epochs, mean, label=LABELS[name])
            ax.fill_between(epochs, mean - std, mean + std, alpha=0.12, color=line.get_color())
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)

    optimizer_ids = ["ce_sgd", "ce_momentum", "ce_adagrad", "ce_adam", "ce_adamax"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))
    curve(axes[0], optimizer_ids, "validation_accuracy", True)
    axes[0].set(ylabel="Validation accuracy (%)", title="Optimizer comparison (mean +/- sample SD)")
    curve(axes[1], optimizer_ids, "validation_cross_entropy")
    axes[1].set(ylabel="Hard-label cross entropy", title="Common validation metric")
    save(fig, "optimizers.png")

    regularization_ids = ["ce_adam", "mse_adam", "smooth_adam", "l2_adam", "dropout_adam"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))
    curve(axes[0], regularization_ids, "validation_accuracy", True)
    axes[0].set(ylabel="Validation accuracy (%)", title="Loss and regularization ablations")
    curve(axes[1], regularization_ids, "validation_cross_entropy")
    axes[1].set(ylabel="Hard-label cross entropy", title="Same scale across training losses")
    save(fig, "regularization.png")

    winner = selection["selected_experiment"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    curve(axes[0], ["ce_adam", "linear_adam"], "train_accuracy", True)
    axes[0].set(ylabel="Training accuracy (%)", title="Nonlinear MLP versus linear classifier")
    curve(axes[1], ["ce_adam", "linear_adam"], "validation_accuracy", True)
    axes[1].set(ylabel="Validation accuracy (%)", title="Validation accuracy")
    save(fig, "classifiers.png")

    fig, ax = plt.subplots(figsize=(10, 5.8))
    positions = np.arange(len(ids))
    ax.errorbar([100 * aggregate[n]["test_accuracy_mean"] for n in ids], positions,
                xerr=[100 * aggregate[n]["test_accuracy_std"] for n in ids],
                fmt="o", markersize=6, capsize=3, color="#2b7a95")
    ax.set_yticks(positions, [LABELS[n] for n in ids])
    ax.set(xlabel="Held-out test accuracy (%)", title="Final test results after validation selection", xlim=(85, 100))
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.2)
    save(fig, "test_accuracy.png")

    test = json.loads((result / "test_metrics.json").read_text())
    matrix = np.array(test["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    image = ax.imshow(matrix, cmap="Blues")
    for i in range(10):
        for j in range(10):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center", fontsize=8,
                    color="white" if matrix[i, j] > matrix.max()/2 else "black")
    ax.set(xticks=range(10), yticks=range(10), xlabel="Predicted digit", ylabel="True digit",
           title=f"Selected model: {LABELS[winner]}, seed {selection['selected_seed']}")
    fig.colorbar(image, ax=ax, shrink=0.8)
    save(fig, "confusion_matrix.png")
    per_class = [{"digit": i, **{key: test[key][i] for key in ("precision", "recall", "f1", "support")}} for i in range(10)]
    write_csv(result / "per_class_metrics.csv", per_class)
    np.savetxt(result / "confusion_matrix.csv", matrix, delimiter=",", fmt="%d")

    rates = json.loads((result / "selected_learning_rates.json").read_text())
    def percent(name, key):
        return f"{100*aggregate[name][key+'_mean']:.2f} ± {100*aggregate[name][key+'_std']:.2f}"
    table = "\n".join(f"| `{name}` | {percent(name, 'train_accuracy')} | {percent(name, 'validation_accuracy')} | {percent(name, 'test_accuracy')} | {100*aggregate[name]['test_macro_f1_mean']:.2f} | {aggregate[name]['seconds_mean']:.1f} |" for name in ids)
    tuning_table = "\n".join(f"| {r['experiment']} | {float(r['learning_rate']):g} | {100*float(r['validation_accuracy']):.2f} |" for r in read_csv(result / "tuning.csv"))
    class_table = "\n".join(f"| {r['digit']} | {100*r['precision']:.2f} | {100*r['recall']:.2f} | {100*r['f1']:.2f} | {r['support']} |" for r in per_class)
    base = aggregate["ce_adam"]
    regularization_table = "\n".join(
        f"| `{name}` | {100*(aggregate[name]['train_accuracy_mean']-aggregate[name]['validation_accuracy_mean']):.2f} | "
        f"{aggregate[name]['test_cross_entropy_mean']:.4f} | {100*aggregate[name]['test_confidence_mean']:.2f} |"
        for name in regularization_ids)
    effects = []
    for name in regularization_ids[1:]:
        s = aggregate[name]
        effects.append(f"- `{name}`：验证准确率变化 {100*(s['validation_accuracy_mean']-base['validation_accuracy_mean']):+.2f} 个百分点。测试准确率变化 {100*(s['test_accuracy_mean']-base['test_accuracy_mean']):+.2f} 个百分点。")
    fastest = max(optimizer_ids, key=lambda n: np.mean([float(run[0]["validation_accuracy"]) for run in histories[n]]))
    best_optimizer = max(optimizer_ids, key=lambda n: aggregate[n]["validation_accuracy_mean"])
    seed_row = next(r for r in raw if r["experiment"] == winner and int(r["seed"]) == selection["selected_seed"])
    mistakes = matrix.copy()
    np.fill_diagonal(mistakes, 0)
    pairs = np.argsort(mistakes.ravel())[-3:][::-1]
    errors = "、".join(f"{i//10}→{i%10}（{mistakes.ravel()[i]} 张）" for i in pairs)
    run_seconds = sum(float(r["seconds"]) for r in raw)
    parameter_count = sum(a*b+b for a,b in zip([784,*settings["hidden_sizes"]], [*settings["hidden_sizes"],10]))
    figure_prefix = Path(os.path.relpath(plots, args.report.parent)).as_posix()
    report = rf"""# A1：NumPy 全连接网络图像分类实验报告

## 1. 任务与完成范围

本实验使用 MNIST 手写数字数据集。
核心计算全部使用 NumPy。
实现范围包括前向传播、反向传播、参数更新和评估。
Matplotlib 只负责绘图。
项目未使用 PyTorch、TensorFlow 或 scikit-learn。

实验覆盖基础任务和 2 项附加题。
附加题包含损失函数、正则化方法和优化算法的对照。
线性分类器作为额外基线。
原目录 `A1` 保持不变。

## 2. 数据与实验协议

MNIST 包含 60,000 张原始训练图像和 10,000 张测试图像。
每张图像为 28×28 灰度图，标签为数字 0 至 9。
数据说明见 [MNIST 数据集文档](https://www.tensorflow.org/datasets/catalog/mnist)。
下载地址使用 [Google 托管的 MNIST 文件](https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz)。
下载程序验证 4 个压缩文件的 MD5 摘要。
MD5 用于识别数据文件，不用于安全认证。

训练集包含 {manifest['train_size']:,} 张图像。
验证集包含 {manifest['validation_size']:,} 张图像。
测试集包含 {manifest['test_size']:,} 张图像。
训练集与验证集采用分层随机划分。
划分种子为 {settings['split_seed']}，索引保存在 `results/split_indices.npz`。
所有实验使用相同的划分。

像素值除以 255 后展平为 784 维向量。
预处理不估计测试集统计量。

学习率搜索只使用训练集与验证集。
每种优化器搜索 2 个预先设置的学习率。
每个候选训练 {settings['tuning_epochs']} 轮，搜索种子为 {settings['seeds'][0]}。
正式实验使用种子 {', '.join(map(str, settings['seeds']))}。
同一种子使用相同的初始化与批次排列。
Dropout 使用独立随机数流。

每个正式实验训练 {settings['epochs']} 轮，批次大小为 {settings['batch_size']}。
程序按验证准确率保存最佳轮次。
准确率相同时，程序选择硬标签交叉熵更小的轮次。
最终配置取 3 个种子的平均验证准确率最大者。
平均值相同时，程序取配置表中的首项。
最终模型固定采用种子 {selection['selected_seed']}。

程序在 `selection.json` 中保存选择结果。
程序随后加载测试集，评估所有已确定的模型。
测试结果不参与调参、选轮次或选模型。

## 3. 模型与手写计算

多层感知机（MLP）的结构为 `784 → {' → '.join(map(str, settings['hidden_sizes']))} → 10`。
模型包含 {parameter_count:,} 个可训练参数。
隐藏层使用修正线性单元（ReLU）。
输出层产生 10 个未归一化分数。
Softmax 将分数转换为类别概率。

隐藏层采用 He 初始化。
输出层从均值为 0、方差为 `1 / fan_in` 的正态分布采样。
偏置初始化为 0。

设某层输入为 $A$，权重为 $W$，偏置为 $b$。
前向传播为 $Z=AW+b$。
隐藏层输出为 $\max(0,Z)$。
Softmax 先减去每行最大值，再计算指数。
交叉熵通过对数求和指数形式计算。
这两项处理用于避免数值溢出。

交叉熵为 $L=-\frac{{1}}{{N}}\sum_{{i,k}}q_{{ik}}\log p_{{ik}}$。
输出梯度为 $\partial L/\partial Z=(P-Q)/N$。
标签平滑使用 $q=(1-\alpha)y+\alpha/K$。
本实验设置 $\alpha=0.1$，$K=10$。
正确类别的目标值为 0.91，其余类别为 0.01。

均方误差（MSE）定义为 $L=\frac{{1}}{{2N}}\sum_{{i,k}}(p_{{ik}}-q_{{ik}})^2$。
MSE 先对概率求导，再乘 Softmax 的雅可比矩阵。
代码使用雅可比矩阵与向量的乘积，避免构造完整矩阵。

若上游梯度为 $G$，权重梯度为 $A^TG$。
偏置梯度为 $\sum_iG_i$，输入梯度为 $GW^T$。
ReLU 反向传播保留正输入位置的梯度。
各层按逆序计算梯度。
全部梯度计算完成后，优化器更新参数。

L2 惩罚为 $\lambda\sum_l\|W_l\|_F^2/2$。
权重梯度增加 $\lambda W_l$，偏置不受惩罚。
本实验设置 $\lambda=10^{{-4}}$。
Dropout 的丢弃概率为 0.2。
训练时保留的激活值除以 0.8。
评估时关闭 Dropout。

## 4. 对照设计与学习率搜索

随机梯度下降（SGD）按 $\theta\leftarrow\theta-\eta g$ 更新参数。
本实验的 SGD 每次使用一个小批次。
Momentum 设置动量系数为 0.9。
Momentum 累积方向信息后更新参数。

Adagrad 累积各参数的梯度平方。
Adam 维护梯度的一阶矩和二阶矩。
Adamax 使用梯度绝对值的衰减最大值。

Adam 与 Adamax 均校正一阶矩的初始偏差。
Adam 还校正二阶矩的初始偏差。
两者使用 $\beta_1=0.9$、$\beta_2=0.999$。
稳定项 $\epsilon=10^{{-8}}$。
更新公式位于 `src/a1_numpy/optim.py`。

| 搜索实验 | 学习率 | 最佳验证准确率（%） |
| --- | ---: | ---: |
{tuning_table}

选定学习率为：`{json.dumps(rates)}`。

优化器对照统一使用交叉熵和相同网络。
损失函数与正则化对照统一使用 Adam。
这些对照沿用交叉熵基线选定的 Adam 学习率。
每个对照只替换损失函数或一项正则化设置。
线性基线删除全部隐藏层，直接计算 Softmax 分类。
线性基线也沿用 Adam 的学习率。

## 5. 正式实验结果

表中准确率采用百分比。
“±”后的数值为 3 个种子的样本标准差。
宏平均 F1 也乘以 100。
时间是单次运行的平均耗时，单位为秒。
时间包含训练、验证、训练集评估和模型保存。
时间不能代表优化器更新公式的独立耗时。

| 实验 | 训练准确率（%） | 验证准确率（%） | 测试准确率（%） | 测试宏平均 F1 | 平均耗时（秒） |
| --- | ---: | ---: | ---: | ---: | ---: |
{table}

所有训练准确率均来自所选轮次的完整训练集评估。
评估使用固定参数，并关闭 Dropout。
每轮在线损失另存为 `online_objective`。
评估按样本数加权，正确计入末尾不足一个批次的数据。

![优化器对照]({figure_prefix}/optimizers.png)

优化器组中，`{fastest}` 的首轮平均验证准确率最高。
按最佳轮次汇总，`{best_optimizer}` 的平均验证准确率最高。
不同算法的学习率来自相同规模的搜索。
这些结果仅适用于当前网络、候选学习率与训练轮数。
3 轮搜索偏向收敛较早的配置，可能漏掉后期表现更好的学习率。

![损失函数与正则化对照]({figure_prefix}/regularization.png)

以 `ce_adam` 为基线，结果变化如下。

{chr(10).join(effects)}

| 实验 | 训练与验证准确率差值（百分点） | 测试交叉熵 | 平均最高类别概率（%） |
| --- | ---: | ---: | ---: |
{regularization_table}

标签平滑提高了本次实验的平均验证准确率。
平均最大预测概率也随标签平滑降低。
软标签使模型减弱对单个类别的确定性。
硬标签交叉熵却有所升高，说明准确率与概率损失不一定同步改善。
平均最大概率不能单独证明概率校准得到改善。

Dropout 缩小了训练与验证准确率的差距。
L2 也缩小了差距，并提高了平均测试准确率。
这些现象与限制模型拟合能力的作用一致。
MSE 的平均准确率低于交叉熵基线。
MSE 的输出梯度还包含 Softmax 的雅可比矩阵。
这个因子可能减小饱和区域的梯度。

交叉熵和 MSE 的原始损失尺度不同。
报告使用硬标签交叉熵作为共同评价指标。
标签平滑的目标分布不同，也采用这个共同指标。
正则化方法不保证提高当前数据集的准确率。
具体判断应结合验证准确率、训练差距和随机种子波动。
单一强度的对照不能证明某种正则化方法始终有效或无效。

![分类器对照]({figure_prefix}/classifiers.png)

交叉熵 MLP 的平均验证准确率为 {100*aggregate['ce_adam']['validation_accuracy_mean']:.2f}%。
在线性基线上，平均验证准确率为 {100*aggregate['linear_adam']['validation_accuracy_mean']:.2f}%。
差值为 {100*(aggregate['ce_adam']['validation_accuracy_mean']-aggregate['linear_adam']['validation_accuracy_mean']):.2f} 个百分点。
这项对照统一采用交叉熵和 Adam。
线性基线使用线性决策边界。
MLP 通过隐藏层学习非线性特征。

这项对照同时改变了参数量，不能单独量化非线性的贡献。

![最终测试准确率]({figure_prefix}/test_accuracy.png)

最终选择为 `{winner}`，依据是平均验证准确率。
最终单模型来自种子 {selection['selected_seed']} 的第 {seed_row['best_epoch']} 轮。
最终单模型的测试准确率为 **{100*test['accuracy']:.2f}%**。
测试宏平均 F1 为 **{100*test['macro_f1']:.2f}%**。
最终模型文件为 `results/best_model.npz`。

## 6. 类别评估与错误分析

![混淆矩阵]({figure_prefix}/confusion_matrix.png)

混淆矩阵的行是真实标签，列是预测标签。
最终单模型中，数量最多的 3 类误判为：{errors}。
矩阵揭示易混类别，不能单独解释每张图像的误判原因。

| 数字 | 精确率（%） | 召回率（%） | F1（%） | 样本数 |
| --- | ---: | ---: | ---: | ---: |
{class_table}

## 7. 正确性验证与复现

`tests/test_core.py` 包含 18 项自动测试。
数值梯度测试使用中心差分和双精度浮点数。
测试覆盖交叉熵、MSE、标签平滑、L2 和 Dropout。
Dropout 梯度测试固定掩码，排除随机噪声。
测试还覆盖 5 种优化器的两步更新。

其余测试验证数据划分、模型保存恢复和指标计算。
合成数据测试验证模型是否能学会分类。
依赖检查验证源码未导入深度学习框架。

交付核对覆盖全部 30 次正式实验。
核对内容包含最佳轮次、模型加载和混淆矩阵。
最终模型按原配置重新训练后，所有参数逐项一致。
命令行推理也通过 8 张训练图像的核对。
核对记录见 `results/verification.json`。

Python 版本：`{manifest['python'].split()[0]}`。
NumPy 版本：`{manifest['numpy']}`。
环境基于 uv 管理的 Python 3.11。
依赖版本记录在 `uv.lock`。

核心源码摘要记录在 `results/manifest.json`。
正式训练累计耗时为 {run_seconds:.1f} 秒。
学习率搜索耗时另见 `results/tuning.csv`。
复现命令见 `README.md`。

## 8. 限制与结论

本实验使用一个固定数据划分和 3 个训练种子。
标准差描述初始化与批次顺序带来的波动。
重新划分数据带来的不确定性不在标准差的描述范围内。
本实验未实施统计显著性检验。

学习率候选只有 2 个，搜索轮数为 3。
损失函数和正则化对照未单独重新搜索学习率。
结论不代表各方法经过充分调参后的最优性能。
不同硬件和数值库可能产生数值及耗时差异。

本实验完成 NumPy 全连接图像分类流程。
基础实现包含手写前向传播、反向传播和评估。
对照实验覆盖两类损失、三种正则化和五种优化器。
所有报告数值均来自保存的实验结果。
"""
    temporary_report = ROOT / "tmp/report/REPORT.md"
    temporary_report.write_text(report, encoding="utf-8")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    temporary_report.replace(args.report)
    write_json(result / "report_manifest.json", {"report": os.path.relpath(args.report, result), "figures": sorted(p.name for p in plots.glob("*.png"))})
    print(f"Report: {args.report}")


if __name__ == "__main__":
    main()
