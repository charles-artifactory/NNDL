"""Generate figures and a source-grounded report from the completed experiment."""
import csv
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "tmp" / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / "tmp" / "cache"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from .data import CLASSES, batch_transform
from .experiment import load_model, write_csv, write_json

COLORS = ["#315f8c", "#c06b46", "#4d8c78", "#92669c", "#b18d36"]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titleweight": "bold", "figure.facecolor": "white",
                     "savefig.facecolor": "white"})


def read_json(path):
    return json.loads(path.read_text())


def save(fig, name):
    temporary = ROOT / "tmp" / f"{name}.png"
    fig.savefig(temporary, dpi=170, bbox_inches="tight")
    plt.close(fig)
    with Image.open(temporary) as image:
        image.verify()
    destination = ROOT / "figures" / f"{name}.png"
    destination.parent.mkdir(exist_ok=True)
    temporary.replace(destination)


# 基础题：展示各组超参数的交叉验证准确率。
def plot_cv(cv, config):
    means = np.array([[next(r["mean_accuracy"] for r in cv if r["normalization"] == n and r["dropout"] == d)
                       for d in config["dropout_rates"]] for n in config["normalizations"]])
    fig, ax = plt.subplots(figsize=(7, 4.1), layout="constrained")
    im = ax.imshow(means * 100, cmap="Blues", aspect="auto", vmin=0, vmax=100)
    ax.set(xticks=range(3), xticklabels=config["dropout_rates"], yticks=range(5),
           yticklabels=config["normalizations"], xlabel="Dropout probability",
           title=f"{config['cv_folds']}-fold CV accuracy | {config['cv_epochs']} epochs per fold")
    for i in range(5):
        for j in range(3):
            row = next(r for r in cv if r["normalization"] == config["normalizations"][i]
                       and r["dropout"] == config["dropout_rates"][j])
            ax.text(j, i, f"{means[i,j]*100:.2f}%\nSD {row['std_accuracy']*100:.2f}",
                    ha="center", va="center", color="white" if means[i,j] > .6 else "#162838")
    fig.colorbar(im, ax=ax, label="Mean validation accuracy (%)")
    save(fig, "cross_validation")


# 基础题：展示模型训练过程中的训练和验证准确率。
def plot_curves(summaries):
    fig, axes = plt.subplots(3, 3, figsize=(12, 9), sharex=True, sharey=True, layout="constrained")
    for ax, (name, summary) in zip(axes.flat, summaries.items()):
        with (ROOT / "results" / "runs" / name / "history.csv").open() as f:
            rows = list(csv.DictReader(f))
        epochs = [int(r["epoch"]) for r in rows]
        ax.plot(epochs, [float(r["train_accuracy"]) * 100 for r in rows], color=COLORS[0], label="Train (augmented)")
        ax.plot(epochs, [float(r["validation_accuracy"]) * 100 for r in rows], color=COLORS[1], label="Validation (clean)")
        ax.axvline(summary["best_epoch"], color="#999999", ls=":", lw=1)
        ax.set(title=name, ylim=(0, 100), xlabel="Epoch", ylabel="Accuracy (%)")
        ax.grid(alpha=.15)
    axes.flat[0].legend(loc="lower right", fontsize=8)
    save(fig, "learning_curves")


def plot_performance(results):
    names = list(results)
    fig, ax = plt.subplots(figsize=(10, 6), layout="constrained")
    y = np.arange(len(names))
    for offset, condition, color in [(-.18, "clean", COLORS[0]), (.18, "occluded", COLORS[1])]:
        values = [results[n][condition]["accuracy"] * 100 for n in names]
        ax.barh(y + offset, values, .32, color=color, label=condition)
        for i, value in enumerate(values):
            ax.text(value + .5, i + offset, f"{value:.2f}", va="center", fontsize=8)
    ax.set(yticks=y, yticklabels=names, xlim=(0, 106), xlabel="Test accuracy (%)",
           title="Clean classification and fixed central 8 x 8 occlusion")
    ax.invert_yaxis()
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False)
    save(fig, "accuracy_comparison")


# 附加题 3：展示真实类别与预测类别的对应数量。
def plot_confusion(result):
    cm = np.array(result["clean"]["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(8, 7), layout="constrained")
    im = ax.imshow(cm / cm.sum(1, keepdims=True), vmin=0, vmax=1, cmap="Blues")
    ax.set(xticks=range(10), xticklabels=CLASSES, yticks=range(10), yticklabels=CLASSES,
           xlabel="Predicted class", ylabel="True class", title="Ensemble confusion matrix | row fractions")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    for i in range(10):
        for j in range(10):
            value = cm[i, j] / cm[i].sum()
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if value > .55 else "#162838")
    fig.colorbar(im, ax=ax, shrink=.8)
    save(fig, "confusion_matrix")


# 附加题 2：比较 3 种卷积核的权重、图像特征和输入梯度。
def visual_features(images):
    names = ["selected", "kernel5", "kernel7"]
    models = [load_model(ROOT / "results" / "runs" / n / "best.pt", "cpu") for n in names]
    fig, axes = plt.subplots(3, 8, figsize=(12, 4.8), layout="constrained")
    for row, (name, model) in enumerate(zip(names, models)):
        weight = model.blocks[0].body[0].weight.detach().numpy()
        for j in range(8):
            w = weight[j].transpose(1, 2, 0)
            w = (w - w.min()) / max(float(np.ptp(w)), 1e-8)
            axes[row, j].imshow(w, interpolation="nearest")
            axes[row, j].set(xticks=[], yticks=[])
            if j == 0:
                axes[row, j].set_ylabel(f"{model.spec.kernel_size} x {model.spec.kernel_size}")
    fig.suptitle("First-layer RGB kernels | independent min-max scaling per filter")
    save(fig, "kernel_weights")
    # Fixed examples: first three classes, same input pixels for all models.
    indices = [0, 1000, 2000]
    x = batch_transform(torch.from_numpy(images[indices].copy()).permute(0, 3, 1, 2))
    maps = []
    with torch.no_grad():
        for model in models:
            maps.append(model.features(x))
    representation = {}
    for name, feature in zip(names, maps):
        representation[name] = {}
        for stage, label in [(0, "early"), (2, "late")]:
            means = feature[stage].mean(1).numpy().reshape(3, -1)
            probabilities = means / np.maximum(means.sum(1, keepdims=True), 1e-12)
            entropy = -(probabilities * np.log(np.maximum(probabilities, 1e-12))).sum(1) / np.log(means.shape[1])
            representation[name][label + "_spatial_entropy"] = float(entropy.mean())
    for stage, stage_name in [(0, "early"), (2, "late")]:
        fig, axes = plt.subplots(3, 4, figsize=(10, 7.6), layout="constrained")
        for row, idx in enumerate(indices):
            axes[row, 0].imshow(images[idx])
            axes[row, 0].set_title(f"Input: {CLASSES[row]}")
            axes[row, 0].axis("off")
            for col, (model, feature) in enumerate(zip(models, maps), start=1):
                feature_map = feature[stage][row].mean(0).numpy()
                axes[row, col].imshow(feature_map, cmap="magma", interpolation="nearest")
                axes[row, col].set_title(f"k={model.spec.kernel_size}, {feature_map.shape[0]} x {feature_map.shape[1]}")
                axes[row, col].axis("off")
        fig.suptitle(f"{stage_name.title()} pooled features | channel means, independent color scales")
        save(fig, f"features_{stage_name}")
    # Measured input-gradient sensitivity, not a theoretical receptive-field diagram.
    fig, axes = plt.subplots(1, 4, figsize=(10, 2.8), layout="constrained")
    axes[0].imshow(images[0]); axes[0].set_title("Fixed input"); axes[0].axis("off")
    concentration = {}
    for ax, name, model in zip(axes[1:], names, models):
        image = x[:1].clone().requires_grad_(True)
        logits = model(image)
        predicted = int(logits.argmax(1))
        cls = 0  # Same true-class target (airplane) for all three models.
        gradient = torch.autograd.grad(logits[0, cls], image)[0].abs().sum(1)[0].detach().numpy()
        gradient /= max(float(gradient.sum()), 1e-12)
        mass = np.sort(gradient.ravel())[::-1]
        count = int(np.searchsorted(np.cumsum(mass), .9) + 1)
        concentration[name] = {"target_class": CLASSES[cls], "predicted_class": CLASSES[predicted],
                               "pixels_for_90_percent_gradient": count,
                               "fraction": count / 1024}
        ax.imshow(gradient, cmap="magma"); ax.set_title(f"k={model.spec.kernel_size} | 90%: {count} pixels"); ax.axis("off")
    fig.suptitle("Input sensitivity to airplane logit | one fixed airplane image")
    save(fig, "input_sensitivity")
    write_json(ROOT / "results" / "feature_analysis.json", {"input_sensitivity": concentration,
                                                              "representation": representation})


# 附加题 3：展示每个类别的误分类图像。
def plot_errors(images, predictions):
    labels = predictions["labels"]
    guessed = predictions["clean"].argmax(1)
    fig, axes = plt.subplots(2, 5, figsize=(11, 5), layout="constrained")
    for cls, ax in enumerate(axes.flat):
        candidates = np.flatnonzero((labels == cls) & (guessed != labels))
        if len(candidates):
            idx = candidates[0]
            ax.imshow(images[idx]); ax.set_title(f"{CLASSES[cls]} -> {CLASSES[guessed[idx]]}", fontsize=9)
        ax.axis("off")
    fig.suptitle("Ensemble errors | first misclassified test image per true class")
    save(fig, "misclassifications")


def paired_interval(a, b, labels):
    # Paired bootstrap over fixed test images: no claim about seed variability.
    delta = ((b.argmax(1) == labels).astype(float) - (a.argmax(1) == labels).astype(float))
    rng = np.random.default_rng(2026)
    bootstrap = [float(delta[rng.integers(0, len(delta), len(delta))].mean()) for _ in range(2000)]
    lo, hi = np.quantile(bootstrap, [.025, .975])
    return {"delta_pp": float(delta.mean() * 100), "lower_pp": float(lo * 100),
            "upper_pp": float(hi * 100), "resamples": 2000,
            "scope": "paired test-image sampling; conditional on trained models; not seed uncertainty"}


def generate_report(config, selection, summaries, results, intervals):
    selected = selection["selected"]
    feature_analysis = read_json(ROOT / "results" / "feature_analysis.json")
    lines = ["# A2：卷积网络", "",
             "## 1. 任务与实验范围", "",
             "本实验独立完成图像分类与 3 项附加任务。",
             "",
             "| 任务 | 实验 |", "| --- | --- |",
             "| 卷积神经网络（CNN）分类 | 3 个卷积块、训练曲线、分类指标 |",
             "| Dropout 与归一化 | 15 组参数的分层交叉验证与消融实验 |",
             "| 超参数选择 | 按交叉验证平均准确率选择 |",
             "| 提升性能 | 深层网络、残差连接、Cutout、模型集成 |",
             "| 卷积核与表征 | 3×3、5×5、7×7，卷积核与特征图可视化 |",
             "| 缺陷分析与验证 | 泛化差距、误分类、固定遮挡测试与改进实验 |", "",
             '### 题目要求与代码对应说明\n\n代码文件均位于 `src/cifar10_lab/`。下面按题目顺序说明实验、代码和报告章节。\n\n**基础题：构建图像分类网络。** `models.py` 的 `Classifier` 定义卷积网络，`ConvBlock` 定义卷积块。`experiment.py` 的 `train_run` 训练模型，`evaluate_all` 计算测试准确率。第 3 节介绍结构，第 5 节展示训练曲线和分类结果。\n\n**基础题：应用 Dropout 和多种归一化方法。** `models.py` 的 `Classifier` 在分类层中使用 `nn.Dropout`，`normalization` 创建 4 种归一化层及对照配置。`experiment.py` 的 `final_specs` 设置对照模型。第 4 节比较 15 组配置；第 5.1 节比较 `selected` 与 `dropout_variant`，说明 Dropout 的影响，再比较 `dropout_variant` 与 `no_normalization`，说明归一化的影响。\n\n**基础题：通过交叉验证选择超参数。** `experiment.py` 的 `cross_validate` 将 9000 张训练图像分成 3 折，对每组配置训练 3 次，再按平均验证准确率选择配置。第 4 节给出各组均值、标准差和选中的配置。\n\n**附加题 1：尝试提高分类准确率的方法。** `experiment.py` 的 `final_specs` 设置加深网络、残差连接和 Cutout 实验；`evaluate_all` 将 3 个模型的预测概率取平均，得到集成预测。`data.py` 的 `batch_transform` 实现训练时的随机遮挡。第 5.2 节比较深度、残差连接和集成，第 7.2 节比较 Cutout。\n\n**附加题 2：比较卷积核大小并展示图像特征。** `models.py` 的 `ConvBlock` 根据 `kernel_size` 创建卷积层，`experiment.py` 的 `final_specs` 设置 3×3、5×5、7×7 对照。`report.py` 的 `visual_features` 绘制卷积核权重、特征图和输入梯度图。第 6 节结合这些图，比较准确率、参数量、运行时间和图像特征。\n\n**附加题 3：分析模型缺陷并提出改进方案。** `data.py` 的 `occlude` 遮挡测试图像中心，`experiment.py` 的 `predict` 计算两种图像条件下的预测。`report.py` 的 `plot_confusion` 和 `plot_errors` 展示分类错误。第 7 节分析训练与验证准确率差距、遮挡影响和类别混淆，并用 Cutout 实验验证改善遮挡表现的方法。延后下采样和保留高分辨率分支列为后续方案。\n\n表格中的 `model` 表示模型名称，`clean` 表示完整图像准确率，`occluded` 表示中心遮挡图像准确率。中心遮挡覆盖 8×8 像素，填充值为中灰色。\n', "",
             "## 2. 数据与评估协议", "",
             "CIFAR-10 包含 60,000 张 32×32 彩色图像。数据集有 10 个类别。[1]",
             "官方训练集含 50,000 张图像，测试集含 10,000 张图像。",
             "本次使用 fast.ai 的无损 PNG 归档。[2] 程序验证归档 SHA-256。",
             "样本先按类别排序，再按文件名排序。数据来源记录见 `results/data_provenance.json`。",
             "训练数据按类别分层划分。训练子集有 40,000 张，验证集有 10,000 张。",
             f"划分种子为 {config['seed']}，索引保存在 `results/split.json`。",
             "训练增强包含反射填充后的随机裁剪，以及随机水平翻转。",
             "全部图像按固定公式 `x / 255 × 2 − 1` 缩放。",
             "验证和测试采用固定缩放公式。",
             "每轮训练后，程序计算验证指标。验证准确率决定最佳权重。",
             "程序在全部模型训练结束后计算测试指标。集成成员在测试评估前固定。",
             "交叉验证指标用于选择配置，测试指标用于比较模型性能。", "",
             "## 3. 模型与训练设置", "",
             f"浅层模型通道数为 {config['width']}、{config['width']*2}、{config['width']*4}。",
             "每个卷积块依次包含卷积、归一化、修正线性单元（ReLU）和 2×2 最大池化。",
             "3 次池化后，特征图尺寸为 4×4。分类头包含 128 个隐藏单元和 10 个输出。",
             "下表给出浅层模型的张量尺寸。B 表示批大小。", "",
             "| 层 | 输出尺寸 |", "| --- | --- |",
             "| 输入 | B×3×32×32 |",
             f"| 第 1 个卷积块与池化 | B×{config['width']}×16×16 |",
             f"| 第 2 个卷积块与池化 | B×{config['width']*2}×8×8 |",
             f"| 第 3 个卷积块与池化 | B×{config['width']*4}×4×4 |",
             f"| 展平 | B×{config['width']*4*4*4} |",
             "| 全连接与 ReLU | B×128 |",
             "| Dropout 与输出层 | B×10 |", "",
             "分类头在隐藏激活后应用 Dropout。损失函数为交叉熵。",
             "Dropout 在训练时以概率 p 将隐藏激活置零。保留的激活除以 `1−p`。",
             "推理时关闭 Dropout，并使用全部隐藏单元。随机失活旨在减少隐藏单元间的依赖。[6]",
             "归一化使用公式 `y = γ(x−μ)/sqrt(σ²+ε)+β`。不同方法的统计维度不同。", "",
             "N、C、H、W 分别表示批大小、通道数、高度和宽度。", "",
             "| 方法 | 均值与方差的计算范围 | 推理时的统计量 |", "| --- | --- | --- |",
             "| BatchNorm | 每个通道的批次与空间维度 N、H、W | 训练时累计的运行统计量 |",
             "| InstanceNorm | 每张图像中，每个通道的 H、W | 当前图像统计量 |",
             "| LayerNorm | 每张图像的全部 C、H、W | 当前图像统计量 |",
             "| GroupNorm | 每张图像的组内通道与 H、W | 当前图像统计量 |", "",
             "本实验通过交叉验证和消融实验比较归一化配置的泛化表现。",
             "BatchNorm 在训练时使用批次统计量，并在推理时使用运行统计量。[7]",
             "深层模型将每个块的卷积数从 1 增至 2。残差模型增加 1×1 投影支路。[3]",
             "残差模型采用自定义结构。",
             "LayerNorm 使用 `nn.LayerNorm((C,H,W))`。程序归一化每张图像的全部特征。",
             "InstanceNorm 启用仿射参数。GroupNorm 固定分为 4 组。",
             "不同归一化方法的仿射参数数量不同。归一化层的仿射参数计入模型总参数量。",
             f"优化器为 AdamW，初始学习率为 {config['learning_rate']}。权重衰减为 {config['weight_decay']}。",
             f"正式训练使用 {config['epochs']} 轮，批大小为 {config['batch_size']}。学习率按余弦曲线衰减。",
             "Cutout 在每张训练图像内随机遮挡一个 8×8 方块。[4] 遮挡值为归一化后的 0。",
             f"正式实验的训练种子为 {config['seed']}。", "",
             "## 4. 交叉验证与超参数选择", "",
             f"程序从训练子集分层抽取 {config['cv_samples']} 张图像。交叉验证使用 {config['cv_folds']} 折。",
             f"每折训练 {config['cv_epochs']} 轮。每个组合记录各折的最佳验证准确率。",
             "搜索空间包含 5 种归一化配置和 3 个丢弃概率。丢弃概率为 0、0.2、0.5。",
             "各组合复用相同折划分。每折重新初始化模型和优化器。",
             f"各折训练种子依次为 {'、'.join(str(config['seed']+i) for i in range(config['cv_folds']))}。各组合复用每折种子。",
             "程序按各折最佳准确率的平均值选择参数。同分时按配置顺序选择。",
             f"本次选中 `{selected['normalization']}`，丢弃概率为 {selected['dropout']}。",
             f"选中组合的平均准确率为 {selected['mean_accuracy']*100:.2f}%，折间标准差为 {selected['std_accuracy']*100:.2f} 个百分点。",
             "折间标准差描述各折最佳验证准确率的离散程度。",
             "各折验证结果用于比较参数组合的早期学习表现。", "",
             "![交叉验证](figures/cross_validation.png)", "",
             "| 归一化 | Dropout | 平均准确率 | 折间标准差（百分点） |", "| --- | ---: | ---: | ---: |"]
    cv = read_json(ROOT / "results" / "cross_validation.json")
    for row in sorted(cv, key=lambda r: r["mean_accuracy"], reverse=True):
        lines.append(f"| {row['normalization']} | {row['dropout']} | {row['mean_accuracy']*100:.2f}% | {row['std_accuracy']*100:.2f} |")
    per_norm_best = [max((r for r in cv if r["normalization"] == n), key=lambda r: r["mean_accuracy"])
                     for n in config["normalizations"]]
    if all(r["dropout"] == 0 for r in per_norm_best):
        lines += ["", "本次筛选中，5 种归一化配置均在丢弃概率 0 时取得最高均值。",
                  "交叉验证按平均准确率对配置排序。",
                  "Dropout 的正式消融结果见第 5.1 节。"]
    lines += ["", "各归一化配置使用相同的优化器和初始学习率。",
              ""]
    lines += ["", "## 5. 正式实验结果", "",
              "测试集包含每类 1,000 张图像。宏平均 F1 对每个类别赋予相同权重。",
              "下表汇总验证集所选权重对应的测试结果。", "",
              "| 模型 | 参数量 | 最佳轮次 | 验证准确率 | 测试准确率 | 宏平均 F1 | 遮挡准确率 |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    comparison = []
    for name, metrics in results.items():
        summary = summaries.get(name)
        params = summary["parameters"] if summary else sum(summaries[n]["parameters"] for n in config["ensemble_members"])
        epoch = summary["best_epoch"] if summary else "—"
        validation = f"{summary['best_validation_accuracy']*100:.2f}%" if summary else "—"
        clean, corrupt = metrics["clean"], metrics["occluded"]
        lines.append(f"| {name} | {params:,} | {epoch} | {validation} | {clean['accuracy']*100:.2f}% | {clean['macro_f1']:.4f} | {corrupt['accuracy']*100:.2f}% |")
        comparison.append({"model": name, "parameters": params, "best_epoch": epoch,
                           "test_accuracy": clean["accuracy"], "macro_f1": clean["macro_f1"],
                           "occluded_accuracy": corrupt["accuracy"]})
    write_csv(ROOT / "results" / "model_comparison.csv", comparison)
    lines += ["", "![模型性能](figures/accuracy_comparison.png)", "",
              "![训练曲线](figures/learning_curves.png)", "",
              "训练曲线记录训练增强下的准确率。验证曲线上记录的是干净图像的准确率。",
              "训练阶段启用 Dropout，验证阶段使用全部隐藏单元。泛化差距采用干净训练准确率与验证准确率之差。",
              "下表使用最佳权重，在关闭增强和 Dropout 后重算训练准确率。", "",
              "| 模型 | 干净训练准确率 | 验证准确率 | 差距（百分点） |", "| --- | ---: | ---: | ---: |"]
    for name in ["baseline", "selected", "deep", "residual", "residual_cutout"]:
        s = summaries[name]
        lines.append(f"| {name} | {s['clean_train_accuracy']*100:.2f}% | {s['best_validation_accuracy']*100:.2f}% | {(s['clean_train_accuracy']-s['best_validation_accuracy'])*100:.2f} |")
    lines += ["", "### 5.1 正则化消融", ""]
    def delta(b, a):
        return (results[b]["clean"]["accuracy"] - results[a]["clean"]["accuracy"]) * 100
    positive = "dropout_variant" if selected["dropout"] == 0 else "selected"
    zero = "selected" if selected["dropout"] == 0 else "dropout_variant"
    dropout_explanation = ("本次交叉验证选中丢弃概率 0，正式对照使用丢弃概率 0.2。"
                           if selected["dropout"] == 0 else
                           f"本次交叉验证选中丢弃概率 {selected['dropout']}，正式对照使用丢弃概率 0。")
    lines += [f"选中模型相对基线的测试差值为 {delta('selected','baseline'):+.2f} 个百分点。",
              dropout_explanation,
              f"保留归一化时，启用 Dropout 的测试差值为 {delta(positive,zero):+.2f} 个百分点。",
              f"归一化的测试差值在保留非零 Dropout 时为 {delta(positive,'no_normalization'):+.2f} 个百分点。",
              "本组对照分别考察 Dropout 与归一化的作用。", "",
              "### 5.2 深度、残差与集成", "",
              f"加深网络的测试差值为 {delta('deep','selected'):+.2f} 个百分点。",
              f"加入残差支路后，测试差值为 {delta('residual','deep'):+.2f} 个百分点。",
              "残差支路同时增加投影参数。",
              "集成固定使用 `selected`、`deep` 和 `residual_cutout`。",
              "集成对 3 个模型的类别概率取等权平均。",
              f"集成相对选中浅层模型的测试差值为 {delta('ensemble','selected'):+.2f} 个百分点。",
              "集成同时增加推理计算量和模型存储量。", "",
              "## 6. 卷积核大小与特征表征", "",
              "3 个模型分别使用 3×3、5×5 和 7×7 主干卷积。通道数和训练预算保持相同。",
              "池化后的理论感受野按卷积与池化的递推公式计算。",
              "理论感受野包含填充区域。分类头综合全部 4×4 特征，覆盖整张输入。", "",
              "| 卷积核 | 最后池化单元的理论感受野边长 | 参数量 | 运行耗时（秒） | 测试准确率 |", "| --- | ---: | ---: | ---: | ---: |"]
    for name in ["selected", "kernel5", "kernel7"]:
        s = summaries[name]
        lines.append(f"| {s['protocol']['model']['kernel_size']}×{s['protocol']['model']['kernel_size']} | {s['receptive_field']} 像素 | {s['parameters']:,} | {s['seconds']:.1f} | {results[name]['clean']['accuracy']*100:.2f}% |")
    lines += ["", "![卷积核权重](figures/kernel_weights.png)", "",
              "耗时包含训练和逐轮验证，也计入权重保存与干净训练集评估。",
              "模型在相同设备上顺序运行。",
              "图中展示第一层的前 8 个 RGB 卷积核。每个卷积核独立缩放颜色。",
              "独立缩放用于观察卷积核内部的相对权重与空间结构。", "",
              "![早期特征](figures/features_early.png)", "",
              "![后期特征](figures/features_late.png)", "",
              "各模型使用相同的 3 张固定测试图像。图中显示池化特征的通道平均值。",
              "早期特征尺寸为 16×16，后期特征尺寸为 4×4。",
              "每张热图按自身数值范围缩放，用于展示通道平均激活的空间分布。", "",
              "![输入敏感性](figures/input_sensitivity.png)", "",
              "输入敏感性图使用飞机类别 logit 对输入的绝对梯度。3 个模型使用相同目标类别。",
              "程序在通道维度求和。输入为固定的第 1 张飞机测试图像。",
              "标题给出累计 90% 梯度质量所需的像素数。",
              "梯度描述飞机类别输出对输入像素变化的局部敏感性。",
              "下表汇总空间响应和输入敏感性。空间熵先归一化至 0 至 1（含）。",
              "空间熵使用通道平均激活在空间位置上的概率分布。表中取 3 张图像的均值。",
              "空间熵接近 1 表示响应分布均匀。空间熵降低时，响应集中于部分位置。", "",
              "| 卷积核 | 早期空间熵 | 后期空间熵 | 90% 输入梯度质量的像素数 |",
              "| --- | ---: | ---: | ---: |",
              *[f"| {summaries[n]['protocol']['model']['kernel_size']}×{summaries[n]['protocol']['model']['kernel_size']} | {feature_analysis['representation'][n]['early_spatial_entropy']:.4f} | {feature_analysis['representation'][n]['late_spatial_entropy']:.4f} | {feature_analysis['input_sensitivity'][n]['pixels_for_90_percent_gradient']} |" for n in ["selected", "kernel5", "kernel7"]],
              "", "空间熵衡量通道平均激活在空间位置上的集中程度。不同模型分别学习各自的通道特征。",
              "通道平均将多个通道的响应汇总为一张空间图。独立色标突出每张图内部的响应差异。",
              "热图呈现各模型的空间响应差异。",
              "分类头汇总全部空间位置，形成整张图像的分类预测。",
              "大卷积核在单层中整合更宽的邻域。堆叠小卷积核在相同感受野内增加非线性。",
              "固定深度下，卷积核边长增加时，理论感受野随之扩大。",
              f"5×5 模型相对 3×3 模型的测试差值为 {delta('kernel5','selected'):+.2f} 个百分点。",
              f"相对 3×3 模型，7×7 模型的测试差值为 {delta('kernel7','selected'):+.2f} 个百分点。",
              "卷积核增大时，感受野、参数量和计算量同步增加。",
              f"深层 3×3 与浅层 5×5 的理论感受野边长均为 {summaries['deep']['receptive_field']} 像素。",
              f"深层 3×3 的参数量比浅层 5×5 多 {(summaries['deep']['parameters']/summaries['kernel5']['parameters']-1)*100:.2f}%。",
              f"深层 3×3 相对浅层 5×5 的测试差值为 {delta('deep','kernel5'):+.2f} 个百分点。",
              "这组对照展示了非线性层数和参数量共同变化时的分类表现。",
              "后续可缩减大卷积核模型的通道数，再验证等参数预算下的差异。", "",
              "## 7. 模型缺陷与实验验证", "",
              "### 7.1 泛化差距", "",
              "训练与验证使用相同预处理后，准确率差距用于描述拟合差异。",
              "本实验通过 Dropout、权重衰减和数据增强开展正则化实验。",
              f"残差模型的干净训练与验证差距为 {(summaries['residual']['clean_train_accuracy']-summaries['residual']['best_validation_accuracy'])*100:.2f} 个百分点。",
              f"加入 Cutout 后，差距为 {(summaries['residual_cutout']['clean_train_accuracy']-summaries['residual_cutout']['best_validation_accuracy'])*100:.2f} 个百分点。",
              "泛化差距与测试准确率分别描述训练拟合差异和测试集分类表现。",
              "", "",
              "### 7.2 遮挡敏感性", "",
              "测试时固定遮挡图像中心的 8×8 区域。遮挡占输入面积的 6.25%。",
              "遮挡值为中灰色。实验采用固定中心遮挡压力测试。",
              "`residual` 与 `residual_cutout` 使用相同结构和训练预算。后者增加训练遮挡。",
              f"Cutout 的干净测试差值为 {delta('residual_cutout','residual'):+.2f} 个百分点。"]
    occlusion_delta = (results["residual_cutout"]["occluded"]["accuracy"] - results["residual"]["occluded"]["accuracy"]) * 100
    lines += [f"Cutout 的遮挡测试差值为 {occlusion_delta:+.2f} 个百分点。",
              f"固定中心遮挡下，残差模型与 Cutout 模型的准确率分别为 {results['residual']['occluded']['accuracy']*100:.2f}% 和 {results['residual_cutout']['occluded']['accuracy']*100:.2f}%。",
              "", "",
              "| 预定对比 | 差值（百分点） | 配对自助法 95% 区间 |", "| --- | ---: | ---: |"]
    for name, row in intervals.items():
        lines.append(f"| {name} | {row['delta_pp']:+.2f} | [{row['lower_pp']:+.2f}, {row['upper_pp']:+.2f}] |")
    lines += ["", "区间使用 2,000 次配对重采样。每次重采样保持同一图像的两组预测配对。",
              "配对重采样区间描述固定模型在测试图像抽样变化下的差值波动。",
              "表中逐项报告 3 组预定对比的 95% 区间。", "",
              "### 7.3 类别混淆与细节丢失", "",
              "![混淆矩阵](figures/confusion_matrix.png)", "",
              "![误分类样本](figures/misclassifications.png)", ""]
    matrix = np.array(results["ensemble"]["clean"]["confusion_matrix"])
    errors = matrix.copy(); np.fill_diagonal(errors, 0)
    pairs = sorted(((int(errors[i,j]),i,j) for i in range(10) for j in range(10) if i != j), reverse=True)[:3]
    lines.append("集成模型出现频次最高的 3 个定向混淆如下。")
    for count, i, j in pairs:
        lines.append(f"- `{CLASSES[i]}` → `{CLASSES[j]}`：{count} 张。")
    lines += ["", "32×32 图像限制可用细节。连续池化进一步降低空间分辨率。",
              "后续改进方案包括延后下采样和保留高分辨率分支。",
              "程序按测试集索引顺序，为每个真实类别选取第 1 个误分类样本。", "",
              "## 8. 运行说明", "",
              "运行命令见 `README.md`。程序保存数据索引和配置，以及环境、权重和逐轮指标。", "",
              "## 参考资料", "",
              "[1] [CIFAR-10 数据集说明](https://www.cs.toronto.edu/~kriz/cifar.html)。",
              "[2] [fast.ai CIFAR-10 图像归档](https://s3.amazonaws.com/fast-ai-imageclas/cifar10.tgz)。",
              "[3] He et al. [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385), 2015。",
              "[4] DeVries & Taylor. [Improved Regularization of Convolutional Neural Networks with Cutout](https://arxiv.org/abs/1708.04552), 2017。",
              "[5] [PyTorch LayerNorm 文档](https://docs.pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html)。", ""]
    lines += ["[6] Srivastava et al. [Dropout: A Simple Way to Prevent Neural Networks from Overfitting](https://jmlr.org/papers/v15/srivastava14a.html), 2014。", "",
              "[7] Ioffe & Szegedy. [Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://proceedings.mlr.press/v37/ioffe15.html), 2015。", ""]
    destination = ROOT / "REPORT.md"
    temporary = ROOT / "tmp" / "REPORT.md"
    # Keep prose paragraphs short without splitting tables or figure blocks.
    formatted, paragraph = [], []
    def flush():
        if paragraph:
            formatted.append("\n".join(paragraph))
            formatted.append("")
            paragraph.clear()
    for line in lines:
        if not line or line.startswith(("#", "|", "![", "- ", "[")):
            flush()
            formatted.append(line)
            if line.startswith("["):
                formatted.append("")
        else:
            paragraph.append(line)
            if len(paragraph) == 2:
                flush()
    flush()
    temporary.write_text("\n".join(formatted))
    temporary.replace(destination)


def write_notebook(config, results):
    cells = []
    def markdown(source):
        cells.append({"id": f"cell-{len(cells):02d}", "cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)})
    def code(source, output=None):
        cells.append({"id": f"cell-{len(cells):02d}", "cell_type": "code", "metadata": {}, "source": source.splitlines(keepends=True),
                      "execution_count": None, "outputs": [] if output is None else [
                          {"output_type": "stream", "name": "stdout", "text": output.splitlines(keepends=True)}]})
    markdown("# A2：卷积网络\n\n本 Notebook 展示各项实验结果。训练代码位于 `src/cifar10_lab/`。\n\n完整解释见 [实验报告](REPORT.md)。运行方法见 [README](README.md)。\n\nNotebook 的代码单元读取保存的实验结果。")
    code('from pathlib import Path\nimport json\nimport sys\n\nROOT = next(p for p in [Path.cwd(), Path.cwd() / "A2_cifar10", Path.cwd().parent]\n            if (p / "configs" / "experiment.json").exists())\nsys.path.insert(0, str(ROOT / "src"))\nprint(ROOT)')
    markdown('## 题目要求与代码对应说明\n\n代码文件均位于 `src/cifar10_lab/`。下面按题目顺序说明实验、代码和报告章节。\n\n**基础题：构建图像分类网络。** `models.py` 的 `Classifier` 定义卷积网络，`ConvBlock` 定义卷积块。`experiment.py` 的 `train_run` 训练模型，`evaluate_all` 计算测试准确率。第 3 节介绍结构，第 5 节展示训练曲线和分类结果。\n\n**基础题：应用 Dropout 和多种归一化方法。** `models.py` 的 `Classifier` 在分类层中使用 `nn.Dropout`，`normalization` 创建 4 种归一化层及对照配置。`experiment.py` 的 `final_specs` 设置对照模型。第 4 节比较 15 组配置；第 5.1 节比较 `selected` 与 `dropout_variant`，说明 Dropout 的影响，再比较 `dropout_variant` 与 `no_normalization`，说明归一化的影响。\n\n**基础题：通过交叉验证选择超参数。** `experiment.py` 的 `cross_validate` 将 9000 张训练图像分成 3 折，对每组配置训练 3 次，再按平均验证准确率选择配置。第 4 节给出各组均值、标准差和选中的配置。\n\n**附加题 1：尝试提高分类准确率的方法。** `experiment.py` 的 `final_specs` 设置加深网络、残差连接和 Cutout 实验；`evaluate_all` 将 3 个模型的预测概率取平均，得到集成预测。`data.py` 的 `batch_transform` 实现训练时的随机遮挡。第 5.2 节比较深度、残差连接和集成，第 7.2 节比较 Cutout。\n\n**附加题 2：比较卷积核大小并展示图像特征。** `models.py` 的 `ConvBlock` 根据 `kernel_size` 创建卷积层，`experiment.py` 的 `final_specs` 设置 3×3、5×5、7×7 对照。`report.py` 的 `visual_features` 绘制卷积核权重、特征图和输入梯度图。第 6 节结合这些图，比较准确率、参数量、运行时间和图像特征。\n\n**附加题 3：分析模型缺陷并提出改进方案。** `data.py` 的 `occlude` 遮挡测试图像中心，`experiment.py` 的 `predict` 计算两种图像条件下的预测。`report.py` 的 `plot_confusion` 和 `plot_errors` 展示分类错误。第 7 节分析训练与验证准确率差距、遮挡影响和类别混淆，并用 Cutout 实验验证改善遮挡表现的方法。延后下采样和保留高分辨率分支列为后续方案。\n\n表格中的 `model` 表示模型名称，`clean` 表示完整图像准确率，`occluded` 表示中心遮挡图像准确率。中心遮挡覆盖 8×8 像素，填充值为中灰色。\n\n')
    markdown("## 实验配置\n\n程序保存固定数据划分和实际依赖版本。程序按验证准确率选择最佳权重。")
    code('config = json.loads((ROOT / "configs" / "experiment.json").read_text())\nprint(json.dumps(config, indent=2, ensure_ascii=False))', json.dumps(config, indent=2, ensure_ascii=False) + "\n")
    markdown("## 模型结构\n\n模型支持 5 种归一化配置，以及 3 种卷积核大小。")
    code('from cifar10_lab.models import Classifier, ModelSpec, parameter_count\nselection = json.loads((ROOT / "results" / "selection.json").read_text())["selected"]\nspec = ModelSpec(normalization=selection["normalization"], dropout=selection["dropout"], width=config["width"])\nmodel = Classifier(spec)\nprint(model)\nprint("Trainable parameters:", parameter_count(model))')
    markdown("## 交叉验证\n\n各组合使用相同的分层折划分。程序使用验证准确率选择超参数。")
    code('selection = json.loads((ROOT / "results" / "selection.json").read_text())\nprint(json.dumps(selection, indent=2, ensure_ascii=False))')
    code('from IPython.display import display, Image\ndisplay(Image(filename=str(ROOT / "figures" / "cross_validation.png")))')
    markdown("## 最终结果\n\n以下输出来自保存的预测结果。准确率以百分比显示。")
    table = "model（模型）            clean（完整图像）    occluded（中心遮挡）\n" + "".join(
        f"{n:24s} {r['clean']['accuracy']*100:6.2f}%  {r['occluded']['accuracy']*100:6.2f}%\n" for n, r in results.items())
    code('metrics = json.loads((ROOT / "results" / "test_metrics.json").read_text())\nprint("model（模型）            clean（完整图像）    occluded（中心遮挡）")\nfor name, row in metrics.items():\n    print(f"{name:24s} {row[\'clean\'][\'accuracy\']*100:6.2f}%  {row[\'occluded\'][\'accuracy\']*100:6.2f}%")', table)
    code('display(Image(filename=str(ROOT / "figures" / "learning_curves.png")))\ndisplay(Image(filename=str(ROOT / "figures" / "accuracy_comparison.png")))')
    markdown("## 卷积核与特征图\n\n不同模型使用相同测试图像。每张热图分别缩放颜色，用于展示图像各位置的响应。")
    code('for name in ["kernel_weights", "features_early", "features_late", "input_sensitivity"]:\n    display(Image(filename=str(ROOT / "figures" / f"{name}.png")))')
    markdown("## 误分类与缺陷\n\n中心遮挡测试覆盖图像中央的 8×8 区域。Cutout 实验比较训练时加入随机遮挡后的准确率。")
    code('display(Image(filename=str(ROOT / "figures" / "confusion_matrix.png")))\ndisplay(Image(filename=str(ROOT / "figures" / "misclassifications.png")))')
    markdown("## 运行与检查\n\n从本目录运行终端命令。`--resume` 跳过设置相同且训练完成的实验。\n\n```bash\n.venv/bin/python -m pytest -q\n.venv/bin/python -m cifar10_lab.experiment --resume\n.venv/bin/python -m cifar10_lab.report\n.venv/bin/python -m cifar10_lab.verify\n```\n\n实验核验结果见 `results/verification.json`。")
    write_json(ROOT / "main.ipynb", {"nbformat": 4, "nbformat_minor": 5, "cells": cells,
                                    "metadata": {"kernelspec": {"display_name": "Python 3 (A2：卷积网络)",
                                                                "language": "python", "name": "python3"},
                                                 "language_info": {"name": "python", "version": "3.11"}}})


def main():
    torch.set_num_threads(2)
    manifest = read_json(ROOT / "results" / "manifest.json")
    config = manifest["config"]
    selection = read_json(ROOT / "results" / "selection.json")
    results = read_json(ROOT / "results" / "test_metrics.json")
    plan = read_json(ROOT / "results" / "evaluation_plan.json")
    summaries = {name: read_json(ROOT / "results" / "runs" / name / "summary.json") for name in plan["models"]}
    plot_cv(read_json(ROOT / "results" / "cross_validation.json"), config)
    plot_curves(summaries)
    plot_performance(results)
    plot_confusion(results["ensemble"])
    with np.load(ROOT / "data" / "cifar10.npz", allow_pickle=False) as data:
        images = data["test_images"]
    visual_features(images)
    predictions = np.load(ROOT / "results" / "ensemble_predictions.npz", allow_pickle=False)
    plot_errors(images, predictions)
    paths = ROOT / "results" / "runs"
    residual = np.load(paths / "residual" / "predictions.npz", allow_pickle=False)
    cutout = np.load(paths / "residual_cutout" / "predictions.npz", allow_pickle=False)
    selected = np.load(paths / "selected" / "predictions.npz", allow_pickle=False)
    intervals = {"Cutout：干净图像": paired_interval(residual["clean"], cutout["clean"], residual["labels"]),
                 "Cutout：遮挡图像": paired_interval(residual["occluded"], cutout["occluded"], residual["labels"]),
                 "集成相对浅层模型：干净图像": paired_interval(selected["clean"], predictions["clean"], residual["labels"])}
    write_json(ROOT / "results" / "paired_intervals.json", intervals)
    generate_report(config, selection, summaries, results, intervals)
    write_notebook(config, results)
    print(f"Report written: {ROOT / 'REPORT.md'}")


if __name__ == "__main__":
    main()
