import os
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Union
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, roc_curve, auc
from torchvision import datasets, transforms
from fcnn import FullyConnectedNN

CLASSES = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
INPUT_SIZE = 32 * 32 * 3
HIDDEN_SIZES = [1024, 512, 256, 128]
NUM_CLASSES = 10
BATCH_SIZE = 256


def plot_confusion_matrix(cm: np.ndarray, classes: List[str], normalize: bool = False,
                          title: str = 'Confusion Matrix', save_to: str = 'confusion_matrix.png',
                          figsize: Tuple[int, int] = (10, 8), cmap: Any = plt.cm.Blues) -> None:
    """
    绘制并保存混淆矩阵。

    参数:
        cm: 混淆矩阵
        classes: 类别名称列表
        normalize: 是否归一化混淆矩阵
        title: 图表标题
        save_to: 保存图表的路径
        figsize: 图表尺寸
        cmap: 使用的颜色映射
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd', cmap=cmap,
                xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(save_to)
    plt.close()


def compute_per_class_metrics(y_true: List[int], y_pred: List[int],
                              classes: List[str]) -> pd.DataFrame:
    """
    计算每个类别的精确率、召回率和F1分数。

    参数:
        y_true: 真实标签
        y_pred: 预测标签
        classes: 类别名称列表

    返回:
        包含每个类别的精确率、召回率和F1分数的DataFrame
    """
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred)

    metrics_df = pd.DataFrame({
        'Class': classes,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'Support': support
    })

    return metrics_df


def compute_roc_auc(model: torch.nn.Module, test_loader: torch.utils.data.DataLoader,
                    device: torch.device, num_classes: int) -> Tuple[Dict[int, np.ndarray],
                                                                     Dict[int, np.ndarray],
                                                                     Dict[int, float]]:
    """
    计算每个类别的ROC曲线和AUC值（采用一对多方法）。

    参数:
        model: 训练好的模型
        test_loader: 测试数据的DataLoader
        device: 运行推理的设备
        num_classes: 类别数量

    返回:
        tuple: (fpr, tpr, roc_auc)
    """
    model.eval()
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.numpy())

    all_probs = np.vstack(all_probs)
    all_labels = np.concatenate(all_labels)

    # 计算每个类别的ROC曲线和AUC值
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(num_classes):
        # 转换为二分类问题（一对多）
        y_true = (all_labels == i).astype(int)
        y_score = all_probs[:, i]

        fpr[i], tpr[i], _ = roc_curve(y_true, y_score)
        roc_auc[i] = auc(fpr[i], tpr[i])

    return fpr, tpr, roc_auc


def plot_roc_curves(fpr: Dict[int, np.ndarray], tpr: Dict[int, np.ndarray],
                    roc_auc: Dict[int, float], classes: List[str],
                    save_to: str = 'roc_curves.png', figsize: Tuple[int, int] = (12, 10)) -> None:
    """
    绘制所有类别的ROC曲线。

    参数:
        fpr: 每个类别的假阳性率
        tpr: 每个类别的真阳性率
        roc_auc: 每个类别的AUC值
        classes: 类别名称列表
        save_to: 保存图表的路径
        figsize: 图表尺寸
    """
    plt.figure(figsize=figsize)

    for i, class_name in enumerate(classes):
        plt.plot(
            fpr[i],
            tpr[i],
            lw=2,
            label=f'{class_name} (AUC = {roc_auc[i]:.2f})'
        )

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves (One-vs-Rest)')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(save_to)
    plt.close()


def evaluate_best_model(model_name: str, model_dir: str, result_dir: str,
                        test_loader: torch.utils.data.DataLoader,
                        device: torch.device) -> Tuple[Optional[float], Optional[np.ndarray], Optional[pd.DataFrame]]:
    """
    加载最佳模型并在测试集上评估。

    参数:
        model_name: 模型名称
        model_dir: 保存模型的目录
        result_dir: 保存评估结果的目录
        test_loader: 测试数据的DataLoader
        device: 运行推理的设备

    返回:
        tuple: (accuracy, confusion matrix, class metrics)
    """
    Path(result_dir).mkdir(parents=True, exist_ok=True)

    model_path = os.path.join(model_dir, "best_model.pth")

    if not os.path.exists(model_path):
        print(f"Best model file does not exist: {model_path}")
        return None, None, None

    # 创建模型实例
    model = FullyConnectedNN(INPUT_SIZE, HIDDEN_SIZES, NUM_CLASSES).to(device)

    # 加载模型权重
    print(f"Loading model weights from {model_path}")
    model.load_state_dict(torch.load(model_path, weights_only=True, map_location=device))
    model.eval()

    # 收集所有预测和真实标签
    all_preds = []
    all_true = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_true.extend(labels.cpu().numpy())

    # 计算总体准确率
    accuracy = 100 * (np.array(all_preds) == np.array(all_true)).mean()

    # 计算混淆矩阵
    cm = confusion_matrix(all_true, all_preds)

    # 将原始混淆矩阵保存为CSV
    cm_df = pd.DataFrame(cm, index=CLASSES, columns=CLASSES)
    cm_df.to_csv(os.path.join(result_dir, "confusion_matrix.csv"))

    # 绘制混淆矩阵
    plot_confusion_matrix(cm, CLASSES, title=f'{model_name} 混淆矩阵',
                          save_to=os.path.join(result_dir, "confusion_matrix.png"))

    # 绘制归一化混淆矩阵
    plot_confusion_matrix(cm, CLASSES, normalize=True, title=f'{model_name} Normalized Confusion Matrix',
                          save_to=os.path.join(result_dir, "normalized_confusion_matrix.png"))

    # 计算每个类别的指标
    class_metrics = compute_per_class_metrics(all_true, all_preds, CLASSES)
    class_metrics.to_csv(os.path.join(result_dir, "class_metrics.csv"), index=False)

    # 绘制类别指标
    plt.figure(figsize=(15, 10))
    metrics = ['Precision', 'Recall', 'F1-Score']
    for i, metric in enumerate(metrics):
        plt.subplot(3, 1, i+1)
        sns.barplot(x='Class', y=metric, data=class_metrics)
        plt.title(f'{model_name} - {metric} by Class')
        plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, "class_metrics_plot.png"))
    plt.close()

    # 计算ROC曲线和AUC
    fpr, tpr, roc_auc = compute_roc_auc(model, test_loader, device, NUM_CLASSES)

    # 保存AUC值
    auc_df = pd.DataFrame({
        'Class': CLASSES,
        'AUC': [roc_auc[i] for i in range(NUM_CLASSES)]
    })
    auc_df.to_csv(os.path.join(result_dir, "auc_values.csv"), index=False)

    # 绘制ROC曲线
    plot_roc_curves(fpr, tpr, roc_auc, CLASSES, save_to=os.path.join(result_dir, "roc_curves.png"))

    print(f"Model {model_name} evaluated:")
    print(f"Overall Accuracy: {accuracy:.2f}%")
    print(f"Results saved to {result_dir}")

    return accuracy, cm, class_metrics


def evaluate_all_best_models(base_model_dir: str = "../model",
                             base_result_dir: str = "../result",
                             batch_size: int = 256) -> None:
    """
    评估所有最佳模型并比较它们。

    参数:
        base_model_dir: 包含模型子目录的基础目录
        base_result_dir: 保存结果的基础目录
        batch_size: 评估的批量大小
    """
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    print(f"Using device: {device} for evaluation")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    test_dataset = datasets.CIFAR10(root='../data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    combined_result_dir = os.path.join(base_result_dir, "combined_evaluation")
    Path(combined_result_dir).mkdir(parents=True, exist_ok=True)

    # 查找所有包含best_model.pth的模型目录
    model_results = []
    for model_dir in os.listdir(base_model_dir):
        full_model_dir = os.path.join(base_model_dir, model_dir)
        best_model_path = os.path.join(full_model_dir, "best_model.pth")

        if os.path.isdir(full_model_dir) and os.path.exists(best_model_path):
            print(f"Found best model in {model_dir}, evaluating...")

            # 为此模型创建结果目录
            model_result_dir = os.path.join(base_result_dir, model_dir, "evaluation")

            # 评估模型
            accuracy, _, _ = evaluate_best_model(model_dir, full_model_dir, model_result_dir, test_loader, device)

            # 存储比较结果
            if accuracy is not None:
                model_results.append({
                    'Model': model_dir,
                    'Accuracy': accuracy
                })

    if model_results:
        # 创建比较DataFrame
        comparison_df = pd.DataFrame(model_results)
        comparison_df = comparison_df.sort_values(by='Accuracy', ascending=False)
        comparison_df.to_csv(os.path.join(combined_result_dir, "model_comparison.csv"), index=False)

        # 绘制比较图
        plt.figure(figsize=(12, 8))
        sns.barplot(x='Model', y='Accuracy', data=comparison_df)
        plt.title('Comparison of Model Accuracies')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(combined_result_dir, "model_comparison.png"))
        plt.close()

        print("\n===== Best Models Accuracy Comparison =====")
        print(comparison_df)
        print(f"\nBest model: {comparison_df.iloc[0]['Model']} with accuracy {comparison_df.iloc[0]['Accuracy']:.2f}%")
