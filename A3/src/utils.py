import time
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import os
import pandas as pd
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import KFold
import itertools
import json

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


def evaluate_model(model: torch.nn.Module,
                   data_loader: torch.utils.data.DataLoader,
                   criterion: torch.nn.Module,
                   device: torch.device) -> tuple[float, float]:
    """
    在给定数据集上评估PyTorch模型。

    参数:
        model: 要评估的神经网络模型
        data_loader: 包含验证/测试数据集的DataLoader
        criterion: 用于计算模型损失的损失函数
        device: 运行评估的设备(CPU/GPU)

    返回:
        tuple[float, float]: 包含以下内容的元组:
            - 数据集上的平均损失
            - 准确率百分比(0-100)
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    loss = running_loss / len(data_loader)
    accuracy = 100 * correct / total

    return loss, accuracy


def train_model(model: torch.nn.Module,
                train_loader: torch.utils.data.DataLoader,
                val_loader: torch.utils.data.DataLoader,
                test_loader: torch.utils.data.DataLoader,
                criterion: torch.nn.Module,
                optimizer: torch.optim.Optimizer,
                num_epochs: int,
                device: torch.device,
                model_name: str,
                scheduler=None) -> dict:
    """
    训练PyTorch模型并返回训练历史。

    参数:
        model: 要训练的神经网络模型
        train_loader: 包含训练数据的DataLoader
        val_loader: 包含验证数据的DataLoader
        test_loader: 包含测试数据的DataLoader
        criterion: 用于计算模型损失的损失函数
        optimizer: 用于更新模型参数的优化器
        num_epochs: 训练轮数
        device: 运行训练的设备(CPU/GPU)
        model_name: 模型名称，用于保存模型和结果
        scheduler: 学习率调度器 (可选)

    返回:
        dict: 包含训练历史的字典，包括:
            - train_losses: 训练损失列表
            - train_accuracies: 训练准确率列表
            - val_losses: 验证损失列表
            - val_accuracies: 验证准确率列表
            - test_accuracies: 测试准确率列表
            - test_class_accuracies: 每个类别的测试准确率字典
            - best_epoch: 最佳模型对应的epoch
    """
    # 创建保存模型的目录
    model_dir = f"../model/{model_name}"
    Path(model_dir).mkdir(parents=True, exist_ok=True)

    # 创建保存结果的目录
    result_dir = f"../result/{model_name}"
    Path(result_dir).mkdir(parents=True, exist_ok=True)

    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    test_accuracies = []
    test_class_accuracies = {}

    # 跟踪最佳模型
    best_val_accuracy = 0
    best_test_accuracy = 0
    best_epoch = 0
    best_metrics = {}

    start_time = time.time()

    for epoch in range(num_epochs):
        model.train()

        running_loss = 0.0
        correct = 0
        total = 0

        progress_bar = tqdm(
            train_loader,
            total=len(train_loader),
            desc=f'Epoch {epoch+1}/{num_epochs}',
        )

        for batch_idx, (images, labels) in enumerate(progress_bar):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            current_loss = running_loss / (batch_idx + 1)
            current_acc = 100 * correct / total

            progress_bar.set_postfix_str(
                f'loss: {current_loss:.4f} - acc: {current_acc:.2f}%'
            )

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_acc)

        val_loss, val_acc = evaluate_model(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        # 学习率调度器更新 - 移到这里，在计算验证损失之后
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

        # 在测试集上评估模型
        test_acc, class_accuracies = test_model(model, test_loader, device, verbose=False)
        test_accuracies.append(test_acc)
        test_class_accuracies[epoch+1] = class_accuracies

        # 保存每个epoch的模型
        epoch_model_path = os.path.join(model_dir, f"epoch_{epoch+1}.pth")
        torch.save(model.state_dict(), epoch_model_path)

        # 使用验证集准确率来确定最佳模型
        is_best = val_acc > best_val_accuracy

        # 如果这是最佳模型，则更新最佳指标
        if is_best:
            best_val_accuracy = val_acc
            best_test_accuracy = test_acc
            best_epoch = epoch + 1
            best_metrics = {
                'epoch': best_epoch,
                'train_accuracy': epoch_acc,
                'val_accuracy': val_acc,
                'test_accuracy': test_acc,
                'class_accuracies': class_accuracies
            }

            # 保存最佳模型
            best_model_path = os.path.join(model_dir, "best_model.pth")
            torch.save(model.state_dict(), best_model_path)

        print(f'Epoch {epoch+1}/{num_epochs} - '
              f'loss: {epoch_loss:.4f} - acc: {epoch_acc:.2f}% - '
              f'val_loss: {val_loss:.4f} - val_acc: {val_acc:.2f}% - '
              f'test_acc: {test_acc:.2f}%')

        # 创建并保存每个epoch的测试结果
        epoch_results = pd.DataFrame({
            'Class': list(classes),
            'Accuracy': list(class_accuracies.values())
        })

        # 保存每个epoch的测试结果
        epoch_results.to_csv(os.path.join(result_dir, f"epoch_{epoch+1}_results.csv"), index=False)

        # 可视化每个epoch的类别准确率
        plt.figure(figsize=(12, 6))
        sns.barplot(x='Class', y='Accuracy', data=epoch_results)
        plt.title(f'Class-wise Accuracy - Epoch {epoch+1}')
        plt.ylabel('Accuracy (%)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(result_dir, f"epoch_{epoch+1}_class_accuracy.png"))
        plt.close()

    elapsed_time = time.time() - start_time
    print(f'Training completed in {elapsed_time:.2f} seconds')
    print(f'Best model achieved at epoch {best_epoch} with validation accuracy: {best_val_accuracy:.2f}% and test accuracy: {best_test_accuracy:.2f}%')

    # 保存所有训练历史
    history_df = pd.DataFrame({
        'Epoch': list(range(1, num_epochs + 1)),
        'Train Loss': train_losses,
        'Train Accuracy': train_accuracies,
        'Validation Loss': val_losses,
        'Validation Accuracy': val_accuracies,
        'Test Accuracy': test_accuracies
    })
    history_df.to_csv(os.path.join(result_dir, "training_history.csv"), index=False)

    # 保存最佳模型的指标
    best_metrics_df = pd.DataFrame([best_metrics])
    best_metrics_df.to_csv(os.path.join(result_dir, "best_model_metrics.csv"), index=False)

    # 绘制训练历史
    plot_learning_curves_with_test({
        'train_losses': train_losses,
        'train_accuracies': train_accuracies,
        'val_losses': val_losses,
        'val_accuracies': val_accuracies,
        'test_accuracies': test_accuracies
    }, os.path.join(result_dir, "learning_curves.png"), f'{model_name} - Learning Curves', best_epoch)

    return {
        'train_losses': train_losses,
        'train_accuracies': train_accuracies,
        'val_losses': val_losses,
        'val_accuracies': val_accuracies,
        'test_accuracies': test_accuracies,
        'test_class_accuracies': test_class_accuracies,
        'best_epoch': best_epoch,
        'best_metrics': best_metrics
    }


def test_model(model: torch.nn.Module,
               test_loader: torch.utils.data.DataLoader,
               device: torch.device,
               verbose: bool = True) -> tuple[float, dict]:
    """
    在测试集上评估模型性能。

    参数:
        model: 要测试的神经网络模型
        test_loader: 包含测试数据的DataLoader
        device: 运行测试的设备(CPU/GPU)
        verbose: 是否打印详细信息

    返回:
        tuple: (总体准确率, 每个类别的准确率字典)
    """
    model.eval()
    correct = 0
    total = 0
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            c = (predicted == labels).squeeze()
            for i in range(labels.size(0)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    accuracy = 100 * correct / total

    # 计算每个类别的准确率
    class_accuracies = {}
    for i in range(10):
        class_acc = 100 * class_correct[i] / class_total[i]
        class_accuracies[classes[i]] = class_acc
        if verbose:
            print(f'Accuracy of {classes[i]}: {class_acc:.2f}%')

    if verbose:
        print(f'Test Accuracy: {accuracy:.2f}%')

    return accuracy, class_accuracies


def plot_learning_curves(history: dict,
                         save_to: str = 'learning_curves.png',
                         title: str = 'Learning Curves') -> None:
    """
    绘制训练和验证的损失曲线与准确率曲线。

    参数:
        history: 包含训练历史的字典，需要包含:
                train_losses, train_accuracies, val_losses, val_accuracies
        save_to: 图像保存路径
        title: 图像标题
    """
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history['train_losses'], label='Training Loss')
    plt.plot(history['val_losses'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'{title} - Loss Curves')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history['train_accuracies'], label='Training Accuracy')
    plt.plot(history['val_accuracies'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.title(f'{title} - Accuracy Curves')
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_to)
    plt.close()


def plot_learning_curves_with_test(history: dict,
                                   save_to: str = 'learning_curves.png',
                                   title: str = 'Learning Curves',
                                   best_epoch: int = None) -> None:
    """
    绘制训练、验证和测试的损失曲线与准确率曲线。

    参数:
        history: 包含训练历史的字典，需要包含:
                train_losses, train_accuracies, val_losses, val_accuracies, test_accuracies
        save_to: 图像保存路径
        title: 图像标题
        best_epoch: 标记最佳epoch
    """
    plt.figure(figsize=(16, 7))

    plt.subplot(1, 2, 1)
    plt.plot(history['train_losses'], label='Training Loss')
    plt.plot(history['val_losses'], label='Validation Loss')
    if best_epoch:
        plt.axvline(x=best_epoch-1, color='r', linestyle='--', label=f'Best Epoch ({best_epoch})')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'{title} - Loss Curves')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history['train_accuracies'], label='Training Accuracy')
    plt.plot(history['val_accuracies'], label='Validation Accuracy')
    plt.plot(history['test_accuracies'], label='Test Accuracy')
    if best_epoch:
        plt.axvline(x=best_epoch-1, color='r', linestyle='--', label=f'Best Epoch ({best_epoch})')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.title(f'{title} - Accuracy Curves')
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_to)
    plt.close()


def show_random_images(dataset, classes, num_images=5):
    """显示数据集中的随机图片"""
    fig, axes = plt.subplots(1, num_images, figsize=(15, 3))
    for ax in axes:
        idx = np.random.randint(len(dataset))
        img, label = dataset[idx]
        img = img / 2 + 0.5  # 反归一化
        ax.imshow(np.transpose(img, (1, 2, 0)))
        ax.set_title(classes[label])
        ax.axis('off')
    plt.show()


def plot_class_distribution(dataset, classes):
    """绘制数据集的类别分布"""
    labels = [label for _, label in dataset]
    unique_labels, counts = np.unique(labels, return_counts=True)

    plt.figure(figsize=(10, 5))
    plt.bar(range(len(classes)), counts)
    plt.xticks(range(len(classes)), classes, rotation=45)
    plt.title('Class Distribution in Dataset')
    plt.xlabel('Classes')
    plt.ylabel('Number of Images')
    plt.tight_layout()
    plt.show()


def cross_validate_hyperparameters(model_class, hyperparams, train_dataset, num_folds=5, batch_size=64, num_epochs=5, device='cpu'):
    """使用交叉验证来评估不同的超参数

    参数:
        model_class: 模型类
        hyperparams: 超参数字典
        train_dataset: 训练数据集
        num_folds: 交叉验证折数
        batch_size: 批次大小
        num_epochs: 每次验证训练的轮数
        device: 训练设备

    返回:
        dict: 包含每种超参数组合的平均验证准确率
    """
    # 生成所有超参数组合
    keys = hyperparams.keys()
    values = hyperparams.values()
    combinations = list(itertools.product(*values))
    param_names = list(keys)

    # 结果记录
    results = {}

    # K折交叉验证
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)

    # 对每种超参数组合进行交叉验证
    for i, combination in enumerate(combinations):
        combo_params = dict(zip(param_names, combination))
        print(f"Testing hyperparameters: {combo_params}")

        # 记录每个fold的验证准确率
        fold_accuracies = []

        # 对每个fold进行训练和验证
        for fold, (train_idx, val_idx) in enumerate(kf.split(train_dataset)):
            print(f"  Fold {fold+1}/{num_folds}")

            # 分割训练集和验证集
            train_subset = torch.utils.data.Subset(train_dataset, train_idx)
            val_subset = torch.utils.data.Subset(train_dataset, val_idx)

            train_loader = torch.utils.data.DataLoader(train_subset, batch_size=batch_size, shuffle=True)
            val_loader = torch.utils.data.DataLoader(val_subset, batch_size=batch_size)

            # 创建模型
            model = model_class(**combo_params).to(device)

            # 损失函数和优化器
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

            # 训练模型
            best_val_acc = 0
            for epoch in range(num_epochs):
                # 训练
                model.train()
                for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
                    images, labels = images.to(device), labels.to(device)

                    optimizer.zero_grad()
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                # 验证
                model.eval()
                correct = 0
                total = 0
                with torch.no_grad():
                    for images, labels in val_loader:
                        images, labels = images.to(device), labels.to(device)
                        outputs = model(images)
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()

                val_acc = 100 * correct / total
                print(f"    Epoch {epoch+1}: Validation Accuracy = {val_acc:.2f}%")

                if val_acc > best_val_acc:
                    best_val_acc = val_acc

            fold_accuracies.append(best_val_acc)

        # 计算该超参数组合的平均准确率
        avg_accuracy = sum(fold_accuracies) / len(fold_accuracies)
        results[str(combo_params)] = {
            "params": combo_params,
            "fold_accuracies": fold_accuracies,
            "avg_accuracy": avg_accuracy
        }
        print(f"  Average validation accuracy: {avg_accuracy:.2f}%")

    # 按平均准确率排序
    sorted_results = sorted(results.items(), key=lambda x: x[1]["avg_accuracy"], reverse=True)
    best_params = sorted_results[0][1]["params"]

    print("\n===== Cross-Validation Results =====")
    for params_str, result in sorted_results:
        print(f"Parameters: {params_str}")
        print(f"  Average accuracy: {result['avg_accuracy']:.2f}%")
        print(f"  Fold accuracies: {result['fold_accuracies']}")

    print(f"\nBest hyperparameters: {best_params}")
    print(f"Best average accuracy: {sorted_results[0][1]['avg_accuracy']:.2f}%")

    # 保存结果
    Path("../result").mkdir(parents=True, exist_ok=True)
    with open("../result/cross_validation_results.json", "w") as f:
        json_results = {k: {"params": str(v["params"]),
                            "fold_accuracies": v["fold_accuracies"],
                            "avg_accuracy": v["avg_accuracy"]} for k, v in results.items()}
        json.dump(json_results, f, indent=2)

    return best_params
