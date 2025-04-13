import os
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, roc_curve, auc
from sklearn.metrics import classification_report
import torchvision
import torchvision.transforms as transforms
from cnn_model import BasicCNN, DeepCNN, ResNet, EnsembleModel

CLASSES = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
BATCH_SIZE = 256


def get_device():
    """Detect and return the available device for computation."""
    if torch.backends.mps.is_available():
        return torch.device('mps')
    elif torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


def load_test_data(batch_size=BATCH_SIZE):
    """Load the CIFAR-10 test dataset."""
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    test_dataset = torchvision.datasets.CIFAR10(
        root='../data', train=False, download=True, transform=transform_test)

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    return test_loader


def plot_confusion_matrix(cm, classes, normalize=False,
                          title='Confusion Matrix',
                          save_path='confusion_matrix.png',
                          figsize=(10, 8), cmap=plt.cm.Blues):
    """
    Plot and save confusion matrix.

    Args:
        cm: Confusion matrix
        classes: List of class names
        normalize: Whether to normalize the confusion matrix
        title: Plot title
        save_path: Path to save the figure
        figsize: Figure size
        cmap: Color map for the plot
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd',
                cmap=cmap, xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def compute_confusion_matrix(model, data_loader, device):
    """
    Compute the confusion matrix for a model on the given dataset.

    Args:
        model: PyTorch model
        data_loader: DataLoader for the dataset
        device: Device to run the model on

    Returns:
        Tuple of (confusion matrix, all predictions, all true labels)
    """
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            outputs = model(images)
            _, predictions = torch.max(outputs, 1)

            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(labels.numpy())

    # Compute confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    return cm, all_preds, all_labels


def compute_class_metrics(y_true, y_pred, classes):
    """
    Compute precision, recall, F1-score for each class.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        classes: List of class names

    Returns:
        DataFrame with precision, recall, F1-score for each class
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


def compute_roc_curves(model, data_loader, device, num_classes=10):
    """
    Compute ROC curves and AUC for each class.

    Args:
        model: PyTorch model
        data_loader: DataLoader for the dataset
        device: Device to run the model on
        num_classes: Number of classes

    Returns:
        Dictionary with ROC curve data and AUC for each class
    """
    model.eval()
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)

            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.numpy())

    all_probs = np.vstack(all_probs)
    all_labels = np.concatenate(all_labels)

    # Compute ROC curve and AUC for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(num_classes):
        # Binary classification: one-vs-rest
        y_true = (all_labels == i).astype(int)
        y_score = all_probs[:, i]

        fpr[i], tpr[i], _ = roc_curve(y_true, y_score)
        roc_auc[i] = auc(fpr[i], tpr[i])

    return {'fpr': fpr, 'tpr': tpr, 'roc_auc': roc_auc}


def plot_roc_curves(roc_data, classes, save_path='roc_curves.png'):
    """
    Plot ROC curves for all classes.

    Args:
        roc_data: Dictionary with ROC curve data
        classes: List of class names
        save_path: Path to save the figure
    """
    fpr = roc_data['fpr']
    tpr = roc_data['tpr']
    roc_auc = roc_data['roc_auc']

    plt.figure(figsize=(12, 10))

    # Plot ROC curves for each class
    for i, class_name in enumerate(classes):
        plt.plot(
            fpr[i],
            tpr[i],
            lw=2,
            label=f'{class_name} (AUC = {roc_auc[i]:.2f})'
        )

    # Plot diagonal line (random classifier)
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves (One-vs-Rest)')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    # Save AUC values to CSV
    auc_df = pd.DataFrame({
        'Class': classes,
        'AUC': [roc_auc[i] for i in range(len(classes))]
    })

    csv_path = os.path.splitext(save_path)[0] + '_auc_values.csv'
    auc_df.to_csv(csv_path, index=False)


def visualize_misclassifications(model, data_loader, device, classes,
                                 save_dir='../result/misclassifications',
                                 num_samples=5):
    """
    Visualize examples of misclassified images.

    Args:
        model: PyTorch model
        data_loader: DataLoader for the dataset
        device: Device to run the model on
        classes: List of class names
        save_dir: Directory to save visualizations
        num_samples: Number of misclassified samples to show per class
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    model.eval()
    misclassified = {i: [] for i in range(len(classes))}

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            outputs = model(images)
            _, predictions = torch.max(outputs, 1)

            # Find misclassified images
            for i, (image, label, pred) in enumerate(zip(images, labels, predictions)):
                if label.item() != pred.item():
                    # Store the image, true label, and predicted label
                    true_class = label.item()
                    if len(misclassified[true_class]) < num_samples:
                        misclassified[true_class].append({
                            'image': image.cpu().numpy(),
                            'true': true_class,
                            'pred': pred.item()
                        })

    # Visualize misclassified images for each class
    for class_idx in range(len(classes)):
        if not misclassified[class_idx]:
            continue

        fig, axes = plt.subplots(1, min(num_samples, len(misclassified[class_idx])),
                                 figsize=(15, 3))

        if len(misclassified[class_idx]) == 1:
            axes = [axes]

        for i, sample in enumerate(misclassified[class_idx]):
            if i >= len(axes):
                break

            # De-normalize the image
            img = sample['image'] / 2 + 0.5
            img = np.transpose(img, (1, 2, 0))

            axes[i].imshow(img)
            axes[i].set_title(f"True: {classes[sample['true']]}\nPred: {classes[sample['pred']]}")
            axes[i].axis('off')

        plt.suptitle(f"Misclassified {classes[class_idx]} Images")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"misclassified_{classes[class_idx]}.png"))
        plt.close()


def evaluate_model(model_name, model_type, test_loader, device, result_dir):
    """
    Evaluate a model using confusion matrix and other metrics.

    Args:
        model_name: Name of the model for loading and saving
        model_type: Type of model (BasicCNN, DeepCNN, ResNet, or EnsembleModel)
        test_loader: DataLoader for the test dataset
        device: Device to run the model on
        result_dir: Directory to save results
    """
    model_path = f"../model/{model_name}/best_model.pth"

    # Create result directory
    evaluation_dir = os.path.join(result_dir, model_name, 'evaluation')
    Path(evaluation_dir).mkdir(parents=True, exist_ok=True)

    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        return None

    # Create model instance based on type
    if model_type == 'BasicCNN':
        # Load hyperparameters from file if available
        try:
            params_path = os.path.join(result_dir, 'cross_validation_results.json')
            if os.path.exists(params_path):
                import json
                with open(params_path, 'r') as f:
                    cv_results = json.load(f)
                    # Get best params from results
                    best_params_str = max(cv_results.items(), key=lambda x: x[1]['avg_accuracy'])[0]
                    # Convert string representation to dict
                    import ast
                    best_params = ast.literal_eval(best_params_str)
                    model = BasicCNN(**best_params)
            else:
                model = BasicCNN()
        except:
            # Fallback to default params
            model = BasicCNN()
    elif model_type == 'DeepCNN':
        model = DeepCNN()
    elif model_type == 'ResNet':
        model = ResNet()
    elif model_type == 'EnsembleModel':
        # For ensemble, load individual models first
        basic_cnn = BasicCNN().to(device)
        deep_cnn = DeepCNN().to(device)
        resnet = ResNet().to(device)

        basic_cnn.load_state_dict(torch.load(f"../model/BasicCNN_BestParams/best_model.pth", weights_only=True, map_location=device))
        deep_cnn.load_state_dict(torch.load(f"../model/DeepCNN/best_model.pth", weights_only=True, map_location=device))
        resnet.load_state_dict(torch.load(f"../model/ResNet/best_model.pth", weights_only=True, map_location=device))

        model = EnsembleModel([basic_cnn, deep_cnn, resnet])
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Load model weights
    if model_type != 'EnsembleModel':
        model.load_state_dict(torch.load(model_path, weights_only=True, map_location=device))

    model = model.to(device)
    model.eval()

    print(f"Evaluating {model_name}...")

    # Compute confusion matrix
    cm, all_preds, all_labels = compute_confusion_matrix(model, test_loader, device)

    # Save raw confusion matrix as CSV
    cm_df = pd.DataFrame(cm, index=CLASSES, columns=CLASSES)
    cm_df.to_csv(os.path.join(evaluation_dir, "confusion_matrix.csv"))

    # Plot and save confusion matrices
    plot_confusion_matrix(
        cm, CLASSES, normalize=False,
        title=f'{model_name} Confusion Matrix',
        save_path=os.path.join(evaluation_dir, "confusion_matrix.png")
    )

    plot_confusion_matrix(
        cm, CLASSES, normalize=True,
        title=f'{model_name} Normalized Confusion Matrix',
        save_path=os.path.join(evaluation_dir, "normalized_confusion_matrix.png")
    )

    # Compute and save class metrics
    metrics_df = compute_class_metrics(all_labels, all_preds, CLASSES)
    metrics_df.to_csv(os.path.join(evaluation_dir, "class_metrics.csv"), index=False)

    # Generate and save classification report
    report = classification_report(all_labels, all_preds, target_names=CLASSES, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(os.path.join(evaluation_dir, "classification_report.csv"))

    # Compute and plot ROC curves
    roc_data = compute_roc_curves(model, test_loader, device)
    plot_roc_curves(
        roc_data, CLASSES,
        save_path=os.path.join(evaluation_dir, "roc_curves.png")
    )

    # Visualize misclassifications
    visualize_misclassifications(
        model, test_loader, device, CLASSES,
        save_dir=os.path.join(evaluation_dir, "misclassifications")
    )

    # Calculate overall accuracy
    accuracy = 100 * (np.array(all_preds) == np.array(all_labels)).mean()

    print(f"{model_name} - Test Accuracy: {accuracy:.2f}%")

    # Save summary statistics
    summary = {
        'Model': model_name,
        'Accuracy': accuracy,
        'Macro Avg Precision': report['macro avg']['precision'],
        'Macro Avg Recall': report['macro avg']['recall'],
        'Macro Avg F1-Score': report['macro avg']['f1-score'],
        'Weighted Avg Precision': report['weighted avg']['precision'],
        'Weighted Avg Recall': report['weighted avg']['recall'],
        'Weighted Avg F1-Score': report['weighted avg']['f1-score'],
    }

    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(os.path.join(evaluation_dir, "summary.csv"), index=False)

    return {
        'name': model_name,
        'accuracy': accuracy,
        'confusion_matrix': cm,
        'metrics': metrics_df,
        'roc_data': roc_data
    }


def evaluate_all_models(result_dir="../result"):
    """
    Evaluate all trained models and compare them.
    """
    # Set up device
    device = get_device()
    print(f"Using device: {device}")

    # Load test data
    test_loader = load_test_data()

    # Models to evaluate
    models_to_evaluate = [
        {'name': 'BasicCNN_BestParams', 'type': 'BasicCNN'},
        {'name': 'DeepCNN', 'type': 'DeepCNN'},
        {'name': 'ResNet', 'type': 'ResNet'},
        {'name': 'ensemble', 'type': 'EnsembleModel'}
    ]

    # Evaluate each model
    evaluation_results = []
    for model_info in models_to_evaluate:
        model_name = model_info['name']
        model_type = model_info['type']

        # Skip if model is ensemble and we're evaluating it separately
        if model_type == 'EnsembleModel' and not os.path.exists(f"../model/{model_name}/best_model.pth"):
            continue

        try:
            result = evaluate_model(model_name, model_type, test_loader, device, result_dir)
            if result:
                evaluation_results.append(result)
        except Exception as e:
            print(f"Error evaluating {model_name}: {e}")

    # Create a comparison report of all models
    comparison_dir = os.path.join(result_dir, 'model_comparison')
    Path(comparison_dir).mkdir(parents=True, exist_ok=True)

    # Compare model accuracies
    accuracies = []
    for result in evaluation_results:
        accuracies.append({
            'Model': result['name'],
            'Accuracy': result['accuracy']
        })

    if accuracies:
        accuracies_df = pd.DataFrame(accuracies)
        accuracies_df = accuracies_df.sort_values('Accuracy', ascending=False)
        accuracies_df.to_csv(os.path.join(comparison_dir, "accuracy_comparison.csv"), index=False)

        # Plot accuracy comparison
        plt.figure(figsize=(12, 6))
        sns.barplot(x='Model', y='Accuracy', data=accuracies_df)
        plt.title('Model Accuracy Comparison')
        plt.ylabel('Accuracy (%)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(comparison_dir, "accuracy_comparison.png"))
        plt.close()

    # Add ensemble model evaluation if not done yet
    try:
        # Check if ensemble was already evaluated
        ensemble_already_evaluated = any(result['name'] == 'ensemble' for result in evaluation_results)

        if not ensemble_already_evaluated:
            # Create and evaluate ensemble model
            basic_cnn = BasicCNN().to(device)
            deep_cnn = DeepCNN().to(device)
            resnet = ResNet().to(device)

            # Load the best models
            basic_cnn.load_state_dict(torch.load("../model/BasicCNN_BestParams/best_model.pth", weights_only=True, map_location=device))
            deep_cnn.load_state_dict(torch.load("../model/DeepCNN/best_model.pth", weights_only=True, map_location=device))
            resnet.load_state_dict(torch.load("../model/ResNet/best_model.pth", weights_only=True, map_location=device))

            ensemble_model = EnsembleModel([basic_cnn, deep_cnn, resnet]).to(device)

            # Create directory for ensemble results
            ensemble_eval_dir = os.path.join(result_dir, 'ensemble', 'evaluation')
            Path(ensemble_eval_dir).mkdir(parents=True, exist_ok=True)

            # Compute confusion matrix for ensemble
            cm, all_preds, all_labels = compute_confusion_matrix(ensemble_model, test_loader, device)

            # Save and plot confusion matrix
            plot_confusion_matrix(
                cm, CLASSES, normalize=False,
                title='Ensemble Model Confusion Matrix',
                save_path=os.path.join(ensemble_eval_dir, "confusion_matrix.png")
            )

            plot_confusion_matrix(
                cm, CLASSES, normalize=True,
                title='Ensemble Model Normalized Confusion Matrix',
                save_path=os.path.join(ensemble_eval_dir, "normalized_confusion_matrix.png")
            )

            # Calculate accuracy
            accuracy = 100 * (np.array(all_preds) == np.array(all_labels)).mean()

            print(f"Ensemble Model - Test Accuracy: {accuracy:.2f}%")

            # Add to comparison
            if accuracies:
                accuracies.append({
                    'Model': 'Ensemble Model',
                    'Accuracy': accuracy
                })
                accuracies_df = pd.DataFrame(accuracies)
                accuracies_df = accuracies_df.sort_values('Accuracy', ascending=False)
                accuracies_df.to_csv(os.path.join(comparison_dir, "accuracy_comparison.csv"), index=False)

                # Update plot
                plt.figure(figsize=(12, 6))
                sns.barplot(x='Model', y='Accuracy', data=accuracies_df)
                plt.title('Model Accuracy Comparison (with Ensemble)')
                plt.ylabel('Accuracy (%)')
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig(os.path.join(comparison_dir, "accuracy_comparison_with_ensemble.png"))
                plt.close()
    except Exception as e:
        print(f"Error evaluating ensemble model: {e}")

    print("\n===== Evaluation Complete =====")
    if accuracies:
        print("Model Accuracy Comparison:")
        print(accuracies_df)
        print(f"\nBest model: {accuracies_df.iloc[0]['Model']} with accuracy {accuracies_df.iloc[0]['Accuracy']:.2f}%")
