import os
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, roc_curve, auc
from torchvision import datasets, transforms
from fcnn import FullyConnectedNN

# Constants
CLASSES = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
INPUT_SIZE = 32 * 32 * 3
HIDDEN_SIZES = [1024, 512, 256, 128]
NUM_CLASSES = 10
BATCH_SIZE = 256


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion Matrix',
                          save_to='confusion_matrix.png', figsize=(10, 8), cmap=plt.cm.Blues):
    """
    Plot and save the confusion matrix.

    Parameters:
        cm: Confusion matrix
        classes: List of class names
        normalize: Whether to normalize the confusion matrix
        title: Title for the plot
        save_to: Path to save the resulting figure
        figsize: Size of the figure
        cmap: Color map to use
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


def compute_per_class_metrics(y_true, y_pred, classes):
    """
    Compute precision, recall, and F1 score for each class.

    Parameters:
        y_true: True labels
        y_pred: Predicted labels
        classes: List of class names

    Returns:
        DataFrame containing precision, recall, and F1 score for each class
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


def compute_roc_auc(model, test_loader, device, num_classes):
    """
    Compute ROC curve and AUC for each class (one-vs-rest approach).

    Parameters:
        model: The trained model
        test_loader: DataLoader for test data
        device: Device to run inference on
        num_classes: Number of classes

    Returns:
        fpr, tpr, roc_auc dictionaries for each class
    """
    # Get all predictions and true labels
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

    # Compute ROC curve and AUC for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(num_classes):
        # Convert to binary classification problem (one-vs-rest)
        y_true = (all_labels == i).astype(int)
        y_score = all_probs[:, i]

        fpr[i], tpr[i], _ = roc_curve(y_true, y_score)
        roc_auc[i] = auc(fpr[i], tpr[i])

    return fpr, tpr, roc_auc


def plot_roc_curves(fpr, tpr, roc_auc, classes, save_to='roc_curves.png', figsize=(12, 10)):
    """
    Plot ROC curves for all classes.

    Parameters:
        fpr: False positive rates for each class
        tpr: True positive rates for each class
        roc_auc: AUC values for each class
        classes: List of class names
        save_to: Path to save the resulting figure
        figsize: Size of the figure
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


def evaluate_best_model(model_name, model_dir, result_dir, test_loader, device):
    """
    Load the best model and evaluate it on the test set.

    Parameters:
        model_name: Name of the model
        model_dir: Directory containing the saved model
        result_dir: Directory to save the evaluation results
        test_loader: DataLoader for test data
        device: Device to run inference on

    Returns:
        tuple: (accuracy, confusion matrix, class metrics)
    """
    # Make sure the result directory exists
    Path(result_dir).mkdir(parents=True, exist_ok=True)

    # Build model path
    model_path = os.path.join(model_dir, "best_model.pth")

    if not os.path.exists(model_path):
        print(f"Best model file does not exist: {model_path}")
        return None, None, None

    # Create model instance
    model = FullyConnectedNN(INPUT_SIZE, HIDDEN_SIZES, NUM_CLASSES).to(device)

    # Load model weights
    print(f"Loading model weights from {model_path}")
    model.load_state_dict(torch.load(model_path, weights_only=True, map_location=device))
    model.eval()

    # Collect all predictions and true labels
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

    # Calculate overall accuracy
    accuracy = 100 * (np.array(all_preds) == np.array(all_true)).mean()

    # Compute confusion matrix
    cm = confusion_matrix(all_true, all_preds)

    # Save raw confusion matrix as CSV
    cm_df = pd.DataFrame(cm, index=CLASSES, columns=CLASSES)
    cm_df.to_csv(os.path.join(result_dir, "confusion_matrix.csv"))

    # Plot confusion matrix
    plot_confusion_matrix(cm, CLASSES, title=f'{model_name} Confusion Matrix',
                          save_to=os.path.join(result_dir, "confusion_matrix.png"))

    # Plot normalized confusion matrix
    plot_confusion_matrix(cm, CLASSES, normalize=True, title=f'{model_name} Normalized Confusion Matrix',
                          save_to=os.path.join(result_dir, "normalized_confusion_matrix.png"))

    # Compute per-class metrics
    class_metrics = compute_per_class_metrics(all_true, all_preds, CLASSES)
    class_metrics.to_csv(os.path.join(result_dir, "class_metrics.csv"), index=False)

    # Plot class metrics
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

    # Compute ROC curves and AUC
    fpr, tpr, roc_auc = compute_roc_auc(model, test_loader, device, NUM_CLASSES)

    # Save AUC values
    auc_df = pd.DataFrame({
        'Class': CLASSES,
        'AUC': [roc_auc[i] for i in range(NUM_CLASSES)]
    })
    auc_df.to_csv(os.path.join(result_dir, "auc_values.csv"), index=False)

    # Plot ROC curves
    plot_roc_curves(fpr, tpr, roc_auc, CLASSES, save_to=os.path.join(result_dir, "roc_curves.png"))

    print(f"Model {model_name} evaluated:")
    print(f"Overall Accuracy: {accuracy:.2f}%")
    print(f"Results saved to {result_dir}")

    return accuracy, cm, class_metrics


def evaluate_all_best_models(base_model_dir="../model", base_result_dir="../result", batch_size=256):
    """
    Evaluate all best models and compare them.

    Parameters:
        base_model_dir: Base directory containing model subdirectories
        base_result_dir: Base directory for saving results
        batch_size: Batch size for evaluation
    """
    # Detect device
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    print(f"Using device: {device} for evaluation")

    # Prepare the test dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    test_dataset = datasets.CIFAR10(root='../data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    # Create a folder for combined results
    combined_result_dir = os.path.join(base_result_dir, "combined_evaluation")
    Path(combined_result_dir).mkdir(parents=True, exist_ok=True)

    # Find all model directories that contain best_model.pth
    model_results = []
    for model_dir in os.listdir(base_model_dir):
        full_model_dir = os.path.join(base_model_dir, model_dir)
        best_model_path = os.path.join(full_model_dir, "best_model.pth")

        if os.path.isdir(full_model_dir) and os.path.exists(best_model_path):
            print(f"Found best model in {model_dir}, evaluating...")

            # Create result directory for this model
            model_result_dir = os.path.join(base_result_dir, model_dir, "evaluation")

            # Evaluate the model
            accuracy, _, _ = evaluate_best_model(model_dir, full_model_dir, model_result_dir, test_loader, device)

            # Store results for comparison
            if accuracy is not None:
                model_results.append({
                    'Model': model_dir,
                    'Accuracy': accuracy
                })

    if model_results:
        # Create comparison dataframe
        comparison_df = pd.DataFrame(model_results)
        comparison_df = comparison_df.sort_values(by='Accuracy', ascending=False)
        comparison_df.to_csv(os.path.join(combined_result_dir, "model_comparison.csv"), index=False)

        # Plot comparison
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
