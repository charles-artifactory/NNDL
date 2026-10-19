"""Sample-weighted evaluation and classification metrics, all in NumPy."""
import numpy as np

from .network import loss_and_gradient, softmax


def classification_metrics(y_true, y_pred, num_classes=10):
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(confusion, (y_true, y_pred), 1)
    tp = np.diag(confusion).astype(float)
    support = confusion.sum(axis=1)
    predicted = confusion.sum(axis=0)
    precision = np.divide(tp, predicted, out=np.zeros_like(tp), where=predicted != 0)
    recall = np.divide(tp, support, out=np.zeros_like(tp), where=support != 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(tp), where=(precision + recall) != 0)
    return {"accuracy": float(tp.sum() / confusion.sum()), "macro_f1": float(f1.mean()),
            "confusion_matrix": confusion.tolist(), "precision": precision.tolist(),
            "recall": recall.tolist(), "f1": f1.tolist(), "support": support.tolist()}


def evaluate(model, x, y, loss="cross_entropy", smoothing=0.0, batch_size=1024, detailed=False):
    total_loss, total_ce, correct = 0.0, 0.0, 0
    predictions = []
    confidence = []
    for start in range(0, len(y), batch_size):
        xb, yb = x[start:start + batch_size], y[start:start + batch_size]
        logits = model.forward(xb)
        value, _ = loss_and_gradient(logits, yb, loss, smoothing)
        ce, _ = loss_and_gradient(logits, yb)
        pred = logits.argmax(axis=1)
        correct += int((pred == yb).sum())
        total_loss += value * len(yb)
        total_ce += ce * len(yb)
        if detailed:
            predictions.append(pred)
            confidence.append(softmax(logits).max(axis=1))
    result = {"data_loss": total_loss / len(y), "cross_entropy": total_ce / len(y),
              "accuracy": correct / len(y)}
    if detailed:
        result.update(classification_metrics(y, np.concatenate(predictions), model.sizes[-1]))
        result["mean_confidence"] = float(np.concatenate(confidence).mean())
    return result
