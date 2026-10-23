"""Check experiment artifacts against independent saved-prediction calculations."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .data import batch_transform
from .experiment import ROOT, load_model, write_json


def read(path):
    return json.loads(path.read_text())


def main():
    torch.set_num_threads(2)
    config = read(ROOT / "results" / "manifest.json")["config"]
    split = read(ROOT / "results" / "split.json")
    train, val = set(split["train_indices"]), set(split["validation_indices"])
    assert len(train) == 40000 and len(val) == 10000
    assert not train & val and train | val == set(range(50000))
    cv_indices = read(ROOT / "results" / "cv_indices.json")
    assert len(set(cv_indices)) == config["cv_samples"]
    assert set(cv_indices) <= train
    with np.load(ROOT / "data" / "cifar10.npz", allow_pickle=False) as data:
        y = data["train_labels"]
        assert np.all(np.bincount(y[split["train_indices"]], minlength=10) == 4000)
        assert np.all(np.bincount(y[split["validation_indices"]], minlength=10) == 1000)
        assert np.all(np.bincount(y[cv_indices], minlength=10) == config["cv_samples"] // 10)
        first_images = torch.from_numpy(data["test_images"][:32].copy()).permute(0, 3, 1, 2)
    cv = read(ROOT / "results" / "cross_validation.json")
    assert len(cv) == len(config["normalizations"]) * len(config["dropout_rates"])
    assert {(r["normalization"], r["dropout"]) for r in cv} == {
        (n, p) for n in config["normalizations"] for p in config["dropout_rates"]}
    completed_folds = 0
    for row in cv:
        assert len(row["fold_accuracies"]) == config["cv_folds"]
        assert np.isclose(np.mean(row["fold_accuracies"]), row["mean_accuracy"])
        assert np.isclose(np.std(row["fold_accuracies"], ddof=1), row["std_accuracy"])
        for fold, accuracy in enumerate(row["fold_accuracies"]):
            directory = ROOT / "results" / "cv" / f"{row['normalization']}_dropout_{row['dropout']:g}_fold_{fold}"
            summary = read(directory / "summary.json")
            with (directory / "history.csv").open() as f:
                history = list(csv.DictReader(f))
            assert len(history) == config["cv_epochs"]
            assert [int(r["epoch"]) for r in history] == list(range(1, config["cv_epochs"] + 1))
            assert summary["protocol"]["epochs"] == config["cv_epochs"]
            assert summary["protocol"]["seed"] == config["seed"] + fold
            assert summary["protocol"]["model"]["normalization"] == row["normalization"]
            assert summary["protocol"]["model"]["dropout"] == row["dropout"]
            assert np.isclose(accuracy, summary["best_validation_accuracy"])
            assert np.isclose(accuracy, max(float(r["validation_accuracy"]) for r in history))
            assert (directory / "best.pt").is_file()
            completed_folds += 1
    selection = read(ROOT / "results" / "selection.json")
    assert selection["selected"] == max(cv, key=lambda r: r["mean_accuracy"])
    assert selection["test_used"] is False
    plan = read(ROOT / "results" / "evaluation_plan.json")
    assert plan["test_used_for_selection"] is False
    results = read(ROOT / "results" / "test_metrics.json")
    assert set(results) == set(plan["models"]) | {"ensemble"}
    all_preds = {}
    summaries = {}
    for name in plan["models"]:
        directory = ROOT / "results" / "runs" / name
        summary = read(directory / "summary.json")
        summaries[name] = summary
        with (directory / "history.csv").open() as f:
            history = list(csv.DictReader(f))
        assert len(history) == config["epochs"]
        assert [int(r["epoch"]) for r in history] == list(range(1, config["epochs"] + 1))
        best = max(history, key=lambda r: float(r["validation_accuracy"]))
        assert int(best["epoch"]) == summary["best_epoch"]
        assert float(best["validation_accuracy"]) == summary["best_validation_accuracy"]
        assert summary["protocol"]["model"] == plan["models"][name]["spec"]
        all_preds[name] = np.load(directory / "predictions.npz", allow_pickle=False)
        checkpoint = torch.load(directory / "best.pt", map_location="cpu", weights_only=True)
        assert checkpoint["epoch"] == summary["best_epoch"]
        model = load_model(directory / "best.pt", "cpu")
        with torch.no_grad():
            cpu_probabilities = model(batch_transform(first_images)).softmax(1).numpy()
        assert np.allclose(cpu_probabilities, all_preds[name]["clean"][:32], atol=1e-3), name
    all_preds["ensemble"] = np.load(ROOT / "results" / "ensemble_predictions.npz", allow_pickle=False)
    for name, data in all_preds.items():
        y = data["labels"]
        assert len(y) == 10000 and np.all(np.bincount(y) == 1000)
        for condition in ["clean", "occluded"]:
            probabilities = data[condition]
            assert probabilities.shape == (10000, 10)
            assert np.all(np.isfinite(probabilities)) and np.all(probabilities >= 0)
            assert np.allclose(probabilities.sum(1), 1, atol=1e-5)
            predicted = probabilities.argmax(1)
            matrix = np.zeros((10, 10), dtype=int)
            np.add.at(matrix, (y, predicted), 1)
            metric = results[name][condition]
            assert np.array_equal(matrix, metric["confusion_matrix"])
            assert np.isclose((predicted == y).mean(), metric["accuracy"])
            f1 = 2 * matrix.diagonal() / (matrix.sum(1) + matrix.sum(0))
            assert np.isclose(f1.mean(), metric["macro_f1"])
    for condition in ["clean", "occluded"]:
        expected = np.mean([all_preds[n][condition] for n in plan["ensemble_members"]], axis=0)
        assert np.allclose(expected, all_preds["ensemble"][condition], atol=1e-7)
    # Coursework coverage is checked separately from numerical consistency.
    specs = {n: item["spec"] for n, item in plan["models"].items()}
    selected, dropout_variant = specs["selected"], specs["dropout_variant"]
    assert min(selected["dropout"], dropout_variant["dropout"]) == 0
    assert max(selected["dropout"], dropout_variant["dropout"]) > 0
    assert {k: v for k, v in selected.items() if k != "dropout"} == {
        k: v for k, v in dropout_variant.items() if k != "dropout"}
    assert {specs[n]["kernel_size"] for n in ("selected", "kernel5", "kernel7")} == {3, 5, 7}
    for name in ("kernel5", "kernel7"):
        assert {k: v for k, v in specs[name].items() if k != "kernel_size"} == {
            k: v for k, v in selected.items() if k != "kernel_size"}
    assert specs["deep"]["depth"] > selected["depth"]
    assert specs["residual"]["residual"] is True
    assert specs["residual"] == specs["residual_cutout"]
    residual_protocol = summaries["residual"]["protocol"]
    cutout_protocol = summaries["residual_cutout"]["protocol"]
    assert residual_protocol["cutout"] is False and cutout_protocol["cutout"] is True
    assert {k: v for k, v in residual_protocol.items() if k != "cutout"} == {
        k: v for k, v in cutout_protocol.items() if k != "cutout"}
    report = (ROOT / "REPORT.md").read_text()
    for section in ("## 3. 模型与训练设置", "## 4. 交叉验证与超参数选择",
                    "### 5.1 正则化消融", "### 5.2 深度、残差与集成",
                    "## 6. 卷积核大小与特征表征", "## 7. 模型缺陷与实验验证"):
        assert section in report
    feature_analysis = read(ROOT / "results" / "feature_analysis.json")
    assert set(feature_analysis["representation"]) == {"selected", "kernel5", "kernel7"}
    requirement_coverage = {
        "basic_cifar10_cnn": {"status": "complete", "evidence": ["src/cifar10_lab/models.py", "results/test_metrics.json", "REPORT.md"]},
        "basic_dropout_and_normalization": {"status": "complete", "normalizations": config["normalizations"],
                                             "dropout_rates": config["dropout_rates"],
                                             "evidence": ["results/cross_validation.json", "results/runs/dropout_variant/", "REPORT.md"]},
        "basic_cross_validation": {"status": "complete", "configurations": len(cv), "completed_folds": completed_folds,
                                    "evidence": ["results/cv/", "results/selection.json", "REPORT.md"]},
        "bonus_performance_methods": {"status": "complete", "methods": ["depth", "residual", "cutout", "ensemble"],
                                        "evidence": ["results/model_comparison.csv", "REPORT.md"]},
        "bonus_kernel_visualization": {"status": "complete", "kernel_sizes": [3, 5, 7],
                                        "evidence": ["figures/kernel_weights.png", "figures/features_early.png", "figures/features_late.png",
                                                     "figures/input_sensitivity.png", "results/feature_analysis.json", "REPORT.md"]},
        "bonus_limitations_and_solutions": {"status": "complete",
                                             "evidence": ["results/paired_intervals.json", "figures/confusion_matrix.png",
                                                          "figures/misclassifications.png", "REPORT.md"]},
    }
    figures = list((ROOT / "figures").glob("*.png"))
    assert len(figures) == 9
    for path in figures:
        with Image.open(path) as image:
            assert min(image.size) > 300
            image.verify()
    notebook = read(ROOT / "main.ipynb")
    assert notebook["nbformat"] == 4
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), "notebook", "exec")
            assert not any(out.get("output_type") == "error" for out in cell.get("outputs", []))
    write_json(ROOT / "results" / "verification.json", {
        "status": "passed",
        "checks": ["stratified disjoint split", "CV restricted to train subset", "CV selection recomputed",
                   "all CV fold histories complete and consistent", "all final training histories complete",
                   "checkpoint epoch selected by validation", "prediction shape and probability sum",
                   "confusion matrices and macro F1 independently recomputed", "equal-weight ensemble recomputed",
                   "CPU checkpoint inference agrees with saved predictions for 32 fixed images",
                   "nine PNGs decoded", "notebook code compiled"],
        "final_models": len(plan["models"]), "test_images": 10000,
        "requirement_coverage": requirement_coverage,
        "notebook_executed_cells": sum(c["cell_type"] == "code" and c.get("execution_count") is not None
                                       for c in notebook["cells"]),
        "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in sorted((ROOT / "src").rglob("*.py"))},
        "figures": [p.name for p in sorted(figures)]})
    print("Artifact verification passed.")


if __name__ == "__main__":
    main()
