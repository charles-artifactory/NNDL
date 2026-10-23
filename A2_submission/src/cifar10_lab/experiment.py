"""Training orchestration. The test split is inaccessible to model selection."""
import argparse
import csv
import hashlib
import itertools
import json
import os
import platform
import random
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
import torchvision
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from torch import nn
from torch.utils.data import DataLoader

from .data import CLASSES, TensorImages, batch_transform, load_data, occlude
from .models import Classifier, ModelSpec, parameter_count, receptive_field

ROOT = Path(__file__).resolve().parents[2]


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = ROOT / "tmp" / f"{path.name}.partial"
    temporary.parent.mkdir(parents=True, exist_ok=True)
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    json.loads(temporary.read_text())
    os.replace(temporary, path)


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = ROOT / "tmp" / f"{path.name}.partial"
    with temporary.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    os.replace(temporary, path)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def choose_device(name):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def loader(images, labels, indices, batch_size, shuffle=False, seed=42):
    return DataLoader(TensorImages(images, labels, indices), batch_size=batch_size,
                      shuffle=shuffle, num_workers=0,
                      generator=torch.Generator().manual_seed(seed))


def epoch_pass(model, batches, device, optimizer=None, cutout=False):
    training = optimizer is not None
    model.train(training)
    loss_sum = torch.zeros((), device=device)
    correct = torch.zeros((), device=device)
    total = 0
    with torch.set_grad_enabled(training):
        for images, labels in batches:
            x = batch_transform(images.to(device), training=training, cutout=cutout)
            y = labels.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = nn.functional.cross_entropy(logits, y)
            if training:
                loss.backward()
                optimizer.step()
            loss_sum += loss.detach() * len(y)
            correct += (logits.argmax(1) == y).sum()
            total += len(y)
    return {"loss": float(loss_sum.cpu()) / total, "accuracy": float(correct.cpu()) / total}


# 基础题：训练网络，记录每轮指标，并按验证准确率保存权重。
def train_run(spec, train_batches, val_batches, config, device, directory,
              epochs, seed, cutout=False, resume=False):
    directory.mkdir(parents=True, exist_ok=True)
    summary_path = directory / "summary.json"
    if resume and summary_path.exists():
        summary = json.loads(summary_path.read_text())
        expected = {"model": spec.to_dict(), "epochs": epochs, "seed": seed, "cutout": cutout,
                    "learning_rate": config["learning_rate"], "weight_decay": config["weight_decay"]}
        if summary["protocol"] != expected:
            raise ValueError(f"Existing run has a different protocol: {directory}")
        return summary
    seed_all(seed)
    model = Classifier(spec).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"],
                                 weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    best_accuracy, best_epoch = -1.0, 0
    history = []
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        t0 = time.perf_counter()
        train = epoch_pass(model, train_batches, device, optimizer, cutout)
        val = epoch_pass(model, val_batches, device)
        row = {"epoch": epoch, "learning_rate": optimizer.param_groups[0]["lr"],
               "train_loss": train["loss"], "train_accuracy": train["accuracy"],
               "validation_loss": val["loss"], "validation_accuracy": val["accuracy"],
               "seconds": time.perf_counter() - t0}
        history.append(row)
        if val["accuracy"] > best_accuracy:
            best_accuracy, best_epoch = val["accuracy"], epoch
            temporary = ROOT / "tmp" / "checkpoint.partial"
            torch.save({"state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                        "spec": spec.to_dict(), "epoch": epoch}, temporary)
            os.replace(temporary, directory / "best.pt")
        scheduler.step()
        write_csv(directory / "history.csv", history)
        print(f"{directory.name} epoch={epoch}/{epochs} train={train['accuracy']:.4f} "
              f"val={val['accuracy']:.4f} seconds={row['seconds']:.1f}", flush=True)
    # Clean, deterministic train-subset accuracy is comparable to validation accuracy.
    state = torch.load(directory / "best.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state["state_dict"])
    train_clean = epoch_pass(model, train_batches, device)
    summary = {"protocol": {"model": spec.to_dict(), "epochs": epochs, "seed": seed,
                            "cutout": cutout, "learning_rate": config["learning_rate"],
                            "weight_decay": config["weight_decay"]},
               "parameters": parameter_count(model), "receptive_field": receptive_field(spec),
               "best_epoch": best_epoch, "best_validation_accuracy": best_accuracy,
               "clean_train_accuracy": train_clean["accuracy"],
               "seconds": time.perf_counter() - started}
    write_json(summary_path, summary)
    del model, optimizer
    if device.type == "mps":
        torch.mps.empty_cache()
    return summary


# 基础题：比较 15 组归一化与 Dropout 配置，按 3 折平均准确率选择配置。
def cross_validate(images, labels, train_indices, config, device, resume):
    cv_indices, _ = next(StratifiedShuffleSplit(
        n_splits=1, train_size=config["cv_samples"], random_state=config["seed"]
    ).split(np.zeros(len(train_indices)), labels.numpy()[train_indices]))
    selected_indices = train_indices[cv_indices]
    write_json(ROOT / "results" / "cv_indices.json", selected_indices.tolist())
    kfold = StratifiedKFold(config["cv_folds"], shuffle=True, random_state=config["seed"])
    folds = list(kfold.split(selected_indices, labels.numpy()[selected_indices]))
    rows = []
    for norm, dropout in itertools.product(config["normalizations"], config["dropout_rates"]):
        spec = ModelSpec(normalization=norm, dropout=dropout, width=config["width"])
        scores = []
        for fold, (tr, va) in enumerate(folds):
            name = f"{norm}_dropout_{dropout:g}_fold_{fold}"
            result = train_run(
                spec,
                loader(images, labels, selected_indices[tr], config["cv_batch_size"], True, config["seed"] + fold),
                loader(images, labels, selected_indices[va], config["cv_batch_size"]),
                config, device, ROOT / "results" / "cv" / name, config["cv_epochs"],
                config["seed"] + fold, resume=resume)
            scores.append(result["best_validation_accuracy"])
        row = {"normalization": norm, "dropout": dropout, "fold_accuracies": scores,
               "mean_accuracy": float(np.mean(scores)), "std_accuracy": float(np.std(scores, ddof=1))}
        rows.append(row)
        write_json(ROOT / "results" / "cross_validation.json", rows)
    # Stable tie-breaks follow the declared grid order. Test data is never accessed.
    selected = max(rows, key=lambda r: r["mean_accuracy"])
    write_json(ROOT / "results" / "selection.json", {
        "criterion": "maximum mean of fold-best validation accuracies; grid order breaks ties",
        "selected": selected, "cv_samples": len(selected_indices), "folds": config["cv_folds"],
        "epochs_per_fold": config["cv_epochs"], "test_used": False})
    return ModelSpec(normalization=selected["normalization"], dropout=selected["dropout"], width=config["width"])


# 基础题及附加题 1、2：建立 Dropout、归一化、深度、残差、遮挡和卷积核对照。
def final_specs(best):
    # Guarantee a positive-dropout final comparison even when CV selects p=0.
    control_dropout = 0.2 if best.dropout == 0 else 0.0
    positive_dropout = best.dropout if best.dropout > 0 else 0.2
    return {
        "baseline": (replace(best, normalization="none", dropout=0), False),
        "selected": (best, False),
        "dropout_variant": (replace(best, dropout=control_dropout), False),
        "no_normalization": (replace(best, normalization="none", dropout=positive_dropout), False),
        "kernel5": (replace(best, kernel_size=5), False),
        "kernel7": (replace(best, kernel_size=7), False),
        "deep": (replace(best, depth=2), False),
        "residual": (replace(best, depth=2, residual=True), False),
        "residual_cutout": (replace(best, depth=2, residual=True), True),
    }


def load_model(path, device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = Classifier(ModelSpec(**checkpoint["spec"]))
    model.load_state_dict(checkpoint["state_dict"])
    return model.to(device).eval()


def metrics(probabilities, labels):
    predictions = probabilities.argmax(1)
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, predictions, labels=np.arange(10), zero_division=0)
    return {"accuracy": float(np.mean(predictions == labels)), "macro_f1": float(f1.mean()),
            "n": len(labels), "confusion_matrix": confusion_matrix(labels, predictions, labels=np.arange(10)).tolist(),
            "per_class": [{"class": CLASSES[i], "precision": float(precision[i]),
                           "recall": float(recall[i]), "f1": float(f1[i]), "support": int(support[i])}
                          for i in range(10)]}


@torch.no_grad()
# 附加题 3：分别计算完整图像与中心遮挡图像的预测。
def predict(model, batches, device, occlusion_size):
    outputs = {"clean": [], "occluded": []}
    for images, _ in batches:
        x = batch_transform(images.to(device))
        outputs["clean"].append(model(x).softmax(1).cpu().numpy())
        outputs["occluded"].append(model(occlude(x, occlusion_size)).softmax(1).cpu().numpy())
    return {k: np.concatenate(v) for k, v in outputs.items()}


# 基础题及附加题 1、3：计算分类指标，并平均多个模型的概率得到集成预测。
def evaluate_all(test_images, test_labels, config, device):
    plan = json.loads((ROOT / "results" / "evaluation_plan.json").read_text())
    batches = loader(test_images, test_labels, np.arange(len(test_labels)), config["batch_size"])
    labels = test_labels.numpy()
    all_probabilities, results = {}, {}
    for name in plan["models"]:
        model = load_model(ROOT / "results" / "runs" / name / "best.pt", device)
        probabilities = predict(model, batches, device, config["occlusion_size"])
        all_probabilities[name] = probabilities
        results[name] = {k: metrics(v, labels) for k, v in probabilities.items()}
        np.savez_compressed(ROOT / "results" / "runs" / name / "predictions.npz",
                            labels=labels, **probabilities)
        print(f"测试 {name}: clean（完整图像准确率）={results[name]['clean']['accuracy']:.4f} "
              f"occluded（中心遮挡准确率）={results[name]['occluded']['accuracy']:.4f}", flush=True)
        del model
    ensemble = {k: np.mean([all_probabilities[n][k] for n in plan["ensemble_members"]], axis=0)
                for k in ("clean", "occluded")}
    results["ensemble"] = {k: metrics(v, labels) for k, v in ensemble.items()}
    np.savez_compressed(ROOT / "results" / "ensemble_predictions.npz", labels=labels, **ensemble)
    write_json(ROOT / "results" / "test_metrics.json", results)
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=ROOT / "configs" / "experiment.json")
    p.add_argument("--stage", choices=["all", "cv", "train", "evaluate"], default="all")
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    config = json.loads(args.config.read_text())
    (ROOT / "tmp").mkdir(parents=True, exist_ok=True)
    (ROOT / "results").mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(config["threads"])
    device = choose_device(config["device"])
    fingerprint = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    manifest_path = ROOT / "results" / "manifest.json"
    if args.resume and manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        if previous["config_sha256"] != fingerprint:
            raise ValueError("Config changed. Use a separate project or explicitly rerun without --resume.")
    if not (args.resume and manifest_path.exists()):
        write_json(manifest_path, {"config": config, "config_sha256": fingerprint,
                                 "python": sys.version, "torch": torch.__version__,
                                 "torchvision": torchvision.__version__, "platform": platform.platform(),
                                 "device": str(device), "strict_bitwise_reproducibility": False})
    images, labels, test_images, test_labels, train_idx, val_idx = load_data(ROOT / "data", config["seed"])
    write_json(ROOT / "results" / "split.json", {"seed": config["seed"],
                                                "train_indices": train_idx.tolist(),
                                                "validation_indices": val_idx.tolist(),
                                                "test_size": len(test_labels)})
    if args.stage in ("all", "cv"):
        best = cross_validate(images, labels, train_idx, config, device, args.resume)
    else:
        selected = json.loads((ROOT / "results" / "selection.json").read_text())["selected"]
        best = ModelSpec(normalization=selected["normalization"], dropout=selected["dropout"], width=config["width"])
    specs = final_specs(best)
    if args.stage in ("all", "train"):
        # Freeze every model and the equal-weight ensemble before seeing test metrics.
        write_json(ROOT / "results" / "evaluation_plan.json", {
            "models": {n: {"spec": s.to_dict(), "cutout": c} for n, (s, c) in specs.items()},
            "ensemble_members": config["ensemble_members"], "ensemble_rule": "mean softmax probabilities",
            "test_used_for_selection": False, "occlusion": {"location": "center", "size": config["occlusion_size"]}})
        for name, (spec, cutout) in specs.items():
            train_run(spec, loader(images, labels, train_idx, config["batch_size"], True, config["seed"]),
                      loader(images, labels, val_idx, config["batch_size"]), config, device,
                      ROOT / "results" / "runs" / name, config["epochs"], config["seed"], cutout, args.resume)
    if args.stage in ("all", "evaluate"):
        evaluate_all(test_images, test_labels, config, device)


if __name__ == "__main__":
    main()
