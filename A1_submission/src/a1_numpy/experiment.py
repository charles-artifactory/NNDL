"""Reproducible validation-first experiment pipeline.

No test images or labels are loaded until every model and selection decision
has been saved. Runs are fixed paths and refuse accidental evidence overwrite.
"""
import argparse
import csv
import hashlib
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .data import FILES, load_mnist, stratified_split
from .metrics import evaluate
from .network import MLP, loss_and_gradient
from .optim import Optimizer

ROOT = Path(__file__).resolve().parents[2]


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.json")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.csv")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def run_training(config, x_train, y_train, x_val, y_val, directory):
    directory.mkdir(parents=True, exist_ok=True)
    write_json(directory / "config.json", config)
    model = MLP([x_train.shape[1], *config["hidden_sizes"], 10], seed=config["seed"], dropout=config["dropout"])
    optimizer = Optimizer(config["optimizer"], config["learning_rate"])
    # Independent streams keep batch ordering identical across dropout settings.
    batch_rng = np.random.default_rng(config["seed"] + 1000)
    dropout_rng = np.random.default_rng(config["seed"] + 2000)
    best_accuracy, best_ce, best_epoch = -1.0, float("inf"), 0
    history = []
    started = time.perf_counter()
    for epoch in range(1, config["epochs"] + 1):
        order = batch_rng.permutation(len(y_train))
        online_loss = 0.0
        for start in range(0, len(order), config["batch_size"]):
            indices = order[start:start + config["batch_size"]]
            logits = model.forward(x_train[indices], training=True, rng=dropout_rng)
            value, gradient = loss_and_gradient(logits, y_train[indices], config["loss"], config["smoothing"])
            online_loss += (value + model.penalty(config["l2"])) * len(indices)
            grads = model.backward(gradient, config["l2"])
            optimizer.step(model.params, grads)
        # Evaluate both splits at end-of-epoch parameters with dropout disabled.
        train = evaluate(model, x_train, y_train, config["loss"], config["smoothing"])
        val = evaluate(model, x_val, y_val, config["loss"], config["smoothing"])
        row = {"epoch": epoch, "online_objective": online_loss / len(y_train),
               "train_loss": train["data_loss"], "validation_loss": val["data_loss"],
               "train_cross_entropy": train["cross_entropy"], "validation_cross_entropy": val["cross_entropy"],
               "train_accuracy": train["accuracy"], "validation_accuracy": val["accuracy"],
               "elapsed_seconds": time.perf_counter() - started}
        if not all(np.isfinite(value) for value in row.values()):
            raise FloatingPointError(f"Non-finite training history: {config['id']}")
        history.append(row)
        # Hard-label CE breaks accuracy ties even for different training losses.
        if (val["accuracy"], -val["cross_entropy"]) > (best_accuracy, -best_ce):
            best_accuracy, best_ce, best_epoch = val["accuracy"], val["cross_entropy"], epoch
            model.save(directory / "best_model.npz", {"config": config, "epoch": epoch})
        write_csv(directory / "history.csv", history)
        print(f"{config['id']} seed={config['seed']} epoch={epoch:02}/{config['epochs']} "
              f"train={train['accuracy']:.4f} val={val['accuracy']:.4f} "
              f"time={row['elapsed_seconds']:.1f}s", flush=True)
    result = {"experiment": config["id"], "seed": config["seed"],
              "best_epoch": best_epoch, "validation_accuracy": best_accuracy,
              "validation_cross_entropy": best_ce,
              "train_accuracy": history[best_epoch - 1]["train_accuracy"],
              "seconds": time.perf_counter() - started, "learning_rate": config["learning_rate"]}
    write_json(directory / "summary.json", result)
    return result


def resolved_config(settings, experiment, seed, learning_rate, epochs=None):
    return {"id": experiment["id"], "optimizer": experiment["optimizer"],
            "hidden_sizes": experiment.get("hidden_sizes", settings["hidden_sizes"]),
            "loss": experiment.get("loss", "cross_entropy"), "smoothing": experiment.get("smoothing", 0.0),
            "l2": experiment.get("l2", 0.0), "dropout": experiment.get("dropout", 0.0),
            "learning_rate": learning_rate, "seed": seed, "epochs": epochs or settings["epochs"],
            "batch_size": settings["batch_size"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/experiments.json")
    parser.add_argument("--output", type=Path, default=ROOT / "results")
    parser.add_argument("--data", type=Path, default=ROOT / "data")
    parser.add_argument("--resume", action="store_true", help="Reuse completed runs only if configuration and source match")
    args = parser.parse_args()
    settings = json.loads(args.config.read_text())
    source_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in sorted(Path(__file__).parent.glob("*.py")) if p.name not in ("report.py", "predict.py")}
    signature = hashlib.sha256(json.dumps({"settings": settings, "source": source_hashes}, sort_keys=True).encode()).hexdigest()
    manifest_path = args.output / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if not args.resume or manifest["signature"] != signature:
            raise SystemExit("Output already exists. Use --resume with identical source/configuration, or inspect it before replacing it.")
        if manifest.get("status") == "complete":
            print("All experiments are already complete.")
            return
    else:
        manifest = {"signature": signature, "settings": settings, "source_sha256": source_hashes,
                    "python": sys.version, "numpy": np.__version__, "platform": platform.platform(),
                    "executable": sys.executable, "base_prefix": sys.base_prefix,
                    "threads": {k: os.environ.get(k) for k in ("OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OMP_NUM_THREADS")},
                    "started_utc": datetime.now(timezone.utc).isoformat(), "status": "training",
                    "dataset_md5": FILES}
        write_json(manifest_path, manifest)
    x, y = load_mnist(args.data, train=True)
    train_indices, val_indices = stratified_split(y, settings["validation_size"], settings["split_seed"])
    np.savez_compressed(args.output / "split_indices.npz", train=train_indices, validation=val_indices)
    x_train, y_train, x_val, y_val = x[train_indices], y[train_indices], x[val_indices], y[val_indices]
    del x

    def train_or_resume(config, directory):
        if args.resume and (directory / "summary.json").exists():
            saved_config = json.loads((directory / "config.json").read_text())
            if saved_config != config or not (directory / "best_model.npz").exists():
                raise ValueError(f"Invalid completed run: {directory}")
            return json.loads((directory / "summary.json").read_text())
        return run_training(config, x_train, y_train, x_val, y_val, directory)

    # A small, predeclared learning-rate grid on validation data for each optimizer.
    tuning, learning_rates = [], {}
    for name, rates in settings["learning_rates"].items():
        candidates = []
        for rate in rates:
            identifier = f"tune_{name}_{rate:g}"
            config = resolved_config(settings, {"id": identifier, "optimizer": name}, settings["seeds"][0], rate, settings["tuning_epochs"])
            result = train_or_resume(config, args.output / "tuning" / identifier)
            candidates.append(result)
            tuning.append(result)
        best = max(candidates, key=lambda r: (r["validation_accuracy"], -r["validation_cross_entropy"]))
        learning_rates[name] = best["learning_rate"]
    write_csv(args.output / "tuning.csv", tuning)
    write_json(args.output / "selected_learning_rates.json", learning_rates)

    runs = []
    for experiment in settings["experiments"]:
        for seed in settings["seeds"]:
            config = resolved_config(settings, experiment, seed, learning_rates[experiment["optimizer"]])
            directory = args.output / "runs" / experiment["id"] / f"seed_{seed}"
            runs.append(train_or_resume(config, directory))

    # Freeze all choices before loading test data. Test rankings never select a model.
    validation_means = {e["id"]: float(np.mean([r["validation_accuracy"] for r in runs if r["experiment"] == e["id"]]))
                        for e in settings["experiments"]}
    winner = max(validation_means, key=validation_means.get)
    selected_seed = settings["seeds"][0]  # Predeclared seed; do not choose by test accuracy.
    freeze = {"selected_experiment": winner, "selected_seed": selected_seed,
              "validation_mean_accuracy": validation_means, "selection_rule": "maximum mean validation accuracy; first listed experiment breaks ties",
              "frozen_utc": datetime.now(timezone.utc).isoformat()}
    freeze_path = args.output / "selection.json"
    if freeze_path.exists() and args.resume:
        previous = json.loads(freeze_path.read_text())
        if any(previous[key] != freeze[key] for key in ("selected_experiment", "selected_seed", "validation_mean_accuracy")):
            raise ValueError("Frozen model selection changed")
        freeze = previous
    write_json(freeze_path, freeze)
    print(f"Selection frozen: {winner}; now loading held-out test data.", flush=True)
    x_test, y_test = load_mnist(args.data, train=False)
    for row in runs:
        directory = args.output / "runs" / row["experiment"] / f"seed_{row['seed']}"
        model, metadata = MLP.load(directory / "best_model.npz")
        config = metadata["config"]
        test_path = directory / "test_metrics.json"
        if args.resume and test_path.exists():
            metrics = json.loads(test_path.read_text())
        else:
            metrics = evaluate(model, x_test, y_test, config["loss"], config["smoothing"], detailed=True)
            write_json(test_path, metrics)
        row.update({"test_accuracy": metrics["accuracy"], "test_macro_f1": metrics["macro_f1"],
                    "test_cross_entropy": metrics["cross_entropy"], "test_confidence": metrics["mean_confidence"]})
        if row["experiment"] == winner and row["seed"] == selected_seed:
            model.save(args.output / "best_model.npz", metadata)
            write_json(args.output / "test_metrics.json", metrics)
    write_csv(args.output / "summary.csv", runs)
    manifest.update({"status": "complete", "completed_utc": datetime.now(timezone.utc).isoformat(),
                     "train_size": len(y_train), "validation_size": len(y_val), "test_size": len(y_test)})
    write_json(manifest_path, manifest)
    print(f"Completed: {len(runs)} final runs; results at {args.output}", flush=True)


if __name__ == "__main__":
    main()
