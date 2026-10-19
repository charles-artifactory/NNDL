"""Explicit dense forward/backward passes and loss derivatives."""
import json
from pathlib import Path

import numpy as np


def softmax(logits):
    shifted = logits - logits.max(axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / exponentials.sum(axis=1, keepdims=True)


def loss_and_gradient(logits, labels, loss="cross_entropy", smoothing=0.0):
    """Return mean data loss and d(loss)/d(logits), including 1/N once.

    Label smoothing convention: q = (1-alpha)*one_hot + alpha/K.
    MSE convention: mean_i 0.5 * sum_k (softmax(z)_ik - q_ik)^2.
    """
    if loss not in ("cross_entropy", "mse"):
        raise ValueError(f"Unknown loss: {loss}")
    if not 0 <= smoothing < 1:
        raise ValueError("smoothing must lie in [0, 1)")
    n, k = logits.shape
    target = np.full_like(logits, smoothing / k)
    target[np.arange(n), labels] += 1 - smoothing
    shifted = logits - logits.max(axis=1, keepdims=True)
    log_probs = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
    probabilities = np.exp(log_probs)
    if loss == "cross_entropy":
        return float(-(target * log_probs).sum() / n), (probabilities - target) / n
    residual = probabilities - target
    value = 0.5 * np.square(residual).sum() / n
    # Softmax Jacobian-vector product; do not treat probabilities as logits.
    gradient = probabilities * (residual - (residual * probabilities).sum(axis=1, keepdims=True)) / n
    return float(value), gradient


class MLP:
    def __init__(self, sizes=(784, 128, 64, 10), seed=42, dropout=0.0, dtype=np.float32):
        if len(sizes) < 2 or any(s <= 0 for s in sizes):
            raise ValueError("sizes requires positive input/output widths")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must lie in [0, 1)")
        self.sizes = list(sizes)
        self.dropout = dropout
        self.params = {}
        rng = np.random.default_rng(seed)
        for i, (fan_in, fan_out) in enumerate(zip(sizes[:-1], sizes[1:])):
            # He initialization for hidden ReLU layers; output variance is 1/fan_in.
            scale = np.sqrt((2 if i < len(sizes) - 2 else 1) / fan_in)
            self.params[f"W{i}"] = (rng.standard_normal((fan_in, fan_out)) * scale).astype(dtype)
            self.params[f"b{i}"] = np.zeros(fan_out, dtype=dtype)
        self.cache = None

    def forward(self, x, training=False, rng=None):
        cache = []
        a = x
        depth = len(self.sizes) - 1
        for i in range(depth):
            z = a @ self.params[f"W{i}"] + self.params[f"b{i}"]
            mask = None
            cache.append([a, z, mask])
            if i < depth - 1:
                a = np.maximum(z, 0)
                if training and self.dropout:
                    if rng is None:
                        raise ValueError("Training with dropout requires an explicit RNG")
                    mask = (rng.random(a.shape) >= self.dropout).astype(a.dtype) / (1 - self.dropout)
                    a *= mask
                    cache[-1][2] = mask
            else:
                a = z
        if training:
            self.cache = cache
        return a

    def backward(self, gradient, l2=0.0):
        if self.cache is None:
            raise RuntimeError("Call forward(training=True) before backward")
        grads = {}
        for i in reversed(range(len(self.cache))):
            inputs, z, mask = self.cache[i]
            if i < len(self.cache) - 1:
                if mask is not None:
                    gradient = gradient * mask
                gradient = gradient * (z > 0)
            grads[f"W{i}"] = inputs.T @ gradient + l2 * self.params[f"W{i}"]
            grads[f"b{i}"] = gradient.sum(axis=0)
            gradient = gradient @ self.params[f"W{i}"].T
        self.cache = None
        return grads

    def penalty(self, l2):
        # Biases are intentionally not penalized.
        return 0.5 * l2 * sum(float(np.square(v).sum()) for k, v in self.params.items() if k.startswith("W"))

    def save(self, path: Path, metadata=None):
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix(".tmp.npz")
        config = {"sizes": self.sizes, "dropout": self.dropout, "metadata": metadata or {}}
        np.savez_compressed(temp, **self.params, config=json.dumps(config))
        temp.replace(path)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as state:
            config = json.loads(str(state["config"]))
            model = cls(config["sizes"], dropout=config["dropout"])
            for key, value in model.params.items():
                loaded = state[key]
                if loaded.shape != value.shape:
                    raise ValueError(f"Invalid parameter shape: {key}")
                model.params[key] = loaded.copy()
        return model, config["metadata"]
