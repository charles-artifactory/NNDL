"""Predict labels for a local .npy array without any ML framework."""
import argparse
from pathlib import Path

import numpy as np

from .network import MLP, softmax


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", type=Path, help=".npy with shape (N,28,28) or (N,784)")
    parser.add_argument("--model", type=Path, default=Path(__file__).resolve().parents[2] / "results/best_model.npz")
    args = parser.parse_args()
    x = np.load(args.images, allow_pickle=False)
    if x.ndim == 3 and x.shape[1:] == (28, 28):
        x = x.reshape(len(x), 784)
    if x.ndim != 2 or x.shape[1] != 784 or len(x) == 0:
        raise ValueError("Expected a nonempty (N,28,28) or (N,784) array")
    if x.dtype == np.uint8:
        x = x.astype(np.float32) / 255
    else:
        x = x.astype(np.float32)
        if not np.isfinite(x).all() or x.min() < 0 or x.max() > 1:
            raise ValueError("Float images must already be normalized to [0,1]")
    model, _ = MLP.load(args.model)
    for start in range(0, len(x), 1024):
        probs = softmax(model.forward(x[start:start + 1024]))
        for index, p in enumerate(probs, start):
            print(f"{index},label={p.argmax()},confidence={p.max():.6f}")


if __name__ == "__main__":
    main()
