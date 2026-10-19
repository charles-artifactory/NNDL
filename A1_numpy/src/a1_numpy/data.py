"""Download verified MNIST IDX files; split without looking at the test set."""
import gzip
import hashlib
import os
import struct
import urllib.request
from pathlib import Path

import numpy as np

BASE_URL = "https://storage.googleapis.com/cvdf-datasets/mnist/"
# Published checksums of the original compressed MNIST files.
FILES = {
    "train-images-idx3-ubyte.gz": "f68b3c2dcbeaaa9fbdd348bbdeb94873",
    "train-labels-idx1-ubyte.gz": "d53e105ee54ea40749a09fcbcd1e9432",
    "t10k-images-idx3-ubyte.gz": "9fb629c4189551a2d022fa330f9573f3",
    "t10k-labels-idx1-ubyte.gz": "ec29112dd5afa0611ce80d1b7f02629c",
}


def download(directory: Path, name: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    if path.exists() and hashlib.md5(path.read_bytes()).hexdigest() == FILES[name]:
        return path
    temporary = directory / (name + ".part")
    try:
        print(f"Downloading {name}", flush=True)
        with urllib.request.urlopen(BASE_URL + name, timeout=90) as response:
            content = response.read()
        if hashlib.md5(content).hexdigest() != FILES[name]:
            raise ValueError(f"MNIST checksum mismatch: {name}")
        temporary.write_bytes(content)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def read_idx(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as handle:
        raw = handle.read()
    zero, dtype_code, dimensions = struct.unpack(">HBB", raw[:4])
    if zero != 0 or dtype_code != 8 or dimensions not in (1, 3):
        raise ValueError(f"Unsupported IDX header: {path}")
    shape = struct.unpack(">" + "I" * dimensions, raw[4:4 + 4 * dimensions])
    values = np.frombuffer(raw, dtype=np.uint8, offset=4 + 4 * dimensions)
    if values.size != np.prod(shape):
        raise ValueError(f"Invalid IDX payload: {path}")
    return values.reshape(shape)


def load_mnist(directory: Path, train: bool = True):
    prefix = "train" if train else "t10k"
    images = read_idx(download(directory, f"{prefix}-images-idx3-ubyte.gz"))
    labels = read_idx(download(directory, f"{prefix}-labels-idx1-ubyte.gz"))
    expected = 60000 if train else 10000
    if images.shape != (expected, 28, 28) or labels.shape != (expected,):
        raise ValueError("Unexpected MNIST dimensions")
    if np.any(labels > 9):
        raise ValueError("MNIST labels must be in [0, 9]")
    return images.reshape(expected, -1).astype(np.float32) / 255, labels.astype(np.int64)


def stratified_split(labels, validation_size=10000, seed=2026):
    """Largest-remainder allocation gives an exact stratified split size."""
    if not 0 < validation_size < len(labels):
        raise ValueError("Validation size must lie between 0 and dataset size")
    rng = np.random.default_rng(seed)
    classes, counts = np.unique(labels, return_counts=True)
    allocation = counts * validation_size / len(labels)
    quotas = np.floor(allocation).astype(int)
    order = np.argsort(-(allocation - quotas), kind="stable")
    quotas[order[:validation_size - quotas.sum()]] += 1
    training, validation = [], []
    for label, count in zip(classes, quotas):
        indices = rng.permutation(np.flatnonzero(labels == label))
        validation.extend(indices[:count])
        training.extend(indices[count:])
    return rng.permutation(training), rng.permutation(validation)
