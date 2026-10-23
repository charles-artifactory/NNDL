"""Prepare a pinned, verified CIFAR-10 image archive."""
import hashlib
import json
import tarfile
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image

from .data import CLASSES

SOURCE = "https://s3.amazonaws.com/fast-ai-imageclas/cifar10.tgz"
SHA256 = "637c5814e11aefcb6ee76d5f59c67ddc8de7f5b5077502a195b0833d1e3e4441"


def prepare(root):
    root = Path(root)
    (root / "tmp").mkdir(parents=True, exist_ok=True)
    data = root / "data"
    data.mkdir(exist_ok=True)
    archive = data / "cifar10.tgz"
    if not archive.exists():
        temporary = root / "tmp" / "cifar10.download"
        temporary.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(SOURCE, temporary)
        if hashlib.sha256(temporary.read_bytes()).hexdigest() != SHA256:
            raise ValueError("Downloaded CIFAR-10 archive checksum mismatch")
        temporary.replace(archive)
    if hashlib.sha256(archive.read_bytes()).hexdigest() != SHA256:
        raise ValueError("CIFAR-10 archive checksum mismatch")
    arrays = {}
    grouped = {split: {name: [] for name in CLASSES} for split in ("train", "test")}
    # Stream the compressed archive exactly once; sorting members before reading
    # would force repeated backwards gzip seeks on this unordered PNG archive.
    with tarfile.open(archive, "r|gz") as tf:
        for member in tf:
            parts = member.name.split("/")
            if not member.isfile() or not member.name.endswith(".png"):
                continue
            if len(parts) != 4 or parts[0] != "cifar10" or parts[1] not in grouped or parts[2] not in CLASSES:
                raise ValueError(f"Unexpected archive member: {member.name}")
            with tf.extractfile(member) as f, Image.open(f) as im:
                image = np.asarray(im.convert("RGB")).copy()
                if image.shape != (32, 32, 3):
                    raise ValueError(f"Unexpected image dimensions: {member.name}")
                grouped[parts[1]][parts[2]].append((parts[3], image))
    for split, per_class in [("train", 5000), ("test", 1000)]:
        images, labels = [], []
        for label, name in enumerate(CLASSES):
            selected = sorted(grouped[split][name], key=lambda item: item[0])
            if len(selected) != per_class:
                raise ValueError(f"Unexpected sample count: {split}/{name}")
            images.extend(image for _, image in selected)
            labels.extend([label] * per_class)
        arrays[split + "_images"] = np.stack(images)
        arrays[split + "_labels"] = np.array(labels, dtype=np.int64)
    temporary = root / "tmp" / "cifar10.npz"
    np.savez_compressed(temporary, **arrays)
    with np.load(temporary, allow_pickle=False) as check:
        assert check["train_images"].shape == (50000, 32, 32, 3)
        assert check["test_images"].shape == (10000, 32, 32, 3)
    temporary.replace(data / "cifar10.npz")
    meta = {"source": SOURCE, "archive_sha256": SHA256,
            "format": "lossless RGB PNG; ordered by declared class then filename",
            "classes": CLASSES, "train_count": 50000, "test_count": 10000,
            "per_train_class": 5000, "per_test_class": 1000, "image_shape": [32, 32, 3]}
    (data / "provenance.json").write_text(json.dumps(meta, indent=2) + "\n")
    (root / "results").mkdir(exist_ok=True)
    (root / "results" / "data_provenance.json").write_text(json.dumps(meta, indent=2) + "\n")


if __name__ == "__main__":
    prepare(Path(__file__).resolve().parents[2])
