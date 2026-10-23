from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import Dataset

CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]


class TensorImages(Dataset):
    """Raw tensors; training-only transforms run on a whole batch on the device."""
    def __init__(self, images, labels, indices):
        self.images = images
        self.labels = labels
        self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = int(self.indices[i])
        return self.images[idx], self.labels[idx]


def load_data(root, seed):
    root = Path(root)
    cached = root / "cifar10.npz"
    if not cached.exists():
        from .prepare import prepare
        prepare(root.parent)
    if cached.exists():
        with np.load(cached, allow_pickle=False) as data:
            images = torch.from_numpy(data["train_images"].copy()).permute(0, 3, 1, 2).contiguous()
            labels = torch.from_numpy(data["train_labels"].copy()).long()
            test_images = torch.from_numpy(data["test_images"].copy()).permute(0, 3, 1, 2).contiguous()
            test_labels = torch.from_numpy(data["test_labels"].copy()).long()
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    train_idx, val_idx = next(split.split(np.zeros(len(labels)), labels.numpy()))
    return images, labels, test_images, test_labels, train_idx, val_idx


# 基础题及附加题 1、3：缩放图像，训练时随机裁剪和翻转，并按配置加入 Cutout。
def batch_transform(x, training=False, cutout=False):
    # Scaling is fixed; no statistics are estimated from validation or test data.
    x = x.float().div(255)
    if training:
        n = len(x)
        padded = torch.nn.functional.pad(x, (4, 4, 4, 4), mode="reflect")
        # Sample independent crop offsets per image without a Python image loop.
        offsets = torch.randint(0, 9, (n, 2), device=x.device)
        rows = torch.arange(32, device=x.device)[None, :, None] + offsets[:, 0, None, None]
        cols = torch.arange(32, device=x.device)[None, None, :] + offsets[:, 1, None, None]
        x = padded.permute(0, 2, 3, 1)[torch.arange(n, device=x.device)[:, None, None], rows, cols]
        x = x.permute(0, 3, 1, 2).contiguous()
        flip = torch.rand(n, 1, 1, 1, device=x.device) < 0.5
        x = torch.where(flip, x.flip(-1), x)
    x = (x - 0.5) / 0.5
    if training and cutout:
        n = len(x)
        # Full 8x8 squares, each sampled independently and fully inside the image.
        starts = torch.randint(0, 25, (n, 2), device=x.device)
        rows = torch.arange(32, device=x.device)[None, :, None]
        cols = torch.arange(32, device=x.device)[None, None, :]
        mask = ((rows >= starts[:, 0, None, None]) & (rows < starts[:, 0, None, None] + 8)
                & (cols >= starts[:, 1, None, None]) & (cols < starts[:, 1, None, None] + 8))
        x = x.masked_fill(mask[:, None], 0)
    return x


# 附加题 3：用中灰色覆盖图像中心，用于测量遮挡后的分类准确率。
def occlude(x, side=8):
    """Deterministic central square, zero in normalized RGB (mid-gray)."""
    out = x.clone()
    start = (32 - side) // 2
    out[:, :, start:start + side, start:start + side] = 0
    return out
