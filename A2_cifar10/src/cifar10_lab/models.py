from dataclasses import asdict, dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class ModelSpec:
    normalization: str = "batch"
    dropout: float = 0.2
    kernel_size: int = 3
    depth: int = 1
    residual: bool = False
    width: int = 32

    def to_dict(self):
        return asdict(self)


# 基础题：创建 BatchNorm、InstanceNorm、LayerNorm、GroupNorm 或对照层。
def normalization(kind, channels, side):
    if kind == "none":
        return nn.Identity()
    if kind == "batch":
        return nn.BatchNorm2d(channels)
    if kind == "instance":
        return nn.InstanceNorm2d(channels, affine=True)
    if kind == "layer":
        # Normalize all C,H,W elements, with genuine elementwise affine parameters.
        return nn.LayerNorm((channels, side, side))
    if kind == "group":
        return nn.GroupNorm(4, channels)
    raise ValueError(f"Unknown normalization: {kind}")


# 基础题：组合卷积、归一化和池化。附加题 1、2：设置深度、残差连接和卷积核大小。
class ConvBlock(nn.Module):
    def __init__(self, in_channels, channels, side, spec):
        super().__init__()
        layers = []
        for i in range(spec.depth):
            layers += [
                nn.Conv2d(in_channels if i == 0 else channels, channels,
                          spec.kernel_size, padding=spec.kernel_size // 2, bias=False),
                normalization(spec.normalization, channels, side),
                nn.ReLU(inplace=False),
            ]
        self.body = nn.Sequential(*layers)
        self.skip = nn.Conv2d(in_channels, channels, 1, bias=False) if spec.residual else None
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        y = self.body(x)
        if self.skip is not None:
            y = torch.relu(y + self.skip(x))
        return self.pool(y)


# 基础题：将 CIFAR-10 图像分成 10 类，分类层包含 Dropout。
class Classifier(nn.Module):
    def __init__(self, spec=ModelSpec()):
        super().__init__()
        if spec.kernel_size not in (3, 5, 7) or spec.depth not in (1, 2):
            raise ValueError("kernel_size must be 3/5/7; depth must be 1/2")
        if not 0 <= spec.dropout < 1 or spec.width % 4:
            raise ValueError("dropout must be in [0,1); width must be a multiple of 4")
        self.spec = spec
        self.blocks = nn.ModuleList([
            ConvBlock(3, spec.width, 32, spec),
            ConvBlock(spec.width, spec.width * 2, 16, spec),
            ConvBlock(spec.width * 2, spec.width * 4, 8, spec),
        ])
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(spec.width * 4 * 4 * 4, 128),
                                  nn.ReLU(), nn.Dropout(spec.dropout), nn.Linear(128, 10))

    def features(self, x):
        maps = []
        for block in self.blocks:
            x = block(x)
            maps.append(x)
        return maps

    def forward(self, x):
        return self.head(self.features(x)[-1])


def parameter_count(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def receptive_field(spec):
    field, jump = 1, 1
    for _ in range(3):
        for _ in range(spec.depth):
            field += (spec.kernel_size - 1) * jump
        field += jump  # 2x2 pooling
        jump *= 2
    return field
