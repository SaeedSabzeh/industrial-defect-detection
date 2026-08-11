"""Model architectures for binary biscuit defect classification.

All three architectures take a single `width` parameter controlling the base
channel count. This replaces the two near-duplicate notebooks in the original
project, where `width=8` ("small") and `width=64` ("big") were maintained as
separate copies of the same code.

Channel progression is (width, 2*width, 4*width) across the three stages.
"""

from __future__ import annotations

import torch
import torch.nn as nn

__all__ = ["BasicCNN", "ResNetBlock", "CustomBaseResNet", "SEBlock", "SEResNetBlock", "SEResNet", "build_model", "count_parameters"]


class BasicCNN(nn.Module):
    """Plain three-block conv net with no residual connections.

    Note the parameter cost: the classifier flattens a
    (4*width, 28, 28) feature map, so the first linear layer holds
    4*width*784*hidden weights. At width=64 this single layer is ~51M
    parameters -- 97% of the whole model, and 33x the entire ResNet.
    """

    def __init__(self, num_classes: int = 2, width: int = 64, dropout: float = 0.5):
        super().__init__()
        w1, w2, w3 = width, width * 2, width * 4
        hidden = w3  # matches the original notebooks: 32 at width=8, 256 at width=64

        self.features = nn.Sequential(
            nn.Conv2d(3, w1, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(w1, w2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(w2, w3, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(w3 * 28 * 28, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


class ResNetBlock(nn.Module):
    """Residual block with an identity or projection shortcut."""

    def __init__(self, in_channels: int, filters: int, downsample: bool = False):
        super().__init__()
        stride = 2 if downsample else 1

        self.conv1 = nn.Conv2d(in_channels, filters, kernel_size=3, padding=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(filters)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(filters, filters, kernel_size=3, padding=1, stride=1, bias=False)
        self.bn2 = nn.BatchNorm2d(filters)

        # Projection shortcut only when shape changes; identity otherwise.
        if downsample or in_channels != filters:
            self.projection: nn.Module = nn.Sequential(
                nn.Conv2d(in_channels, filters, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(filters),
            )
        else:
            self.projection = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.projection(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)


class SEBlock(nn.Module):
    """Squeeze-and-Excitation channel attention (Hu et al., 2018).

    Implemented with 1x1 convolutions rather than Linear layers so no
    reshaping is needed. The bottleneck is clamped to at least 1 channel,
    which matters at small widths: at width=8 with reduction=16 the
    bottleneck would otherwise round down to zero.
    """

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        bottleneck = max(1, channels // reduction)
        self.fc1 = nn.Conv2d(channels, bottleneck, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(bottleneck, channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = x.mean(dim=(2, 3), keepdim=True)   # squeeze
        w = self.sigmoid(self.fc2(self.relu(self.fc1(w))))  # excite
        return x * w                            # scale


class SEResNetBlock(nn.Module):
    """Residual block with SE recalibration applied before the shortcut add."""

    def __init__(self, in_channels: int, filters: int, downsample: bool = False, reduction: int = 16):
        super().__init__()
        stride = 2 if downsample else 1

        self.conv1 = nn.Conv2d(in_channels, filters, kernel_size=3, padding=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(filters)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(filters, filters, kernel_size=3, padding=1, stride=1, bias=False)
        self.bn2 = nn.BatchNorm2d(filters)
        self.se = SEBlock(filters, reduction=reduction)

        if downsample or in_channels != filters:
            self.projection: nn.Module = nn.Sequential(
                nn.Conv2d(in_channels, filters, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(filters),
            )
        else:
            self.projection = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.projection(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        return self.relu(out + identity)


class _ResNetTrunk(nn.Module):
    """Shared stem + 3-stage trunk + classifier head."""

    def __init__(self, block_fn, num_classes: int, width: int):
        super().__init__()
        w1, w2, w3 = width, width * 2, width * 4

        self.init_conv = nn.Sequential(
            nn.Conv2d(3, w1, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(w1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.layer1 = block_fn(w1, w1, False)
        self.layer2 = block_fn(w1, w2, True)
        self.layer3 = block_fn(w2, w3, True)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(w3, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.init_conv(x)
        x = self.layer3(self.layer2(self.layer1(x)))
        x = torch.flatten(self.avgpool(x), 1)
        return self.fc(x)


class CustomBaseResNet(_ResNetTrunk):
    def __init__(self, num_classes: int = 2, width: int = 64):
        super().__init__(
            lambda i, f, d: ResNetBlock(i, f, downsample=d),
            num_classes,
            width,
        )


class SEResNet(_ResNetTrunk):
    def __init__(self, num_classes: int = 2, width: int = 64, reduction: int = 16):
        super().__init__(
            lambda i, f, d: SEResNetBlock(i, f, downsample=d, reduction=reduction),
            num_classes,
            width,
        )


_REGISTRY = {
    "cnn": BasicCNN,
    "resnet": CustomBaseResNet,
    "se_resnet": SEResNet,
}


def build_model(name: str, width: int = 64, num_classes: int = 2, reduction: int = 16) -> nn.Module:
    """Construct a model by name. `reduction` is ignored for non-SE models."""
    if name not in _REGISTRY:
        raise ValueError(f"unknown model {name!r}; choose from {sorted(_REGISTRY)}")
    if name == "se_resnet":
        return SEResNet(num_classes=num_classes, width=width, reduction=reduction)
    return _REGISTRY[name](num_classes=num_classes, width=width)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
