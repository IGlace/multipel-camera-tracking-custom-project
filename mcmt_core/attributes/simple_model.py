"""Simple attribute baseline used for unified checkpoint-backed inference."""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights, ResNet50_Weights, resnet18, resnet50


_BACKBONES = {
    "resnet18": (resnet18, ResNet18_Weights.IMAGENET1K_V1),
    "resnet50": (resnet50, ResNet50_Weights.IMAGENET1K_V2),
}


class AttributeBaseline(nn.Module):
    def __init__(
        self,
        num_classes: int,
        *,
        backbone: str = "resnet50",
        pretrained_backbone: bool = False,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if backbone not in _BACKBONES:
            raise ValueError(f"Unsupported attribute backbone: {backbone}")
        builder, weights_enum = _BACKBONES[backbone]
        model = builder(weights=weights_enum if pretrained_backbone else None)
        feature_dim = model.fc.in_features
        self.backbone_name = backbone
        self.backbone = nn.Sequential(*list(model.children())[:-2])
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.bn = nn.BatchNorm1d(feature_dim)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.head = nn.Linear(feature_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = self.pool(x).flatten(1)
        x = self.bn(x)
        x = self.dropout(x)
        x = self.head(x)
        return x
