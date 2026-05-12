from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import (
    ShuffleNet_V2_X0_5_Weights,
    ShuffleNet_V2_X1_0_Weights,
    shufflenet_v2_x0_5,
    shufflenet_v2_x1_0,
)

from .eca import ECALayer


def _replace_classifier(model: nn.Module, num_classes: int) -> None:
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)


def _freeze_stem_and_early_stages(model: nn.Module) -> None:
    for name, p in model.named_parameters():
        freeze = (
            name.startswith("conv1")
            or name.startswith("maxpool")
            or name.startswith("stage2")
        )
        p.requires_grad = not freeze


def build_shufflenet_baseline(num_classes: int, pretrained: bool = True) -> nn.Module:
    weights = ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1 if pretrained else None
    model = shufflenet_v2_x1_0(weights=weights)
    _replace_classifier(model, num_classes)
    _freeze_stem_and_early_stages(model)
    return model


def _infer_stage_channels(backbone: nn.Module, size: int = 224) -> tuple[int, int, int]:
    with torch.no_grad():
        x = torch.zeros(1, 3, size, size)
        x = backbone.conv1(x)
        x = backbone.maxpool(x)
        x = backbone.stage2(x)
        x = backbone.stage3(x)
        c3 = int(x.shape[1])
        x = backbone.stage4(x)
        c4 = int(x.shape[1])
        x = backbone.conv5(x)
        c5 = int(x.shape[1])
    return c3, c4, c5


class ShuffleNetV2WithECA(nn.Module):
    def __init__(
        self, num_classes: int, pretrained: bool = True, image_size: int = 224
    ) -> None:
        super().__init__()
        weights = ShuffleNet_V2_X0_5_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = shufflenet_v2_x0_5(weights=weights)
        c3, c4, c5 = _infer_stage_channels(self.backbone, size=image_size)
        self.eca3 = ECALayer(c3)
        self.eca4 = ECALayer(c4)
        self.eca5 = ECALayer(c5)
        _replace_classifier(self.backbone, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = self.backbone
        x = b.conv1(x)
        x = b.maxpool(x)
        x = b.stage2(x)
        x = b.stage3(x)
        x = self.eca3(x)
        x = b.stage4(x)
        x = self.eca4(x)
        x = b.conv5(x)
        x = self.eca5(x)
        x = x.mean([2, 3])
        return b.fc(x)

    def freeze_early(self) -> None:
        _freeze_stem_and_early_stages(self.backbone)
        for m in (self.eca3, self.eca4, self.eca5):
            for p in m.parameters():
                p.requires_grad = True


def build_shufflenet_eca_variant(
    num_classes: int, pretrained: bool = True, image_size: int = 224
) -> nn.Module:
    model = ShuffleNetV2WithECA(
        num_classes=num_classes, pretrained=pretrained, image_size=image_size
    )
    model.freeze_early()
    return model


def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def unfreeze_all(model: nn.Module) -> None:
    for p in model.parameters():
        p.requires_grad = True
