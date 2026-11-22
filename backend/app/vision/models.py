from dataclasses import dataclass
from typing import Callable

import torch
from torch import nn
from torchvision import models

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


@dataclass(frozen=True)
class Arch:
    id: str
    name: str
    build: Callable[[int, bool], nn.Module]
    target_layer: Callable[[nn.Module], nn.Module]  # last convolutional block, used by Grad-CAM


def _mobilenet_v2(num_classes: int, pretrained: bool) -> nn.Module:
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V2 if pretrained else None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model


def _efficientnet_b0(num_classes: int, pretrained: bool) -> nn.Module:
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model


def _resnet50(num_classes: int, pretrained: bool) -> nn.Module:
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


ARCHS = {
    a.id: a
    for a in (
        Arch("mobilenet_v2", "MobileNetV2", _mobilenet_v2, lambda m: m.features[-1]),
        Arch("efficientnet_b0", "EfficientNet-B0", _efficientnet_b0, lambda m: m.features[-1]),
        Arch("resnet50", "ResNet-50", _resnet50, lambda m: m.layer4),
    )
}


def head_parameters(model: nn.Module):
    head = model.fc if hasattr(model, "fc") else model.classifier
    return list(head.parameters())


def device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def normalize(batch_uint8: torch.Tensor) -> torch.Tensor:
    x = batch_uint8.float() / 255.0
    return (x - MEAN.to(x.device)) / STD.to(x.device)
