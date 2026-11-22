"""Grad-CAM (Selvaraju et al., 2017) for any CNN with a known last convolutional block."""
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None
        target_layer.register_forward_hook(self._capture)

    def _capture(self, _module, _inputs, output):
        if not output.requires_grad:
            return
        self.activations = output
        output.register_hook(self._save_grad)

    def _save_grad(self, grad):
        self.gradients = grad

    def __call__(self, x: torch.Tensor, target: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (probabilities [B, C], heatmaps [B, H, W] scaled to 0..1) for the target classes
        (default: the predicted class)."""
        self.model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            x = x.detach().requires_grad_(False)
            logits = self.model(x)
            if target is None:
                target = logits.argmax(dim=1)
            logits.gather(1, target[:, None]).sum().backward()
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # global-average-pooled gradients
        cam = F.relu((weights * self.activations).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[:, 0]
        flat = cam.flatten(1)
        low, high = flat.min(dim=1).values, flat.max(dim=1).values
        cam = (cam - low[:, None, None]) / (high - low + 1e-8)[:, None, None]
        probs = logits.softmax(dim=1)
        self.activations = self.gradients = None
        return probs.detach(), cam.detach()


def colormap(heat: np.ndarray) -> np.ndarray:
    """Map 0..1 values to an RGB jet-like palette (blue -> cyan -> yellow -> red)."""
    h = np.clip(heat, 0, 1)
    r = np.clip(1.5 - np.abs(4 * h - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * h - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * h - 1), 0, 1)
    return (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)


def overlay(image: np.ndarray, heat: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """image: HxWx3 uint8, heat: HxW in 0..1."""
    colored = colormap(heat).astype(np.float32)
    weight = alpha * np.clip(heat, 0.15, 1.0)[..., None]
    blended = image.astype(np.float32) * (1 - weight) + colored * weight
    return blended.clip(0, 255).astype(np.uint8)
