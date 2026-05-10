"""Efficient Channel Attention (ECA-Net, CVPR 2020)."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class ECALayer(nn.Module):
    """1D conv on globally pooled channels; adaptive kernel size from channel count."""

    def __init__(self, channels: int, gamma: float = 2.0, b: float = 1.0) -> None:
        super().__init__()
        t = int(abs((math.log2(max(channels, 2)) + b) / gamma))
        k = t if (t % 2 == 1) else t + 1
        self.k = max(k, 3)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(
            1, 1, kernel_size=self.k, padding=(self.k - 1) // 2, bias=False
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        return x * self.sigmoid(y)
