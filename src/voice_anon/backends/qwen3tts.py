"""Small optional speaker-conditioning adapter for a compatible Qwen run."""

from __future__ import annotations

import torch
from torch import nn


class SpeakerConditioningAdapter(nn.Module):
    """Residual adapter; does not alter the pretrained text/codec backbone."""

    def __init__(self, dimension: int = 1024, bottleneck: int = 256) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dimension)
        self.down = nn.Linear(dimension, bottleneck, bias=False)
        self.activation = nn.SiLU()
        self.up = nn.Linear(bottleneck, dimension, bias=False)

    def forward(self, vector: torch.Tensor) -> torch.Tensor:
        return vector + self.up(self.activation(self.down(self.norm(vector))))
