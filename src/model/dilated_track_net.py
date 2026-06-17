"""Dilated residual-conv student for base-resolution multi-track distillation.

Why this exists: the BPNet-stem student's receptive field (~hundreds of bp for `medium`, ~1-2 kb
for `large`) is far below the 16 kb input window, so broad-domain marks (H3K27me3 Polycomb domains,
H3K36me3 gene bodies, which span kb-10s of kb) cannot be captured no matter the loss or data volume.

This stacks dilated residual conv blocks with exponentially growing dilation, so the receptive
field covers the entire window at base resolution. Drop-in for ``BPNetRegressor``: same I/O contract
``input_ids [B, L]`` (integer DNA 0-3) -> per-bp tracks ``[B, T, L]``.

Receptive field (kernel 3 blocks, dilations 1,2,...,2^(n-1)):
    RF = 1 + (stem_kernel - 1) + sum_blocks (kernel - 1) * dilation
With the defaults (stem 15, 14 blocks, dilations capped at 8192) RF ~= 32.8 kb >> 16 kb window.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class DilatedTrackNetConfig:
    """Config for :class:`DilatedTrackNet`. Defaults cover a 16 kb window's receptive field."""

    num_tracks: int
    hidden: int = 256
    n_blocks: int = 14  # dilations 1,2,...,8192 -> RF ~32.8 kb (>= 16 kb window)
    kernel: int = 3
    stem_kernel: int = 15
    dropout: float = 0.1
    max_dilation: int = 8192


class _DilatedResBlock(nn.Module):
    """Length-preserving residual block: Conv(dilated)->BN->GELU->Conv(1x1)->BN, then x + drop(h)."""

    def __init__(self, ch: int, kernel: int, dilation: int, dropout: float):
        super().__init__()
        assert kernel % 2 == 1, "kernel must be odd for symmetric same-padding"
        pad = dilation * (kernel - 1) // 2  # 'same' padding for odd kernel
        self.conv1 = nn.Conv1d(ch, ch, kernel, padding=pad, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(ch)
        self.conv2 = nn.Conv1d(ch, ch, kernel_size=1)
        self.bn2 = nn.BatchNorm1d(ch)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.gelu(self.bn1(self.conv1(x)))
        h = self.bn2(self.conv2(h))
        return F.gelu(x + self.drop(h))


class DilatedTrackNet(nn.Module):
    """Dilated residual conv tower; ``input_ids [B, L]`` (0-3) -> tracks ``[B, T, L]``."""

    def __init__(self, config: DilatedTrackNetConfig):
        super().__init__()
        self.config = config
        c = config.hidden
        self.stem = nn.Conv1d(4, c, config.stem_kernel, padding=config.stem_kernel // 2)
        self.dilations = [min(2 ** i, config.max_dilation) for i in range(config.n_blocks)]
        self.blocks = nn.ModuleList(
            [_DilatedResBlock(c, config.kernel, d, config.dropout) for d in self.dilations]
        )
        self.track_head = nn.Conv1d(c, config.num_tracks, kernel_size=1)

    @property
    def receptive_field(self) -> int:
        """Number of input positions each output position can see."""
        rf = (self.config.stem_kernel - 1) + sum(
            (self.config.kernel - 1) * d for d in self.dilations
        )
        return rf + 1

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """input_ids: [B, L] integer DNA (0-3). Returns tracks [B, T, L] (base resolution)."""
        x = F.one_hot(input_ids, num_classes=4).float().permute(0, 2, 1)  # [B, 4, L]
        x = F.gelu(self.stem(x))
        for blk in self.blocks:
            x = blk(x)
        return self.track_head(x)  # [B, T, L]
