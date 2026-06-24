"""Per-base-pair multi-track regressor student (for the NTv3 multi-track task).

Reuses the existing BPNet dilated-conv stem (all size variants from BPNetClassifier) and
swaps the global-pool + linear classifier head for a per-position 1x1-conv track head, so the
output is ``[B, T, L_out]`` — T functional tracks predicted at each position. This is the
student that distills NTv3's per-bp ``bigwig_tracks_logits``.

Minimal/DRY: the backbone is built by the existing ``BPNetClassifier`` (we keep its backbone and
discard its classifier head), so every size option stays in one place.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Literal, Optional

from .bpnet_classifier import BPNetClassifier, BPNetClassifierConfig, VariableBPNet
from .nn.bpnet_pytorch import BPNet


@dataclass
class BPNetRegressorConfig:
    num_tracks: int  # number of per-bp output tracks T (e.g. the chosen NTv3 track subset)
    model_type: Literal["bpnet", "cnn", "bilstm"] = "bpnet"
    model_size: str = "original"
    hidden_dim: Optional[int] = None
    # Downsample the per-bp output to match the teacher's track bin size (NTv3 U-Net bins via
    # num_downsamples). 1 = base resolution; set to the teacher's bin factor (e.g. 128).
    out_resolution: int = 1


class BPNetRegressor(nn.Module):
    """BPNet stem -> 1x1 conv -> T tracks per position. Output: ``[B, T, L_out]``."""

    def __init__(self, config: BPNetRegressorConfig):
        super().__init__()
        self.config = config
        # Reuse BPNetClassifier's backbone builder (covers every size); drop its classifier head.
        clf = BPNetClassifier(
            BPNetClassifierConfig(
                model_type=config.model_type,
                model_size=config.model_size,
                hidden_dim=config.hidden_dim,
                num_labels=1,
            )
        )
        self.backbone = clf.backbone
        feat_dim = getattr(self.backbone, "feature_dim", None) or 64
        self.track_head = nn.Conv1d(feat_dim, config.num_tracks, kernel_size=1)
        self.pool = (
            nn.AvgPool1d(config.out_resolution) if config.out_resolution > 1 else nn.Identity()
        )

    def _features(self, x: torch.Tensor) -> torch.Tensor:
        """Run the backbone; returns [B, C, L]. BPNet returns a dict, variants return a tensor."""
        if isinstance(self.backbone, BPNet):
            return self.backbone(x)["x"]
        if isinstance(self.backbone, VariableBPNet):
            return self.backbone(x)
        return self.backbone(x)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """input_ids: [B, L] integer-encoded DNA (0-3) -> tracks [B, T, L_out]."""
        one_hot = F.one_hot(input_ids, num_classes=4).float().permute(0, 2, 1)  # [B, 4, L]
        feats = self._features(one_hot)  # [B, C, L]
        tracks = self.track_head(feats)  # [B, T, L]
        return self.pool(tracks)  # [B, T, L_out]
