import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Literal, Tuple
from dataclasses import dataclass
from .nn.bpnet_pytorch import BPNet


@dataclass
class BPNetClassifierConfig:
    num_labels: int = 2
    teacher_hidden_size: Optional[int] = None
    teacher_projection_opt: Literal["down", "up"] = "down"

    # Extended for multi-architecture support
    model_type: Literal["bpnet", "bilstm", "cnn"] = "bpnet"
    model_size: Literal["original", "tiny", "small", "medium", "large"] = (
        "original"  # Default to original
    )
    hidden_dim: Optional[int] = None  # Override default hidden dimensions


class SimpleResidual(nn.Module):
    """Simple residual block"""

    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return x + self.fn(x)


class SimpleCNN(nn.Module):
    """Simple CNN backbone as alternative to BPNet"""

    def __init__(self, hidden_dim: int = 128):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv1d(4, hidden_dim // 2, 15, padding="same"),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Conv1d(hidden_dim // 2, hidden_dim, 9, padding="same"),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, 5, padding="same"),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
        )
        self.feature_dim = hidden_dim

    def forward(self, x):
        return self.conv_layers(x)


class SimpleBiLSTM(nn.Module):
    """Simple BiLSTM backbone"""

    def __init__(self, hidden_dim: int = 128):
        super().__init__()
        self.conv = nn.Conv1d(4, 64, 9, padding="same")
        self.lstm = nn.LSTM(
            64,
            hidden_dim // 2,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.2,
        )
        self.feature_dim = hidden_dim

    def forward(self, x):
        x = F.relu(self.conv(x))
        x = x.permute(0, 2, 1)  # (batch, seq, channels)
        x, _ = self.lstm(x)
        return x.permute(0, 2, 1)  # (batch, channels, seq)


class BPNetClassifier(nn.Module):
    """
    Unified classifier supporting BPNet, CNN, and BiLSTM architectures
    Maintains backward compatibility while adding new model types
    """

    def __init__(self, config: BPNetClassifierConfig):
        super().__init__()
        self.config = config

        # Select backbone based on model type and size
        if config.model_type == "bpnet":
            self.backbone = self._create_bpnet_backbone(config.model_size)
        elif config.model_type == "cnn":
            hidden_dim = config.hidden_dim or self._get_hidden_dim(config.model_size)
            self.backbone = SimpleCNN(hidden_dim)
        elif config.model_type == "bilstm":
            hidden_dim = config.hidden_dim or self._get_hidden_dim(config.model_size)
            self.backbone = SimpleBiLSTM(hidden_dim)
        else:
            # Default to original BPNet for backward compatibility
            self.backbone = BPNet()

        # Get feature dimension
        if hasattr(self.backbone, "feature_dim"):
            C = self.backbone.feature_dim
        elif hasattr(self.backbone, "stem"):
            C = self.backbone.stem[0].out_channels
        else:
            C = 64  # Default

        self.pool = nn.AdaptiveAvgPool1d(1)
        self._define_classifier(C)

    def _create_bpnet_backbone(self, model_size: str):
        """Create BPNet variant based on size"""
        if model_size == "tiny":
            # Tiny BPNet with fewer channels
            return nn.Sequential(
                nn.Conv1d(4, 32, 15, padding="same"),
                nn.ReLU(),
                SimpleResidual(
                    nn.Sequential(
                        nn.Conv1d(32, 32, 3, padding="same", dilation=2), nn.ReLU()
                    )
                ),
                SimpleResidual(
                    nn.Sequential(
                        nn.Conv1d(32, 32, 3, padding="same", dilation=4), nn.ReLU()
                    )
                ),
            )
        elif model_size == "large":
            # Large BPNet with more channels and layers
            layers = [nn.Conv1d(4, 256, 25, padding="same"), nn.ReLU()]
            for i in range(1, 10):
                layers.append(
                    SimpleResidual(
                        nn.Sequential(
                            nn.Conv1d(
                                256, 256, 3, padding="same", dilation=2 ** min(i, 8)
                            ),
                            nn.ReLU(),
                        )
                    )
                )
            return nn.Sequential(*layers)
        elif model_size == "medium":
            # Medium BPNet
            layers = [nn.Conv1d(4, 128, 21, padding="same"), nn.ReLU()]
            for i in range(1, 7):
                layers.append(
                    SimpleResidual(
                        nn.Sequential(
                            nn.Conv1d(
                                128, 128, 3, padding="same", dilation=2 ** min(i, 6)
                            ),
                            nn.ReLU(),
                        )
                    )
                )
            return nn.Sequential(*layers)
        elif model_size == "original":
            # Default small or use original BPNet
            return BPNet()
        else:
            try:
                return BPNet()
            except:
                # Fallback if original BPNet not available
                return SimpleCNN(64)

    def _get_hidden_dim(self, model_size: str) -> int:
        """Get hidden dimension based on model size"""
        size_map = {"tiny": 32, "small": 64, "medium": 128, "large": 256}
        return size_map.get(model_size, 64)

    def _define_classifier(self, C: int):
        """Define classifier head with optional teacher projection"""
        if self.config.teacher_hidden_size is not None:
            if self.config.teacher_projection_opt == "down":
                self.teacher_proj = nn.Linear(self.config.teacher_hidden_size, C)
                self.classifier = nn.Linear(C, self.config.num_labels)
            elif self.config.teacher_projection_opt == "up":
                self.teacher_proj = nn.Sequential(
                    nn.Linear(C, C * 2),
                    nn.ReLU(),
                    nn.Linear(C * 2, self.config.teacher_hidden_size),
                )
                self.classifier = nn.Linear(
                    self.config.teacher_hidden_size, self.config.num_labels
                )
            else:
                raise ValueError(
                    f"Invalid teacher projection: {self.config.teacher_projection_opt}"
                )
        else:
            self.teacher_proj = None
            self.classifier = nn.Linear(C, self.config.num_labels)

    def forward(self, input_ids, return_feats: bool = False):
        """Forward pass with optional feature return"""
        one_hot = F.one_hot(input_ids, num_classes=4).float()
        x = one_hot.permute(0, 2, 1)

        # Process through backbone
        if isinstance(self.backbone, BPNet):
            out = self.backbone(x)
            feats = out["x"]
        else:
            feats = self.backbone(x)

        pooled = self.pool(feats).squeeze(-1)

        # Apply projection if needed
        if self.teacher_proj is not None:
            if self.config.teacher_projection_opt == "up":
                pooled = self.teacher_proj(pooled)

        logits = self.classifier(pooled)

        if return_feats:
            return logits, pooled
        return logits

    def aligned_feats(
        self, sfeats: torch.Tensor, tfeats: Optional[torch.Tensor] = None
    ):
        """Align student and teacher features for distillation"""
        if self.teacher_proj is not None and tfeats is not None:
            if self.config.teacher_projection_opt == "down":
                return sfeats, self.teacher_proj(tfeats)
            elif self.config.teacher_projection_opt == "up":
                return sfeats, tfeats
        return sfeats, tfeats
