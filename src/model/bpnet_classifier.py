import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Literal
from dataclasses import dataclass
from .nn.bpnet_pytorch import BPNet


@dataclass
class BPNetClassifierConfig:
    num_labels: int = 2
    teacher_hidden_size: Optional[int] = None
    teacher_projection_opt: Literal["down", "up"] = "down"

    # Extended for multi-architecture support
    model_type: Literal["bpnet", "bilstm", "cnn"] = "bpnet"
    model_size: Literal[
        "original",
        "tiny",
        "small",
        "medium",
        "large",
        "ultra_tiny",
        "extra_tiny",
        "medium_small",
        "pico",
        "medium_large",  # NEW: ~0.4M params
        "extra_large",  # NEW: ~0.8M params
        "xxlarge",  # NEW: ~3.6M params
    ] = "original"
    hidden_dim: Optional[int] = None


class SimpleResidual(nn.Module):
    """Simple residual block"""

    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return x + self.fn(x)


class VariableBPNet(nn.Module):
    """Wrapper for variable-size BPNet backbones with feature_dim attribute"""

    def __init__(self, layers, feature_dim):
        super().__init__()
        self.layers = layers
        self.feature_dim = feature_dim

    def forward(self, x):
        return self.layers(x)


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
            if config.model_size == "original":
                # Use original BPNet() for backward compatibility
                self.backbone = BPNet()
            else:
                # Use variable-size BPNet for new experiments
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

        # Print model parameters
        self._print_model_info()

    def _create_bpnet_backbone(self, model_size: str):
        """Create BPNet variant based on size - returns VariableBPNet with correct feature_dim"""
        if model_size == "tiny":
            # Tiny BPNet with fewer channels (32)
            layers = nn.Sequential(
                nn.Conv1d(4, 32, 15, padding="same"),
                nn.ReLU(),
                SimpleResidual(
                    nn.Sequential(nn.Conv1d(32, 32, 3, padding="same", dilation=2), nn.ReLU())
                ),
                SimpleResidual(
                    nn.Sequential(nn.Conv1d(32, 32, 3, padding="same", dilation=4), nn.ReLU())
                ),
            )
            return VariableBPNet(layers, feature_dim=32)

        elif model_size == "small":
            # Small BPNet with 64 channels (similar to original)
            layers = nn.Sequential(
                nn.Conv1d(4, 64, 21, padding="same"),
                nn.ReLU(),
                SimpleResidual(
                    nn.Sequential(nn.Conv1d(64, 64, 3, padding="same", dilation=2), nn.ReLU())
                ),
                SimpleResidual(
                    nn.Sequential(nn.Conv1d(64, 64, 3, padding="same", dilation=4), nn.ReLU())
                ),
                SimpleResidual(
                    nn.Sequential(nn.Conv1d(64, 64, 3, padding="same", dilation=8), nn.ReLU())
                ),
            )
            return VariableBPNet(layers, feature_dim=64)

        elif model_size == "medium":
            # Medium BPNet with 128 channels
            layers = [nn.Conv1d(4, 128, 25, padding="same"), nn.ReLU()]
            for i in range(1, 10):
                layers.append(
                    SimpleResidual(
                        nn.Sequential(
                            nn.Conv1d(128, 128, 3, padding="same", dilation=2 ** min(i, 6)),
                            nn.ReLU(),
                        )
                    )
                )
            return VariableBPNet(nn.Sequential(*layers), feature_dim=128)

        elif model_size == "large":
            # Large BPNet with 256 channels and more layers
            layers = [nn.Conv1d(4, 256, 25, padding="same"), nn.ReLU()]
            for i in range(1, 10):
                layers.append(
                    SimpleResidual(
                        nn.Sequential(
                            nn.Conv1d(256, 256, 3, padding="same", dilation=2 ** min(i, 8)),
                            nn.ReLU(),
                        )
                    )
                )
            return VariableBPNet(nn.Sequential(*layers), feature_dim=256)

        elif model_size == "medium_small":
            # Medium-Small BPNet with 90 channels (~0.48M params)
            layers = [nn.Conv1d(4, 90, 25, padding="same"), nn.ReLU()]
            for i in range(1, 10):
                layers.append(
                    SimpleResidual(
                        nn.Sequential(
                            nn.Conv1d(90, 90, 3, padding="same", dilation=2 ** min(i, 6)),
                            nn.ReLU(),
                        )
                    )
                )
            return VariableBPNet(nn.Sequential(*layers), feature_dim=90)

        elif model_size == "extra_tiny":
            # Extra Tiny BPNet with 30 channels (~0.1m/4 = 25k params)
            # Following the successful medium/large/medium_small structure
            layers = [nn.Conv1d(4, 30, 25, padding="same"), nn.ReLU()]
            for i in range(1, 10):
                layers.append(
                    SimpleResidual(
                        nn.Sequential(
                            nn.Conv1d(30, 30, 3, padding="same", dilation=2 ** min(i, 6)),
                            nn.ReLU(),
                        )
                    )
                )
            return VariableBPNet(nn.Sequential(*layers), feature_dim=30)

        elif model_size == "ultra_tiny":
            # Ultra Tiny BPNet with 14 channels (~0.1m/4/4 = 6.25k params)
            # Following the successful medium/large/medium_small structure
            layers = [nn.Conv1d(4, 14, 25, padding="same"), nn.ReLU()]
            for i in range(1, 10):
                layers.append(
                    SimpleResidual(
                        nn.Sequential(
                            nn.Conv1d(14, 14, 3, padding="same", dilation=2 ** min(i, 6)),
                            nn.ReLU(),
                        )
                    )
                )
            return VariableBPNet(nn.Sequential(*layers), feature_dim=14)

        elif model_size == "pico":
            # Pico BPNet with 7 channels (~0.1m/4/4/4 = 1.5625k params)
            # Following the successful medium/large/medium_small structure
            layers = [nn.Conv1d(4, 7, 25, padding="same"), nn.ReLU()]
            for i in range(1, 10):
                layers.append(
                    SimpleResidual(
                        nn.Sequential(
                            nn.Conv1d(7, 7, 3, padding="same", dilation=2 ** min(i, 6)),
                            nn.ReLU(),
                        )
                    )
                )
            return VariableBPNet(nn.Sequential(*layers), feature_dim=7)
        elif model_size == "medium_large":
            # Medium-Large BPNet with 120 channels (~0.4M params)
            layers = [nn.Conv1d(4, 120, 25, padding="same"), nn.ReLU()]
            for i in range(1, 10):
                layers.append(
                    SimpleResidual(
                        nn.Sequential(
                            nn.Conv1d(120, 120, 3, padding="same", dilation=2 ** min(i, 6)),
                            nn.ReLU(),
                        )
                    )
                )
            return VariableBPNet(nn.Sequential(*layers), feature_dim=120)

        elif model_size == "extra_large":
            # Extra-Large BPNet with 170 channels (~0.8M params)
            layers = [nn.Conv1d(4, 170, 25, padding="same"), nn.ReLU()]
            for i in range(1, 10):
                layers.append(
                    SimpleResidual(
                        nn.Sequential(
                            nn.Conv1d(170, 170, 3, padding="same", dilation=2 ** min(i, 6)),
                            nn.ReLU(),
                        )
                    )
                )
            return VariableBPNet(nn.Sequential(*layers), feature_dim=170)

        elif model_size == "xxlarge":
            # XX-Large BPNet with 363 channels (~3.6M params)
            layers = [nn.Conv1d(4, 363, 25, padding="same"), nn.ReLU()]
            for i in range(1, 10):
                layers.append(
                    SimpleResidual(
                        nn.Sequential(
                            nn.Conv1d(363, 363, 3, padding="same", dilation=2 ** min(i, 8)),
                            nn.ReLU(),
                        )
                    )
                )
            return VariableBPNet(nn.Sequential(*layers), feature_dim=363)
        else:
            # Default to small
            return self._create_bpnet_backbone("small")

    def _get_hidden_dim(self, model_size: str) -> int:
        """Get hidden dimension based on model size"""
        size_map = {
            "pico": 7,
            "ultra_tiny": 14,
            "extra_tiny": 30,
            "tiny": 32,
            "small": 64,
            "medium_small": 90,
            "medium": 128,
            "medium_large": 120,  # NEW
            "extra_large": 170,  # NEW
            "large": 256,
            "xxlarge": 363,  # NEW
        }
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
                self.classifier = nn.Linear(self.config.teacher_hidden_size, self.config.num_labels)
            else:
                raise ValueError(
                    f"Invalid teacher projection: {self.config.teacher_projection_opt}"
                )
        else:
            self.teacher_proj = None
            self.classifier = nn.Linear(C, self.config.num_labels)

    def _print_model_info(self):
        """Print model architecture and parameter count"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        print(f"\n{'=' * 60}")
        print(
            f"BPNet Classifier - {self.config.model_type.upper()} ({self.config.model_size.upper()})"
        )
        print(f"{'=' * 60}")
        print(f"Total parameters:      {total_params:,} ({total_params / 1e6:.2f}M)")
        print(f"Trainable parameters:  {trainable_params:,} ({trainable_params / 1e6:.2f}M)")
        print(
            f"Feature dimension:     {self.backbone.feature_dim if hasattr(self.backbone, 'feature_dim') else 'N/A'}"
        )
        print(f"Number of labels:      {self.config.num_labels}")
        print(f"{'=' * 60}\n")

    def forward(self, input_ids, return_feats: bool = False):
        """Forward pass with optional feature return"""
        one_hot = F.one_hot(input_ids, num_classes=4).float()
        x = one_hot.permute(0, 2, 1)

        # Process through backbone
        if isinstance(self.backbone, BPNet):
            out = self.backbone(x)
            feats = out["x"]
        elif isinstance(self.backbone, VariableBPNet):
            feats = self.backbone(x)
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

    def aligned_feats(self, sfeats: torch.Tensor, tfeats: Optional[torch.Tensor] = None):
        """
        align the features from the student and the teacher
        :param sfeats: the features from the student
        :param tfeats: the features from the teacher
        :return: the aligned features
        """
        if self.teacher_proj is not None and tfeats is not None:
            if self.config.teacher_projection_opt == "down":
                return sfeats, self.teacher_proj(tfeats)
            elif self.config.teacher_projection_opt == "up":
                return sfeats, tfeats
        return sfeats, tfeats
