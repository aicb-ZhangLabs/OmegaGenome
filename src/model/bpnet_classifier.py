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
        "deploy_120k",  # ~0.12M deploy params (65 ch) — NOTE: dilation CAPPED at 64 (small RF). For
                        # Carbon-3B distillation use `original` instead (full-RF, matches other teachers).
        "pico",
        "medium_large",
        "extra_large",
        "extra_large_fix",  # NEW: Fixed extra_large with proper dilation
        "xxlarge",
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
        """Create BPNet variant based on size - returns VariableBPNet with correct feature_dim

        ARCHITECTURE NOTES:
        - The original BPNet uses dilation 2^i for i in range(1,10), giving dilations up to 512
        - This large receptive field is crucial for capturing long-range dependencies in DNA
        - Models with dilation caps (e.g., min(i, 6) caps at 64) have reduced receptive fields
        - The 'extra_large' model uses cap 6, which is why it underperforms vs original
        - The 'extra_large_fix' removes the cap to match original BPNet's architecture
        """
        if model_size == "tiny":
            # Tiny BPNet with fewer channels (32)
            layers = nn.Sequential(
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
            return VariableBPNet(layers, feature_dim=32)

        elif model_size == "small":
            # Small BPNet with 64 channels (similar to original)
            layers = nn.Sequential(
                nn.Conv1d(4, 64, 21, padding="same"),
                nn.ReLU(),
                SimpleResidual(
                    nn.Sequential(
                        nn.Conv1d(64, 64, 3, padding="same", dilation=2), nn.ReLU()
                    )
                ),
                SimpleResidual(
                    nn.Sequential(
                        nn.Conv1d(64, 64, 3, padding="same", dilation=4), nn.ReLU()
                    )
                ),
                SimpleResidual(
                    nn.Sequential(
                        nn.Conv1d(64, 64, 3, padding="same", dilation=8), nn.ReLU()
                    )
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
                            nn.Conv1d(
                                128, 128, 3, padding="same", dilation=2 ** min(i, 6)
                            ),
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
                            nn.Conv1d(
                                256, 256, 3, padding="same", dilation=2 ** min(i, 8)
                            ),
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
                            nn.Conv1d(
                                90, 90, 3, padding="same", dilation=2 ** min(i, 6)
                            ),
                            nn.ReLU(),
                        )
                    )
                )
            return VariableBPNet(nn.Sequential(*layers), feature_dim=90)

        elif model_size == "deploy_120k":
            # ~0.12M deployment-param BPNet (65 channels). NOTE: dilation is CAPPED at 2**min(i,6)=64,
            # NOT uncapped like the `original` backbone (which goes to 2**9=512). The cap shrinks the
            # receptive field ~8x and underperforms on long-range tasks — most acutely splice_donor
            # (0.63 vs 0.85 for `original`). Kept for backward-compat with prior checkpoints; for
            # Carbon-3B distillation use `original` (full RF, and matches the NT/Enformer/Caduceus/
            # DNABERT-2 students). Dilation adds no params, so the cap was pure downside.
            layers = [nn.Conv1d(4, 65, 25, padding="same"), nn.ReLU()]
            for i in range(1, 10):
                layers.append(
                    SimpleResidual(
                        nn.Sequential(
                            nn.Conv1d(65, 65, 3, padding="same", dilation=2 ** min(i, 6)),
                            nn.ReLU(),
                        )
                    )
                )
            return VariableBPNet(nn.Sequential(*layers), feature_dim=65)

        elif model_size == "extra_tiny":
            # Extra Tiny BPNet with 30 channels (~0.1m/4 = 25k params)
            # Following the successful medium/large/medium_small structure
            layers = [nn.Conv1d(4, 30, 25, padding="same"), nn.ReLU()]
            for i in range(1, 10):
                layers.append(
                    SimpleResidual(
                        nn.Sequential(
                            nn.Conv1d(
                                30, 30, 3, padding="same", dilation=2 ** min(i, 6)
                            ),
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
                            nn.Conv1d(
                                14, 14, 3, padding="same", dilation=2 ** min(i, 6)
                            ),
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
                            nn.Conv1d(
                                120, 120, 3, padding="same", dilation=2 ** min(i, 6)
                            ),
                            nn.ReLU(),
                        )
                    )
                )
            return VariableBPNet(nn.Sequential(*layers), feature_dim=120)

        elif model_size == "extra_large":
            # Extra-Large BPNet with 170 channels (~0.8M params)
            # NOTE: This model has POOR performance due to dilation cap of 6
            # Use 'extra_large_fix' instead for better results
            layers = [nn.Conv1d(4, 170, 25, padding="same"), nn.ReLU()]
            for i in range(1, 10):
                layers.append(
                    SimpleResidual(
                        nn.Sequential(
                            nn.Conv1d(
                                170, 170, 3, padding="same", dilation=2 ** min(i, 6)
                            ),
                            nn.ReLU(),
                        )
                    )
                )
            return VariableBPNet(nn.Sequential(*layers), feature_dim=170)

        elif model_size == "extra_large_fix":
            # ================================================================
            # FIXED Extra-Large BPNet with 170 channels (~0.8M params)
            # ================================================================
            # KEY FIX: No dilation cap! Uses 2**i like original BPNet
            #
            # The original 'extra_large' uses dilation cap of 6 (max dilation 64),
            # which limits the receptive field and causes poor performance.
            #
            # Original BPNet with 64 channels achieves MCC 0.9078
            # extra_large with 170 channels only achieves MCC 0.8612 (WORSE!)
            #
            # This is because the dilation cap prevents the model from capturing
            # long-range dependencies in DNA sequences, which are crucial for
            # genomic tasks like splice site prediction.
            #
            # This fixed version removes the dilation cap to match the original
            # BPNet architecture, which should restore proper scaling behavior.
            # ================================================================
            layers = [nn.Conv1d(4, 170, 25, padding="same"), nn.ReLU()]
            for i in range(1, 10):
                layers.append(
                    SimpleResidual(
                        nn.Sequential(
                            # NO CAP: dilation goes 2, 4, 8, 16, 32, 64, 128, 256, 512
                            nn.Conv1d(170, 170, 3, padding="same", dilation=2**i),
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
                            nn.Conv1d(
                                363, 363, 3, padding="same", dilation=2 ** min(i, 8)
                            ),
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
            "medium_large": 120,
            "extra_large": 170,
            "extra_large_fix": 170,  # Same as extra_large
            "large": 256,
            "xxlarge": 363,
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

    def _print_model_info(self):
        """Print model architecture and parameter count"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        # Calculate parameters excluding teacher projection (deployment size)
        backbone_params = sum(p.numel() for p in self.backbone.parameters())
        pool_params = 0  # AdaptiveAvgPool1d has no parameters
        classifier_params = sum(p.numel() for p in self.classifier.parameters())

        deployment_params = backbone_params + pool_params + classifier_params

        # Calculate teacher projection params (if exists)
        teacher_proj_params = 0
        if self.teacher_proj is not None:
            teacher_proj_params = sum(p.numel() for p in self.teacher_proj.parameters())

        print(f"\n{'=' * 60}")
        print(
            f"BPNet Classifier - {self.config.model_type.upper()} ({self.config.model_size.upper()})"
        )
        print(f"{'=' * 60}")
        print(
            f"Total parameters (with teacher proj):  {total_params:,} ({total_params / 1e6:.2f}M)"
        )
        print(
            f"Trainable parameters:                  {trainable_params:,} ({trainable_params / 1e6:.2f}M)"
        )

        # Show deployment size (without teacher projection)
        if teacher_proj_params > 0:
            print("\n--- Deployment Configuration (teacher proj excluded) ---")
            print(
                f"Deployment parameters: {deployment_params:,} ({deployment_params / 1e6:.2f}M)"
            )
            print(
                f"  └─ Backbone:         {backbone_params:,} ({backbone_params / 1e6:.2f}M)"
            )
            print(
                f"  └─ Classifier head:  {classifier_params:,} ({classifier_params / 1e6:.2f}M)"
            )
            print(
                f"\nTeacher projection:    {teacher_proj_params:,} ({teacher_proj_params / 1e6:.2f}M) (training only)"
            )
        else:
            print(
                "\n(No teacher projection - all parameters are deployment parameters)"
            )

        print(
            f"\nFeature dimension:     {self.backbone.feature_dim if hasattr(self.backbone, 'feature_dim') else 'N/A'}"
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

    def aligned_feats(
        self, sfeats: torch.Tensor, tfeats: Optional[torch.Tensor] = None
    ):
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
