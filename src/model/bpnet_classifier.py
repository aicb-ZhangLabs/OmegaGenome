import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Literal
from dataclasses import dataclass
from .nn.bpnet_pytorch import BPNet, BPNetWidth


@dataclass
class BPNetClassifierConfig:
    num_labels: int = 2
    teacher_hidden_size: Optional[int] = None
    teacher_projection_opt: Literal["down", "up"] = "down"

    # --- INPUT MODE (one-hot vs pretrained-embedding student input) -------------
    # "onehot"        : DEFAULT, unchanged. forward() one-hot encodes input_ids (A/C/G/T -> 4 ch).
    # "nt_embedding"  : forward() instead receives PER-POSITION pretrained DNA-model embeddings
    #                   (shape [B, L, embedding_dim]) and maps them to the backbone's 4-ch input via a
    #                   small learned 1x1 conv front-end (`input_adapter`). Everything downstream of the
    #                   backbone stem (dilated convs, pool, classifier, KD losses) is IDENTICAL to the
    #                   one-hot path, so this is an apples-to-apples input-representation swap.
    input_mode: Literal["onehot", "nt_embedding"] = "onehot"
    # Per-position teacher-embedding dim (NT-2.5B = 2560). Only used when input_mode == "nt_embedding".
    embedding_dim: Optional[int] = None

    # --- EMBEDDING FRONT-END (PI fix: the D->4 1x1 conv is a SEVERE bottleneck that may erase the
    # pretrained embedding's info, making "embedding ~ one-hot" an ARTIFACT). Only used when
    # input_mode == "nt_embedding"; the cached [L,D] embeddings are reused unchanged (no re-precompute).
    #   "replace4"  : DEFAULT/control. adapter D->4, feed BPNet's one-hot 4-ch stem (capacity-matched).
    #   "replaceK"  : adapter D->K (adapter_width), widen the BPNet first conv to consume K channels.
    #   "latefuse_emb"    : stem consumes a D->4 projection of the EMBEDDING (dataset has no one-hot) +
    #                 early dilated blocks -> F[L,C]; project embedding D->fuse_width -> E[L,fw]; CONCAT
    #                 [F,E] (next conv widened to C+fw). Embedding is full-width at the deep concat.
    #   "latefuse_onehot" : FAITHFUL PI version. Input packs BOTH the cached embedding and the REAL
    #                 one-hot ([L, D+4]); the stem runs on the REAL one-hot (4-ch, like the baseline) ->
    #                 F[L,C]; the embedding is projected D->fuse_width and CONCAT'd deep. Tests if the
    #                 pretrained embedding ADDS value on top of a real one-hot BPNet (the motivating question).
    front_end: Literal["replace4", "replaceK", "latefuse_emb", "latefuse_onehot"] = "replace4"
    adapter_width: int = 32  # K for replaceK (adapter output / widened stem in-channels)
    fuse_width: int = 32  # embedding projection width concatenated in latefuse
    fuse_after_block: int = 2  # # of early dilated Residual blocks before the latefuse concat
    # input_adapter projection depth (replace4 / replaceK). 0 (DEFAULT) = a single 1x1 conv = a per-
    # position LINEAR map D->out. >0 = a per-position 2-layer MLP: Conv1d(D->adapter_mlp_hidden) -> ReLU
    # -> Conv1d(adapter_mlp_hidden->out). Still point-wise (kernel_size=1): no mixing across positions,
    # only a nonlinear map of the per-bp embedding vector. NOTE the D->hidden layer is ~D*hidden params
    # (D=2560), so even a small hidden dim makes the adapter large.
    adapter_mlp_hidden: int = 0

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
        "original_emb_matched",  # PARAM-MATCH (replace4 arm, SECONDARY): original BPNet shrunk 64->61
        # ch (full-RF, uncapped dilation, profile/total_count heads kept). backbone+cls=
        # 110,477; +replace4 adapter (2560->4 = 10,244) -> deployable total 120,721 ~= one-hot
        # baseline 121,094 (diff -373). NOTE: replace4 is a 2560->4 BOTTLENECK that mostly
        # discards the embedding -> kept only as a secondary point. Uses BPNetWidth (a BPNet
        # subclass) so latefuse_onehot / replaceK arms work unchanged.
        "emb_matched_replaceK",  # PARAM-MATCH (replaceK arm, HEADLINE): BPNet shrunk to 25 ch. With the
        # replaceK front-end (D->K=32 adapter 81,952 + first conv widened 4->K) the TOTAL =
        # 120,433 ~= one-hot 121,094 (diff -661). Fair "same param budget" question: spend
        # params on a rich NT-embedding front-end (tiny backbone) vs a deep one-hot backbone?
        "emb_matched_latefuse",  # PARAM-MATCH (latefuse_onehot arm, HEADLINE): BPNet shrunk to 34 ch.
        # With the latefuse front-end (embed_proj D->fw=32 + fuse_block) the TOTAL = 120,238
        # ~= one-hot 121,094 (diff -856). Real one-hot stem + deep NT-embedding concat at the
        # SAME 121K param budget as the one-hot baseline.
        "pico",
        "medium_large",
        "extra_large",
        "extra_large_fix",  # NEW: Fixed extra_large with proper dilation
        "xxlarge",
        "bpnet_10m",  # minimal-NT comparison: ~10.5M full-RF BPNet (C=621), matched to NT token-emb+cls
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
            elif config.model_size == "bpnet_10m":
                # minimal-NT comparison: a ~10.5M full-RF BPNet (C=621), matched to the NT
                # token-embedding + classifier size (10,508,800 + 5,122 = 10,513,922). Uses BPNetWidth
                # (good architecture: uncapped dilation 2**i, i=1..9), distilled from the NT-2.5B teacher.
                self.backbone = BPNetWidth(channels=621)
            elif config.model_size in (
                "original_emb_matched",
                "emb_matched_replaceK",
                "emb_matched_latefuse",
            ):
                # PER-ARM param-match: shrink the BPNet width so that backbone + THAT arm's embedding
                # front-end + classifier = TOTAL ~= the one-hot baseline 121,094. Each arm's front-end has a
                # different (mostly C-independent) cost, so each needs a different width. BPNetWidth subclass
                # keeps the BPNet interface (isinstance + .stem) so replaceK / latefuse front-ends work.
                #   replace4  (2560->4 bottleneck, SECONDARY) : C=61 -> total 120,721
                #   replaceK  (D->32 adapter + widened stem)  : C=25 -> total 120,433
                #   latefuse  (real one-hot stem + deep concat): C=34 -> total 120,238
                _matched_ch = {
                    "original_emb_matched": 61,
                    "emb_matched_replaceK": 25,
                    "emb_matched_latefuse": 34,
                }[config.model_size]
                self.backbone = BPNetWidth(channels=_matched_ch)
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

        # --- nt_embedding front-end -----------------------------------------------------
        # Map per-position teacher embeddings (embedding_dim) -> the 4 input channels the backbone's
        # first conv expects, via a learned 1x1 conv (point-wise, no mixing across positions). This is
        # the ONLY architectural addition for the embedding-input variant; its params are reported
        # separately as the "embedding front-end" cost vs the one-hot student. For onehot mode it is None.
        self.input_adapter = None
        self.embed_proj = None  # latefuse: D -> fuse_width projection of the embedding
        self.stem_adapter = None  # latefuse_emb: D -> 4 stem input (None for latefuse_onehot)
        self._latefuse = False
        self._latefuse_onehot = False
        if config.input_mode == "nt_embedding":
            if config.embedding_dim is None:
                raise ValueError(
                    "input_mode='nt_embedding' requires config.embedding_dim (e.g. 2560 for NT-2.5B)"
                )
            self._build_embedding_front_end(config, C)

        # Print model parameters
        self._print_model_info()

    def _make_input_adapter(self, D: int, out_ch: int) -> nn.Module:
        """Per-position embedding->channel projection used by replace4 (out_ch=4) / replaceK (out_ch=K).

        Both variants are point-wise (kernel_size=1): they only transform the per-bp D-vector, never
        mixing across positions.
          - adapter_mlp_hidden == 0 (default): a single 1x1 conv = a LINEAR map D->out_ch.
          - adapter_mlp_hidden  > 0         : a 2-layer MLP  Conv1d(D->h) -> ReLU -> Conv1d(h->out_ch),
            i.e. a nonlinear per-position projection. (First layer is ~D*h params, D=2560, so large.)
        """
        h = getattr(self.config, "adapter_mlp_hidden", 0) or 0
        if h > 0:
            return nn.Sequential(
                nn.Conv1d(D, h, kernel_size=1),
                nn.ReLU(),
                nn.Conv1d(h, out_ch, kernel_size=1),
            )
        return nn.Conv1d(D, out_ch, kernel_size=1)

    def _build_embedding_front_end(self, config, C: int):
        """Construct the chosen embedding front-end (PI fix). Reuses the cached [L,D] embeddings.

        replace4 : adapter D->4, unchanged BPNet 4-ch stem (capacity-matched control).
        replaceK : adapter D->K, widen the stem's first conv in_channels 4->K (input-replace, no D->4
                   bottleneck). Only the first conv's in_channels changes; everything else identical.
        latefuse : keep the 4-ch one-hot stem + `fuse_after_block` early dilated Residual blocks, then
                   CONCAT a D->fuse_width projection of the embedding onto those features and widen the
                   next block's conv in_channels C -> C+fuse_width. Injects the embedding deeper to test
                   if it ADDS to one-hot. Only supported for model_size='original' (the embedding-input student).
        """
        D = config.embedding_dim
        fe = config.front_end
        if fe == "replace4":
            self.input_adapter = self._make_input_adapter(D, 4)
            return
        if fe == "replaceK":
            K = config.adapter_width
            self.input_adapter = self._make_input_adapter(D, K)
            # Widen the backbone's FIRST conv to consume K channels (4 -> K). original BPNet: stem[0];
            # VariableBPNet: layers[0]. Preserves out_channels / kernel / padding; only in_channels grows.
            first = self._first_conv()
            new_first = nn.Conv1d(
                K,
                first.out_channels,
                first.kernel_size[0],
                padding=first.padding,
                dilation=first.dilation,
            )
            self._set_first_conv(new_first)
            return
        if fe in ("latefuse_emb", "latefuse_onehot"):
            if not isinstance(self.backbone, BPNet):
                raise ValueError("latefuse front_end currently supports model_size='original' only")
            self._latefuse = True
            self._latefuse_onehot = fe == "latefuse_onehot"
            # Stem branch input differs by arm:
            #   latefuse_emb    : dataset yields only the embedding [L,D]; the stem consumes a learned
            #                     D->4 projection of it (stem_adapter). The embedding's FULL info is still
            #                     injected non-bottlenecked at the deep concat (embed_proj).
            #   latefuse_onehot : dataset packs [L, D+4] (embedding ++ REAL one-hot); the stem consumes
            #                     the REAL one-hot directly (no stem_adapter) -- exactly the baseline stem.
            self.stem_adapter = None if self._latefuse_onehot else nn.Conv1d(D, 4, kernel_size=1)
            self.embed_proj = nn.Conv1d(D, config.fuse_width, kernel_size=1)
            # Split the original stem: [conv0, relu, Residual x9]. Keep the first `fuse_after_block`
            # Residual blocks (plus conv0+relu = the first 2 modules) as the EARLY one-hot path; the
            # rest run AFTER the concat. The block right after the concat gets a widened in_channels conv.
            stem = list(self.backbone.stem)  # [Conv1d, ReLU, Res, Res, ..., Res]  (2 + 9)
            n_early = 2 + max(0, config.fuse_after_block)  # conv0 + relu + N residual blocks
            self._early = nn.Sequential(*stem[:n_early])
            late_blocks = stem[
                n_early:
            ]  # remaining Residual blocks (each: Residual(Sequential(Conv,ReLU)))
            if len(late_blocks) == 0:
                raise ValueError("fuse_after_block too large: no blocks left after the concat")
            # The first late block's inner conv must consume C + fuse_width instead of C, and (since it is
            # Residual) its OUTPUT must go back to C so the residual add + subsequent blocks stay valid.
            # We therefore replace that Residual block with a NON-residual widen-conv (C+fw -> C) + ReLU,
            # then keep the remaining (still-residual) blocks unchanged.
            fw = config.fuse_width
            first_late = late_blocks[0]  # Residual(Sequential(Conv1d(C,C,3,dil), ReLU))
            inner = first_late.fn  # Sequential(Conv1d, ReLU)
            old_conv = inner[0]
            fuse_conv = nn.Sequential(
                nn.Conv1d(
                    C + fw,
                    C,
                    old_conv.kernel_size[0],
                    padding=old_conv.padding,
                    dilation=old_conv.dilation,
                ),
                nn.ReLU(),
            )
            self._fuse_block = fuse_conv
            self._late = nn.Sequential(*late_blocks[1:])  # remaining residual blocks (in/out = C)
            # The early/late residual blocks are SHARED references into the original stem (kept alive by
            # _early/_late). Drop self.backbone so its unused parts (profile/total_count heads + the
            # replaced first-late block) don't double-register or inflate the param count. The latefuse
            # forward never touches self.backbone.
            self.backbone = None
            return
        raise ValueError(f"unknown front_end {fe}")

    def _first_conv(self) -> nn.Conv1d:
        """Return the backbone's first Conv1d (original BPNet stem[0] or VariableBPNet layers[0])."""
        if isinstance(self.backbone, BPNet):
            return self.backbone.stem[0]
        return self.backbone.layers[0]

    def _set_first_conv(self, conv: nn.Conv1d):
        """Replace the backbone's first Conv1d in place (mirror of _first_conv)."""
        if isinstance(self.backbone, BPNet):
            self.backbone.stem[0] = conv
        else:
            self.backbone.layers[0] = conv

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
            # NOTE: This model has POOR performance due to dilation cap of 6
            # Use 'extra_large_fix' instead for better results
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
            "original_emb_matched": 61,  # param-matched replace4 (64->61 ch)
            "emb_matched_replaceK": 25,  # param-matched replaceK arm
            "emb_matched_latefuse": 34,  # param-matched latefuse arm
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

        # Calculate parameters excluding teacher projection (deployment size). latefuse sets
        # self.backbone=None (its conv stack lives in _early/_fuse_block/_late), so sum those instead.
        if self.backbone is not None:
            backbone_params = sum(p.numel() for p in self.backbone.parameters())
        else:
            _lf_mods = [self._early, self._fuse_block, self._late, self.embed_proj]
            if self.stem_adapter is not None:
                _lf_mods.append(self.stem_adapter)
            backbone_params = sum(p.numel() for m in _lf_mods for p in m.parameters())
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
            print(f"Deployment parameters: {deployment_params:,} ({deployment_params / 1e6:.2f}M)")
            print(f"  └─ Backbone:         {backbone_params:,} ({backbone_params / 1e6:.2f}M)")
            print(f"  └─ Classifier head:  {classifier_params:,} ({classifier_params / 1e6:.2f}M)")
            print(
                f"\nTeacher projection:    {teacher_proj_params:,} ({teacher_proj_params / 1e6:.2f}M) (training only)"
            )
        else:
            print("\n(No teacher projection - all parameters are deployment parameters)")

        print(
            f"\nFeature dimension:     {getattr(self.backbone, 'feature_dim', 'N/A') if self.backbone is not None else 'N/A (latefuse)'}"
        )
        print(f"Number of labels:      {self.config.num_labels}")
        print(f"{'=' * 60}\n")

    def forward(self, input_ids, return_feats: bool = False):
        """Forward pass with optional feature return.

        Input contract depends on ``config.input_mode``:
          - "onehot"       : ``input_ids`` is a LongTensor [B, L] of A/C/G/T indices (DEFAULT, unchanged);
                             one-hot encoded to [B, 4, L] here.
          - "nt_embedding" : ``input_ids`` is a FloatTensor [B, L, embedding_dim] of per-position teacher
                             embeddings; mapped to [B, 4, L] by the learned 1x1 ``input_adapter`` conv.
        Both branches produce a [B, 4, L] tensor ``x`` fed to the (identical) backbone.
        """
        if self.config.input_mode == "nt_embedding" and self._latefuse:
            # latefuse: stem (4-ch) + early dilated blocks -> F[B,C,L]; concat a full-width
            # D->fuse_width projection of the EMBEDDING -> widened fuse conv + remaining residual blocks.
            # No D->4 bottleneck on the embedding's deep injection. Stem input differs by arm:
            if self._latefuse_onehot:
                # input_ids packs [B, L, D+4] = embedding ++ REAL one-hot. Split and run the stem on the
                # REAL one-hot (exactly the baseline 4-ch stem).
                D = self.config.embedding_dim
                emb = input_ids[..., :D].permute(0, 2, 1).float()  # [B, D, L]
                x = input_ids[..., D:].permute(0, 2, 1).float()  # [B, 4, L] real one-hot
            else:
                # latefuse_emb: input_ids is [B, L, D]; stem consumes a D->4 projection of the embedding.
                emb = input_ids.permute(0, 2, 1).float()  # [B, D, L]
                x = self.stem_adapter(emb)  # [B, 4, L]
            f = self._early(x)  # [B, C, L]
            e = self.embed_proj(emb)  # [B, fuse_width, L]
            fused = torch.cat([f, e], dim=1)  # [B, C+fuse_width, L]
            h = self._fuse_block(fused)  # [B, C, L]
            feats = self._late(h)  # [B, C, L]
        else:
            if self.config.input_mode == "nt_embedding":
                # replace4 (D->4) / replaceK (D->K): input_ids [B,L,D] -> [B,D,L] -> adapter -> [B,*,L]
                x = input_ids.permute(0, 2, 1).float()
                x = self.input_adapter(x)
            else:
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
