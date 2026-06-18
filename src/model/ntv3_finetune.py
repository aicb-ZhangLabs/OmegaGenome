"""NTv3 backbone + a new per-bp track head, for fine-tuning on the NTv3 Benchmark tasks.

The benchmark tracks are NOT in NTv3's native output (held out from post-training), so reproducing
them requires a task-specific head — exactly what NTv3 does internally (`LinearHead`). This mirrors
that head (LayerNorm -> Linear -> softplus) on the model's ``embedding`` (the post-deconv, pre-head
per-bp features, shape ``[B, L_full, embed_dim]``), cropped to the central 37.5% to match the native
bigwig head's output region.

    forward(input_ids, species_ids) -> tracks [B, T, L_out]   (non-negative; L_out = round(0.375*L_full))

Reuses the loaded NTv3 model as the backbone (load via ``NTv3Teacher``); the new head is the only
required new parameter set. ``freeze_backbone`` toggles head-only (cheap) vs full fine-tune (paper).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# NTv3 crops track outputs to the central 37.5% of the input (per the GitHub v3 doc; verified: a
# 2048-bp input gives a 768-bp bigwig output, 768/2048 = 0.375).
NTV3_CROP_FRAC = 0.375

# NTv3 transformer-tower linears (per audit): Q/K/V (`linear`), attn-out (`mha_output`), FFN (fc1/fc2).
NTV3_LORA_TARGETS = ["linear", "mha_output", "fc1", "fc2"]


def apply_lora(backbone, r: int = 16, alpha: int = 32, dropout: float = 0.05):
    """Wrap the NTv3 backbone with PEFT LoRA on the transformer linears (base frozen, adapters train).

    Cheap alternative to full fine-tune: trains ~1% of params. Returns the PEFT-wrapped backbone,
    which forwards identically (``out.embedding`` preserved) but with LoRA deltas applied.
    """
    from peft import LoraConfig, get_peft_model

    cfg = LoraConfig(r=r, lora_alpha=alpha, lora_dropout=dropout,
                     target_modules=NTV3_LORA_TARGETS, bias="none")
    return get_peft_model(backbone, cfg)


def central_crop(x: torch.Tensor, out_len: int) -> torch.Tensor:
    """Crop the length axis (dim 1) of ``[B, L, C]`` to the centered ``out_len``."""
    start = (x.shape[1] - out_len) // 2
    return x[:, start : start + out_len, :]


class NTv3FineTune(nn.Module):
    """NTv3 backbone + a per-bp track head (mirrors NTv3 ``LinearHead``). Output ``[B, T, L_out]``."""

    def __init__(self, backbone: nn.Module, embed_dim: int, num_tracks: int, freeze_backbone: bool = False):
        super().__init__()
        self.backbone = backbone
        self.freeze_backbone = freeze_backbone
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad_(False)
        # mirror NTv3 LinearHead: LayerNorm(fp32) -> Linear -> softplus (non-negative track signal)
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_tracks)

    def forward(self, input_ids: torch.Tensor, species_ids: torch.Tensor) -> torch.Tensor:
        if self.freeze_backbone:
            with torch.no_grad():
                emb = self.backbone(input_ids=input_ids, species_ids=species_ids).embedding
        else:
            emb = self.backbone(input_ids=input_ids, species_ids=species_ids).embedding
        out_len = round(emb.shape[1] * NTV3_CROP_FRAC)
        emb = central_crop(emb, out_len).float()  # [B, L_out, embed_dim]
        x = F.softplus(self.head(self.norm(emb)))  # [B, L_out, T]
        return x.transpose(1, 2)  # [B, T, L_out]

    def head_parameters(self):
        return list(self.norm.parameters()) + list(self.head.parameters())
