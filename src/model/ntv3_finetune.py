"""NTv3 post-trained backbone + a fresh per-position track head, for fine-tuning on the NTv3
Benchmark bigWig tracks.

Faithful port of the model in InstaDeepAI's official notebook
``03_fine_tuning_posttrained_model_biwig.ipynb`` (``HFModelWithHead``): the benchmark's 34 human
tracks are NOT in NTv3's native output, so fine-tuning rebuilds the *headless* conditioned backbone
(``core``), loads the post-trained weights into it, and attaches a new ``LinearHead`` predicting the
benchmark tracks at single-nucleotide resolution over the central ``keep_target_center_fraction``
(0.375) of the input.

    forward(tokens) -> {"bigwig_tracks_logits": [B, L_out, num_tracks]}   (non-negative; L_out = 0.375*L)

Full fine-tuning by default (the paper recipe); ``use_lora`` offers a cheap LoRA alternative.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data.ntv3_ft_data import crop_center

# NTv3 crops track outputs to the central 37.5% of the input (per the GitHub v3 doc / notebook).
NTV3_CROP_FRAC = 0.375

# NTv3 transformer-tower linears (per audit): Q/K/V (`linear`), attn-out (`mha_output`), FFN (fc1/fc2).
NTV3_LORA_TARGETS = ["linear", "mha_output", "fc1", "fc2"]


def apply_lora(backbone, r: int = 16, alpha: int = 32, dropout: float = 0.05):
    """Wrap an NTv3 backbone with PEFT LoRA on the transformer linears (base frozen, adapters train)."""
    from peft import LoraConfig, get_peft_model

    cfg = LoraConfig(r=r, lora_alpha=alpha, lora_dropout=dropout,
                     target_modules=NTV3_LORA_TARGETS, bias="none")
    return get_peft_model(backbone, cfg)


class LinearHead(nn.Module):
    """NTv3 per-position track head: LayerNorm -> Linear -> softplus (non-negative track signal)."""

    def __init__(self, embed_dim: int, num_labels: int):
        super().__init__()
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_labels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.softplus(self.head(self.layer_norm(x)))


class NTv3BigWigModel(nn.Module):
    """NTv3 post-trained backbone (headless ``core``) + a fresh ``LinearHead`` over the central crop.

    Mirrors the official notebook's ``HFModelWithHead``: rebuild the conditioned backbone's base class
    (drops the native bigwig/bed/LM heads), load the post-trained weights (``strict=False``), condition
    on the species token, and predict the benchmark tracks at single-nt resolution.
    """

    def __init__(self, model_name: str, num_tracks: int, species_str: str = "human",
                 keep_target_center_fraction: float = NTV3_CROP_FRAC, use_lora: bool = False,
                 lora_r: int = 16, lora_alpha: int = 32, local_files_only: bool = False):
        super().__init__()
        from transformers import AutoConfig, AutoModel

        self.config = AutoConfig.from_pretrained(model_name, trust_remote_code=True,
                                                 local_files_only=local_files_only)
        base = AutoModel.from_pretrained(model_name, trust_remote_code=True, config=self.config,
                                         local_files_only=local_files_only)
        # Rebuild the headless conditioned backbone (parent class of `core`) and load the post-trained
        # weights; strict=False drops the native heads we don't fine-tune.
        discrete_conditioned_model = type(base.core).__bases__[0]
        self.core = discrete_conditioned_model(self.config)
        self.load_state_dict(base.state_dict(), strict=False)
        del base

        # Species conditioning token (fall back to the mask token id 2 for unsupported species).
        species_id = self.config.species_to_token_id.get(species_str, 2) \
            if hasattr(self.config, "species_to_token_id") else 2
        self.register_buffer("species_ids", torch.LongTensor([species_id]), persistent=False)

        self.keep_target_center_fraction = keep_target_center_fraction
        self.bigwig_head = LinearHead(self.config.embed_dim, num_tracks)
        if use_lora:
            self.core = apply_lora(self.core, r=lora_r, alpha=lora_alpha)

    def forward(self, tokens: torch.Tensor) -> dict:
        species_tokens = torch.repeat_interleave(self.species_ids, tokens.shape[0]).to(tokens.device)
        outputs = self.core(tokens, [species_tokens], output_hidden_states=True)
        emb = outputs["hidden_states"][-1]  # [B, seq_len, embed_dim] @ single-nt resolution
        if self.keep_target_center_fraction < 1.0:
            emb = crop_center(emb, self.keep_target_center_fraction)
        return {"bigwig_tracks_logits": self.bigwig_head(emb)}  # [B, L_out, num_tracks]
