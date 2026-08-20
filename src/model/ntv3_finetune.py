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

# Scaled RANDOM-INIT NTv3-pretrained variants for the from-scratch size-ladder (InstaDeep publishes
# NTv3 ONLY at 8M/100M/650M — no ckpt exists at 4M/30M/300M, so those tiers can only be built random-init
# then distilled). Each reuses the 8m_pre snapshot's modeling .py + tokenizer as a TEMPLATE; only the
# config width/depth dims below change (family invariants held: ffn=4*embed, conv_init_embed_dim=embed,
# num_downsamples=7, token_embed_dim=16; head_dim=embed/heads=key_size). The 8m/100m entries carry the
# EXISTING native dims so the from-scratch series can also instantiate those sizes random-init (the
# same-protocol reference points that de-confound the pretrained-vs-scratch curve). Measured CPU param
# counts (see design §3): 4m=4.34M, 8m=7.69M, 30m=29.87M, 100m=106.46M, 300m=303.05M.
NTV3_TEMPLATE = "/extra/zhanglab0/INDV/pengchx3/ntv3_local/8m_pre"  # prepare_local_snapshot -> _loadable
NTV3_SCALED_SIZES = {  # embed_dim, num_layers, attention_heads, ffn_embed_dim, key_size
    "ntv3-4m":   dict(embed_dim=192,  num_layers=2, attention_heads=6,  ffn_embed_dim=768,  key_size=32),
    "ntv3-8m":   dict(embed_dim=256,  num_layers=2, attention_heads=8,  ffn_embed_dim=1024, key_size=32),
    "ntv3-30m":  dict(embed_dim=448,  num_layers=4, attention_heads=8,  ffn_embed_dim=1792, key_size=56),
    "ntv3-100m": dict(embed_dim=768,  num_layers=6, attention_heads=12, ffn_embed_dim=3072, key_size=64),
    "ntv3-300m": dict(embed_dim=1152, num_layers=9, attention_heads=18, ffn_embed_dim=4608, key_size=64),
}


def scaled_ntv3_config(size_key: str, template: str = NTV3_TEMPLATE):
    """Build an NTv3 PRE config for a random-init scaled student size (a ``NTV3_SCALED_SIZES`` key),
    using the 8m_pre snapshot as the modeling/tokenizer TEMPLATE and overriding only the width/depth
    dims. ``conv_init_embed_dim`` is tied to ``embed_dim`` (family invariant). Returns a config ready for
    ``AutoModelForMaskedLM.from_config(cfg, trust_remote_code=True)`` (random weights, no ckpt loaded)."""
    from transformers import AutoConfig
    from src.model.ntv3_teacher import prepare_local_snapshot

    dims = NTV3_SCALED_SIZES[size_key]
    cfg = AutoConfig.from_pretrained(prepare_local_snapshot(template), trust_remote_code=True,
                                     local_files_only=True)
    cfg.embed_dim = dims["embed_dim"]
    cfg.conv_init_embed_dim = dims["embed_dim"]  # family invariant: conv stem width == embed_dim
    cfg.num_layers = dims["num_layers"]
    cfg.attention_heads = dims["attention_heads"]
    cfg.ffn_embed_dim = dims["ffn_embed_dim"]
    cfg.key_size = dims["key_size"]
    return cfg


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
        # `features` = the per-bp hidden embedding feeding the head (cropped to L_out, so teacher and
        # student features share positions) — used for FitNets-style KD feature alignment.
        return {"bigwig_tracks_logits": self.bigwig_head(emb), "features": emb}  # [B, L_out, *]


class NTv3PreBigWigModel(nn.Module):
    """PRETRAINED NTv3 backbone (e.g. 8M) + a fresh ``LinearHead`` — sibling of ``NTv3BigWigModel``.

    PRE checkpoints (``NTv3PreTrained``) register as ``AutoModelForMaskedLM`` (NOT ``AutoModel``),
    have no ``.core`` and no species conditioning — but their last hidden state is a per-nt embedding
    ``[B, L, embed_dim]``, so we attach the *same* track head over the central crop. Used for the 8M
    baseline fine-tune and as the KD student. Identical forward contract to ``NTv3BigWigModel``:
    ``{"bigwig_tracks_logits": [B, L_out, num_tracks]}`` (non-negative, ``L_out = 0.375*L``).
    """

    def __init__(self, model_name: str, num_tracks: int,
                 keep_target_center_fraction: float = NTV3_CROP_FRAC, use_lora: bool = False,
                 lora_r: int = 16, lora_alpha: int = 32, local_files_only: bool = False, **_ignored):
        super().__init__()
        from transformers import AutoConfig, AutoModelForMaskedLM

        if model_name in NTV3_SCALED_SIZES:
            # RANDOM-INIT scaled tier (no InstaDeep ckpt at this size): build the config off the 8m_pre
            # template with the tier's width/depth overrides and instantiate FRESH weights via from_config
            # (the pretrained `from_pretrained` path below is untouched). Head + forward are identical —
            # both read self.config.embed_dim, so the fresh backbone drops into the same distill/eval loop.
            self.config = scaled_ntv3_config(model_name)
            self.backbone = AutoModelForMaskedLM.from_config(self.config, trust_remote_code=True)
        else:
            self.config = AutoConfig.from_pretrained(model_name, trust_remote_code=True,
                                                     local_files_only=local_files_only)
            self.backbone = AutoModelForMaskedLM.from_pretrained(
                model_name, trust_remote_code=True, config=self.config, local_files_only=local_files_only)
        self.keep_target_center_fraction = keep_target_center_fraction
        self.bigwig_head = LinearHead(self.config.embed_dim, num_tracks)
        if use_lora:
            self.backbone = apply_lora(self.backbone, r=lora_r, alpha=lora_alpha)

    def forward(self, tokens: torch.Tensor) -> dict:
        out = self.backbone(tokens, output_hidden_states=True)
        emb = out.hidden_states[-1] if hasattr(out, "hidden_states") else out["hidden_states"][-1]
        if self.keep_target_center_fraction < 1.0:
            emb = crop_center(emb, self.keep_target_center_fraction)
        # `features` = per-bp hidden embedding (cropped to L_out) for KD feature alignment; see sibling.
        return {"bigwig_tracks_logits": self.bigwig_head(emb), "features": emb}  # [B, L_out, *]


# NTv3 single-nucleotide tokenizer ids (vocab.json): A=6, C=8, G=9, T=7 (one token per bp). The
# BPNet student one-hots these from the SAME ``tokens`` the transformer students receive, so it is a
# drop-in alternative — special/pad/N tokens map to an all-zero (absent-base) column.
NTV3_NUC_IDS = {"A": 6, "C": 8, "G": 9, "T": 7}
NTV3_VOCAB_SIZE = 11


def _load_carbon_bpnet_module():
    """Load the sibling ``code_carbon`` BPNet module by file path (it is self-contained: torch + einops
    only), so we reuse the tested architecture without coupling the two ``src`` packages."""
    import importlib.util
    import os

    here = os.path.dirname(os.path.abspath(__file__))
    bp_path = os.path.normpath(os.path.join(
        here, "..", "..", "..", "code_carbon", "src", "model", "nn", "bpnet_pytorch.py"))
    spec = importlib.util.spec_from_file_location("carbon_bpnet_pytorch", bp_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _carbon_bpnet_backbone(model_size: str = "original", channels: int = None, n_dilated: int = None):
    """REUSE the tested ``code_carbon`` BPNet dilated-conv tower (the same one the Carbon-3B distillation
    distills into — full receptive field, dilation 2^i, NO cap). Returns ``(backbone_module, feat_dim)``
    where the module maps one-hot ``[B, 4, L]`` -> per-bp features ``[B, C, L]``.

    ``channels`` + ``n_dilated`` knobs build a custom-width/-depth tower with the SAME block structure
    (stem Conv1d(4,C,25) + ``n_dilated`` ``Residual(Conv1d(C,C,3,dilation=2^i)+ReLU)`` blocks), reusing
    the carbon ``Residual`` primitive so it stays one architecture. With both unset, returns the stock
    ``original`` 64-ch / 9-layer ``BPNet().stem``.
    """
    mod = _load_carbon_bpnet_module()
    if channels is None and n_dilated is None:
        if model_size != "original":
            raise ValueError(f"named size {model_size!r} not wired; pass channels/n_dilated instead")
        return mod.BPNet().stem, 64  # stock original tower (64 ch, dilation 2..512)
    channels = channels or 64
    n_dilated = n_dilated or 9
    layers = [nn.Conv1d(4, channels, 25, padding="same"), nn.ReLU()]
    for i in range(1, n_dilated + 1):  # dilation 2^i, uncapped -> full receptive field (matches original)
        layers.append(mod.Residual(
            nn.Sequential(nn.Conv1d(channels, channels, 3, padding="same", dilation=2 ** i), nn.ReLU())))
    return nn.Sequential(*layers), channels


class BPNetTrackStudent(nn.Module):
    """BPNet dilated-CNN student for per-bp track regression — the genomics-native architecture, tiny
    (~0.5M params) and fast to train, as an alternative to the transformer NTv3 student.

    REUSES the existing, tested ``code_carbon`` BPNet tower (``_carbon_bpnet_backbone``); this wrapper
    only adds what the NTv3 track task needs on top: (1) one-hot the SAME single-nt ``tokens`` the NTv3
    students receive (the carbon BPNet expects one-hot ``[B,4,L]``, NTv3 tokens are ids 6-9), (2) crop to
    the central ``keep_target_center_fraction``, (3) the per-bp track head. Identical forward contract to
    the NTv3 students so it drops into the same training / KD loop and metrics unchanged:
        forward(tokens) -> {"bigwig_tracks_logits": [B, L_out, num_tracks], "features": [B, L_out, C]}
    """

    def __init__(self, model_name: str = None, num_tracks: int = 1,
                 keep_target_center_fraction: float = NTV3_CROP_FRAC, model_size: str = "original",
                 channels: int = None, n_dilated: int = None,
                 nuc_ids: dict = None, vocab_size: int = NTV3_VOCAB_SIZE, **_ignored):
        super().__init__()
        self.keep_target_center_fraction = keep_target_center_fraction
        self.backbone, channels = _carbon_bpnet_backbone(model_size, channels, n_dilated)

        class _Cfg:  # lightweight stand-in for the HF config the NTv3 students carry (KD reads embed_dim)
            pass
        self.config = _Cfg()
        self.config.embed_dim = channels

        # One-hot lookup [vocab_size, 4] in ACGT channel order; non-base tokens -> all-zero row. (NTv3
        # tokens are 6-9, vs the carbon BPNet's F.one_hot(input_ids, 4) on 0-3 — same 4 channels.)
        nuc_ids = nuc_ids or NTV3_NUC_IDS
        lut = torch.zeros(vocab_size, 4)
        for ch, base in enumerate("ACGT"):
            lut[nuc_ids[base], ch] = 1.0
        self.register_buffer("nuc_lut", lut, persistent=False)

        self.head = LinearHead(channels, num_tracks)  # LayerNorm -> Linear -> softplus (same NTv3 head)

    def forward(self, tokens: torch.Tensor) -> dict:
        x = self.nuc_lut[tokens].transpose(1, 2)        # [B, 4, L]  (Conv1d wants channels-first)
        x = self.backbone(x)                            # [B, C, L]  (reused carbon dilated tower)
        emb = x.transpose(1, 2)                         # [B, L, C]  (channels-last, like NTv3 emb)
        if self.keep_target_center_fraction < 1.0:
            emb = crop_center(emb, self.keep_target_center_fraction)
        return {"bigwig_tracks_logits": self.head(emb), "features": emb}  # [B, L_out, *]


def _is_pretrained_ckpt(config) -> bool:
    """PRE (masked-LM) vs POST: PRE checkpoints (NTv3-8M) have no native bigwig/species head and load
    via AutoModelForMaskedLM; POST (100M/650M) carry ``bigwigs_per_species`` + a track ``core``."""
    arch = (getattr(config, "architectures", None) or [""])[0]
    return "Pre" in arch or not hasattr(config, "bigwigs_per_species")


def build_bigwig_model(model_name: str, num_tracks: int, **kwargs):
    """One factory for every NTv3 size: dispatches to NTv3PreBigWigModel (PRE, e.g. 8M) or
    NTv3BigWigModel (POST, e.g. 100M/650M) by inspecting the config — so the same fine-tune / distill
    entrypoint works for the 8M baseline, the teachers, and KD students with strictly-correct loading.
    Local snapshot dirs are auto-prepared (gated auto_map stripped) before loading.
    """
    import os
    from transformers import AutoConfig
    from src.model.ntv3_teacher import prepare_local_snapshot

    # BPNet student: a from-scratch dilated CNN — no pretrained backbone to load, so dispatch early.
    # bpnet-only kwargs are stripped here so they never reach the NTv3 constructors (the POST 650M
    # model has no **kwargs); the NTv3 path simply ignores them.
    is_bpnet = kwargs.pop("student_arch", None) == "bpnet"
    bp_keys = ("model_size", "channels", "n_dilated", "nuc_ids", "vocab_size")
    bp_kwargs = {k: kwargs.pop(k) for k in bp_keys if k in kwargs}
    if is_bpnet:
        if "keep_target_center_fraction" in kwargs:
            bp_kwargs["keep_target_center_fraction"] = kwargs["keep_target_center_fraction"]
        return BPNetTrackStudent(model_name, num_tracks, **bp_kwargs)

    # Random-init scaled NTv3 tier (from-scratch size-ladder): the from_config branch in
    # NTv3PreBigWigModel builds it fresh from NTV3_SCALED_SIZES — no path/config on disk to inspect.
    if model_name in NTV3_SCALED_SIZES:
        return NTv3PreBigWigModel(model_name, num_tracks, **kwargs)

    if os.path.isdir(model_name):
        model_name = prepare_local_snapshot(model_name)
        kwargs["local_files_only"] = True
    cfg = AutoConfig.from_pretrained(model_name, trust_remote_code=True,
                                     local_files_only=kwargs.get("local_files_only", False))
    cls = NTv3PreBigWigModel if _is_pretrained_ckpt(cfg) else NTv3BigWigModel
    return cls(model_name, num_tracks, **kwargs)


def load_finetuned_bigwig_teacher(ckpt_path: str, base_model: str, num_tracks: int,
                                  device="cpu", **kwargs):
    """Load a FINE-TUNED NTv3 bigWig model as a frozen KD teacher (e.g. the reproduced 650M
    ``best_model.pth``). Reuses ``build_bigwig_model`` to construct the arch on ``base_model``, then
    overwrites with the fine-tuned ``state_dict`` (best_model.pth is a raw state_dict). Returns the
    model on ``device`` in eval mode with grads off — same ``{"bigwig_tracks_logits": [B,L_out,T]}``
    forward contract, so it drops into the KD loss directly.
    """
    model = build_bigwig_model(base_model, num_tracks, **kwargs)
    sd = torch.load(ckpt_path, map_location="cpu")
    sd = sd["model"] if isinstance(sd, dict) and "model" in sd else sd
    # ckpts saved from a torch.compile'd model carry a "_orig_mod." prefix; strip it.
    sd = {k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k: v for k, v in sd.items()}
    # strict=False to tolerate derived rotary-cache buffers (registered on first forward, not saved);
    # but guard that nothing REAL is missing/unexpected (mirrors the resume loader).
    info = model.load_state_dict(sd, strict=False)
    _rot = ("rotary", "cos_cached", "sin_cached")
    real_missing = [k for k in info.missing_keys if not any(t in k for t in _rot)]
    real_unexpected = [k for k in info.unexpected_keys if not any(t in k for t in _rot)]
    if real_missing or real_unexpected:
        raise RuntimeError(f"teacher load mismatch: missing={real_missing[:5]} unexpected={real_unexpected[:5]}")
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model
