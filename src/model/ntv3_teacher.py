"""NTv3 teacher for per-base-pair multi-track distillation.

Loads an NTv3 post-trained model and exposes its per-bp ``bigwig_tracks_logits`` as the
distillation targets for the ``BPNetRegressor`` student. Reuses the loading recipe from the
biomodel repo's ``adapters/ntv3_adapter.py``: HF ``AutoModel`` with ``trust_remote_code``,
species-conditioned forward (``encode_species``), and tokenization padded to a multiple of 128.

Default teacher is the 650M post-trained model (strong SOTA). Larger variants or the local
100M snapshot can be selected via ``model_name_or_path``.
"""

import json
import os

import torch
from dataclasses import dataclass
from typing import List, Optional, Sequence

NTV3_INPUT_MULTIPLE = 128  # NTv3 requires input length a multiple of 128

# Known checkpoints (override model_name_or_path to point elsewhere):
NTV3_650M_POST = "InstaDeepAI/NTv3_650M_post"   # default teacher (HF; gated -> needs HF token)
NTV3_100M_POST_LOCAL = "/extra/zhanglab0/INDV/pengchx3/ntv3_local/100m_post"  # quick-test snapshot

_GATED_PREFIX = "InstaDeepAI/ntv3_base_model--"


def _strip_auto_map(value):
    """Remove the gated-repo prefix from one auto_map entry (str or [str, None])."""
    if isinstance(value, str):
        return value.replace(_GATED_PREFIX, "")
    if isinstance(value, list):
        return [v.replace(_GATED_PREFIX, "") if isinstance(v, str) else v for v in value]
    return value


def prepare_local_snapshot(path: str) -> str:
    """Local-snapshot bypass for NTv3's gated dynamic modeling files.

    The snapshot's config points auto_map at the gated ``InstaDeepAI/ntv3_base_model`` repo, so
    HF tries to download it. The modeling/tokenizer ``.py`` files are already in the snapshot, so
    we make a patched copy with the gated prefix stripped and load that with local_files_only.
    Returns the patched dir (or the original path if it's an HF repo id / has no gated auto_map).
    """
    if not os.path.isdir(path):
        return path  # HF repo id (gated 650M etc.) -> relies on HF_TOKEN auth
    patched = path.rstrip("/") + "_loadable"  # sibling dir, avoids self-recursion
    os.makedirs(patched, exist_ok=True)
    cfg_files = {"config.json", "tokenizer_config.json"}
    # Symlink everything (weights etc.) — no copy — except the configs we rewrite.
    for fname in os.listdir(path):
        src, dst = os.path.join(path, fname), os.path.join(patched, fname)
        if os.path.isfile(src) and fname not in cfg_files and not os.path.exists(dst):
            os.symlink(src, dst)
    for cfg_name in cfg_files:
        orig = os.path.join(path, cfg_name)
        if not os.path.exists(orig):
            continue
        with open(orig) as f:
            d = json.load(f)
        if d.get("auto_map"):
            d["auto_map"] = {k: _strip_auto_map(v) for k, v in d["auto_map"].items()}
        with open(os.path.join(patched, cfg_name), "w") as f:
            json.dump(d, f, indent=2)
    return patched


@dataclass
class NTv3TeacherConfig:
    model_name_or_path: str = NTV3_650M_POST
    species: str = "human"  # NTv3 is species-conditioned; "human" has 7362 bigwig tracks
    trust_remote_code: bool = True
    bf16: bool = True
    # HF token for the gated NTv3 repos (650M etc.). Falls back to the HF_TOKEN env var.
    hf_token_path: Optional[str] = "/home/pengchx3/text-dna/huggingface-token-0616.txt"
    # Optional subset of bigwig track indices to distill (human has 7362; a paper subset is
    # far smaller). None = all tracks for the species.
    track_subset: Optional[Sequence[int]] = None


class NTv3Teacher:
    """Wraps NTv3 post-trained. ``predict_tracks(seqs)`` -> per-bp track logits ``[B, T, L_out]``."""

    def __init__(self, config: NTv3TeacherConfig, device: str = "cuda"):
        from transformers import AutoModel, AutoTokenizer

        self.config = config
        self.device = device
        dtype = torch.bfloat16 if (config.bf16 and device != "cpu") else None
        # Local-snapshot bypass for the gated dynamic modeling files (no-op for HF repo ids).
        resolved = prepare_local_snapshot(config.model_name_or_path)
        local = os.path.isdir(resolved)
        # HF token for gated repos (650M etc.); ignored for local_files_only loads.
        token = os.environ.get("HF_TOKEN")
        if not token and config.hf_token_path and os.path.exists(config.hf_token_path):
            token = open(config.hf_token_path).read().strip()
        kw = {"trust_remote_code": config.trust_remote_code, "local_files_only": local}
        if token and not local:
            kw["token"] = token
        self.tokenizer = AutoTokenizer.from_pretrained(resolved, **kw)
        self.model = (
            AutoModel.from_pretrained(resolved, torch_dtype=dtype, **kw).to(device).eval()
        )

    @property
    def all_track_names(self) -> List[str]:
        return list(self.model.config.bigwigs_per_species.get(self.config.species, []))

    @property
    def num_tracks(self) -> int:
        if self.config.track_subset is not None:
            return len(self.config.track_subset)
        return len(self.all_track_names)

    @torch.no_grad()
    def predict_tracks(
        self, sequences: List[str], species: Optional[List[str]] = None
    ) -> torch.Tensor:
        """Return NTv3's per-bp bigwig track logits (detached, usable as fixed distill targets).

        NOTE: orientation of the track axis (``[B, L_out, T]`` vs ``[B, T, L_out]``) should be
        confirmed by a one-batch probe; ``track_subset`` selects along the last dim by default.
        """
        species = species or [self.config.species] * len(sequences)
        batch = self.tokenizer(
            sequences,
            add_special_tokens=False,
            padding=True,
            pad_to_multiple_of=NTV3_INPUT_MULTIPLE,
            return_tensors="pt",
        ).to(self.device)
        species_ids = self.model.encode_species(species).to(self.device)
        out = self.model(input_ids=batch["input_ids"], species_ids=species_ids)
        tracks = out.bigwig_tracks_logits
        if self.config.track_subset is not None:
            idx = torch.as_tensor(list(self.config.track_subset), device=tracks.device)
            tracks = tracks.index_select(-1, idx)
        return tracks.detach().float()
