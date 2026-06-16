"""NTv3 teacher for per-base-pair multi-track distillation.

Loads an NTv3 post-trained model and exposes its per-bp ``bigwig_tracks_logits`` as the
distillation targets for the ``BPNetRegressor`` student. Reuses the loading recipe from the
biomodel repo's ``adapters/ntv3_adapter.py``: HF ``AutoModel`` with ``trust_remote_code``,
species-conditioned forward (``encode_species``), and tokenization padded to a multiple of 128.

Default teacher is the 650M post-trained model (strong SOTA). Larger variants or the local
100M snapshot can be selected via ``model_name_or_path``.
"""

import torch
from dataclasses import dataclass
from typing import List, Optional, Sequence

NTV3_INPUT_MULTIPLE = 128  # NTv3 requires input length a multiple of 128

# Known checkpoints (override model_name_or_path to point elsewhere):
NTV3_650M_POST = "InstaDeepAI/NTv3_650M_post"   # default teacher (HF download)
NTV3_100M_POST_LOCAL = "/extra/zhanglab0/INDV/pengchx3/ntv3_local/100m_post"  # quick-test snapshot


@dataclass
class NTv3TeacherConfig:
    model_name_or_path: str = NTV3_650M_POST
    species: str = "human"  # NTv3 is species-conditioned; "human" has 7362 bigwig tracks
    trust_remote_code: bool = True
    bf16: bool = True
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
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model_name_or_path, trust_remote_code=config.trust_remote_code
        )
        self.model = (
            AutoModel.from_pretrained(
                config.model_name_or_path,
                trust_remote_code=config.trust_remote_code,
                torch_dtype=dtype,
            )
            .to(device)
            .eval()
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
