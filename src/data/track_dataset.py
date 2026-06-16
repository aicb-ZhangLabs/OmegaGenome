"""Dataset for per-bp multi-track distillation: DNA windows + NTv3 teacher track targets.

For distillation the targets are NTv3's per-bp ``bigwig_tracks_logits`` (cached once), so no
external benchmark labels are required to train the student. Reuses ``encode_seq`` so the student
sees the same integer encoding as the classification tasks.
"""

import torch
from torch.utils.data import Dataset

from .dataset import encode_seq


class TrackDataset(Dataset):
    """Returns ``(input_ids [L], target_tracks [L_teacher, T])`` per item.

    sequences: list of DNA strings (all same length -> default collate stacks cleanly).
    targets:   array/tensor ``[N, L_teacher, T]`` of teacher per-bp track values.
    """

    def __init__(self, sequences, targets, max_len: int):
        if len(sequences) != len(targets):
            raise ValueError(f"sequences ({len(sequences)}) != targets ({len(targets)})")
        self.ids = torch.tensor(
            [encode_seq(s, max_len) for s in sequences], dtype=torch.long
        )
        self.targets = torch.as_tensor(targets, dtype=torch.float)
        if self.targets.dim() != 3:
            raise ValueError(f"targets must be [N, L_teacher, T], got {tuple(self.targets.shape)}")

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx):
        return self.ids[idx], self.targets[idx]


def build_teacher_targets(teacher, sequences, batch_size: int = 4):
    """Run an NTv3 teacher over ``sequences`` and stack its per-bp tracks: ``[N, L_teacher, T]``.

    ``teacher`` is anything with ``predict_tracks(list[str]) -> [b, L_teacher, T]`` (NTv3Teacher
    or a stub). Targets are moved to CPU float and concatenated.
    """
    chunks = []
    for i in range(0, len(sequences), batch_size):
        out = teacher.predict_tracks(sequences[i : i + batch_size])
        chunks.append(out.detach().to("cpu", torch.float))
    return torch.cat(chunks, dim=0)
