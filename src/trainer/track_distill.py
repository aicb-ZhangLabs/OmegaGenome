"""Per-bp multi-track distillation loss + student/teacher resolution alignment.

NTv3 (teacher) emits ``bigwig_tracks_logits`` as ``[B, L_teacher, T]`` (positions, then tracks),
binned by its U-Net (e.g. 1024 bp -> 384 bins). The BPNetRegressor (student) emits
``[B, T, L_student]`` at input resolution. To compare them we put both in ``[B, T, L_teacher]``
(adaptive-avg-pool the student's length down to the teacher's, permute the teacher's track axis
to the middle) and apply a per-position regression loss.
"""

import torch
import torch.nn.functional as F


def _teacher_to_btl(teacher: torch.Tensor) -> torch.Tensor:
    """NTv3 teacher tracks ``[B, L_teacher, T]`` -> ``[B, T, L_teacher]``."""
    if teacher.dim() != 3:
        raise ValueError(f"teacher tracks must be 3D [B, L, T], got {tuple(teacher.shape)}")
    return teacher.permute(0, 2, 1).contiguous()


def align_student_to_teacher(student: torch.Tensor, target_len: int) -> torch.Tensor:
    """Resample student ``[B, T, L_student]`` to ``[B, T, target_len]`` (adaptive avg pool).

    Works for downsampling (L_student > target_len, the usual case) and upsampling.
    """
    if student.dim() != 3:
        raise ValueError(f"student must be 3D [B, T, L], got {tuple(student.shape)}")
    if student.shape[-1] == target_len:
        return student
    return F.adaptive_avg_pool1d(student, target_len)


def track_distill_loss(
    student: torch.Tensor,
    teacher: torch.Tensor,
    kind: str = "mse",
) -> torch.Tensor:
    """Per-bp multi-track regression loss between student and (frozen) teacher tracks.

    student: ``[B, T, L_student]`` (BPNetRegressor output).
    teacher: ``[B, L_teacher, T]`` (NTv3 ``bigwig_tracks_logits``); treated as a fixed target.
    Returns a scalar. ``kind`` in {"mse", "poisson"} (poisson treats student as a log-rate).
    """
    teacher = _teacher_to_btl(teacher).detach()  # [B, T, L_teacher]; never backprop into teacher
    if student.shape[1] != teacher.shape[1]:
        raise ValueError(
            f"track-count mismatch: student T={student.shape[1]} vs teacher T={teacher.shape[1]}"
        )
    student = align_student_to_teacher(student, teacher.shape[-1])  # [B, T, L_teacher]
    if kind == "mse":
        return F.mse_loss(student, teacher)
    if kind == "poisson":
        # student = log-rate, teacher = non-negative target counts/signal
        return F.poisson_nll_loss(student, teacher, log_input=True, full=False)
    raise ValueError(f"unknown loss kind: {kind!r}")
