"""Per-bp multi-track distillation loss + student/teacher resolution alignment.

NTv3 (teacher) emits ``bigwig_tracks_logits`` as ``[B, L_teacher, T]`` (positions, then tracks),
binned by its U-Net (e.g. 1024 bp -> 384 bins). The BPNetRegressor (student) emits
``[B, T, L_student]`` at input resolution. To compare them we put both in ``[B, T, L_teacher]``
(adaptive-avg-pool the student's length down to the teacher's, permute the teacher's track axis
to the middle) and apply a per-position regression loss.
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from src.trainer.track_losses import poisson_multinomial_loss


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
    Returns a scalar. ``kind`` in {"mse", "poisson", "pearson", "mse+pearson"}:
    poisson treats student as a log-rate; pearson = 1 - mean per-(window,track) correlation along
    position (directly optimizes the eval metric); "mse+pearson" sums both.
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
    if kind in ("pearson", "mse+pearson"):
        # Pearson along the position axis, per (window, track); maximize -> 1 - mean corr.
        s = student - student.mean(dim=-1, keepdim=True)
        t = teacher - teacher.mean(dim=-1, keepdim=True)
        num = (s * t).sum(dim=-1)
        den = s.norm(dim=-1) * t.norm(dim=-1) + 1e-8
        corr = (num / den).mean()
        return (1.0 - corr) + (F.mse_loss(student, teacher) if kind == "mse+pearson" else 0.0)
    raise ValueError(f"unknown loss kind: {kind!r}")


# =============================================================================================
# 3-term KD loss — the REGRESSION analog of the classification CE+KL+MSE recipe
# =============================================================================================
# Classification KD (src/model/distillation.py, UNTOUCHED) uses: CE(label) + KL(teacher softmax/T)
# + MSE(features). Per-bp track prediction is REGRESSION (continuous, non-negative signal), so there
# is no softmax/KL — we replace them with regression losses, while keeping the same 3-term shape:
#
#     L = w_ce · L_gt(student, real_bigWig)          # "ce"  -> ground-truth supervision
#       + w_kl · L_distill(student, teacher_tracks)  # "kl"  -> distill from the 650M (soft targets)
#       + w_mse · L_feat(student_feat, teacher_feat) # "mse" -> intermediate feature matching
#
# Loss-type choices follow the SOTA references:
#   * poisson_multinomial — Borzoi/Enformer/Flashzoi/Enigma objective; the *multinomial* (profile)
#     term treats the per-position track as a distribution, so distilling the teacher's profile this
#     way is the genuine regression analog of Hinton's KL (soft targets over positions).
#   * teacher_bounded — Chen et al. 2017: only penalise the student where it is WORSE than the
#     teacher (vs ground truth), so it never copies the teacher's errors.
#   * mse / pearson — simple value match / direct optimisation of the eval metric.
# EVERY weight and EVERY per-term loss type is configurable (TrackKDConfig); nothing hard-coded.

@dataclass
class TrackKDConfig:
    """Config for the 3-term per-bp track KD loss. All weights + per-term loss types configurable.

    Defaults mirror the carbon classification recipe (0.5 / 0.5 / 0.2) with the faithful
    Poisson-multinomial (Borzoi/NTv3) losses so the student stays comparable to the 650M teacher.
    """
    w_ce: float = 0.5          # ground-truth term weight (student vs real bigWig)
    w_kl: float = 0.5          # distill term weight (student vs frozen 650M teacher tracks)
    w_mse: float = 0.2         # feature-matching term weight (0 disables; needs *_feat args)
    gt_loss: str = "poisson_multinomial"       # {"poisson_multinomial", "mse", "pearson"}
    distill_loss: str = "poisson_multinomial"  # {"poisson_multinomial", "mse", "pearson", "teacher_bounded"}
    multinomial_weight: float = 5.0            # Poisson-multinomial shape/scale coefficient (Borzoi=5)
    teacher_bound_margin: float = 0.0          # margin for the teacher_bounded distill loss


def _pearson_term(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """1 - mean per-(window,track) Pearson along the POSITION axis. pred/target: [B, L, T]."""
    s = pred - pred.mean(dim=1, keepdim=True)
    t = target - target.mean(dim=1, keepdim=True)
    num = (s * t).sum(dim=1)
    den = s.norm(dim=1) * t.norm(dim=1) + 1e-8
    return 1.0 - (num / den).mean()


def _teacher_bounded_term(student: torch.Tensor, teacher: torch.Tensor, gt: torch.Tensor,
                          margin: float) -> torch.Tensor:
    """Chen et al. 2017 teacher-bounded regression: squared student error vs GT, counted ONLY where
    the student is worse than the teacher (by > margin). student/teacher/gt: [B, L, T]."""
    s_err = (student - gt) ** 2
    t_err = (teacher - gt) ** 2
    mask = (s_err > t_err + margin).to(student.dtype)
    return (mask * s_err).sum() / (mask.sum() + 1e-8)


def _regression_term(pred: torch.Tensor, target: torch.Tensor, kind: str, mw: float) -> torch.Tensor:
    """One regression loss between pred and (fixed) target, both [B, L, T] (positions, tracks)."""
    if kind == "mse":
        return F.mse_loss(pred, target)
    if kind == "poisson_multinomial":
        return poisson_multinomial_loss(pred, target, shape_loss_coefficient=mw)
    if kind == "pearson":
        return _pearson_term(pred, target)
    raise ValueError(f"unknown regression loss kind: {kind!r}")


def _align_positions(student: torch.Tensor, target_len: int) -> torch.Tensor:
    """Resample student [B, L, T] along positions to target_len (adaptive avg pool)."""
    if student.shape[1] == target_len:
        return student
    pooled = F.adaptive_avg_pool1d(student.permute(0, 2, 1), target_len)  # [B,T,L']
    return pooled.permute(0, 2, 1).contiguous()


def track_kd_loss(student: torch.Tensor, teacher: torch.Tensor, gt: torch.Tensor, *,
                  student_feat: torch.Tensor = None, teacher_feat: torch.Tensor = None,
                  cfg: TrackKDConfig = None, student_layout: str = "BLT"):
    """3-term KD loss for per-bp track regression. Returns ``(total_loss, components_dict)``.

    student / teacher / gt: ``[B, L, T]`` (positions, tracks). ``teacher`` and ``gt`` are targets;
    the teacher is ``.detach()``-ed (never backprop into it), and the GT supervises the "ce" term.
    ``student_layout='BTL'`` permutes a ``[B, T, L]`` student (DilatedTrackNet/BPNetRegressor) to
    ``[B, L, T]`` first; the NTv3 student already emits ``[B, L, T]``. Student positions are
    adaptive-pooled to the teacher/GT length if they differ.
    """
    cfg = cfg or TrackKDConfig()
    if student_layout == "BTL":
        student = student.permute(0, 2, 1).contiguous()
    elif student_layout != "BLT":
        raise ValueError(f"student_layout must be 'BLT' or 'BTL', got {student_layout!r}")
    teacher = teacher.detach()
    gt = gt.detach()
    student = _align_positions(student, teacher.shape[1])
    if not (student.shape == teacher.shape == gt.shape):
        raise ValueError(f"shape mismatch student {tuple(student.shape)} teacher {tuple(teacher.shape)} gt {tuple(gt.shape)}")

    L_gt = _regression_term(student, gt, cfg.gt_loss, cfg.multinomial_weight)
    if cfg.distill_loss == "teacher_bounded":
        L_distill = _teacher_bounded_term(student, teacher, gt, cfg.teacher_bound_margin)
    else:
        L_distill = _regression_term(student, teacher, cfg.distill_loss, cfg.multinomial_weight)
    if cfg.w_mse > 0 and student_feat is not None and teacher_feat is not None:
        # FitNets feature alignment. Channels must already match (caller projects student->teacher dim);
        # adaptive-pool the student's SEQUENCE axis to the teacher's if they differ (mirrors the track
        # path), so unequal student/teacher emb lengths never crash the MSE.
        if student_feat.shape[1] != teacher_feat.shape[1]:
            student_feat = _align_positions(student_feat, teacher_feat.shape[1])
        L_feat = F.mse_loss(student_feat, teacher_feat.detach())
    else:
        L_feat = student.new_zeros(())

    total = cfg.w_ce * L_gt + cfg.w_kl * L_distill + cfg.w_mse * L_feat
    return total, {"gt": L_gt.detach(), "distill": L_distill.detach(),
                   "feat": L_feat.detach(), "total": total.detach()}
