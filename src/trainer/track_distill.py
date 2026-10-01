"""Per-bp multi-track distillation loss.

Everything here is SINGLE-NUCLEOTIDE resolution. Teacher, student and ground truth are all
``[B, L, T]`` (positions, then tracks) over the central 0.375 crop — e.g. a 32768-bp window ->
``L = 12288`` per-bp predictions, one per nucleotide. The NTv3 teacher/student take the per-nt last
hidden state; the BPNet student uses stride-1 'same'-padded dilated convs (length-preserving, no
binning), so its output is per-bp too. Predictions align 1:1 with the per-bp bigWig targets, so the
losses are applied element-wise per position.

``track_kd_loss`` accepts a ``[B, T, L]`` student via ``student_layout='BTL'`` (permuted to ``[B, L, T]``)
and, as a safety net, adaptive-pools the student's position axis to the teacher's if they ever differ
(a no-op when both are the same per-bp length, which is the normal case).
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from src.trainer.track_losses import poisson_multinomial_loss


def resolve_kd_track_targets(subset_idx, num_native_tracks: int, teacher_specialist: bool = False):
    """Resolve the KD teacher's track count + index-select indices so its per-bp output lines up with
    a (possibly specialist) student's tracks. Pure function — no torch/device, so it is unit-testable.

    - ``subset_idx`` None/empty (full-model student): teacher used whole -> ``(num_native_tracks, None)``.
    - subset + GENERALIST teacher (default): teacher emits all native tracks, pick the subset out of its
      output -> ``(num_native_tracks, list(subset_idx))``.
    - subset + SPECIALIST teacher (``teacher_specialist``): teacher already has a ``len(subset)``-track
      head (trained on the same subset), use directly, no index-select -> ``(len(subset_idx), None)``.

    Returns ``(teacher_num_tracks, kd_track_idx_or_None)``; the caller wraps the int list in a tensor.
    """
    if not subset_idx:
        return num_native_tracks, None
    if teacher_specialist:
        return len(subset_idx), None
    return num_native_tracks, list(subset_idx)


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

    w_ce: float = 0.5  # ground-truth term weight (student vs real bigWig)
    w_kl: float = 0.5  # distill term weight (student vs frozen 650M teacher tracks)
    w_mse: float = 0.2  # feature-matching term weight (0 disables; needs *_feat args)
    gt_loss: str = "poisson_multinomial"  # {"poisson_multinomial", "mse", "pearson", "dist", "standardized_mse"}
    # distill term vs the frozen teacher's tracks. SOTA-grounded options:
    #   poisson_multinomial (Borzoi/Enigma) · mse · pearson (1−corr over positions = the eval metric) ·
    #   teacher_bounded (Chen 2017) · dist (NeurIPS'22 correlation-matching, robust to capacity gap) ·
    #   standardized_mse (CVPR'24 logit-standardization, per-track z-score → scale-invariant) ·
    #   cwd (ICCV'21 channel-wise distillation, per-track softmax-KL over positions → scale-invariant).
    distill_loss: str = "poisson_multinomial"
    multinomial_weight: float = 5.0  # Poisson-multinomial shape/scale coefficient (Borzoi=5)
    teacher_bound_margin: float = 0.0  # margin for the teacher_bounded distill loss
    cwd_temperature: float = 4.0  # softmax temperature for the cwd distill loss (ICCV'21 default)
    distill_target_gt_mix: float = (
        0.0  # Enigma-style: distill target = (1-m)·teacher + m·ground-truth
    )


def _corr_along(pred: torch.Tensor, target: torch.Tensor, dim: int) -> torch.Tensor:
    """1 - mean Pearson correlation between pred and target along ``dim`` (both [B, L, T])."""
    s = pred - pred.mean(dim=dim, keepdim=True)
    t = target - target.mean(dim=dim, keepdim=True)
    num = (s * t).sum(dim=dim)
    den = s.norm(dim=dim) * t.norm(dim=dim) + 1e-8
    return 1.0 - (num / den).mean()


def _pearson_term(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """1 - mean per-(window,track) Pearson along the POSITION axis. pred/target: [B, L, T]. This is
    exactly the eval metric (per-track Pearson over positions), but computed vs the teacher's tracks."""
    return _corr_along(pred, target, dim=1)


def _dist_term(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """DIST distillation (Tang et al., "Knowledge Distillation from A Stronger Teacher", NeurIPS 2022;
    github.com/hunto/DIST_KD) adapted to multitrack regression. Matches the teacher's CORRELATION
    STRUCTURE rather than exact values, so it is robust to the teacher↔student magnitude/capacity gap
    (the regime where exact MSE/Poisson matching fails). Two terms, both 1−Pearson:
      • intra: per-track correlation along POSITIONS — i.e. distil directly to the eval metric;
      • inter: per-position correlation across the 34 TRACKS — preserves cross-track relationships.
    pred/target: [B, L, T]."""
    return _corr_along(pred, target, dim=1) + _corr_along(pred, target, dim=2)


def _standardized_mse_term(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Per-track standardized MSE (logit-standardization, Sun et al., CVPR 2024, adapted for regression):
    z-score each track over POSITIONS before the MSE so the loss is scale-invariant and not dominated by
    the high-count tracks (ATAC/RNA) at the expense of the low-count high-headroom ones (PRO-cap/eCLIP).
    pred/target: [B, L, T]."""
    ps = (pred - pred.mean(dim=1, keepdim=True)) / (pred.std(dim=1, keepdim=True) + 1e-7)
    ts = (target - target.mean(dim=1, keepdim=True)) / (target.std(dim=1, keepdim=True) + 1e-7)
    return F.mse_loss(ps, ts)


def _cwd_term(pred: torch.Tensor, target: torch.Tensor, temperature: float) -> torch.Tensor:
    """Channel-Wise Distillation (Shu et al., ICCV 2021; github.com/irfanICMLL/TorchDistiller), adapted
    per TRACK over POSITIONS. Softmax each track's profile over the L positions at ``temperature`` (so the
    loss is scale-invariant — it ignores absolute counts, matching only the positional SHAPE, which is
    what the per-track Pearson eval rewards), then KL-match teacher→student:
        L = (T²/C)·Σ_c Σ_i softmax_i(t^c/T)·log[softmax_i(t^c/T)/softmax_i(s^c/T)].
    pred/target: [B, L, T_tracks]; softmax/KL are along the position axis (dim=1)."""
    Tm = temperature
    s_logp = F.log_softmax(pred / Tm, dim=1)  # [B, L, C]: log-prob over positions per track
    t_logp = F.log_softmax(
        target / Tm, dim=1
    )  # (log-softmax both -> KL is exactly 0 when identical)
    kl = (t_logp.exp() * (t_logp - s_logp)).sum(dim=1)  # [B, C]: KL over positions per track
    return (Tm * Tm) * kl.mean()


def _teacher_bounded_term(
    student: torch.Tensor, teacher: torch.Tensor, gt: torch.Tensor, margin: float
) -> torch.Tensor:
    """Chen et al. 2017 teacher-bounded regression: squared student error vs GT, counted ONLY where
    the student is worse than the teacher (by > margin). student/teacher/gt: [B, L, T]."""
    s_err = (student - gt) ** 2
    t_err = (teacher - gt) ** 2
    mask = (s_err > t_err + margin).to(student.dtype)
    return (mask * s_err).sum() / (mask.sum() + 1e-8)


def _regression_term(
    pred: torch.Tensor, target: torch.Tensor, kind: str, mw: float, cwd_temp: float = 4.0
) -> torch.Tensor:
    """One regression loss between pred and (fixed) target, both [B, L, T] (positions, tracks)."""
    if kind == "mse":
        return F.mse_loss(pred, target)
    if kind == "poisson_multinomial":
        return poisson_multinomial_loss(pred, target, shape_loss_coefficient=mw)
    if kind == "pearson":
        return _pearson_term(pred, target)
    if kind == "dist":
        return _dist_term(pred, target)
    if kind == "standardized_mse":
        return _standardized_mse_term(pred, target)
    if kind == "cwd":
        return _cwd_term(pred, target, cwd_temp)
    raise ValueError(f"unknown regression loss kind: {kind!r}")


def _align_positions(student: torch.Tensor, target_len: int) -> torch.Tensor:
    """Resample student [B, L, T] along positions to target_len (adaptive avg pool)."""
    if student.shape[1] == target_len:
        return student
    pooled = F.adaptive_avg_pool1d(student.permute(0, 2, 1), target_len)  # [B,T,L']
    return pooled.permute(0, 2, 1).contiguous()


def track_kd_loss(
    student: torch.Tensor,
    teacher: torch.Tensor,
    gt: torch.Tensor,
    *,
    student_feat: torch.Tensor = None,
    teacher_feat: torch.Tensor = None,
    cfg: TrackKDConfig = None,
    student_layout: str = "BLT",
):
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
        raise ValueError(
            f"shape mismatch student {tuple(student.shape)} teacher {tuple(teacher.shape)} gt {tuple(gt.shape)}"
        )

    L_gt = _regression_term(student, gt, cfg.gt_loss, cfg.multinomial_weight, cfg.cwd_temperature)
    # Enigma-style soft target: blend a little ground truth into the teacher target (m=0.1 typical), so
    # the distill term is anchored to truth where the teacher errs. m=0 (default) = pure teacher target.
    distill_target = teacher
    if cfg.distill_target_gt_mix > 0:
        m = cfg.distill_target_gt_mix
        distill_target = (1.0 - m) * teacher + m * gt
    if cfg.distill_loss == "teacher_bounded":
        L_distill = _teacher_bounded_term(student, teacher, gt, cfg.teacher_bound_margin)
    else:
        L_distill = _regression_term(
            student, distill_target, cfg.distill_loss, cfg.multinomial_weight, cfg.cwd_temperature
        )
    if cfg.w_mse > 0 and student_feat is not None and teacher_feat is not None:
        # FitNets feature alignment. Channels must already match (caller projects student->teacher dim);
        # adaptive-pool the student's SEQUENCE axis to the teacher's if they differ (mirrors the track
        # path), so unequal student/teacher emb lengths never crash the MSE.
        if student_feat.shape[1] != teacher_feat.shape[1]:
            student_feat = _align_positions(student_feat, teacher_feat.shape[1])
        # L2-normalise each per-position feature vector before the MSE. NTv3 hidden states are large-
        # magnitude (raw MSE ~O(100s)), which would swamp the O(1) Poisson track terms and make w_mse
        # uninterpretable. Normalising transfers the teacher's representation DIRECTION (cosine-style,
        # MSE in [0,4]) so w_mse trades off against the track losses on a comparable scale.
        sf = F.normalize(student_feat, dim=-1)
        tf = F.normalize(teacher_feat.detach(), dim=-1)
        L_feat = F.mse_loss(sf, tf)
    else:
        L_feat = student.new_zeros(())

    total = cfg.w_ce * L_gt + cfg.w_kl * L_distill + cfg.w_mse * L_feat
    return total, {
        "gt": L_gt.detach(),
        "distill": L_distill.detach(),
        "feat": L_feat.detach(),
        "total": total.detach(),
    }
