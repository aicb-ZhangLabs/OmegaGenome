"""
COMPREHENSIVE DEBUGGING VERSION for Logit Standardization KD
=============================================================

INSTRUCTIONS:
1. Replace your src/model/distillation.py with this file
2. Run training with logit_standard method
3. Check console output and log file for detailed diagnostics

This version logs EVERY step to identify:
- Where the bug is occurring
- What values are being computed at each step
- Why MCC might be low

POTENTIAL BUG SOURCES TO CHECK:
1. std() using wrong unbiased parameter (FIXED in this version)
2. Teacher logits being None or corrupt
3. Label distribution issues
4. Loss weighting problems
5. Temperature scaling issues
6. Gradient flow problems

Reference: https://github.com/sunshangquan/logit-standardization-KD
Paper: "Logit Standardization in Knowledge Distillation" (CVPR 2024)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
from dataclasses import dataclass
from typing import List, Tuple, Dict, Literal


# ============================================================================
# GLOBAL DEBUG SETTINGS
# ============================================================================
import os as _os

DEBUG = _os.environ.get("DISTILL_DEBUG", "0") == "1"  # default OFF (was hardcoded True -> ~3k log
# lines/job of "[DEBUG batch=..] Teacher logits=None"; 574 grid jobs => GBs of /home log spam).
# Set DISTILL_DEBUG=1 to restore the per-batch debug logging. Logging-only; no effect on results.
DEBUG_LOG_EVERY = 50  # Log every N batches
DEBUG_BATCH = 0
DEBUG_LOG_FIRST_N = 5  # Always log first N batches


def log(msg, force=False):
    """Debug logging function."""
    global DEBUG_BATCH
    if DEBUG and (force or DEBUG_BATCH < DEBUG_LOG_FIRST_N or DEBUG_BATCH % DEBUG_LOG_EVERY == 0):
        print(f"[DEBUG batch={DEBUG_BATCH}] {msg}", file=sys.stderr)


def stats(t, name):
    """Get tensor statistics string."""
    if t is None:
        return f"{name}=None"
    with torch.no_grad():
        f = t.float()
        return (
            f"{name}: shape={list(t.shape)}, "
            f"[{f.min():.4f}, {f.max():.4f}], "
            f"μ={f.mean():.4f}, σ={f.std():.4f}, "
            f"nan={torch.isnan(f).any()}, inf={torch.isinf(f).any()}"
        )


def per_sample(t, name):
    """Per-sample stats for [B, C] tensor."""
    if t is None or len(t.shape) != 2:
        return stats(t, name)
    with torch.no_grad():
        f = t.float()
        B, C = f.shape
        means = f.mean(dim=-1)
        stds_pop = f.std(dim=-1, unbiased=False)
        stds_samp = f.std(dim=-1, unbiased=True)
        return (
            f"{name}: [{B}x{C}]\n"
            f"  means: [{means.min():.4f}, {means.max():.4f}] avg={means.mean():.4f}\n"
            f"  std_pop: [{stds_pop.min():.4f}, {stds_pop.max():.4f}] avg={stds_pop.mean():.4f}\n"
            f"  std_samp: [{stds_samp.min():.4f}, {stds_samp.max():.4f}] avg={stds_samp.mean():.4f}\n"
            f"  ratio(samp/pop)={((stds_samp) / (stds_pop + 1e-10)).mean():.4f}"
        )


# ============================================================================
# CONFIG
# ============================================================================


@dataclass
class DistillationModelConfig:
    weight_ce: float = 0.5
    weight_kl: float = 0.5
    weight_mse: float = 0.0
    temperature: float = 2.0
    zscore: bool = False
    kl_method: Literal["kl", "mse"] = "kl"
    mse_normalize: bool = False  # L2-normalize features before MSE (cosine-style) vs raw MSE

    # Extended for new distillation methods
    distill_method: Literal["vanilla", "logit_standard", "dkd", "dist"] = "vanilla"
    dkd_alpha: float = 1.0
    dkd_beta: float = 8.0

    def __post_init__(self):
        pass


# ============================================================================
# DKD HELPERS
# ============================================================================


def _get_gt_mask(logits, target):
    target = target.reshape(-1)
    return torch.zeros_like(logits).scatter_(1, target.unsqueeze(1), 1).bool()


def _get_other_mask(logits, target):
    """Create binary mask for non-target classes."""
    target = target.reshape(-1)
    return torch.ones_like(logits).scatter_(1, target.unsqueeze(1), 0).bool()


def cat_mask(t, mask1, mask2):
    """
    Concatenate masked probabilities into binary distribution.

    Args:
        t: Probability tensor (B, C)
        mask1: Ground truth mask (B, C)
        mask2: Other classes mask (B, C)

    Returns:
        Binary probability tensor (B, 2) = [P(target), P(non-target)]
    """
    t1 = (t * mask1).sum(dim=1, keepdims=True)  # P(target class)
    t2 = (t * mask2).sum(dim=1, keepdims=True)  # P(all non-target classes)
    rt = torch.cat([t1, t2], dim=1)
    return rt


def dkd_loss(logits_student, logits_teacher, target, alpha, beta, temperature):
    """
    Decoupled Knowledge Distillation Loss (CVPR 2022)

    Official implementation from:
    https://github.com/megvii-research/mdistiller/blob/master/mdistiller/distillers/DKD.py

    Args:
        logits_student: Student logits (B, C)
        logits_teacher: Teacher logits (B, C)
        target: Ground truth labels (B,)
        alpha: Weight for TCKD
        beta: Weight for NCKD
        temperature: Temperature for softmax

    Returns:
        DKD loss = alpha * TCKD + beta * NCKD
    """
    gt_mask = _get_gt_mask(logits_student, target)
    other_mask = _get_other_mask(logits_student, target)

    # Get softmax probabilities
    pred_student = F.softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)

    # ===== TCKD (Target Class KD) =====
    # Create binary distribution: [P(target), P(non-target)]
    pred_student = cat_mask(pred_student, gt_mask, other_mask)
    pred_teacher = cat_mask(pred_teacher, gt_mask, other_mask)
    log_pred_student = torch.log(pred_student)

    # KL divergence on binary distribution
    tckd_loss = (
        F.kl_div(log_pred_student, pred_teacher, size_average=False)
        * (temperature**2)
        / target.shape[0]
    )

    # ===== NCKD (Non-Target Class KD) =====
    # Mask out target class by subtracting large number before softmax
    # This effectively suppresses the target class probability
    pred_teacher_part2 = F.softmax(logits_teacher / temperature - 1000.0 * gt_mask, dim=1)
    log_pred_student_part2 = F.log_softmax(logits_student / temperature - 1000.0 * gt_mask, dim=1)

    # KL divergence on masked distribution
    nckd_loss = (
        F.kl_div(log_pred_student_part2, pred_teacher_part2, size_average=False)
        * (temperature**2)
        / target.shape[0]
    )

    return alpha * tckd_loss + beta * nckd_loss


# ============================================================================
# MAIN MODEL
# ============================================================================


class DistillationModel(nn.Module):
    def __init__(
        self,
        config: DistillationModelConfig,
        teacher_model: nn.Module,
        student_model: nn.Module,
        device: str,
    ):
        super().__init__()
        self.config = config
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.device = device

        print("\n" + "=" * 70, file=sys.stderr)
        print("DISTILLATION DEBUG VERSION INITIALIZED", file=sys.stderr)
        print("=" * 70, file=sys.stderr)
        print(f"  method={config.distill_method}", file=sys.stderr)
        print(f"  temperature={config.temperature}", file=sys.stderr)
        print(
            f"  weight_ce={config.weight_ce}, weight_kl={config.weight_kl}",
            file=sys.stderr,
        )
        print("=" * 70 + "\n", file=sys.stderr)

    def prepare_batch(self, batch: List[torch.Tensor]):
        ids, labs = batch[0].to(self.device), batch[1].to(self.device)
        tlog = batch[2].to(self.device) if len(batch) > 2 and self.config.weight_kl > 0 else None
        tfeats = batch[3].to(self.device) if len(batch) > 3 and self.config.weight_mse > 0 else None
        return {"ids": ids, "labs": labs, "tlog": tlog, "tfeats": tfeats}

    def get_student_knowledge(self, inputs: Dict[str, torch.Tensor]):
        """get the knowledge from the student

        :param inputs: the inputs from the prepare_batch function
        :return: the logits and features from the student
        """
        s_logits, s_feats = self.student_model(inputs["ids"], return_feats=True)
        return s_logits, s_feats

    def get_teacher_knowledge(self, inputs: Dict[str, torch.Tensor]):
        """get the knowledge from the teacher

        :param inputs: the inputs from the prepare_batch function
        :return: the logits and features from the teacher
        """
        tlog = inputs["tlog"]
        tfeats = inputs["tfeats"]
        return tlog, tfeats

    def forward(self, batch) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        global DEBUG_BATCH
        DEBUG_BATCH += 1

        inputs = self.prepare_batch(batch)
        loss, metrics = self.distillation_loss(inputs)
        return loss, metrics

    def distillation_loss(self, inputs) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        s_logits, s_feats = self.get_student_knowledge(inputs)
        tlog, tfeats = self.get_teacher_knowledge(inputs)

        # --- loss computation (unconditional; identical math + order to before) ---
        labels = inputs["labs"]

        # CE Loss
        ce = F.cross_entropy(s_logits, labels)
        loss = self.config.weight_ce * ce

        # KL Loss
        kl = self.kl_term(s_logits, tlog, labels)
        loss += kl

        # MSE Loss
        mse = self.mse_term(s_feats, tfeats)
        loss += mse

        # --- per-step debug logging (gated on DISTILL_DEBUG, default OFF) ---
        # Python evaluates these f-string args EAGERLY before log() decides whether to
        # print, so the .item()/.tolist() (GPU->CPU syncs) and the extra reductions
        # (per_sample x2, abs().mean(), torch.unique) ran every single step even with
        # logging disabled -- stalling the GPU pipeline (solo util ~31%). Guarding the
        # whole block makes the disabled path sync-free. DEBUG=1 prints the same values
        # (only the KL line, emitted inside kl_term, now precedes this block).
        if DEBUG:
            log(f"\n{'=' * 60}")
            log(f"METHOD: {self.config.distill_method}")
            log(per_sample(s_logits, "Student logits"))
            log(per_sample(tlog, "Teacher logits"))

            # CHECK: Are teacher logits actually different from student?
            if tlog is not None and s_logits.shape == tlog.shape:
                diff = (s_logits - tlog).abs().mean()
                log(f"Student-Teacher logit diff: {diff.item():.6f}")
                if diff < 0.01:
                    log(
                        "!!! WARNING: Student and teacher logits are nearly identical !!!",
                        force=True,
                    )

            # Label info
            unique, counts = torch.unique(labels, return_counts=True)
            log(f"Labels: unique={unique.tolist()}, counts={counts.tolist()}")

            log(f"CE: raw={ce.item():.6f}, weighted={(self.config.weight_ce * ce).item():.6f}")
            log(f"TOTAL: {loss.item():.6f}")
            log(f"{'=' * 60}\n")

        return loss, {
            "loss": loss.item(),
            "ce": ce.item(),
            "kl": kl.item(),
            "mse": mse.item(),
        }

    def kl_term(self, s_logits, tlog, labels=None):
        if self.config.weight_kl <= 0 or tlog is None:
            log("KL SKIPPED (weight=0 or tlog=None)")
            return torch.tensor(0.0, device=self.device)

        method = self.config.distill_method

        if method == "logit_standard":
            # Logit Standardization (CVPR 2024)
            kl = self._logit_standard_kl(s_logits, tlog)
        elif method == "logit_standard_debug":
            kl = self._logit_standard_kl_debug(s_logits, tlog)
        elif method == "dkd" and labels is not None:
            # Decoupled Knowledge Distillation (CVPR 2022) - IMPROVED
            kl = self._dkd_loss(s_logits, tlog, labels)
        elif method == "dist":
            # DIST (NeurIPS 2022) - FIXED
            kl = self._dist_loss(s_logits, tlog)
        else:
            kl = self._vanilla_kl(s_logits, tlog)

        weighted = self.config.weight_kl * kl
        if DEBUG:  # avoid 2 per-step GPU->CPU syncs (.item()) when logging is disabled
            log(f"KL: raw={kl.item():.6f}, weighted={weighted.item():.6f}")
        return weighted

    def _vanilla_kl(self, s_logits, tlog):
        temp = self.config.temperature
        s_scaled = s_logits / temp
        t_scaled = tlog / temp
        kl = F.kl_div(
            F.log_softmax(s_scaled, dim=-1),
            F.softmax(t_scaled, dim=-1),
            reduction="batchmean",
        ) * (temp**2)
        return kl

    def _logit_standard_kl(self, s_logits, t_logits):
        """Logit Standardization KD (CVPR 2024) — clean production implementation.

        Z-score each logit vector with POPULATION std (paper Algorithm 1), divide by temperature,
        then KL(softmax(student) || softmax(teacher)) * T². Numerically identical to the verbose
        ``_logit_standard_kl_debug`` below (same mean, unbiased=False std + 1e-7 eps, /T, batchmean).
        Reference: https://github.com/sunshangquan/logit-standardization-KD
        """
        temp = self.config.temperature

        def standardize(z):
            mu = z.mean(dim=-1, keepdim=True)
            sigma = z.std(dim=-1, keepdim=True, unbiased=False) + 1e-7
            return (z - mu) / sigma / temp

        s_norm, t_norm = standardize(s_logits), standardize(t_logits)
        return F.kl_div(
            F.log_softmax(s_norm, dim=-1), F.softmax(t_norm, dim=-1), reduction="batchmean"
        ) * (temp**2)

    def _logit_standard_kl_debug(self, s_logits, t_logits):
        """
        LOGIT STANDARDIZATION with comprehensive debugging.
        Logit Standardization KD Loss (CVPR 2024) - CORRECTED IMPLEMENTATION

        Paper: "Logit Standardization in Knowledge Distillation"
        Reference: https://github.com/sunshangquan/logit-standardization-KD
        Algorithm from paper:
        1. μ = mean(z)
        2. σ = sqrt(mean((z - μ)²))  # POPULATION std (divide by K, not K-1)
        3. z_norm = (z - μ) / σ / τ
        4. KL = KL_div(softmax(z_s_norm), softmax(z_t_norm)) * τ²
        """
        temp = self.config.temperature
        K = s_logits.shape[-1]
        B = s_logits.shape[0]

        log(f"\n[LOGIT_STD] K={K}, B={B}, temp={temp}")
        log(
            f"[LOGIT_STD] Expected std ratio (sample/pop) = sqrt({K}/{K - 1}) = {np.sqrt(K / (K - 1)):.6f}"
        )

        # STEP 1: Compute mean
        s_mean = s_logits.mean(dim=-1, keepdim=True)
        t_mean = t_logits.mean(dim=-1, keepdim=True)
        log(stats(s_mean, "[LOGIT_STD] s_mean"))
        log(stats(t_mean, "[LOGIT_STD] t_mean"))

        # STEP 2: Compute std - CRITICAL: Use population std (unbiased=False)
        # Paper's Algorithm 1: σ(x) = sqrt((1/K) Σ (x^(k) - x̄)²)
        s_std_pop = s_logits.std(dim=-1, keepdim=True, unbiased=False)  # CORRECT
        t_std_pop = t_logits.std(dim=-1, keepdim=True, unbiased=False)  # CORRECT
        s_std_samp = s_logits.std(dim=-1, keepdim=True, unbiased=True)  # WRONG (PyTorch default)

        log(stats(s_std_pop, "[LOGIT_STD] s_std POPULATION (unbiased=False) [CORRECT]"))
        log(stats(s_std_samp, "[LOGIT_STD] s_std SAMPLE (unbiased=True) [WRONG]"))
        log(stats(t_std_pop, "[LOGIT_STD] t_std POPULATION (unbiased=False) [CORRECT]"))

        ratio = (s_std_samp / (s_std_pop + 1e-10)).mean().item()
        log(f"[LOGIT_STD] Actual std ratio (sample/pop): {ratio:.6f}")

        # Use POPULATION std (add small epsilon)
        s_std = s_std_pop + 1e-7
        t_std = t_std_pop + 1e-7

        # STEP 3: Z-score normalize then divide by temperature
        s_centered = s_logits - s_mean
        t_centered = t_logits - t_mean

        s_normalized = s_centered / s_std
        t_normalized = t_centered / t_std

        # Check: after normalization, per-sample std should be ~1
        s_norm_std = s_normalized.std(dim=-1, unbiased=False).mean().item()
        t_norm_std = t_normalized.std(dim=-1, unbiased=False).mean().item()
        log(
            f"[LOGIT_STD] After normalization: s_std={s_norm_std:.6f}, t_std={t_norm_std:.6f} (should be ~1.0)"
        )

        s_norm = s_normalized / temp
        t_norm = t_normalized / temp

        log(per_sample(s_norm, "[LOGIT_STD] s_norm (final)"))
        log(per_sample(t_norm, "[LOGIT_STD] t_norm (final)"))

        # Show sample values
        if B >= 1:
            log(f"[LOGIT_STD] s_norm[0] = {s_norm[0].detach().cpu().numpy()}")
            log(f"[LOGIT_STD] t_norm[0] = {t_norm[0].detach().cpu().numpy()}")

        # STEP 4: Softmax
        s_softmax = F.softmax(s_norm, dim=-1)
        t_softmax = F.softmax(t_norm, dim=-1)

        log(per_sample(s_softmax, "[LOGIT_STD] s_softmax"))
        log(per_sample(t_softmax, "[LOGIT_STD] t_softmax"))

        if B >= 1:
            log(f"[LOGIT_STD] s_softmax[0] = {s_softmax[0].detach().cpu().numpy()}")
            log(f"[LOGIT_STD] t_softmax[0] = {t_softmax[0].detach().cpu().numpy()}")

        # Entropy check (critical for understanding if distributions are too uniform)
        max_entropy = np.log(K)
        s_entropy = -(s_softmax * torch.log(s_softmax + 1e-10)).sum(dim=-1).mean()
        t_entropy = -(t_softmax * torch.log(t_softmax + 1e-10)).sum(dim=-1).mean()

        log(f"[LOGIT_STD] max_entropy (K={K}): {max_entropy:.6f}")
        log(
            f"[LOGIT_STD] s_entropy: {s_entropy.item():.6f} ({100 * s_entropy.item() / max_entropy:.1f}%)"
        )
        log(
            f"[LOGIT_STD] t_entropy: {t_entropy.item():.6f} ({100 * t_entropy.item() / max_entropy:.1f}%)"
        )

        if s_entropy.item() / max_entropy > 0.95:
            log(
                "[LOGIT_STD] !!!! CRITICAL: Student softmax >95% max entropy - nearly uniform !!!!",
                force=True,
            )
        if t_entropy.item() / max_entropy > 0.95:
            log(
                "[LOGIT_STD] !!!! CRITICAL: Teacher softmax >95% max entropy - nearly uniform !!!!",
                force=True,
            )

        # STEP 5: KL divergence
        kl = F.kl_div(F.log_softmax(s_norm, dim=-1), t_softmax, reduction="batchmean")
        kl_scaled = kl * (temp**2)

        log(f"[LOGIT_STD] KL (before T²): {kl.item():.6f}")
        log(f"[LOGIT_STD] KL (after T²={temp**2}): {kl_scaled.item():.6f}")

        # Compare with vanilla KD
        vanilla_kl = F.kl_div(
            F.log_softmax(s_logits / temp, dim=-1),
            F.softmax(t_logits / temp, dim=-1),
            reduction="batchmean",
        ) * (temp**2)

        log(f"[LOGIT_STD] Vanilla KL: {vanilla_kl.item():.6f}")
        log(
            f"[LOGIT_STD] Ratio (logit_std / vanilla): {kl_scaled.item() / (vanilla_kl.item() + 1e-10):.6f}"
        )

        return kl_scaled

    def _dkd_loss(self, s_logits, t_logits, labels):
        return dkd_loss(
            s_logits,
            t_logits,
            labels,
            self.config.dkd_alpha,
            self.config.dkd_beta,
            self.config.temperature,
        )

    def _dist_loss(self, s_logits, t_logits):
        s_probs = F.softmax(s_logits / self.config.temperature, dim=1)
        t_probs = F.softmax(t_logits / self.config.temperature, dim=1)

        s_t, t_t = s_probs.t(), t_probs.t()
        s_mean, t_mean = s_t.mean(dim=1, keepdim=True), t_t.mean(dim=1, keepdim=True)
        s_c, t_c = s_t - s_mean, t_t - t_mean
        inter_corr = (s_c * t_c).sum(dim=1)
        s_n = torch.sqrt((s_c**2).sum(dim=1) + 1e-8)
        t_n = torch.sqrt((t_c**2).sum(dim=1) + 1e-8)
        inter_corr = inter_corr / (s_n * t_n + 1e-8)
        inter_loss = (1 - inter_corr).mean()

        s_mean, t_mean = s_probs.mean(dim=1, keepdim=True), t_probs.mean(dim=1, keepdim=True)
        s_c, t_c = s_probs - s_mean, t_probs - t_mean
        intra_corr = (s_c * t_c).sum(dim=1)
        s_n = torch.sqrt((s_c**2).sum(dim=1) + 1e-8)
        t_n = torch.sqrt((t_c**2).sum(dim=1) + 1e-8)
        intra_corr = intra_corr / (s_n * t_n + 1e-8)
        intra_loss = (1 - intra_corr).mean()

        return (inter_loss + intra_loss) / 2.0

    def mse_term(self, s_feats, tfeats):
        if self.config.weight_mse > 0 and tfeats is not None:
            s_feats, tfeats = self.student_model.aligned_feats(s_feats, tfeats)
            if self.config.mse_normalize:
                # L2-normalize each feature vector before MSE: removes the scale mismatch between
                # a 3B-LLM hidden state and the tiny conv's pooled features (cosine-style matching).
                s_feats = F.normalize(s_feats, dim=-1)
                tfeats = F.normalize(tfeats, dim=-1)
            mse = F.mse_loss(s_feats, tfeats)
            mse = self.config.weight_mse * mse
        else:
            mse = torch.tensor(0.0, device=self.device)
        return mse
