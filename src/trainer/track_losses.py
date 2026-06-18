"""Faithful Poisson-multinomial loss for NTv3 bigWig-track fine-tuning.

Ported verbatim (behaviour) from InstaDeepAI's official notebook
``03_fine_tuning_posttrained_model_biwig.ipynb``. The loss decomposes a track into a *scale* term
(Poisson on the per-track total signal) and a *shape* term (multinomial cross-entropy on the
normalised per-position distribution), combined as ``shape + scale / shape_loss_coefficient``.
This is the Enformer/Borzoi-style objective for count-like genomic signal.
"""

import torch


def poisson_loss(ytrue: torch.Tensor, ypred: torch.Tensor, epsilon: float = 1e-7) -> torch.Tensor:
    """Per-element Poisson loss ``ypred - ytrue*log(ypred)`` (no Stirling term; constant in y)."""
    return ypred - ytrue * torch.log(ypred + epsilon)


def _safe_for_grad_log(x: torch.Tensor) -> torch.Tensor:
    """log defined for all x (uses 1 where x<=0) in a backprop-safe way."""
    return torch.log(torch.where(x > 0.0, x, torch.ones_like(x)))


def poisson_multinomial_loss(logits: torch.Tensor, targets: torch.Tensor,
                             shape_loss_coefficient: float = 5.0, epsilon: float = 1e-7) -> torch.Tensor:
    """Poisson-multinomial loss for ``(batch, seq_length, num_tracks)`` predictions and targets."""
    batch_size, seq_length, num_tracks = logits.shape

    # Scale term: Poisson on the total signal per (batch, track), normalised by sequence length.
    scale_loss = poisson_loss(targets.sum(dim=1), logits.sum(dim=1), epsilon=epsilon)  # (B, T)
    scale_loss = (scale_loss / (seq_length + epsilon)).mean()

    # Shape term: multinomial cross-entropy on the per-position distribution over the sequence axis.
    predicted_counts = logits + epsilon
    p_pred = predicted_counts / (predicted_counts.sum(dim=1, keepdim=True) + epsilon)
    shape_loss = -((targets + epsilon) * _safe_for_grad_log(p_pred))
    shape_loss = shape_loss.sum() / (batch_size * seq_length * num_tracks + epsilon)

    return shape_loss + scale_loss / shape_loss_coefficient
