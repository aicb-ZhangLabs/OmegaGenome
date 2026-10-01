"""Per-track regression metrics for the NTv3 multi-track (per-bp) task.

Track prediction is scored by correlation between predicted and target signal, per track,
then averaged — the standard sequence-to-function metric (vs MCC for the classification tasks).
"""

import numpy as np
import torch


def per_track_pearson(pred, target):
    """Pearson r per track between predicted and target per-bp tracks.

    pred, target: arrays of shape ``[N, T, L]`` (samples, tracks, positions). Correlation is
    computed per track over all (sample, position) pairs.
    Returns ``(mean_r, per_track_r)`` where per_track_r has length T (NaN tracks excluded from mean).
    """
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    assert pred.shape == target.shape, f"shape mismatch {pred.shape} vs {target.shape}"
    T = pred.shape[1]
    p = np.transpose(pred, (1, 0, 2)).reshape(T, -1)  # [T, N*L]
    t = np.transpose(target, (1, 0, 2)).reshape(T, -1)
    rs = np.full(T, np.nan)
    eps = 1e-8  # treat tracks constant up to float noise (e.g. all-zero) as undefined -> NaN
    for k in range(T):
        pk, tk = p[k], t[k]
        if pk.std() > eps and tk.std() > eps:
            rs[k] = float(np.corrcoef(pk, tk)[0, 1])
    mean_r = float(np.nanmean(rs)) if np.isfinite(rs).any() else float("nan")
    return mean_r, rs


def compute_track_metrics(eval_pred):
    """HF Trainer-style ``compute_metrics`` adapter: returns {'track_pearson': mean_r}."""
    preds, labels = eval_pred
    mean_r, _ = per_track_pearson(preds, labels)
    return {"track_pearson": mean_r}


class TracksMetrics:
    """Streaming multi-track Pearson, faithful to the official NTv3 FT notebook.

    Uses ``torchmetrics.PearsonCorrCoef(num_outputs=T)`` in float64 (for numerical stability) and
    accumulates over flattened ``(batch*seq_len, T)`` (prediction, target) pairs across batches —
    i.e. correlation is pooled over all positions, computed in the SAME (scaled) space the targets
    live in (no log1p). ``compute()`` returns per-track + mean Pearson and the mean loss.
    """

    def __init__(self, track_names, device="cpu"):
        from torchmetrics import PearsonCorrCoef

        self.track_names = list(track_names)
        self.num_tracks = len(self.track_names)
        self.pearson = PearsonCorrCoef(num_outputs=self.num_tracks).to(device)
        self.pearson.set_dtype(torch.float64)
        self.losses = []

    def reset(self):
        self.pearson.reset()
        self.losses = []

    def update(self, predictions: "torch.Tensor", targets: "torch.Tensor", loss: float = None):
        """Accumulate one batch. predictions/targets: (..., num_tracks)."""
        self.pearson.update(
            predictions.detach().reshape(-1, self.num_tracks).to(torch.float64),
            targets.detach().reshape(-1, self.num_tracks).to(torch.float64),
        )
        if loss is not None:
            self.losses.append(loss)

    def compute(self):
        """Return {track/pearson, mean/pearson, loss}."""
        # atleast_1d: torchmetrics PearsonCorrCoef(num_outputs=1) returns a 0-d scalar, which would crash
        # the per-track indexing below for a single-track (specialist) student. No-op for T>1.
        corr = np.atleast_1d(self.pearson.compute().cpu().numpy())
        out = {f"{n}/pearson": float(corr[i]) for i, n in enumerate(self.track_names)}
        out["mean/pearson"] = float(corr.mean())
        out["loss"] = float(np.mean(self.losses)) if self.losses else float("nan")
        return out
