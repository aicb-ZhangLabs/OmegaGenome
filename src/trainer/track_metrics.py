"""Per-track regression metrics for the NTv3 multi-track (per-bp) task.

Track prediction is scored by correlation between predicted and target signal, per track,
then averaged — the standard sequence-to-function metric (vs MCC for the classification tasks).
"""

import numpy as np


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
