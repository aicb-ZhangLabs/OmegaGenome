"""Audit of the single-track metrics fix (np.atleast_1d): TracksMetrics must not crash when a specialist
student has T=1 (torchmetrics returns a 0-d scalar there). 3 angles."""

import numpy as np
import torch
from src.trainer.track_metrics import TracksMetrics, per_track_pearson


def test_single_track_compute_no_crash():
    """T=1: compute() returns a scalar mean/pearson without IndexError (the atleast_1d fix)."""
    m = TracksMetrics(["PRO-cap"], device="cpu")
    pred = torch.rand(4, 32, 1)
    tgt = pred * 0.9 + 0.1 * torch.rand(4, 32, 1)
    m.update(pred, tgt)
    out = m.compute()
    assert "mean/pearson" in out and np.isfinite(out["mean/pearson"])
    assert "PRO-cap/pearson" in out  # the single track is named, not dropped


def test_multi_track_still_per_track():
    """T>1 still yields one entry per track + the mean (no regression from the atleast_1d change)."""
    m = TracksMetrics(["a", "b", "c"], device="cpu")
    pred = torch.rand(4, 16, 3)
    m.update(pred, pred.clone())
    out = m.compute()
    assert all(f"{n}/pearson" in out for n in "abc") and "mean/pearson" in out


def test_per_track_pearson_single_track_shape():
    """The numpy reference also handles T=1: per-track array length 1, finite mean."""
    pred = np.random.rand(3, 1, 50)
    tgt = pred * 0.8 + 0.05
    mean_r, rs = per_track_pearson(pred, tgt)
    assert rs.shape == (1,) and np.isfinite(mean_r)
