"""Orchestration tests for the NTv3 pipeline scripts (window tiling + distill_tracks load/eval).

Synthetic cached data only -> no NTv3 download / no network. Covers the parts of the CLIs that
were previously only syntax-checked. Run: python -m tests.test_ntv3_orchestration
"""

import os
import sys
import tempfile

import numpy as np
import torch

from src.data.ntv3_windows import tile_windows
from src.data.track_dataset import TrackDataset, build_teacher_targets
from src.model.bpnet_regressor import BPNetRegressor, BPNetRegressorConfig
from src.train.distill_tracks import _evaluate, _load_raw, _track_stats


class _StubTeacher:
    """predict_tracks -> [b, L, T] proportional to seq length (mimics NTv3's per-bp output)."""

    def predict_tracks(self, seqs):
        L, T = len(seqs[0]) // 4, 3
        return torch.randn(len(seqs), L, T)


_n = 0


def ok(c, m):
    global _n
    assert c, "FAIL: " + m
    _n += 1


def test_tiling():
    ok(
        tile_windows("chr1", 0, 1000, 100) == [("chr1", i * 100, (i + 1) * 100) for i in range(10)],
        "non-overlapping tiling",
    )
    ok(len(tile_windows("chr1", 0, 1000, 100, n=3)) == 3, "n cap")
    ok(
        tile_windows("chr1", 0, 1000, 300, stride=100)[:2]
        == [("chr1", 0, 300), ("chr1", 100, 400)],
        "stride/overlap",
    )
    ok(tile_windows("chr1", 0, 50, 100) == [], "window > range -> empty")
    ok(
        tile_windows("chr1", 100, 100 + 16384 * 2, 16384, n=2)[0] == ("chr1", 100, 16484),
        "offset start",
    )


def test_load_and_eval():
    with tempfile.TemporaryDirectory() as d:
        N, Lt, T, W = 6, 48, 3, 512
        seqs = ["ACGT" * (W // 4) for _ in range(N)]
        torch.save(
            {
                "sequences": seqs,
                "targets": torch.randn(N, Lt, T),
                "labels": ["a", "b", "c"],
                "window": W,
            },
            os.path.join(d, "test.pt"),
        )
        tgt_raw = (
            torch.randn(N, Lt, T) * torch.tensor([8.0, 2.0, 0.4]) + 3.0
        )  # heterogeneous scales
        torch.save(
            {"sequences": seqs, "targets": tgt_raw, "labels": ["a", "b", "c"], "window": W},
            os.path.join(d, "test.pt"),
        )
        seqs2, tgt, labels, window = _load_raw(os.path.join(d, "test.pt"))
        ok(len(seqs2) == N and labels == ["a", "b", "c"] and window == W, "load_raw fields")
        # per-track z-norm should give ~unit std / ~zero mean per track
        mu, sd = _track_stats(tgt)
        z = (tgt - mu) / sd
        ok(tuple(mu.shape) == (1, 1, T), "track_stats shape")
        ok(bool((z.std(dim=(0, 1)) - 1.0).abs().max() < 0.05), "z-norm unit std per track")
        ds = TrackDataset(seqs2, z, max_len=window)
        ids, t0 = ds[0]
        ok(tuple(ids.shape) == (W,) and tuple(t0.shape) == (Lt, T), "dataset item shapes")
        student = BPNetRegressor(BPNetRegressorConfig(num_tracks=T, model_size="small"))
        mean_r, per = _evaluate(student, ds, "cpu", batch_size=2)
        ok(len(per) == T and np.isfinite(mean_r), "_evaluate per-track + finite")


def test_build_teacher_targets_guard():
    teacher = _StubTeacher()
    tgt = build_teacher_targets(teacher, ["ACGT" * 64] * 3, batch_size=2)  # equal length
    ok(tuple(tgt.shape) == (3, 64, 3), "equal-length build shape")
    try:
        build_teacher_targets(teacher, ["ACGT" * 64, "ACGT" * 65], batch_size=1)
        ok(False, "unequal lengths must raise")
    except ValueError:
        ok(True, "unequal lengths raise ValueError")


if __name__ == "__main__":
    failed = 0
    fns = [test_tiling, test_load_and_eval, test_build_teacher_targets_guard]
    for fn in fns:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except Exception as e:
            failed += 1
            print(f"FAIL  {fn.__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed, {_n} assertions")
    sys.exit(1 if failed else 0)
