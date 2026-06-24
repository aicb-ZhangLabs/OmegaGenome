"""Comprehensive tests for the NTv3 per-bp multi-track distillation pieces.

Audited from several angles: math correctness, shape/orientation (incl. transpose traps),
edge cases, gradient/detach behaviour, and end-to-end learning. CPU-only, no NTv3 download
(teacher-dependent paths use a stub). Run: python -m tests.test_ntv3_multitrack
"""

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.model.bpnet_regressor import BPNetRegressor, BPNetRegressorConfig
from src.trainer.track_distill import (
    align_student_to_teacher,
    track_distill_loss,
    _teacher_to_btl,
)
from src.trainer.track_metrics import per_track_pearson
from src.data.track_dataset import TrackDataset, build_teacher_targets

torch.manual_seed(0)
_checks = 0


def ok(cond, msg):
    global _checks
    assert cond, "FAIL: " + msg
    _checks += 1


# ---------------------------------------------------------------- student
def test_regressor():
    for size, T, L in [("original", 8, 512), ("small", 4, 256), ("medium", 16, 1024)]:
        m = BPNetRegressor(BPNetRegressorConfig(num_tracks=T, model_size=size))
        y = m(torch.randint(0, 4, (3, L)))
        ok(tuple(y.shape) == (3, T, L), f"regressor {size}: shape {tuple(y.shape)} != (3,{T},{L})")
        ok(y.dtype == torch.float32, "regressor output not float32")
    # out_resolution downsamples the position axis
    m = BPNetRegressor(BPNetRegressorConfig(num_tracks=5, model_size="original", out_resolution=4))
    y = m(torch.randint(0, 4, (2, 512)))
    ok(tuple(y.shape) == (2, 5, 128), f"out_resolution pool: {tuple(y.shape)} != (2,5,128)")


# ---------------------------------------------------------------- alignment
def test_align():
    s = torch.randn(2, 7, 1024)
    ok(tuple(align_student_to_teacher(s, 384).shape) == (2, 7, 384), "downsample shape")
    ok(tuple(align_student_to_teacher(s, 2048).shape) == (2, 7, 2048), "upsample shape")
    ok(align_student_to_teacher(s, 1024) is s, "identity should return same tensor")
    # mean-preserving on a constant signal
    c = torch.full((1, 3, 1024), 2.5)
    pooled = align_student_to_teacher(c, 384)
    ok(torch.allclose(pooled, torch.full_like(pooled, 2.5)), "avg-pool not mean-preserving")
    try:
        align_student_to_teacher(torch.randn(2, 10), 5)
        ok(False, "2D student should raise")
    except ValueError:
        ok(True, "2D guard")


# ---------------------------------------------------------------- loss
def test_loss():
    B, T, Lt, Ls = 2, 3, 5, 20
    student = torch.randn(B, T, Ls)
    teacher = torch.randn(B, Lt, T)  # NTv3 orientation [B, L, T]
    loss = track_distill_loss(student, teacher, "mse")
    ok(loss.dim() == 0 and torch.isfinite(loss), "mse loss not finite scalar")

    # orientation/transpose trap: T(3) != Lt(5). If teacher weren't permuted, the track-count
    # check (student T=3 vs teacher dim1) would see 5 and raise -> a clean run proves the permute.
    ok(_teacher_to_btl(teacher).shape == (B, T, Lt), "teacher permute wrong")

    # track-count mismatch must raise
    try:
        track_distill_loss(torch.randn(B, 4, Ls), teacher, "mse")
        ok(False, "track mismatch should raise")
    except ValueError:
        ok(True, "track-count guard")

    # 2D teacher must raise
    try:
        track_distill_loss(student, torch.randn(B, T), "mse")
        ok(False, "2D teacher should raise")
    except ValueError:
        ok(True, "teacher-dim guard")

    # exact-match -> zero MSE (constant teacher; pooled constant student equals it)
    t_const = torch.full((1, Lt, T), 1.7)
    s_const = torch.full((1, T, Ls), 1.7)
    ok(track_distill_loss(s_const, t_const, "mse").item() < 1e-10, "matched MSE not ~0")

    # differentiable wrt student; teacher detached (no grad)
    s = torch.randn(B, T, Ls, requires_grad=True)
    t = torch.randn(B, Lt, T, requires_grad=True)
    track_distill_loss(s, t, "mse").backward()
    ok(s.grad is not None and torch.isfinite(s.grad).all(), "student grad missing")
    ok(t.grad is None, "teacher must NOT receive gradient (should be detached)")

    # poisson finite
    ok(torch.isfinite(track_distill_loss(student, teacher.abs(), "poisson")), "poisson not finite")
    try:
        track_distill_loss(student, teacher, "bogus")
        ok(False, "unknown loss kind should raise")
    except ValueError:
        ok(True, "loss-kind guard")


# ---------------------------------------------------------------- metric
def test_pearson():
    a = np.random.randn(3, 5, 64)
    ok(abs(per_track_pearson(a, a)[0] - 1.0) < 1e-9, "self pearson != 1")
    ok(abs(per_track_pearson(a, -a)[0] + 1.0) < 1e-9, "negated pearson != -1")
    # constant track -> NaN, excluded from the mean
    pred = a.copy()
    tgt = a.copy()
    tgt[:, 0, :] = 4.2  # track 0 constant
    mean_r, per = per_track_pearson(pred, tgt)
    ok(np.isnan(per[0]), "constant track should be NaN")
    ok(np.isfinite(mean_r), "mean should ignore NaN tracks")
    try:
        per_track_pearson(np.zeros((2, 2, 2)), np.zeros((2, 2, 3)))
        ok(False, "shape mismatch should raise")
    except AssertionError:
        ok(True, "pearson shape guard")


# ---------------------------------------------------------------- dataset
class _StubTeacher:
    def __init__(self, Lt, T):
        self.Lt, self.T = Lt, T

    def predict_tracks(self, seqs):
        return torch.randn(len(seqs), self.Lt, self.T)


def test_dataset():
    seqs = ["ACGT" * 64 for _ in range(5)]  # 256 bp each
    targets = torch.randn(5, 96, 4)  # [N, Lt, T]
    ds = TrackDataset(seqs, targets, max_len=256)
    ok(len(ds) == 5, "dataset len")
    ids, tgt = ds[0]
    ok(tuple(ids.shape) == (256,) and tuple(tgt.shape) == (96, 4), "getitem shapes")
    bi, bt = next(iter(DataLoader(ds, batch_size=2)))
    ok(tuple(bi.shape) == (2, 256) and tuple(bt.shape) == (2, 96, 4), "batched shapes")
    # length-mismatch guard
    try:
        TrackDataset(seqs, torch.randn(3, 96, 4), 256)
        ok(False, "len mismatch should raise")
    except ValueError:
        ok(True, "dataset len guard")
    # build_teacher_targets stacks correctly across batches
    built = build_teacher_targets(_StubTeacher(96, 4), seqs, batch_size=2)
    ok(tuple(built.shape) == (5, 96, 4), f"build_teacher_targets shape {tuple(built.shape)}")


# ---------------------------------------------------------------- end-to-end
def test_e2e_learns():
    """Tiny student should fit a *learnable* teacher target (a function of the input) -> loss
    drops a lot. (Random noise unrelated to x is not learnable, so we derive the target from x.)"""
    import torch.nn.functional as F

    B, T, L, Lt = 4, 4, 256, 64
    torch.manual_seed(1)
    student = BPNetRegressor(BPNetRegressorConfig(num_tracks=T, model_size="small"))
    x = torch.randint(0, 4, (B, L))
    # teacher target = smooth function of x: pool one-hot to Lt, mix to T tracks (fixed map)
    onehot = F.one_hot(x, 4).float().permute(0, 2, 1)  # [B, 4, L]
    pooled = F.adaptive_avg_pool1d(onehot, Lt)  # [B, 4, Lt]
    W = torch.randn(T, 4)
    teacher = torch.einsum("tc,bcl->blt", W, pooled).detach()  # [B, Lt, T]
    opt = torch.optim.Adam(student.parameters(), lr=1e-2)
    first = last = None
    for step in range(60):
        opt.zero_grad()
        loss = track_distill_loss(student(x), teacher, "mse")
        loss.backward()
        opt.step()
        if step == 0:
            first = loss.item()
        last = loss.item()
    ok(last < 0.5 * first, f"e2e did not learn: {first:.3f} -> {last:.3f}")


if __name__ == "__main__":
    import sys

    fns = [test_regressor, test_align, test_loss, test_pearson, test_dataset, test_e2e_learns]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except Exception as e:
            failed += 1
            print(f"FAIL  {fn.__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} test groups passed, {_checks} assertions")
    sys.exit(1 if failed else 0)
