"""Comprehensive audit of DilatedTrackNet (the multi-track student).

Angles covered: (1) I/O shape, (2) receptive-field formula, (3) EMPIRICAL receptive field via
finite differences, (4) gradient flow, (5) numerical stability incl. batch=1 / train+eval, (6) loss
integration (mse/poisson/pearson) + align + Pearson, (7) drop-in contract vs BPNetRegressor,
(8) eval determinism, (9) learnability (overfits a tiny set), (10) padding preserves length.

Run: python -m tests.test_dilated_track_net
"""

import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.track_dataset import TrackDataset
from src.model.bpnet_regressor import BPNetRegressor, BPNetRegressorConfig
from src.model.dilated_track_net import DilatedTrackNet, DilatedTrackNetConfig
from src.trainer.track_distill import _teacher_to_btl, align_student_to_teacher, track_distill_loss
from src.trainer.track_metrics import per_track_pearson

_n = 0


def ok(c, m):
    global _n
    assert c, "FAIL: " + m
    _n += 1


def _small(num_tracks=3, **kw):
    return DilatedTrackNet(
        DilatedTrackNetConfig(num_tracks=num_tracks, hidden=16, n_blocks=4, dropout=0.0, **kw)
    )


def test_shape():
    net = _small(num_tracks=9)
    for B, L in [(2, 512), (1, 256), (3, 1024)]:
        out = net(torch.randint(0, 4, (B, L)))
        ok(tuple(out.shape) == (B, 9, L), f"shape {B}x{L} -> [B,9,L] (got {tuple(out.shape)})")


def test_receptive_field_formula():
    # defaults must cover a 16 kb window
    net = DilatedTrackNet(DilatedTrackNetConfig(num_tracks=9))
    ok(net.receptive_field >= 16384, f"default RF {net.receptive_field} >= 16384")
    # small net: stem 1, dilations 1,2,4 -> RF = 1 + 0 + 2*(1+2+4) = 15
    n = DilatedTrackNetConfig(
        num_tracks=3, hidden=8, n_blocks=3, kernel=3, stem_kernel=1, dropout=0.0
    )
    ok(DilatedTrackNet(n).receptive_field == 15, "RF formula 15 for stem1/dils[1,2,4]")


def test_receptive_field_empirical():
    """Finite-difference: perturbing input within RF changes center output; beyond RF does not."""
    net = DilatedTrackNet(
        DilatedTrackNetConfig(
            num_tracks=3, hidden=8, n_blocks=3, kernel=3, stem_kernel=1, dropout=0.0
        )
    ).eval()
    half = net.receptive_field // 2  # receptive field 15 -> half 7
    L, c = 200, 100
    base = torch.zeros(1, L, dtype=torch.long)
    with torch.no_grad():
        out0 = net(base)[0, :, c].clone()
        din = base.clone()
        din[0, c + half] = 1  # just inside RF
        d_in = (net(din)[0, :, c] - out0).abs().max().item()
        dout = base.clone()
        dout[0, c + half + 1] = 1  # just outside RF
        d_out = (net(dout)[0, :, c] - out0).abs().max().item()
    ok(d_in > 1e-6, f"perturb within RF changes center (delta={d_in:.2e})")
    ok(d_out < 1e-9, f"perturb beyond RF leaves center unchanged (delta={d_out:.2e})")


def test_grad_flow():
    net = _small(num_tracks=4)
    out = net(torch.randint(0, 4, (2, 256)))
    out.pow(2).mean().backward()
    bad = [
        name
        for name, p in net.named_parameters()
        if p.grad is None or not torch.isfinite(p.grad).all()
    ]
    ok(not bad, f"all params have finite grads (offenders: {bad[:3]})")


def test_numerical_stability():
    net = _small(num_tracks=5)
    for mode in ("train", "eval"):
        net.train(mode == "train")
        for B in (1, 4):
            out = net(torch.randint(0, 4, (B, 384)))
            ok(torch.isfinite(out).all(), f"{mode} B={B} output finite")


def test_loss_integration():
    net = _small(num_tracks=6)
    ids = torch.randint(0, 4, (3, 512))
    teacher = torch.rand(3, 192, 6) * 5  # [B, Lt, T] non-negative (poisson-valid)
    for kind in ("mse", "poisson", "pearson", "mse+pearson"):
        loss = track_distill_loss(net(ids), teacher, kind)
        ok(torch.isfinite(loss) and loss.ndim == 0, f"loss '{kind}' is finite scalar")
    aligned = align_student_to_teacher(net(ids), 192)
    mr, per = per_track_pearson(aligned.detach().numpy(), _teacher_to_btl(teacher).numpy())
    ok(len(per) == 6 and np.isfinite(mr), "per_track_pearson finite over aligned output")


def test_dropin_contract():
    """Same I/O contract as BPNetRegressor: input_ids [B,L] -> [B,T,L_out]."""
    ids = torch.randint(0, 4, (2, 512))
    dil = _small(num_tracks=7)(ids)
    bp = BPNetRegressor(BPNetRegressorConfig(num_tracks=7, model_size="small"))(ids)
    ok(dil.shape[:2] == bp.shape[:2] == (2, 7), "both emit [B=2, T=7, L]")
    ok(dil.dim() == 3 and bp.dim() == 3, "both 3-D [B,T,L]")


def test_eval_determinism():
    net = _small(num_tracks=3).eval()
    ids = torch.randint(0, 4, (2, 256))
    with torch.no_grad():
        ok(torch.allclose(net(ids), net(ids)), "eval output deterministic")


def test_padding_preserves_length():
    for k, nb, sk in [(3, 5, 7), (5, 3, 9), (3, 8, 15)]:
        net = DilatedTrackNet(
            DilatedTrackNetConfig(
                num_tracks=2, hidden=8, n_blocks=nb, kernel=k, stem_kernel=sk, dropout=0.0
            )
        )
        out = net(torch.randint(0, 4, (1, 333)))
        ok(out.shape[-1] == 333, f"k={k},nb={nb},sk={sk} preserves L=333")


def test_can_learn():
    """Overfit a tiny set -> train per-track Pearson should climb well above 0."""
    torch.manual_seed(0)
    N, L, Lt, T = 6, 512, 128, 3
    seqs = ["".join("ACGT"[i] for i in torch.randint(0, 4, (L,)).tolist()) for _ in range(N)]
    targets = torch.randn(N, Lt, T)
    ds = TrackDataset(seqs, targets, max_len=L)
    net = DilatedTrackNet(DilatedTrackNetConfig(num_tracks=T, hidden=32, n_blocks=4, dropout=0.0))
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    for _ in range(120):
        for ids, t in DataLoader(ds, batch_size=3, shuffle=True):
            opt.zero_grad()
            track_distill_loss(net(ids), t, "mse").backward()
            opt.step()
    net.eval()
    with torch.no_grad():
        P = [
            align_student_to_teacher(net(ids), _teacher_to_btl(t).shape[-1]).numpy()
            for ids, t in DataLoader(ds, batch_size=3)
        ]
        G = [_teacher_to_btl(t).numpy() for _, t in DataLoader(ds, batch_size=3)]
    mr, _ = per_track_pearson(np.concatenate(P), np.concatenate(G))
    ok(mr > 0.5, f"overfits tiny set: train mean Pearson {mr:.3f} > 0.5")


if __name__ == "__main__":
    tests = [
        test_shape,
        test_receptive_field_formula,
        test_receptive_field_empirical,
        test_grad_flow,
        test_numerical_stability,
        test_loss_integration,
        test_dropin_contract,
        test_eval_determinism,
        test_padding_preserves_length,
        test_can_learn,
    ]
    failed = 0
    for fn in tests:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except Exception as e:
            failed += 1
            print(f"FAIL  {fn.__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed, {_n} assertions")
    sys.exit(1 if failed else 0)
