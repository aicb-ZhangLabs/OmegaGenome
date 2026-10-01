"""Audit of the faithful NTv3 benchmark fine-tuning pipeline (CPU-only; no heavy NTv3 load).

Covers the official-notebook port across angles: (1) crop_center fraction/centering, (2) LinearHead
shape + non-negativity, (3) Poisson-multinomial loss shape/finiteness/optimum, (4) TracksMetrics
pooled Pearson, (5) target scaling (x/mean + softclip), (6) region-budget sampler, (7) the dense
GenomeBigWigDataset on a synthetic genome+bigWig, (8) optimizer/scheduler warmup->decay shape.
"""

import os
import sys
import tempfile

import numpy as np
import torch

from src.data.ntv3_ft_data import (
    GenomeBigWigDataset,
    crop_center,
    make_target_scaling_fn,
    sample_regions_for_total_length,
)
from src.model.ntv3_finetune import LinearHead
from src.trainer.ntv3_optim import build_optimizer_and_scheduler
from src.trainer.track_losses import poisson_multinomial_loss
from src.trainer.track_metrics import TracksMetrics

_n = 0


def ok(cond, msg):
    global _n
    _n += 1
    assert cond, msg


def test_crop_center():
    x = torch.arange(8).view(1, 8, 1).float()  # length 8
    c = crop_center(x, 0.375)  # offset=int(8*0.625//2)=2, len=4
    ok(c.shape == (1, 4, 1), f"crop shape {c.shape}")
    ok(c[0, :, 0].tolist() == [2, 3, 4, 5], f"crop centered {c[0, :, 0].tolist()}")
    ok(crop_center(np.arange(8).reshape(1, 8, 1), 0.375).shape == (1, 4, 1), "numpy crop ok")


def test_linear_head():
    h = LinearHead(16, 5)
    y = h(torch.randn(2, 10, 16))
    ok(y.shape == (2, 10, 5), f"head shape {y.shape}")
    ok((y >= 0).all(), "softplus non-negative")


def test_poisson_multinomial_loss():
    torch.manual_seed(0)
    targets = torch.rand(2, 12, 3) * 5
    perfect = poisson_multinomial_loss(targets.clone(), targets.clone())
    far = poisson_multinomial_loss(torch.rand(2, 12, 3) * 50, targets)
    ok(torch.isfinite(perfect) and torch.isfinite(far), "loss finite")
    ok(perfect.item() < far.item(), "loss lower when prediction matches target")
    pred = (torch.rand(2, 12, 3) * 5).requires_grad_(True)
    poisson_multinomial_loss(pred, targets).backward()
    ok(pred.grad is not None and torch.isfinite(pred.grad).all(), "grad finite")


def test_tracks_metrics():
    m = TracksMetrics(["a", "b"], device="cpu")
    pred = torch.randn(20, 2)
    m.update(pred, pred.clone(), loss=0.5)  # perfectly correlated
    out = m.compute()
    ok(
        abs(out["a/pearson"] - 1.0) < 1e-6 and abs(out["mean/pearson"] - 1.0) < 1e-6,
        f"perfect corr -> 1.0, got {out['mean/pearson']}",
    )
    ok(abs(out["loss"] - 0.5) < 1e-9, "loss recorded")


def test_target_scaling():
    fn = make_target_scaling_fn(np.array([2.0, 4.0]))
    x = torch.tensor([[2.0, 4.0], [40.0, 4.0]])  # row1 -> [1,1]; row2 col0 -> 20 (>10 clip)
    y = fn(x)
    ok(abs(y[0, 0] - 1.0) < 1e-6 and abs(y[0, 1] - 1.0) < 1e-6, "below-clip = x/mean")
    expected = 2.0 * (20.0 * 10.0) ** 0.5 - 10.0
    ok(abs(y[1, 0].item() - expected) < 1e-4, f"softclip >10: {y[1, 0].item()} vs {expected}")


def test_sample_regions_budget():
    regions = [("chr1", 0, 100), ("chr2", 0, 100), ("chr3", 0, 1000)]
    s = sample_regions_for_total_length(regions, 250)
    ok(sum(e - st for _, st, e in s) == 250, f"budget met: {s}")
    ok(s[0] == ("chr1", 0, 100) and s[2][0] == "chr3", "accumulates across regions in order")


def _write_synth_genome(d, length=4000):
    fasta = os.path.join(d, "genome.fasta")
    with open(fasta, "w") as f:
        f.write(">chr1\n")
        seq = "ACGT" * (length // 4)
        for i in range(0, len(seq), 80):
            f.write(seq[i : i + 80] + "\n")
    import pyBigWig

    bw_path = os.path.join(d, "t0.bigwig")
    bw = pyBigWig.open(bw_path, "w")
    bw.addHeader([("chr1", length)])
    bw.addEntries("chr1", [0], values=[3.0], span=length)  # constant signal
    bw.close()
    return fasta, [bw_path]


class _StubTokenizer:
    def __call__(self, seq, padding=None, truncation=None, max_length=None, return_tensors=None):
        return {"input_ids": torch.zeros(1, max_length, dtype=torch.long)}


def test_genome_bigwig_dataset():
    with tempfile.TemporaryDirectory() as d:
        fasta, bw_paths = _write_synth_genome(d, length=4000)
        regions = [("chr1", 0, 4000)]
        fn = make_target_scaling_fn(np.array([3.0]))  # mean=3 -> constant signal scales to 1.0
        ds = GenomeBigWigDataset(
            fasta,
            bw_paths,
            regions,
            sequence_length=1000,
            tokenizer=_StubTokenizer(),
            transform_fn=fn,
            overlap=0.0,
            keep_target_center_fraction=0.375,
        )
        ok(len(ds) == 4, f"4 non-overlapping 1000bp windows in 4000bp, got {len(ds)}")
        s = ds[0]
        l_out = 1000 - 2 * int(1000 * 0.625 // 2)
        ok(s["tokens"].shape == (1000,), f"tokens {s['tokens'].shape}")
        ok(s["bigwig_targets"].shape == (l_out, 1), f"targets {s['bigwig_targets'].shape}")
        ok(
            torch.allclose(s["bigwig_targets"], torch.ones(l_out, 1), atol=1e-4),
            "scaled constant -> 1.0",
        )
        ds2 = GenomeBigWigDataset(fasta, bw_paths, regions, 1000, _StubTokenizer(), fn, overlap=0.9)
        ok(len(ds2) > len(ds), f"overlap increases windows: {len(ds2)} > {len(ds)}")


def test_optimizer_scheduler():
    model = torch.nn.Linear(4, 4)
    opt, sched = build_optimizer_and_scheduler(
        model, initial_lr=1e-5, end_lr=5e-5, weight_decay=0.01, num_warmup=10, num_steps=100
    )
    ok(abs(opt.param_groups[0]["lr"] - 1e-5) < 1e-7, "starts near initial lr")
    for _ in range(10):
        opt.step()
        sched.step()
    peak = opt.param_groups[0]["lr"]
    ok(abs(peak - 5e-5) < 5e-6, f"reaches peak after warmup, got {peak}")
    for _ in range(89):
        opt.step()
        sched.step()
    ok(opt.param_groups[0]["lr"] < peak, "decays after warmup")


if __name__ == "__main__":
    tests = [
        test_crop_center,
        test_linear_head,
        test_poisson_multinomial_loss,
        test_tracks_metrics,
        test_target_scaling,
        test_sample_regions_budget,
        test_genome_bigwig_dataset,
        test_optimizer_scheduler,
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
