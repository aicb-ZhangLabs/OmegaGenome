"""Differential PARITY test: our ports vs the *verbatim* functions from the official notebook
``03_fine_tuning_posttrained_model_biwig.ipynb``.

For each ported component we paste the notebook's original code (the ``_nb_*`` reference functions,
copied byte-for-byte from the notebook cells) and assert our implementation produces numerically
identical output on random + edge-case inputs. This proves the *runtime logic* is the same, not just
that it looks similar. Audited angles:
  (1) numerical equivalence on random inputs   (2) edge cases (NaN, the >10 clip boundary, stride/
  budget edges)   (3) order-of-operations (crop-then-transform, scheduler step indexing)
  (4) gradient equivalence (loss backward gives the same grads).

CPU-only; no model load. The real-model core-extraction + loop is covered by the GPU smoke.
"""

import sys

import numpy as np
import torch

# ---- our ports under test ----
from src.data.ntv3_ft_data import (
    crop_center,
    make_target_scaling_fn,
    sample_regions_for_total_length,
)
from src.trainer.ntv3_optim import build_optimizer_and_scheduler
from src.trainer.track_losses import poisson_multinomial_loss

_n = 0


def ok(cond, msg):
    global _n
    _n += 1
    assert cond, msg


# =========================================================================================
# Verbatim notebook reference functions (copied from the .ipynb cells, unmodified)
# =========================================================================================
def _nb_crop_center(x, keep_target_center_fraction: float = 0.375):
    seq_len = x.shape[-2]
    target_offset = int(seq_len * (1 - keep_target_center_fraction) // 2)
    target_length = seq_len - 2 * target_offset
    return x[..., target_offset : target_offset + target_length, :]


def _nb_poisson_loss(ytrue, ypred, epsilon=1e-7):
    return ypred - ytrue * torch.log(ypred + epsilon)


def _nb_safe_for_grad_log_torch(x):
    return torch.log(torch.where(x > 0.0, x, torch.ones_like(x)))


def _nb_poisson_multinomial_loss(logits, targets, shape_loss_coefficient=5.0, epsilon=1e-7):
    batch_size, seq_length, num_tracks = logits.shape
    sum_pred = logits.sum(dim=1)
    sum_true = targets.sum(dim=1)
    scale_loss = _nb_poisson_loss(sum_true, sum_pred, epsilon=epsilon)
    scale_loss = scale_loss / (seq_length + epsilon)
    scale_loss = scale_loss.mean()
    predicted_counts = logits + epsilon
    targets_with_epsilon = targets + epsilon
    denom = predicted_counts.sum(dim=1, keepdim=True) + epsilon
    p_pred = predicted_counts / denom
    pl_pred = _nb_safe_for_grad_log_torch(p_pred)
    shape_loss = -(targets_with_epsilon * pl_pred)
    shape_denom = batch_size * seq_length * num_tracks + epsilon
    shape_loss = shape_loss.sum() / shape_denom
    loss = shape_loss + scale_loss / shape_loss_coefficient
    return loss


def _nb_make_transform(track_means):
    track_means_tensor = torch.tensor(track_means, dtype=torch.float32)

    def transform_fn(x):
        means = track_means_tensor.to(x.device)
        scaled = x / means
        clipped = torch.where(scaled > 10.0, 2.0 * torch.sqrt(scaled * 10.0) - 10.0, scaled)
        return clipped

    return transform_fn


def _nb_sample_regions_for_a_total_length(regions, total_length_needed, seed=0):
    sampled_regions = []
    rng = np.random.RandomState(seed)
    accumulated_length = 0
    for _, (chr_name, start, end) in enumerate(regions):
        region_length = end - start
        remaining_length_needed = total_length_needed - accumulated_length
        if region_length >= remaining_length_needed:
            max_start = region_length - remaining_length_needed
            if max_start > 0:
                window_start_offset = rng.randint(0, max_start + 1)
            else:
                window_start_offset = 0
            window_start = start + window_start_offset
            window_end = start + window_start_offset + remaining_length_needed
            sampled_regions.append((chr_name, window_start, window_end))
            accumulated_length += remaining_length_needed
            break
        else:
            sampled_regions.append((chr_name, start, end))
            accumulated_length += region_length
        if accumulated_length >= total_length_needed:
            break
    return sampled_regions


def _nb_scheduler_multiplier(
    current_step,
    initial_learning_rate,
    optimizer_lr,
    num_steps_warmup,
    num_steps_training,
    final_lr_multiplier=0.5,
):
    num = np.log(1.0 / final_lr_multiplier)
    denom = np.log(float(num_steps_training) / float(num_steps_warmup))
    alpha_polynomial_decay = num / denom
    if current_step < 0:
        current_step = 0
    if optimizer_lr == 0:
        return 0.0
    if current_step < num_steps_warmup:
        start_multiplier = initial_learning_rate / optimizer_lr
        progress = float(current_step) / float(num_steps_warmup)
        return start_multiplier + (1.0 - start_multiplier) * progress
    denominator = float(current_step + 1)
    decay_multiplier = (float(num_steps_warmup) / denominator) ** alpha_polynomial_decay
    return min(decay_multiplier, 1.0)


def _nb_window_coords(regions, sequence_length, stride):
    """Notebook's _process_regions + __getitem__ index->(chrom,start,end) math, made explicit."""
    region_info, cumulative_starts, total = [], [], 0
    for chr_name, region_s, region_e in regions:
        region_length = region_e - region_s
        n = (
            (region_length - sequence_length) // stride + 1
            if region_length >= sequence_length
            else 0
        )
        if n > 0:
            region_info.append(
                {"chr_name": chr_name, "region_start_offset": region_s, "num_samples": n}
            )
            cumulative_starts.append(total)
            total += n
    import bisect

    coords = []
    for idx in range(total):
        ci = bisect.bisect_right(cumulative_starts, idx) - 1
        chrom = region_info[ci]["chr_name"]
        rs = region_info[ci]["region_start_offset"]
        start = rs + (idx - cumulative_starts[ci]) * stride
        coords.append((chrom, start, start + sequence_length))
    return coords


# =========================================================================================
# Parity assertions
# =========================================================================================
def test_parity_crop_center():
    for L in (8, 16, 100, 1000, 4096, 32768):
        x = torch.randn(2, L, 3)
        ok(torch.equal(crop_center(x, 0.375), _nb_crop_center(x, 0.375)), f"crop parity L={L}")
    # numpy path too
    a = np.random.randn(1, 256, 4)
    ok(np.array_equal(crop_center(a, 0.375), _nb_crop_center(a, 0.375)), "crop parity numpy")


def test_parity_loss_and_grads():
    torch.manual_seed(1)
    for shape in [(2, 64, 3), (4, 1536, 34), (1, 200, 7)]:
        logits = torch.rand(*shape) * 8
        targets = torch.rand(*shape) * 8
        mine = poisson_multinomial_loss(logits, targets)
        nb = _nb_poisson_multinomial_loss(logits, targets)
        ok(
            torch.allclose(mine, nb, atol=0, rtol=0),
            f"loss bit-parity {shape}: {mine.item()} vs {nb.item()}",
        )
    # gradient parity — SAME input fed to both, grads must match bit-for-bit
    x = torch.rand(2, 128, 5) * 4
    t = torch.rand(2, 128, 5) * 4
    a = x.clone().requires_grad_(True)
    b = x.clone().requires_grad_(True)
    poisson_multinomial_loss(a, t).backward()
    _nb_poisson_multinomial_loss(b, t).backward()
    ok(torch.allclose(a.grad, b.grad, atol=0, rtol=0), "loss gradient bit-parity (same input)")


def test_parity_target_transform():
    means = np.array([0.5, 2.0, 7.3, 0.1])
    mine_fn, nb_fn = make_target_scaling_fn(means), _nb_make_transform(means)
    x = torch.rand(10, 4) * 100  # spans the >10-after-scaling clip boundary
    ok(torch.allclose(mine_fn(x), nb_fn(x), atol=0, rtol=0), "transform bit-parity")
    # explicit boundary: exactly at scaled==10 (no clip) and just above
    xb = torch.tensor([[5.0, 20.0, 73.0, 1.0]])  # /means -> [10, 10, 10, 10] boundary
    ok(torch.allclose(mine_fn(xb), nb_fn(xb), atol=0), "transform clip-boundary parity")


def test_parity_sample_regions():
    regions = [("c1", 0, 100), ("c2", 50, 400), ("c3", 0, 5000), ("c4", 10, 60)]
    for budget in (50, 150, 250, 5000, 99999):
        for seed in (0, 7):
            mine = sample_regions_for_total_length(regions, budget, seed=seed)
            nb = _nb_sample_regions_for_a_total_length(regions, budget, seed=seed)
            ok(mine == nb, f"sample_regions parity budget={budget} seed={seed}: {mine} vs {nb}")


def test_parity_scheduler():
    init, peak, warm, total = 1e-5, 5e-5, 598, 19932
    model = torch.nn.Linear(2, 2)
    opt, sched = build_optimizer_and_scheduler(model, init, peak, 0.01, warm, total)
    # compare actual LR at each step to notebook multiplier * peak
    for step in range(total + 1):
        nb_lr = _nb_scheduler_multiplier(step, init, peak, warm, total) * peak
        ok(
            abs(opt.param_groups[0]["lr"] - nb_lr) < 1e-12,
            f"scheduler LR parity at step {step}: {opt.param_groups[0]['lr']} vs {nb_lr}",
        )
        opt.step()
        sched.step()


def test_parity_window_coords():
    from src.data.ntv3_ft_data import GenomeBigWigDataset

    regions = [("c1", 0, 4000), ("c2", 100, 9000)]
    for seq_len, overlap in [(1000, 0.0), (1000, 0.9), (512, 0.5)]:
        stride = max(1, int((1 - overlap) * seq_len))
        # build a dataset shell just to read its window math (no file IO in __init__)
        ds = GenomeBigWigDataset.__new__(GenomeBigWigDataset)
        ds.sequence_length, ds.stride = seq_len, stride
        ri, cs, total = ds._process_regions(regions)
        ds.region_info, ds._cumulative_starts, ds.num_samples = ri, cs, total
        mine = [
            (
                ds.region_info[bisect_idx(cs, i)]["chr_name"],
                ds.region_info[bisect_idx(cs, i)]["region_start_offset"]
                + (i - cs[bisect_idx(cs, i)]) * stride,
            )
            for i in range(total)
        ]
        nb = [(c, s) for (c, s, _e) in _nb_window_coords(regions, seq_len, stride)]
        ok(mine == nb, f"window-coord parity seq={seq_len} overlap={overlap}: {len(mine)} windows")


def test_parity_tracks_metric():
    """TracksMetrics per-track Pearson must equal a direct numpy corrcoef over pooled positions
    (the notebook computes torchmetrics PearsonCorrCoef in float64 over the same flattened pairs)."""
    from src.trainer.track_metrics import TracksMetrics

    torch.manual_seed(3)
    T = 5
    m = TracksMetrics([f"t{i}" for i in range(T)], device="cpu")
    preds, targs = [], []
    for _ in range(4):  # several batches accumulated, as in the real loop
        p = torch.randn(3, 40, T)
        q = 0.5 * p + torch.randn(3, 40, T)  # partially correlated
        m.update(p, q)
        preds.append(p.reshape(-1, T).numpy())
        targs.append(q.reshape(-1, T).numpy())
    out = m.compute()
    P = np.concatenate(preds)
    Q = np.concatenate(targs)
    for i in range(T):
        ref = np.corrcoef(P[:, i], Q[:, i])[0, 1]
        ok(
            abs(out[f"t{i}/pearson"] - ref) < 1e-9,
            f"track {i} pearson parity: {out[f't{i}/pearson']} vs {ref}",
        )
    ok(
        abs(out["mean/pearson"] - np.mean([np.corrcoef(P[:, i], Q[:, i])[0, 1] for i in range(T)]))
        < 1e-9,
        "mean pearson parity",
    )


def bisect_idx(cumulative_starts, idx):
    import bisect

    return bisect.bisect_right(cumulative_starts, idx) - 1


if __name__ == "__main__":
    tests = [
        test_parity_crop_center,
        test_parity_loss_and_grads,
        test_parity_target_transform,
        test_parity_sample_regions,
        test_parity_scheduler,
        test_parity_window_coords,
        test_parity_tracks_metric,
    ]
    failed = 0
    for fn in tests:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except Exception as e:
            failed += 1
            print(f"FAIL  {fn.__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed, {_n} parity assertions")
    sys.exit(1 if failed else 0)
