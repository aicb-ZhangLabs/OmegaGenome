"""Distill NTv3 per-bp tracks into a compact BPNetRegressor student; report per-track Pearson.

Loads the cached (sequences, NTv3 targets) from ntv3_gen_targets, trains the student to match the
teacher's per-bp tracks (resolution-aligned MSE), and evaluates per-track Pearson on the held-out
split — the OmegaGenome per-bp regression result.

  python -m src.train.distill_tracks --data <dir> --out <dir> [--model_size medium] ...
"""

import argparse
import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.track_dataset import TrackDataset
from src.model.bpnet_regressor import BPNetRegressor, BPNetRegressorConfig
from src.model.dilated_track_net import DilatedTrackNet, DilatedTrackNetConfig
from src.trainer.track_distill import _teacher_to_btl, align_student_to_teacher, track_distill_loss
from src.trainer.track_metrics import per_track_pearson


def _load_raw(path: str):
    d = torch.load(path, weights_only=False)
    return d["sequences"], d["targets"], d["labels"], d["window"]


def _track_stats(targets):
    """Per-track mean/std over (windows, positions) -> [1, 1, T] each."""
    mu = targets.mean(dim=(0, 1), keepdim=True)
    sd = targets.std(dim=(0, 1), keepdim=True).clamp_min(1e-6)
    return mu, sd


@torch.no_grad()
def _evaluate(student, ds, device, batch_size, loss_kind="mse"):
    """Per-track Pearson between student and teacher tracks on a dataset."""
    student.eval()
    preds, tgts = [], []
    for ids, tgt in DataLoader(ds, batch_size=batch_size):
        out = student(ids.to(device))  # [B, T, Ls]
        if loss_kind == "poisson":
            out = out.exp()  # student emits a log-rate under poisson; correlate the rate itself
        teacher = _teacher_to_btl(tgt.to(device))  # [B, T, Lt]
        out = align_student_to_teacher(out, teacher.shape[-1])  # [B, T, Lt]
        preds.append(out.cpu().numpy())
        tgts.append(teacher.cpu().numpy())
    mean_r, per = per_track_pearson(np.concatenate(preds), np.concatenate(tgts))
    return mean_r, per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="dir with train.pt / test.pt")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model_size", default="medium")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--no-normalize", action="store_true", help="disable per-track target z-norm")
    ap.add_argument("--student", choices=["bpnet", "dilated"], default="bpnet")
    ap.add_argument("--loss", choices=["mse", "poisson", "pearson", "mse+pearson"], default="mse")
    ap.add_argument("--hidden", type=int, default=256, help="dilated student channels")
    ap.add_argument("--n_blocks", type=int, default=14, help="dilated student blocks (sets receptive field)")
    args = ap.parse_args()
    if args.loss == "poisson" and not args.no_normalize:
        raise SystemExit("poisson needs raw (non-negative) targets; pass --no-normalize with --loss poisson")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tr_seqs, tr_tgt, labels, window = _load_raw(os.path.join(args.data, "train.pt"))
    te_seqs, te_tgt, _, _ = _load_raw(os.path.join(args.data, "test.pt"))
    # Per-track z-normalization with TRAIN stats (heterogeneous track scales otherwise let the
    # high-magnitude tracks dominate MSE). Pearson is scale-invariant, so the metric is unchanged;
    # this only conditions training. Disable with --no-normalize.
    if not args.no_normalize:
        mu, sd = _track_stats(tr_tgt)
        tr_tgt, te_tgt = (tr_tgt - mu) / sd, (te_tgt - mu) / sd
    train_ds = TrackDataset(tr_seqs, tr_tgt, max_len=window)
    test_ds = TrackDataset(te_seqs, te_tgt, max_len=window)
    T = len(labels)
    if args.student == "dilated":
        student = DilatedTrackNet(
            DilatedTrackNetConfig(num_tracks=T, hidden=args.hidden, n_blocks=args.n_blocks)
        ).to(device)
        print(f"student=dilated hidden={args.hidden} blocks={args.n_blocks} "
              f"receptive_field={student.receptive_field} (window={window})")
    else:
        student = BPNetRegressor(BPNetRegressorConfig(num_tracks=T, model_size=args.model_size)).to(device)
    opt = torch.optim.Adam(student.parameters(), lr=args.lr)
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)

    os.makedirs(args.out, exist_ok=True)
    best = -1.0
    for ep in range(args.epochs):
        student.train()
        tot = 0.0
        for ids, tgt in loader:
            opt.zero_grad()
            loss = track_distill_loss(student(ids.to(device)), tgt.to(device), args.loss)
            loss.backward()
            opt.step()
            tot += loss.item()
        mean_r, _ = _evaluate(student, test_ds, device, args.batch_size, args.loss)
        print(f"epoch {ep:3d}  train_loss={tot / len(loader):.4f}  test_mean_pearson={mean_r:.4f}")
        if mean_r > best:
            best = mean_r
            torch.save(student.state_dict(), os.path.join(args.out, "student_best.pt"))

    mean_r, per = _evaluate(student, test_ds, device, args.batch_size, args.loss)
    result = {
        "student": args.student,
        "loss": args.loss,
        "model_size": args.model_size,
        "n_tracks": T,
        "labels": labels,
        "best_test_mean_pearson": best,
        "final_test_mean_pearson": float(mean_r),
        "per_track_pearson": {labels[i]: (None if np.isnan(per[i]) else float(per[i])) for i in range(T)},
        "student_params": sum(p.numel() for p in student.parameters()),
    }
    with open(os.path.join(args.out, "track_distill_result.json"), "w") as f:
        json.dump(result, f, indent=2)
    print("\nper-track Pearson:")
    for i in range(T):
        print(f"  {labels[i]:12s} {per[i]:.4f}")
    print(f"mean (best): {best:.4f} | student params: {result['student_params'] / 1e6:.3f}M")


if __name__ == "__main__":
    main()
