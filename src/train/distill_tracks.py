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
from src.trainer.track_distill import _teacher_to_btl, align_student_to_teacher, track_distill_loss
from src.trainer.track_metrics import per_track_pearson


def _load_split(path: str):
    d = torch.load(path, weights_only=False)
    return TrackDataset(d["sequences"], d["targets"], max_len=d["window"]), d["labels"]


@torch.no_grad()
def _evaluate(student, ds, device, batch_size):
    """Per-track Pearson between student and teacher tracks on a dataset."""
    student.eval()
    preds, tgts = [], []
    for ids, tgt in DataLoader(ds, batch_size=batch_size):
        out = student(ids.to(device))  # [B, T, Ls]
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
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_ds, labels = _load_split(os.path.join(args.data, "train.pt"))
    test_ds, _ = _load_split(os.path.join(args.data, "test.pt"))
    T = len(labels)
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
            loss = track_distill_loss(student(ids.to(device)), tgt.to(device), "mse")
            loss.backward()
            opt.step()
            tot += loss.item()
        mean_r, _ = _evaluate(student, test_ds, device, args.batch_size)
        print(f"epoch {ep:3d}  train_mse={tot / len(loader):.4f}  test_mean_pearson={mean_r:.4f}")
        if mean_r > best:
            best = mean_r
            torch.save(student.state_dict(), os.path.join(args.out, "student_best.pt"))

    mean_r, per = _evaluate(student, test_ds, device, args.batch_size)
    result = {
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
