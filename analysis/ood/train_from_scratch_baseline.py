#!/usr/bin/env python
"""Train a SINGLE from-scratch (no-teacher) BPNet baseline for the OOD baseline matrix.

The from-scratch baseline tree (.../NT/...onehot-laniakea/<task>/model_final.pt) is missing H3K27ac
(17/18). This script reproduces an architecture-IDENTICAL from-scratch baseline via the OmegaGenome
infra so the cross-task comparison stays apples-to-apples:
  * model  : ``BPNetClassifier(model_size='original')`` -- the SAME 0.12M one-hot BPNet as the distilled
             students and the existing baselines (no teacher_proj; pure classifier head).
  * data   : ``build_data_splits_from_huggingface`` (same loader/seed) + ``SeqDataset`` at MAX_LEN=1024
             (the verified training length; the legacy baseline script new-distillation-4-21-onehot.py
             also used 1024).
  * loss   : PURE cross-entropy (NO teacher, weight_kl=weight_mse=0) -- a from-scratch expert.
  * recipe : lr=1e-4, batch_size=16, AdamW, up to ``--epochs`` (default 200, matching the distilled
             H3K27ac best-HP schedule), best-VAL-MCC checkpoint reported as ``model_final.pt`` (the
             baseline tree's filename), so the diagonal of the baseline matrix == this model's own
             in-task test MCC.

SLURM only (GPU). Saves <out_dir>/<task>/model_final.pt and prints the held-out TEST MCC.
"""

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import matthews_corrcoef

REPO = os.environ.get("OG_ROOT", os.getcwd())
sys.path.insert(0, REPO)

from src.data.dataset import (  # noqa: E402
    DatasetConfig,
    build_data_splits_from_huggingface,
    get_num_labels,
    SeqDataset,
)
from src.model.bpnet_classifier import BPNetClassifier, BPNetClassifierConfig  # noqa: E402

MAX_LEN = 1024  # matches the from-scratch baseline tree + the distilled students (verified).


@torch.no_grad()
def eval_mcc(model, loader, device):
    """Argmax MCC of the model over a (SeqDataset) DataLoader yielding (ids, labels)."""
    model.eval()
    preds, labs = [], []
    for x, y in loader:
        logits = model(x.to(device))
        preds.append(logits.argmax(-1).cpu().numpy())
        labs.append(y.numpy())
    return matthews_corrcoef(np.concatenate(labs), np.concatenate(preds))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument(
        "--out-dir",
        required=True,
        help="parent dir; checkpoint saved at <out-dir>/<task>/model_final.pt",
    )
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    nl = get_num_labels(args.task)
    print(
        f"device={device} task={args.task} num_labels={nl} max_len={MAX_LEN} "
        f"epochs={args.epochs} lr={args.lr} bs={args.batch_size}",
        flush=True,
    )

    cfg = DatasetConfig(task_name=args.task, data_path="", random_state=args.seed)
    X_tr, y_tr, X_va, y_va, X_te, y_te = build_data_splits_from_huggingface(cfg)
    tr = DataLoader(
        SeqDataset(list(X_tr), list(y_tr), MAX_LEN), batch_size=args.batch_size, shuffle=True
    )
    va = DataLoader(SeqDataset(list(X_va), list(y_va), MAX_LEN), batch_size=64)
    te = DataLoader(SeqDataset(list(X_te), list(y_te), MAX_LEN), batch_size=64)

    model = BPNetClassifier(BPNetClassifierConfig(num_labels=nl, model_size="original")).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    ce = nn.CrossEntropyLoss()

    out_task = os.path.join(args.out_dir, args.task)
    os.makedirs(out_task, exist_ok=True)
    ckpt = os.path.join(out_task, "model_final.pt")
    best_val, best_test_at_best_val = -2.0, None

    for ep in range(1, args.epochs + 1):
        model.train()
        for x, y in tr:
            opt.zero_grad()
            loss = ce(model(x.to(device)), y.to(device))
            loss.backward()
            opt.step()
        vmcc = eval_mcc(model, va, device)
        if vmcc > best_val:
            best_val = vmcc
            torch.save(model.state_dict(), ckpt)  # bare state_dict, like the baseline tree
            best_test_at_best_val = eval_mcc(model, te, device)
            print(
                f"epoch {ep:3d} val_mcc={vmcc:.4f} * NEW BEST  test_mcc={best_test_at_best_val:.4f} "
                f"-> saved {ckpt}",
                flush=True,
            )
        elif ep % 10 == 0:
            print(f"epoch {ep:3d} val_mcc={vmcc:.4f} (best {best_val:.4f})", flush=True)

    print(
        f"\nDONE task={args.task} best_val_mcc={best_val:.4f} "
        f"test_mcc_at_best_val={best_test_at_best_val:.4f}",
        flush=True,
    )
    print(f"SAVED {ckpt}", flush=True)


if __name__ == "__main__":
    main()
