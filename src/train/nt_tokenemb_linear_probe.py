#!/usr/bin/env python
"""Linear probe on NT-2.5B TOKEN embeddings (minimal-NT baseline, R1.3 rebuttal).

The smallest possible use of NT-2.5B: use ONLY its FROZEN word-embedding table (the [vocab, 2560]
lookup -- NO transformer layers, no attention, no FFN), mean-pool it over the sequence, and train a
single ``nn.Linear(2560 -> n_classes)`` FROM SCRATCH with cross-entropy. Answers "how much task signal
is in NT's pretrained per-6-mer embeddings alone, read by the simplest linear classifier?"

Only the ~10.5M embedding table is touched (a gather), so this needs NO 2.5B forward. The table is
loaded from a small precomputed tensor (scratchpad/dl_nt_wordemb.py extracts it once):
    <wordemb>.pt  ->  [vocab=4105, 2560] fp16

Data splits + labels come from the SAME loaders the other R1.3 arms use, so the test MCC is directly
comparable to the one-hot / replaceK / replace4 numbers. Trained from scratch (no KD).
"""
import os
import csv
import argparse
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import matthews_corrcoef
from transformers import AutoTokenizer

from src.data.dataset import get_num_labels, build_data_splits_from_huggingface, DatasetConfig

NT_BASE = "InstaDeepAI/nucleotide-transformer-2.5b-multi-species"
NT_DIM = 2560


@torch.no_grad()
def featurize(seqs, tok, W, device, max_length, bs=128):
    """Mean-pool the frozen NT word embeddings over valid (non-pad) tokens -> [N, 2560]."""
    out = []
    for i in range(0, len(seqs), bs):
        enc = tok(list(seqs[i:i + bs]), return_tensors="pt", padding=True,
                  truncation=True, max_length=max_length)
        ids = enc["input_ids"].to(device)
        mask = enc["attention_mask"].to(device).to(W.dtype)       # [b, T]
        emb = W[ids]                                              # [b, T, 2560] (frozen lookup)
        summed = (emb * mask.unsqueeze(-1)).sum(dim=1)            # [b, 2560]
        cnt = mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        out.append((summed / cnt).float().cpu())
    return torch.cat(out, dim=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task-name", required=True)
    ap.add_argument("--results-csv", required=True)
    ap.add_argument("--wordemb", required=True, help="path to the [vocab,2560] NT word-embedding tensor (.pt)")
    ap.add_argument("--max-length", type=int, default=1002, help="tokenizer max tokens (NT max_pos=1002)")
    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--patience", type=int, default=50)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    task = a.task_name
    num_labels = get_num_labels(task)
    print(f"[probe] task={task} num_labels={num_labels} device={device}", flush=True)

    # ---- data (identical splits/labels to the other R1.3 arms) ----
    ds = DatasetConfig(task_name=task, data_path="", random_state=a.seed)
    Xtr, ytr, Xva, yva, Xte, yte = build_data_splits_from_huggingface(ds)
    if a.smoke:
        Xtr, ytr, Xva, yva, Xte, yte = Xtr[:256], ytr[:256], Xva[:128], yva[:128], Xte[:128], yte[:128]
        a.epochs = min(a.epochs, 40)

    # ---- frozen NT token (word) embedding + tokenizer ----
    W = torch.load(a.wordemb, map_location="cpu").float().to(device)   # [vocab, 2560]
    tok = AutoTokenizer.from_pretrained(NT_BASE, trust_remote_code=True)
    print(f"[probe] word-emb {tuple(W.shape)} | train/val/test={len(Xtr)}/{len(Xva)}/{len(Xte)}", flush=True)

    Ftr = featurize(Xtr, tok, W, device, a.max_length).to(device)
    Fva = featurize(Xva, tok, W, device, a.max_length).to(device)
    Fte = featurize(Xte, tok, W, device, a.max_length).to(device)
    ytr_t = torch.as_tensor(np.asarray(ytr), dtype=torch.long, device=device)

    # ---- simplest linear classifier, trained from scratch (CE, full-batch) ----
    clf = nn.Linear(NT_DIM, num_labels).to(device)
    opt = torch.optim.Adam(clf.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    lossf = nn.CrossEntropyLoss()
    best_val, best_test, best_ep, wait = -2.0, None, -1, 0
    for ep in range(1, a.epochs + 1):
        clf.train()
        opt.zero_grad()
        loss = lossf(clf(Ftr), ytr_t)
        loss.backward()
        opt.step()
        clf.eval()
        with torch.no_grad():
            vp = clf(Fva).argmax(dim=1).cpu().numpy()
            tp = clf(Fte).argmax(dim=1).cpu().numpy()
        vmcc = matthews_corrcoef(yva, vp)
        tmcc = matthews_corrcoef(yte, tp)
        if vmcc > best_val:
            best_val, best_test, best_ep, wait = vmcc, tmcc, ep, 0
        else:
            wait += 1
        if ep == 1 or ep % 25 == 0:
            print(f"  epoch {ep}/{a.epochs} loss={loss.item():.4f} val_mcc={vmcc:.4f} "
                  f"test_mcc={tmcc:.4f} best_val={best_val:.4f}", flush=True)
        if a.patience > 0 and wait >= a.patience:
            print(f"  early stop @ {ep} (patience {a.patience})", flush=True)
            break

    trainable = sum(p.numel() for p in clf.parameters())
    print(f"[probe] DONE task={task} best_val_mcc={best_val:.4f} best_test_mcc={best_test:.4f} "
          f"@ep{best_ep} trainable={trainable} frozen_emb={W.numel()}", flush=True)

    os.makedirs(os.path.dirname(a.results_csv) or ".", exist_ok=True)
    newf = not os.path.exists(a.results_csv)
    with open(a.results_csv, "a", newline="") as f:
        w = csv.writer(f)
        if newf:
            w.writerow(["task", "method", "embedding_source", "best_val_mcc", "best_test_mcc",
                        "trainable_params", "frozen_emb_params", "best_epoch", "seed", "timestamp"])
        w.writerow([task, "nt_tokenemb_linear", "nt_base", f"{best_val:.6f}", f"{best_test:.6f}",
                    trainable, W.numel(), best_ep, a.seed, datetime.now().isoformat()])
    print(f"[probe] wrote {a.results_csv}", flush=True)


if __name__ == "__main__":
    main()
