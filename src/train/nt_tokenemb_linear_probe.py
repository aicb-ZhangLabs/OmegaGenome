#!/usr/bin/env python
"""Linear classifier on NT-2.5B TOKEN embeddings (minimal-NT baselines, R1.3 rebuttal).

Two modes, both: NT-2.5B word-embedding lookup ([vocab,2560], NO transformer layers) -> mean-pool over
the sequence -> nn.Linear(2560 -> n_classes). Trained from scratch (CE). Data splits/labels come from
the SAME loaders as the other R1.3 arms, so test MCC is directly comparable.

  --trainable-emb OFF (default, "frozen"):  the NT token embedding is FROZEN (only the linear head
      trains, ~5k params). Deployed size ~10.5M (frozen table) + linear. Fast: pooled features are
      precomputed once, then full-batch linear training.
  --trainable-emb ON:  the token embedding is a trainable nn.Embedding INITIALIZED from NT's table and
      fine-tuned end-to-end (~10.5M trainable) alongside the linear head. Needs per-epoch forward, so it
      mini-batches. Fairer same-#trainable-params comparison to a ~10M distilled BPNet.

The word-embedding table is a precomputed tensor (scratchpad/dl_nt_wordemb.py extracts it once):
    <wordemb>.pt  ->  [vocab=4105, 2560]
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


def tokenize_all(seqs, tok, max_length):
    """Tokenize once -> padded token ids [N, Tmax] (long) + attention mask [N, Tmax] (float, CPU)."""
    ids_list, mask_list, Tmax = [], [], 0
    for i in range(0, len(seqs), 256):
        enc = tok(list(seqs[i:i + 256]), padding=False, truncation=True, max_length=max_length)
        for row in enc["input_ids"]:
            ids_list.append(row)
            Tmax = max(Tmax, len(row))
    N = len(ids_list)
    ids = torch.zeros(N, Tmax, dtype=torch.long)
    mask = torch.zeros(N, Tmax, dtype=torch.float32)
    pad_id = tok.pad_token_id or 0
    ids.fill_(pad_id)
    for r, row in enumerate(ids_list):
        ids[r, :len(row)] = torch.tensor(row, dtype=torch.long)
        mask[r, :len(row)] = 1.0
    return ids, mask


def masked_mean(emb, mask):
    """emb [B,T,H], mask [B,T] -> [B,H] mean over valid tokens."""
    summed = (emb * mask.unsqueeze(-1)).sum(dim=1)
    cnt = mask.sum(dim=1, keepdim=True).clamp(min=1.0)
    return summed / cnt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task-name", required=True)
    ap.add_argument("--results-csv", required=True)
    ap.add_argument("--wordemb", required=True, help="path to the [vocab,2560] NT word-embedding tensor (.pt)")
    ap.add_argument("--trainable-emb", action="store_true", help="fine-tune the NT token embedding (init from NT) end-to-end")
    ap.add_argument("--max-length", type=int, default=1002, help="tokenizer max tokens (NT max_pos=1002)")
    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--patience", type=int, default=50)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=256, help="mini-batch size (trainable-emb mode)")
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    task = a.task_name
    num_labels = get_num_labels(task)
    method = "nt_tokenemb_linear_trainable" if a.trainable_emb else "nt_tokenemb_linear_frozen"
    print(f"[probe] task={task} num_labels={num_labels} device={device} mode={method}", flush=True)

    ds = DatasetConfig(task_name=task, data_path="", random_state=a.seed)
    Xtr, ytr, Xva, yva, Xte, yte = build_data_splits_from_huggingface(ds)
    if a.smoke:
        Xtr, ytr, Xva, yva, Xte, yte = Xtr[:256], ytr[:256], Xva[:128], yva[:128], Xte[:128], yte[:128]
        a.epochs = min(a.epochs, 40)

    W = torch.load(a.wordemb, map_location="cpu").float()          # [vocab, 2560]
    tok = AutoTokenizer.from_pretrained(NT_BASE, trust_remote_code=True)
    ytr_np, yva_np, yte_np = np.asarray(ytr), np.asarray(yva), np.asarray(yte)
    print(f"[probe] word-emb {tuple(W.shape)} | train/val/test={len(Xtr)}/{len(Xva)}/{len(Xte)}", flush=True)

    if not a.trainable_emb:
        # -------- FROZEN: precompute pooled features once, full-batch linear --------
        Wd = W.to(device)
        @torch.no_grad()
        def feats(seqs):
            out = []
            for i in range(0, len(seqs), 128):
                enc = tok(list(seqs[i:i + 128]), return_tensors="pt", padding=True, truncation=True, max_length=a.max_length)
                ids = enc["input_ids"].to(device)
                mask = enc["attention_mask"].to(device).float()
                out.append(masked_mean(Wd[ids], mask).cpu())
            return torch.cat(out, 0)
        Ftr, Fva, Fte = feats(Xtr).to(device), feats(Xva).to(device), feats(Xte).to(device)
        ytr_t = torch.as_tensor(ytr_np, dtype=torch.long, device=device)
        clf = nn.Linear(NT_DIM, num_labels).to(device)
        opt = torch.optim.Adam(clf.parameters(), lr=a.lr, weight_decay=a.weight_decay)
        lossf = nn.CrossEntropyLoss()
        best_val, best_test, best_ep, wait = -2.0, None, -1, 0
        for ep in range(1, a.epochs + 1):
            clf.train(); opt.zero_grad()
            loss = lossf(clf(Ftr), ytr_t); loss.backward(); opt.step()
            clf.eval()
            with torch.no_grad():
                vp = clf(Fva).argmax(1).cpu().numpy(); tp = clf(Fte).argmax(1).cpu().numpy()
            vmcc, tmcc = matthews_corrcoef(yva_np, vp), matthews_corrcoef(yte_np, tp)
            if vmcc > best_val:
                best_val, best_test, best_ep, wait = vmcc, tmcc, ep, 0
            else:
                wait += 1
            if ep == 1 or ep % 25 == 0:
                print(f"  epoch {ep}/{a.epochs} loss={loss.item():.4f} val={vmcc:.4f} test={tmcc:.4f} best_val={best_val:.4f}", flush=True)
            if a.patience > 0 and wait >= a.patience:
                print(f"  early stop @ {ep}", flush=True); break
        trainable = sum(p.numel() for p in clf.parameters())
        frozen = W.numel()
    else:
        # -------- TRAINABLE: nn.Embedding init NT, fine-tuned end-to-end, mini-batched --------
        ids_tr, mask_tr = tokenize_all(Xtr, tok, a.max_length)
        ids_va, mask_va = tokenize_all(Xva, tok, a.max_length)
        ids_te, mask_te = tokenize_all(Xte, tok, a.max_length)
        ytr_t = torch.as_tensor(ytr_np, dtype=torch.long)
        emb = nn.Embedding(W.shape[0], W.shape[1]).to(device)
        emb.weight.data.copy_(W.to(device))              # init from NT token embedding
        clf = nn.Linear(NT_DIM, num_labels).to(device)
        opt = torch.optim.Adam(list(emb.parameters()) + list(clf.parameters()), lr=a.lr, weight_decay=a.weight_decay)
        lossf = nn.CrossEntropyLoss()

        @torch.no_grad()
        def evalmcc(ids, mask, y):
            emb.eval(); clf.eval(); preds = []
            for i in range(0, ids.shape[0], 512):
                e = emb(ids[i:i+512].to(device))
                pooled = masked_mean(e, mask[i:i+512].to(device))
                preds.append(clf(pooled).argmax(1).cpu().numpy())
            return matthews_corrcoef(y, np.concatenate(preds))

        best_val, best_test, best_ep, wait = -2.0, None, -1, 0
        N = ids_tr.shape[0]
        for ep in range(1, a.epochs + 1):
            emb.train(); clf.train()
            perm = torch.randperm(N)
            tot = 0.0
            for i in range(0, N, a.batch_size):
                idx = perm[i:i + a.batch_size]
                e = emb(ids_tr[idx].to(device))
                pooled = masked_mean(e, mask_tr[idx].to(device))
                loss = lossf(clf(pooled), ytr_t[idx].to(device))
                opt.zero_grad(); loss.backward(); opt.step()
                tot += loss.item()
            vmcc, tmcc = evalmcc(ids_va, mask_va, yva_np), evalmcc(ids_te, mask_te, yte_np)
            if vmcc > best_val:
                best_val, best_test, best_ep, wait = vmcc, tmcc, ep, 0
            else:
                wait += 1
            if ep == 1 or ep % 10 == 0:
                print(f"  epoch {ep}/{a.epochs} loss={tot/max(1,N//a.batch_size):.4f} val={vmcc:.4f} test={tmcc:.4f} best_val={best_val:.4f}", flush=True)
            if a.patience > 0 and wait >= a.patience:
                print(f"  early stop @ {ep}", flush=True); break
        trainable = sum(p.numel() for p in emb.parameters()) + sum(p.numel() for p in clf.parameters())
        frozen = 0

    print(f"[probe] DONE task={task} mode={method} best_val_mcc={best_val:.4f} best_test_mcc={best_test:.4f} "
          f"@ep{best_ep} trainable={trainable} frozen_emb={frozen}", flush=True)
    os.makedirs(os.path.dirname(a.results_csv) or ".", exist_ok=True)
    newf = not os.path.exists(a.results_csv)
    with open(a.results_csv, "a", newline="") as f:
        w = csv.writer(f)
        if newf:
            w.writerow(["task", "method", "embedding_source", "best_val_mcc", "best_test_mcc",
                        "trainable_params", "frozen_emb_params", "best_epoch", "seed", "timestamp"])
        w.writerow([task, method, "nt_base", f"{best_val:.6f}", f"{best_test:.6f}",
                    trainable, frozen, best_ep, a.seed, datetime.now().isoformat()])
    print(f"[probe] wrote {a.results_csv}", flush=True)


if __name__ == "__main__":
    main()
