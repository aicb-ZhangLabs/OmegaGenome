#!/usr/bin/env python
"""Linear classifier on NT-2.5B TOKEN embeddings (minimal-NT baselines).

Two modes, both: NT-2.5B word-embedding lookup ([vocab,2560], NO transformer layers) -> mean-pool over
the sequence -> nn.Linear(2560 -> n_classes). Trained from scratch (CE). Data splits/labels come from
the SAME loaders as the other embedding arms, so test MCC is directly comparable.

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
import math
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
    ids_list, Tmax = [], 0
    for i in range(0, len(seqs), 256):
        enc = tok(list(seqs[i : i + 256]), padding=False, truncation=True, max_length=max_length)
        for row in enc["input_ids"]:
            ids_list.append(row)
            Tmax = max(Tmax, len(row))
    N = len(ids_list)
    ids = torch.zeros(N, Tmax, dtype=torch.long)
    mask = torch.zeros(N, Tmax, dtype=torch.float32)
    pad_id = tok.pad_token_id or 0
    ids.fill_(pad_id)
    for r, row in enumerate(ids_list):
        ids[r, : len(row)] = torch.tensor(row, dtype=torch.long)
        mask[r, : len(row)] = 1.0
    return ids, mask


def masked_mean(emb, mask):
    """emb [B,T,H], mask [B,T] -> [B,H] mean over valid tokens."""
    summed = (emb * mask.unsqueeze(-1)).sum(dim=1)
    cnt = mask.sum(dim=1, keepdim=True).clamp(min=1.0)
    return summed / cnt


def pool_feature_cache(cache_dir, split, device, n_limit=None, chunk=512):
    """(e) Load cached FINETUNED-NT mid-layer per-token features and pool to one vector per sequence.

    <split>_mid_token_embeddings.npy is [N, T_tokens, 2560] fp16; <split>_mid_char_lens.npy is [N, T_tokens]
    giving each token's base-pair span (CLS=0, 6-mers=6, trailing 1s; padding tokens=0). We take the
    char_len-WEIGHTED token mean, which equals the per-base-pair mean of the mid-layer representation --
    the exact per-bp signal the BPNet arms consume -- and drops CLS/padding automatically (weight 0).
    Chunked over rows so the 13.7GB train file is never fully materialised. n_limit caps rows (smoke).
    """
    emb = np.load(os.path.join(cache_dir, f"{split}_mid_token_embeddings.npy"), mmap_mode="r")
    cl = np.load(os.path.join(cache_dir, f"{split}_mid_char_lens.npy"), mmap_mode="r")
    assert emb.shape[0] == cl.shape[0], (
        f"{split}: emb N={emb.shape[0]} != char_lens N={cl.shape[0]}"
    )
    N = emb.shape[0] if n_limit is None else min(n_limit, emb.shape[0])
    out = torch.empty(N, emb.shape[2], dtype=torch.float32)
    for i in range(0, N, chunk):
        j = min(i + chunk, N)
        e = torch.from_numpy(np.ascontiguousarray(emb[i:j])).float()  # [b,T,2560]
        w = torch.from_numpy(np.ascontiguousarray(cl[i:j])).float()  # [b,T]
        out[i:j] = (e * w.unsqueeze(-1)).sum(1) / w.sum(1, keepdim=True).clamp(min=1.0)
    return out.to(device)


class CompactTransformerClassifier(nn.Module):
    """Compact NT-style transformer head on the FROZEN NT token embedding (arm (d), ~0.12M params).

    Faithful to how NT's own layers work (multi-head QKV self-attention + FFN over the 6-mer TOKEN
    sequence), but sized to the same ~0.12M budget as the one-hot / (c) BPNet students so it is a fair
    param-matched comparison. The 2560-dim frozen token embedding is the (free) input; every parameter
    here is trainable from scratch:
        Linear(2560 -> d_model) -> +sinusoidal pos -> N x TransformerEncoderLayer(d_model, nhead, ff)
        -> masked mean-pool over valid tokens -> Linear(d_model -> n_classes).
    At d_model=32, nhead=4, ff=128, layers=4 this is ~132.8k params (proj 81,952 + 4x12,704 + head),
    matching the (c) BPNet student (131,338) within ~1%.

    Arm (f) "1-layer NT" reuses this class at FULL NT width (d_model=in_dim=2560, nhead=20, ff=10240,
    layers=1, ~78.7M): when d_model==in_dim the token embedding feeds the transformer DIRECTLY (no
    projection, like a real NT layer) with an input LayerNorm for training stability, and no sqrt scaling.
    """

    def __init__(
        self, in_dim, n_classes, d_model=32, nhead=4, ff=128, layers=4, dropout=0.1, max_len=1024
    ):
        super().__init__()
        if d_model == in_dim:
            # full-NT-width (f): feed the token embedding straight into the layer (no proj), LN for stability
            self.proj, self.in_ln, self.scale = nn.Identity(), nn.LayerNorm(d_model), 1.0
        else:
            # compact (d): project in_dim -> d_model and scale by sqrt(d_model)
            self.proj, self.in_ln, self.scale = (
                nn.Linear(in_dim, d_model),
                nn.Identity(),
                math.sqrt(d_model),
            )
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("posenc", pe, persistent=False)  # [max_len, d_model], 0 params
        enc = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward=ff, dropout=dropout, batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(enc, num_layers=layers)
        self.head = nn.Linear(d_model, n_classes)

    def forward(self, emb, mask):
        """emb [B,T,in_dim] (frozen token embedding), mask [B,T] float (1=valid) -> logits [B,n_classes]."""
        T = emb.shape[1]
        x = self.in_ln(self.proj(emb)) * self.scale + self.posenc[:T].unsqueeze(0)
        key_pad = mask == 0  # [B,T] True where pad -> ignored by attn
        x = self.encoder(x, src_key_padding_mask=key_pad)
        return self.head(masked_mean(x, mask))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task-name", required=True)
    ap.add_argument("--results-csv", required=True)
    ap.add_argument(
        "--wordemb",
        default=None,
        help="path to the [vocab,2560] NT word-embedding tensor (.pt); required unless --feature-cache",
    )
    ap.add_argument(
        "--feature-cache",
        default=None,
        help="(e) per-task dir of cached FINETUNED-NT mid-layer per-token features "
        "(<split>_mid_token_embeddings.npy + <split>_mid_char_lens.npy). When set, the probe "
        "trains a linear head on the char_len-weighted (=per-bp) mean of these frozen features "
        "instead of the base word embedding -- no 2.5B load.",
    )
    ap.add_argument(
        "--trainable-emb",
        action="store_true",
        help="fine-tune the NT token embedding (init from NT) end-to-end",
    )
    ap.add_argument(
        "--head",
        choices=["linear", "transformer"],
        default="linear",
        help="'linear' = mean-pool + Linear (a/a'); 'transformer' = frozen tok-emb + compact NT-style transformer (d)",
    )
    ap.add_argument("--d-model", type=int, default=32, help="transformer head width (arm d)")
    ap.add_argument("--nhead", type=int, default=4, help="transformer head attention heads (arm d)")
    ap.add_argument("--ff", type=int, default=128, help="transformer head FFN dim (arm d)")
    ap.add_argument("--layers", type=int, default=4, help="transformer head depth (arm d)")
    ap.add_argument("--dropout", type=float, default=0.1, help="transformer head dropout (arm d)")
    ap.add_argument(
        "--max-length", type=int, default=1002, help="tokenizer max tokens (NT max_pos=1002)"
    )
    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--patience", type=int, default=50)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="mini-batch size (trainable-emb / transformer modes)",
    )
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    task = a.task_name
    num_labels = get_num_labels(task)
    if a.feature_cache:
        method = "nt_finetuned_feat_linear"
    elif a.head == "transformer":
        method = "nt_tokenemb_transformer"
    else:
        method = "nt_tokenemb_linear_trainable" if a.trainable_emb else "nt_tokenemb_linear_frozen"
    print(f"[probe] task={task} num_labels={num_labels} device={device} mode={method}", flush=True)

    ds = DatasetConfig(task_name=task, data_path="", random_state=a.seed)
    Xtr, ytr, Xva, yva, Xte, yte = build_data_splits_from_huggingface(ds)
    if a.smoke:
        Xtr, ytr, Xva, yva, Xte, yte = (
            Xtr[:256],
            ytr[:256],
            Xva[:128],
            yva[:128],
            Xte[:128],
            yte[:128],
        )
        a.epochs = min(a.epochs, 40)

    # (e) feature-cache mode needs neither the word-embedding table nor the tokenizer (features precomputed).
    if a.feature_cache:
        W, tok = None, None
    else:
        assert a.wordemb, "--wordemb is required unless --feature-cache is set"
        W = torch.load(a.wordemb, map_location="cpu").float()  # [vocab, 2560]
        tok = AutoTokenizer.from_pretrained(NT_BASE, trust_remote_code=True)
    ytr_np, yva_np, yte_np = np.asarray(ytr), np.asarray(yva), np.asarray(yte)
    print(
        f"[probe] {'feature-cache ' + a.feature_cache if a.feature_cache else 'word-emb ' + str(tuple(W.shape))} "
        f"| train/val/test={len(Xtr)}/{len(Xva)}/{len(Xte)}",
        flush=True,
    )

    if a.head == "transformer":
        # -------- (d) FROZEN token-emb + compact NT-style transformer (~0.12M), from scratch --------
        Wd = W.to(device)  # [vocab,2560] frozen (not a Parameter)
        ids_tr, mask_tr = tokenize_all(Xtr, tok, a.max_length)
        ids_va, mask_va = tokenize_all(Xva, tok, a.max_length)
        ids_te, mask_te = tokenize_all(Xte, tok, a.max_length)
        ytr_t = torch.as_tensor(ytr_np, dtype=torch.long)
        model = CompactTransformerClassifier(
            NT_DIM,
            num_labels,
            a.d_model,
            a.nhead,
            a.ff,
            a.layers,
            a.dropout,
            max_len=a.max_length + 2,
        ).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
        lossf = nn.CrossEntropyLoss()

        @torch.no_grad()
        def evalmcc(ids, mask, y):
            model.eval()
            preds = []
            for i in range(0, ids.shape[0], 256):
                e = Wd[ids[i : i + 256].to(device)]  # frozen lookup [b,T,2560]
                preds.append(model(e, mask[i : i + 256].to(device)).argmax(1).cpu().numpy())
            return matthews_corrcoef(y, np.concatenate(preds))

        best_val, best_test, best_ep, wait = -2.0, None, -1, 0
        N = ids_tr.shape[0]
        for ep in range(1, a.epochs + 1):
            model.train()
            perm = torch.randperm(N)
            tot = 0.0
            for i in range(0, N, a.batch_size):
                idx = perm[i : i + a.batch_size]
                e = Wd[ids_tr[idx].to(device)]  # frozen lookup, no grad to embedding
                loss = lossf(model(e, mask_tr[idx].to(device)), ytr_t[idx].to(device))
                opt.zero_grad()
                loss.backward()
                opt.step()
                tot += loss.item()
            vmcc, tmcc = evalmcc(ids_va, mask_va, yva_np), evalmcc(ids_te, mask_te, yte_np)
            if vmcc > best_val:
                best_val, best_test, best_ep, wait = vmcc, tmcc, ep, 0
            else:
                wait += 1
            if ep == 1 or ep % 10 == 0:
                print(
                    f"  epoch {ep}/{a.epochs} loss={tot / max(1, N // a.batch_size):.4f} val={vmcc:.4f} "
                    f"test={tmcc:.4f} best_val={best_val:.4f}",
                    flush=True,
                )
            if a.patience > 0 and wait >= a.patience:
                print(f"  early stop @ {ep}", flush=True)
                break
        trainable = sum(p.numel() for p in model.parameters())
        frozen = W.numel()
        print(
            f"[probe] transformer head params={trainable} (target ~131k, matches (c) BPNet)",
            flush=True,
        )
    elif not a.trainable_emb:
        # -------- FROZEN: precompute pooled features once, full-batch linear --------
        if a.feature_cache:
            # (e) frozen FINETUNED-NT mid-layer features -> char_len-weighted (=per-bp) mean-pool -> linear.
            # Cache row order matches build_data_splits (same pipeline that wrote the verified KD cache);
            # smoke truncates labels to 256, so pool only the first len(y) rows to stay aligned.
            Ftr = pool_feature_cache(a.feature_cache, "train", device, n_limit=len(ytr_np))
            Fva = pool_feature_cache(a.feature_cache, "val", device, n_limit=len(yva_np))
            Fte = pool_feature_cache(a.feature_cache, "test", device, n_limit=len(yte_np))
            for nm, F, y in (("train", Ftr, ytr_np), ("val", Fva, yva_np), ("test", Fte, yte_np)):
                assert F.shape[0] == len(y), f"cache {nm} pooled N={F.shape[0]} != labels {len(y)}"
            print(
                f"[probe] pooled finetuned-NT mid features: train{tuple(Ftr.shape)} "
                f"val{tuple(Fva.shape)} test{tuple(Fte.shape)}",
                flush=True,
            )
            frozen_ct = 0  # features precomputed upstream
        else:
            Wd = W.to(device)

            @torch.no_grad()
            def feats(seqs):
                out = []
                for i in range(0, len(seqs), 128):
                    enc = tok(
                        list(seqs[i : i + 128]),
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=a.max_length,
                    )
                    ids = enc["input_ids"].to(device)
                    mask = enc["attention_mask"].to(device).float()
                    out.append(masked_mean(Wd[ids], mask).cpu())
                return torch.cat(out, 0)

            Ftr, Fva, Fte = feats(Xtr).to(device), feats(Xva).to(device), feats(Xte).to(device)
            frozen_ct = W.numel()
        ytr_t = torch.as_tensor(ytr_np, dtype=torch.long, device=device)
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
                vp = clf(Fva).argmax(1).cpu().numpy()
                tp = clf(Fte).argmax(1).cpu().numpy()
            vmcc, tmcc = matthews_corrcoef(yva_np, vp), matthews_corrcoef(yte_np, tp)
            if vmcc > best_val:
                best_val, best_test, best_ep, wait = vmcc, tmcc, ep, 0
            else:
                wait += 1
            if ep == 1 or ep % 25 == 0:
                print(
                    f"  epoch {ep}/{a.epochs} loss={loss.item():.4f} val={vmcc:.4f} test={tmcc:.4f} best_val={best_val:.4f}",
                    flush=True,
                )
            if a.patience > 0 and wait >= a.patience:
                print(f"  early stop @ {ep}", flush=True)
                break
        trainable = sum(p.numel() for p in clf.parameters())
        frozen = frozen_ct
    else:
        # -------- TRAINABLE: nn.Embedding init NT, fine-tuned end-to-end, mini-batched --------
        ids_tr, mask_tr = tokenize_all(Xtr, tok, a.max_length)
        ids_va, mask_va = tokenize_all(Xva, tok, a.max_length)
        ids_te, mask_te = tokenize_all(Xte, tok, a.max_length)
        ytr_t = torch.as_tensor(ytr_np, dtype=torch.long)
        emb = nn.Embedding(W.shape[0], W.shape[1]).to(device)
        emb.weight.data.copy_(W.to(device))  # init from NT token embedding
        clf = nn.Linear(NT_DIM, num_labels).to(device)
        opt = torch.optim.Adam(
            list(emb.parameters()) + list(clf.parameters()), lr=a.lr, weight_decay=a.weight_decay
        )
        lossf = nn.CrossEntropyLoss()

        @torch.no_grad()
        def evalmcc(ids, mask, y):
            emb.eval()
            clf.eval()
            preds = []
            for i in range(0, ids.shape[0], 512):
                e = emb(ids[i : i + 512].to(device))
                pooled = masked_mean(e, mask[i : i + 512].to(device))
                preds.append(clf(pooled).argmax(1).cpu().numpy())
            return matthews_corrcoef(y, np.concatenate(preds))

        best_val, best_test, best_ep, wait = -2.0, None, -1, 0
        N = ids_tr.shape[0]
        for ep in range(1, a.epochs + 1):
            emb.train()
            clf.train()
            perm = torch.randperm(N)
            tot = 0.0
            for i in range(0, N, a.batch_size):
                idx = perm[i : i + a.batch_size]
                e = emb(ids_tr[idx].to(device))
                pooled = masked_mean(e, mask_tr[idx].to(device))
                loss = lossf(clf(pooled), ytr_t[idx].to(device))
                opt.zero_grad()
                loss.backward()
                opt.step()
                tot += loss.item()
            vmcc, tmcc = evalmcc(ids_va, mask_va, yva_np), evalmcc(ids_te, mask_te, yte_np)
            if vmcc > best_val:
                best_val, best_test, best_ep, wait = vmcc, tmcc, ep, 0
            else:
                wait += 1
            if ep == 1 or ep % 10 == 0:
                print(
                    f"  epoch {ep}/{a.epochs} loss={tot / max(1, N // a.batch_size):.4f} val={vmcc:.4f} test={tmcc:.4f} best_val={best_val:.4f}",
                    flush=True,
                )
            if a.patience > 0 and wait >= a.patience:
                print(f"  early stop @ {ep}", flush=True)
                break
        trainable = sum(p.numel() for p in emb.parameters()) + sum(
            p.numel() for p in clf.parameters()
        )
        frozen = 0

    print(
        f"[probe] DONE task={task} mode={method} best_val_mcc={best_val:.4f} best_test_mcc={best_test:.4f} "
        f"@ep{best_ep} trainable={trainable} frozen_emb={frozen}",
        flush=True,
    )
    os.makedirs(os.path.dirname(a.results_csv) or ".", exist_ok=True)
    newf = not os.path.exists(a.results_csv)
    with open(a.results_csv, "a", newline="") as f:
        w = csv.writer(f)
        if newf:
            w.writerow(
                [
                    "task",
                    "method",
                    "embedding_source",
                    "best_val_mcc",
                    "best_test_mcc",
                    "trainable_params",
                    "frozen_emb_params",
                    "best_epoch",
                    "seed",
                    "timestamp",
                ]
            )
        w.writerow(
            [
                task,
                method,
                "nt_base",
                f"{best_val:.6f}",
                f"{best_test:.6f}",
                trainable,
                frozen,
                best_ep,
                a.seed,
                datetime.now().isoformat(),
            ]
        )
    print(f"[probe] wrote {a.results_csv}", flush=True)


if __name__ == "__main__":
    main()
