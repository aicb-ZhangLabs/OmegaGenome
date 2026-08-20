"""Ablation: distillation loss x student architecture on the cached 2000-window 100M targets.

Answers "would a different loss / bigger student help the multi-track distillation Pearson?"
empirically, reusing cached targets (no re-gen). Eval = per-track Pearson vs the RAW teacher
(correlation is affine-invariant, so z-norm/raw train targets are comparable; poisson predicts a
log-rate so we exponentiate before correlating).
"""
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.track_dataset import TrackDataset
from src.model.bpnet_regressor import BPNetRegressor, BPNetRegressorConfig
from src.trainer.track_distill import _teacher_to_btl, align_student_to_teacher, track_distill_loss
from src.trainer.track_metrics import per_track_pearson

D = "/tmp/galaxy_srv_disk00/pengchx3/ntv3_targets/v1_100m"
N_TR, N_TE, EPOCHS = 800, 300, 30
dev = "cuda" if torch.cuda.is_available() else "cpu"
tr = torch.load(f"{D}/train.pt", weights_only=False)
te = torch.load(f"{D}/test.pt", weights_only=False)
W = tr["window"]
tr_seq, tr_raw = tr["sequences"][:N_TR], tr["targets"][:N_TR]
te_seq, te_raw = te["sequences"][:N_TE], te["targets"][:N_TE]
mu, sd = tr_raw.mean((0, 1), keepdim=True), tr_raw.std((0, 1), keepdim=True).clamp_min(1e-6)


def make_ds(seq, raw, norm):
    tgt = (raw - mu) / sd if norm == "z" else raw
    return TrackDataset(seq, tgt, max_len=W)


@torch.no_grad()
def evaluate(model, ds, kind):
    model.eval()
    P, G = [], []
    for ids, _ in DataLoader(ds, batch_size=16):
        out = model(ids.to(dev))                       # [B,T,Ls] (log-rate if poisson)
        if kind == "poisson":
            out = out.exp()
        P.append(align_student_to_teacher(out, te_rawb.shape[-1]).cpu().numpy())
    pred = np.concatenate(P)
    return per_track_pearson(pred, np.concatenate(G_raw))[0]


# raw teacher aligned, for eval target (same for every config)
te_rawb = _teacher_to_btl(te_raw)                       # [N,T,Lt]
G_raw = [te_rawb.numpy()]

CONFIGS = [
    ("mse",         "medium", "z"),
    ("pearson",     "medium", "z"),
    ("mse+pearson", "medium", "z"),
    ("mse",         "large",  "z"),
    ("pearson",     "large",  "z"),
    ("poisson",     "medium", "raw"),
]
print(f"device={dev}  train={N_TR} test={N_TE} epochs={EPOCHS}\n")
print(f"{'loss':14s} {'arch':8s} {'norm':4s} {'best_test_pearson':>17s}  {'params':>9s}")
for kind, size, norm in CONFIGS:
    dtr, dte = make_ds(tr_seq, tr_raw, norm), make_ds(te_seq, te_raw, norm)
    m = BPNetRegressor(BPNetRegressorConfig(num_tracks=9, model_size=size)).to(dev)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    ld = DataLoader(dtr, batch_size=8, shuffle=True)
    best = -1.0
    for ep in range(EPOCHS):
        m.train()
        for ids, t in ld:
            opt.zero_grad()
            track_distill_loss(m(ids.to(dev)), t.to(dev), kind).backward()
            opt.step()
        r = evaluate(m, dte, kind)
        best = max(best, r)
    n = sum(p.numel() for p in m.parameters())
    print(f"{kind:14s} {size:8s} {norm:4s} {best:17.4f}  {n/1e6:8.3f}M", flush=True)
print("\n(baseline mse/medium/z should reproduce ~0.17 from job 235979)")
