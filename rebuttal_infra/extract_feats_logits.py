#!/usr/bin/env python
"""GPU extraction for the PI-faithful R1.9/R1.13 viz: teacher + OmegaGenome + DKD + FROM-SCRATCH
baseline penultimate FEATURES *and* class LOGITS on the same splice_sites_all test seqs.

Reuses the loaders/forward helpers from feature_space_compare.py (same recipe, same seqs/seed 42).
Adds the from-scratch BPNet baseline (no teacher_proj) and caches logits so feature_viz.py can do
both FEATURE-space and LOGIT-space comparisons (the PI script distinguishes the two).

SLURM only (GPU). Smoke with --n 8.
"""
import argparse
import os
import sys

import numpy as np
import torch

REPO = "/home/pengchx3/text-dna/OmegaGenome_Revise_202606/code_carbon"
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "rebuttal_infra"))

from feature_space_compare import (  # noqa: E402  reuse the PI-recipe loaders
    load_student, load_splice_test_seqs, TEACHER_CKPT, OMEGA_CKPT, DKD_CKPT,
    NUM_LABELS, MAX_LEN,
)
from src.model.bpnet_classifier import BPNetClassifier, BPNetClassifierConfig  # noqa: E402
from src.model.enformer import load_enformer_model, EnformerTokenizer  # noqa: E402
from src.data.dataset import encode_seq  # noqa: E402

FROM_SCRATCH_CKPT = (
    "/extra/zhanglab0/INDV/pengchx3/NT/nucleotide-transformer-student-results-3-11-v4-"
    "cnn-layer16-kernel-5-bpnet-4-21-onehot-laniakea/splice_sites_all/model_final.pt"
)


def load_scratch_bpnet(ckpt_path, device):
    """Load the from-scratch (no-distillation) BPNet baseline. Same `original` BPNet as the
    students but trained WITHOUT a teacher -> NO teacher_proj in the state_dict. `bpnet.` prefix
    remaps to `backbone.`; profile/total_count heads stay uninitialised (features from trunk only)."""
    cfg = BPNetClassifierConfig(num_labels=NUM_LABELS, model_size="original")  # no teacher_*
    model = BPNetClassifier(cfg)
    sd = torch.load(ckpt_path, map_location="cpu")
    sd = {(k.replace("bpnet.", "backbone.", 1) if k.startswith("bpnet.") else k): v
          for k, v in sd.items()}
    res = model.load_state_dict(sd, strict=False)
    leftover = [k for k in res.missing_keys if "profile" not in k and "total_count" not in k
                and "teacher_proj" not in k]
    assert not leftover, f"unexpected missing keys: {leftover}"
    return model.to(device).eval()


@torch.no_grad()
def student_feats_logits(model, seqs, device, bs=32):
    """(logits[N,3], pooled feats[N,64]) for a BPNet student/baseline."""
    fo, lo = [], []
    for i in range(0, len(seqs), bs):
        ids = torch.tensor([encode_seq(s, MAX_LEN) for s in seqs[i:i + bs]],
                           dtype=torch.long, device=device)
        logits, feats = model(ids, return_feats=True)
        lo.append(logits.float().cpu().numpy())
        fo.append(feats.float().cpu().numpy())
    return np.concatenate(lo, 0), np.concatenate(fo, 0)


@torch.no_grad()
def teacher_feats_logits(model, tok, seqs, device, bs=8):
    """(logits[N,3], SECOND_TO_LAST feats[N,3072]) for the Enformer teacher.

    PAPER-FAITHFUL FIX (matches umap-tsne-260114_panel_v13_260204.py, the script that produced
    fig:embedding-panel4 / Combined_KD_splice_sites_all_distinct5_hist_260204): the panel's cosine /
    t-SNE use the ``second_to_last`` feature, defined for Enformer as the mean over the LAST QUARTER of
    sequence positions of the trunk embeddings -- ``embeddings[:, -L//4:, :].mean(dim=1)`` -- NOT the
    full-sequence pooled mean. ``return_multi_features=True`` exposes raw embeddings [B,L,D] (its first
    element); we slice the last quarter and mean over positions. (The full-mean ``return_features`` path
    gave teacher-student cosine ~0.055 even at full sample; the last-quarter feature reproduces ~0.411.)
    BPNet students' ``second_to_last`` == their pooled feature (both AdaptiveAvgPool over the stem
    output), so the student/baseline extraction is unchanged."""
    fo, fo_last, lo = [], [], []
    for i in range(0, len(seqs), bs):
        enc = tok(seqs[i:i + bs], padding="max_length", truncation=True, max_length=MAX_LEN)
        ids = enc["input_ids"].to(device)
        mask = enc["attention_mask"].to(device)
        logits, multi = model(input_ids=ids, attention_mask=mask, return_multi_features=True)
        embeddings = multi[0]  # [B, L, D] raw trunk embeddings
        L = embeddings.shape[1]
        second_to_last = embeddings[:, -L // 4:, :].mean(dim=1)  # [B, D] last-quarter-positions mean
        # MSE-aligned analog: full-sequence mean of the trunk embedding -- this is exactly the feature
        # the Enformer distillation MSE term aligns the student to (precompute_teacher_logits ->
        # EnformerFeatureExtractor.return_features == hidden_states[-1].mean(dim=1)). For the CNN-trunk
        # Enformer there is no transformer [-1]/[-2]; the "last/MSE-aligned" panel is the full pooled mean
        # vs the paper-method last-quarter mean (second_to_last).
        last = embeddings.mean(dim=1)  # [B, D] full-sequence mean (MSE target)
        lo.append(logits.float().cpu().numpy())
        fo.append(second_to_last.float().cpu().numpy())
        fo_last.append(last.float().cpu().numpy())
    return np.concatenate(lo, 0), np.concatenate(fo, 0), np.concatenate(fo_last, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--out", default=os.path.join(REPO, "rebuttal_infra/figs"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}", flush=True)

    seqs, labs = load_splice_test_seqs(args.n, os.environ.get("HF_HOME", ""))
    print(f"loaded {len(seqs)} splice_sites_all test seqs", flush=True)

    print("loading students + from-scratch baseline...", flush=True)
    omega = load_student(OMEGA_CKPT, device)
    dkd = load_student(DKD_CKPT, device)
    scratch = load_scratch_bpnet(FROM_SCRATCH_CKPT, device)

    print("extracting student/baseline feats+logits...", flush=True)
    L_omega, F_omega = student_feats_logits(omega, seqs, device)
    L_dkd, F_dkd = student_feats_logits(dkd, seqs, device)
    L_base, F_baseline = student_feats_logits(scratch, seqs, device)

    print("loading enformer teacher (1GB, may stall on NFS)...", flush=True)
    wrapped, _, _ = load_enformer_model(TEACHER_CKPT, NUM_LABELS, device)
    tok = EnformerTokenizer()
    print("extracting teacher feats+logits...", flush=True)
    L_teacher, F_teacher, F_teacher_last = teacher_feats_logits(wrapped, tok, seqs, device)
    print(f"feat shapes: teacher={F_teacher.shape} teacher_last={F_teacher_last.shape} "
          f"omega={F_omega.shape} dkd={F_dkd.shape} base={F_baseline.shape}", flush=True)

    out = os.path.join(args.out, "feature_logit_data.npz")
    np.savez_compressed(
        out,
        F_teacher=F_teacher, F_teacher_last=F_teacher_last,
        F_omega=F_omega, F_dkd=F_dkd, F_baseline=F_baseline,
        L_teacher=L_teacher, L_omega=L_omega, L_dkd=L_dkd, L_base=L_base,
        labels=np.array(labs),
    )
    print(f"SAVED {out}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
