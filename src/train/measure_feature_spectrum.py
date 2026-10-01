"""Measure 650M teacher feature compressibility BEFORE building any feature cache, so we know EMPIRICALLY how
much info PCA-channel-compression + position-pooling lose vs full-resolution (instead of guessing). Decides the
cache's D' (channel rank) and pooling factor with a variance-retention guarantee.

Outputs (stdout + --out json):
  1. PCA cumulative explained variance vs D' (channel compressibility) -> D' needed for 90/95/99% retention.
  2. relative reconstruction error at D'={16..512}: PCA vs random-projection (confirms PCA >> JL at equal size).
  3. positional cosine autocorrelation at lags {1,2,4,8,16,32} (whether position-pooling is safe / how lossy).
  4. logit-subspace residual: fraction of feature variance ORTHOGONAL to the 34 track-logit read-out directions
     (= the incremental info feature-KD would add beyond the already-cached logit distillation).

Reuses build_cache_dataset + load_finetuned_bigwig_teacher (the SAME window construction the real cache uses),
so the measured spectrum is exactly the distribution a cache would store.

  python -m src.train.measure_feature_spectrum --data_dir <bench> --teacher <650M ckpt> \
      --tokenizer <8m_pre dir> --n_windows 16 --sample_positions 80000 --amp --out <results.json>
"""

import argparse
import json

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.ntv3_ft_data import load_benchmark_frames
from src.model.ntv3_finetune import load_finetuned_bigwig_teacher
from src.trainer.teacher_cache import build_cache_dataset


def pca_eigenvalues(X: np.ndarray) -> np.ndarray:
    """Per-component variance (descending) of centered ``X`` [M, D], via the DxD covariance eigendecomposition
    (memory-stable: DxD, never MxM). Eigenvalues = variance captured by each principal direction."""
    Xc = X - X.mean(axis=0, keepdims=True)
    cov = (Xc.T @ Xc) / (Xc.shape[0] - 1)  # [D, D]
    eig = np.linalg.eigvalsh(cov)  # ascending
    return np.clip(eig[::-1], 0.0, None)  # descending, non-negative


def pca_recon_error(eig: np.ndarray, d: int) -> float:
    """Relative Frobenius reconstruction error of keeping the top-``d`` PCA components = sqrt(dropped/total).
    This IS the fractional information loss of a rank-``d`` PCA channel compression."""
    if d >= len(eig):
        return 0.0
    return float(np.sqrt(eig[d:].sum() / (eig.sum() + 1e-12)))


def randproj_recon_error(X: np.ndarray, d: int, seed: int = 0) -> float:
    """Relative error of a Gaussian random projection D->d followed by the best linear reconstruction (the JL
    baseline PCA is compared against). Uses a position subsample to keep the lstsq tractable."""
    Xc = X - X.mean(axis=0, keepdims=True)
    rng = np.random.default_rng(seed)
    sub = Xc[rng.choice(Xc.shape[0], size=min(20000, Xc.shape[0]), replace=False)]
    R = rng.standard_normal((sub.shape[1], d)).astype(np.float32) / np.sqrt(d)
    Z = sub @ R  # [m, d]
    W, *_ = np.linalg.lstsq(Z, sub, rcond=None)  # best Z->X map
    return float(np.linalg.norm(sub - Z @ W) / (np.linalg.norm(sub) + 1e-12))


def positional_autocorr(feat_LD: np.ndarray, lags) -> dict:
    """Mean cosine similarity between feature vectors at positions i and i+lag, for a single window [L, D].
    High at lag=16 => 16x pooling is near-lossless; low => pooling destroys real signal."""
    f = feat_LD / (np.linalg.norm(feat_LD, axis=1, keepdims=True) + 1e-8)
    out = {}
    for k in lags:
        if k < f.shape[0]:
            out[str(k)] = float((f[:-k] * f[k:]).sum(axis=1).mean())
    return out


def logit_subspace_residual(X: np.ndarray, head_weight: np.ndarray) -> float:
    """Fraction of feature variance ORTHOGONAL to the row-space of the 34x1536 track head (the directions
    already captured by the cached logits). 1 - this = redundant with logit-KD; this = feature-KD's extra info.
    Approximation: ignores the head's LayerNorm (documented)."""
    Xc = X - X.mean(axis=0, keepdims=True)
    q, _ = np.linalg.qr(head_weight.T)  # [D, r] orthonormal basis of the logit read-out
    proj = Xc @ q  # variance inside the logit subspace
    var_in = (proj**2).sum()
    var_tot = (Xc**2).sum()
    return float(1.0 - var_in / (var_tot + 1e-12))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--teacher_base", default="InstaDeepAI/NTv3_650M_post")
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--species", default="human")
    ap.add_argument("--sequence_length", type=int, default=32768)
    ap.add_argument(
        "--n_windows", type=int, default=16, help="teacher windows to forward (features are big)"
    )
    ap.add_argument(
        "--sample_positions", type=int, default=80000, help="positions subsampled for the PCA"
    )
    ap.add_argument("--amp", action="store_true")
    ap.add_argument(
        "--dry_run",
        action="store_true",
        help="skip the teacher; run the math on synthetic features",
    )
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dims = [8, 16, 32, 64, 128, 256, 512]
    lags = [1, 2, 4, 8, 16, 32]

    if args.dry_run:
        # synthetic low-rank features (rank ~50) + smooth positional structure -> exercises every code path
        rng = np.random.default_rng(0)
        D, L = 1536, 4096
        basis = rng.standard_normal((50, D)).astype(np.float32)
        coeffs = rng.standard_normal((args.n_windows * L, 50)).astype(np.float32) * np.linspace(
            5, 0.1, 50
        )
        feats = (coeffs @ basis).reshape(args.n_windows, L, D)
        head_w = rng.standard_normal((34, D)).astype(np.float32)
        print("[dry_run] synthetic features", feats.shape, "(injected rank ~50)")
    else:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
        fasta, bw_paths, bw_ids, regions, _, _ = load_benchmark_frames(args.data_dir, args.species)
        teacher = load_finetuned_bigwig_teacher(
            args.teacher, args.teacher_base, len(bw_ids), device=device
        )
        ds = build_cache_dataset(
            fasta, bw_paths, regions["train"], args.sequence_length, tok, args.n_windows
        )
        loader = DataLoader(ds, batch_size=2, shuffle=False, num_workers=4)
        teacher.eval()
        chunks = []
        for batch in loader:
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.amp):
                out = teacher(batch["tokens"].to(device))
            chunks.append(out["features"].float().cpu().numpy())
            if sum(c.shape[0] for c in chunks) >= args.n_windows:
                break
        feats = np.concatenate(chunks, axis=0)[: args.n_windows]  # [N, L, D]
        head_w = teacher.bigwig_head.head.weight.detach().float().cpu().numpy()  # [34, D]
        del teacher
        torch.cuda.empty_cache()
        print(f"teacher features {feats.shape} | head {head_w.shape}", flush=True)

    N, L, D = feats.shape
    flat = feats.reshape(N * L, D)
    rng = np.random.default_rng(0)
    idx = rng.choice(flat.shape[0], size=min(args.sample_positions, flat.shape[0]), replace=False)
    Xs = flat[idx]  # [M, D] position subsample

    eig = pca_eigenvalues(Xs)
    cum = np.cumsum(eig) / (eig.sum() + 1e-12)
    d_for = {f"{int(p * 100)}%": int(np.searchsorted(cum, p) + 1) for p in (0.90, 0.95, 0.99)}
    pca_err = {str(d): round(pca_recon_error(eig, d), 4) for d in dims}
    rp_err = {str(d): round(randproj_recon_error(Xs, d), 4) for d in dims}
    autoc = positional_autocorr(feats[0], lags)
    resid = logit_subspace_residual(Xs, head_w)

    results = {
        "n_windows": N,
        "L_out": L,
        "embed_dim": D,
        "sampled_positions": len(idx),
        "pca_dims_for_variance": d_for,
        "pca_recon_error": pca_err,  # fractional info loss at each D' (channels)
        "randproj_recon_error": rp_err,  # JL baseline at each D'
        "positional_cosine_autocorr": autoc,  # >0.9 at lag16 => 16x pooling safe
        "logit_subspace_residual_var_frac": round(
            resid, 4
        ),  # feature-KD's extra info beyond logits
    }
    print(json.dumps(results, indent=2), flush=True)
    print("\nINTERPRET:", flush=True)
    print(
        f"  channels: D'=64 loses {pca_err['64'] * 100:.1f}% (PCA) vs {rp_err['64'] * 100:.1f}% (random-proj); "
        f"99% variance needs D'={d_for['99%']}.",
        flush=True,
    )
    print(
        f"  positions: cosine(i,i+16)={autoc.get('16', 'n/a')} -> "
        f"{'16x pooling SAFE' if autoc.get('16', 0) > 0.9 else 'pooling LOSSY, keep resolution'}.",
        flush=True,
    )
    print(f"  feature-KD adds {resid * 100:.1f}% variance beyond the cached logits.", flush=True)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
