"""Build the GENERALIST teacher FEATURE cache (one epoch over all train windows) for fast feature-alignment KD.

The teacher backbone hidden state is SHARED across all 34 tracks, so this single cache serves every track's
feature-alignment term — no per-track feature files. Stored in the separate `feat/` namespace, so it never
conflicts with the per-track LOGIT caches at `teacher_logits/{gen,spec}/t{i}.pt`. Compression (per the
feasibility analysis): PCA-project the low-rank 1536-dim channels to d', avg-pool positions by `pool`, fp16 —
keeps the cache RAM-loadable while preserving the alignment-relevant information.

  python -m src.train.cache_teacher_features --data_dir <bench> --teacher <650M ckpt> \
      --tokenizer <8m_pre> --out_dir <ntv3_cache/teacher_feat> --d_prime 128 --pool 8 --amp
"""

import argparse

import torch

from src.data.ntv3_ft_data import load_benchmark_frames
from src.model.ntv3_finetune import load_finetuned_bigwig_teacher
from src.trainer.teacher_cache import (
    build_cache_dataset,
    cache_teacher_features,
    feature_cache_dir,
    window_coords,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--teacher_base", default="InstaDeepAI/NTv3_650M_post")
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--species", default="human")
    ap.add_argument("--sequence_length", type=int, default=32768)
    ap.add_argument(
        "--n_windows", type=int, default=64000, help="train windows to cache (caps at ~63707)"
    )
    ap.add_argument("--d_prime", type=int, default=128, help="PCA channel rank to keep")
    ap.add_argument(
        "--pool", type=int, default=8, help="position avg-pool factor (1 = full per-bp)"
    )
    ap.add_argument(
        "--n_fit_windows", type=int, default=16, help="windows used to fit the PCA basis"
    )
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument(
        "--out_dir",
        required=True,
        help="feature-cache ROOT; the actual dir is "
        "<root>/<source>/<method>_d<d'>_p<pool> (encodes source+compression, no conflicts)",
    )
    ap.add_argument(
        "--source", default="gen", help="teacher source tag: 'gen' (generalist) or 'spec<i>'"
    )
    ap.add_argument(
        "--method", default="svd", help="channel-compression method tag (svd; encodes the dir)"
    )
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # ENCODE source + compression into the path so generalist/specialist and different methods never collide.
    out_dir = feature_cache_dir(args.out_dir, args.source, args.method, args.d_prime, args.pool)
    print(f"feature-cache dir: {out_dir}", flush=True)

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    fasta, bw_paths, bw_ids, regions, _, _ = load_benchmark_frames(args.data_dir, args.species)
    # SAME window construction as the logit cache (overlap=0, limit=n) -> idx-aligned across both caches.
    ds = build_cache_dataset(
        fasta, bw_paths, regions["train"], args.sequence_length, tok, args.n_windows
    )
    coords = window_coords(ds)
    teacher = load_finetuned_bigwig_teacher(
        args.teacher, args.teacher_base, len(bw_ids), device=device
    )
    print(
        f"feature cache: {len(ds)} windows, teacher={args.teacher}, d'={args.d_prime}, pool={args.pool}",
        flush=True,
    )
    cache_teacher_features(
        teacher,
        ds,
        out_dir,
        args.d_prime,
        args.pool,
        args.n_fit_windows,
        args.sequence_length,
        coords,
        device,
        args.amp,
        args.batch_size,
        limit_num_samples=args.n_windows,
    )
    print("=== feature cache build done ===", flush=True)


if __name__ == "__main__":
    main()
