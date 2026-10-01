"""Audit a teacher-logit cache: rebuild the IDENTICAL window set + the teacher, run the teacher LIVE on
the first few windows, and assert cache[idx] == live teacher output. Proves the cache values are correct
(not just that the windows align). Also re-checks the coord fingerprint.

  python -m src.train.validate_teacher_cache --cache <cache.pt> --teacher <ckpt> --data_dir <bench> \
      --tokenizer <8m_pre> [--amp]
"""

import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.ntv3_ft_data import GenomeBigWigDataset, load_benchmark_frames, make_target_scaling_fn
from src.model.ntv3_finetune import NTV3_CROP_FRAC, load_finetuned_bigwig_teacher
from src.trainer.track_distill import resolve_kd_track_targets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--teacher_base", default="InstaDeepAI/NTv3_650M_post")
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--species", default="human")
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--n_check", type=int, default=16, help="windows to verify live")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    c = torch.load(args.cache, map_location="cpu")
    idx, N, limit = c["track_subset"], c["num_windows"], c["limit_num_samples"]
    cache_logits, coords = c["logits"], c["coords"]
    print(
        f"cache: track_subset={idx} N={N} limit={limit} overlap={c['overlap']} logits={tuple(cache_logits.shape)}"
    )

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    fasta, bw_paths, bw_ids, regions, means, assays = load_benchmark_frames(
        args.data_dir, args.species
    )
    native_n = len(bw_ids)
    sub_means = (
        means[idx]
        if hasattr(means, "__getitem__") and not isinstance(means, list)
        else [means[i] for i in idx]
    )
    # IDENTICAL reconstruction to the precompute + the cached trainer
    ds = GenomeBigWigDataset(
        fasta,
        [bw_paths[i] for i in idx],
        regions["train"],
        c["sequence_length"],
        tok,
        make_target_scaling_fn(sub_means),
        overlap=c["overlap"],
        keep_target_center_fraction=NTV3_CROP_FRAC,
        limit_num_samples=limit,
    )
    assert len(ds) == N, f"rebuilt {len(ds)} != cache N {N}"
    print(f"rebuilt dataset len {len(ds)} == cache N {N}  [OK]")

    tnt, kd_idx = resolve_kd_track_targets(idx, native_n, c["teacher_specialist"])
    kd_track_idx = torch.tensor(kd_idx, device=device) if kd_idx is not None else None
    teacher = load_finetuned_bigwig_teacher(args.teacher, args.teacher_base, tnt, device=device)

    loader = DataLoader(ds, batch_size=4, shuffle=False, num_workers=4)
    max_diff, checked = 0.0, 0
    for batch in loader:
        tokens = batch["tokens"].to(device)
        idxs = batch["idx"].numpy()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.amp):
            t = teacher(tokens)["bigwig_tracks_logits"]
        if kd_track_idx is not None:
            t = t.index_select(-1, kd_track_idx)
        t = t.float().cpu()
        for j, wi in enumerate(idxs):
            # coord fingerprint re-check
            ci = max(0, np.searchsorted(ds._cumulative_starts, wi, side="right") - 1)
            ri = ds.region_info[ci]
            start = ri["region_start_offset"] + (wi - ds._cumulative_starts[ci]) * ds.stride
            assert (ri["chr_name"], int(start)) == tuple(coords[wi]), f"coord mismatch @ {wi}"
            d = (t[j] - cache_logits[wi]).abs().max().item()
            max_diff = max(max_diff, d)
            checked += 1
        if checked >= args.n_check:
            break
    rel = max_diff / (cache_logits.abs().max().item() + 1e-8)
    print(
        f"checked {checked} windows | max|cache-live|={max_diff:.6f} (rel {rel:.4%}) | coords all match"
    )
    tol = 0.1 if args.amp else 1e-3  # bf16 amp matmuls vary slightly run-to-run; fp32 is ~exact
    assert max_diff < tol, f"CACHE MISMATCH: max diff {max_diff} >= tol {tol}"
    print(f"PASS: cache values match the live teacher within tol {tol} (amp={args.amp})")


if __name__ == "__main__":
    main()
