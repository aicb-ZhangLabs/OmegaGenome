"""Precompute a frozen teacher's per-bp track logits over a FIXED set of training windows, so a KD
student can train WITHOUT a per-step teacher forward (the dominant cost when the teacher is the 650M).

Correctness: windows are deterministic given (regions, seq_length, limit, seed=0) — the SAME construction
the cached KD run uses — so the cache, indexed by the dataset window index, aligns 1:1 with what the
student sees. We also store the window count + a coord fingerprint so the trainer can assert alignment.

Reuses: load_benchmark_frames / GenomeBigWigDataset (data), load_finetuned_bigwig_teacher (model),
resolve_kd_track_targets (the generalist index-select vs specialist-direct logic). Cache = logit-only
(features are too large to cache), so a cached KD run does gt + distill, no feature alignment.

  python -m src.train.precompute_teacher_logits --data_dir <bench> --teacher <ckpt> \
      --track_subset 12 --num_windows 8000 --tokenizer <8m_pre dir> --out <cache.pt>
"""

import argparse
import numpy as np
import torch

from src.data.ntv3_ft_data import load_benchmark_frames, validate_subset_bigwigs
from src.model.ntv3_finetune import load_finetuned_bigwig_teacher
from src.trainer.track_distill import resolve_kd_track_targets
from src.trainer.teacher_cache import (
    build_cache_dataset,
    compute_logits,
    save_per_track,
    window_coords,
)


def _parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--teacher", required=True, help="finetuned NTv3 bigWig teacher ckpt")
    ap.add_argument("--teacher_base", default="InstaDeepAI/NTv3_650M_post")
    ap.add_argument(
        "--teacher_specialist",
        action="store_true",
        help="teacher trained on the SAME --track_subset (1-track head; no index-select)",
    )
    ap.add_argument(
        "--track_subset",
        default=None,
        help="comma-sep dataset indices, e.g. '12'. OMIT for the JOINT all-track cache "
        "(all native tracks; stored track_subset=None so it matches a joint student).",
    )
    ap.add_argument("--num_windows", type=int, default=8000, help="fixed train windows to cache")
    ap.add_argument("--sequence_length", type=int, default=32768)
    ap.add_argument("--species", default="human")
    ap.add_argument("--tokenizer", required=True, help="any NTv3 tokenizer dir (e.g. 8m_pre)")
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument(
        "--amp", action="store_true", help="bf16 teacher forward (matches the --amp KD run)"
    )
    ap.add_argument(
        "--out", default=None, help="output cache .pt path (single combined file, RAM-loaded)"
    )
    ap.add_argument(
        "--memmap_out",
        default=None,
        help="DIR for a disk-streamed fp16 memmap cache (logits.npy + meta.pt) instead of a "
        "RAM-loaded .pt — for the JOINT 34-track cache that is too large (~53GB) to hold "
        "in RAM. Teacher-free student runs stream it via --cached_teacher_logits <dir>.",
    )
    ap.add_argument(
        "--save_per_track",
        action="store_true",
        help="from ONE teacher pass, save a separate single-track cache per subset track to "
        "--out_dir/t{idx}.pt (efficient 'cache generalist by track'; no file conflicts)",
    )
    ap.add_argument(
        "--out_dir", default=None, help="dir for per-track files (with --save_per_track)"
    )
    args = ap.parse_args()
    if not (args.out or args.memmap_out or (args.save_per_track and args.out_dir)):
        ap.error("need --out, --memmap_out, or --save_per_track with --out_dir")
    return args


def main():
    args = _parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)

    fasta, bw_paths, bw_ids, regions, track_means, track_assays = load_benchmark_frames(
        args.data_dir, args.species
    )
    native_n = len(bw_ids)
    # JOINT all-track cache (no --track_subset): use every native track and store track_subset=None so it
    # matches a joint student (idx=None). Otherwise the explicit subset. `stored_subset` is what goes on disk.
    if args.track_subset:
        idx = [int(x) for x in args.track_subset.split(",")]
        bad = [i for i in idx if i < 0 or i >= native_n]
        if bad:  # out-of-range index would mis-target the cache (cache i must mean dataset track i)
            raise ValueError(
                f"--track_subset {idx} out of range for {native_n} tracks (bad: {bad})"
            )
        stored_subset = idx
    else:
        idx, stored_subset = list(range(native_n)), None
    # RIGOR GUARD #2 (shared): never cache a 0-byte placeholder / corrupt bigWig as if it were real data —
    # the teacher only reads tokens here, but the bigWig must exist as real data for the later cached-KD run.
    validate_subset_bigwigs(idx, [bw_ids[i] for i in idx], [bw_paths[i] for i in idx])
    teacher_num_tracks, kd_idx = resolve_kd_track_targets(
        stored_subset, native_n, args.teacher_specialist
    )
    kd_track_idx = torch.tensor(kd_idx, device=device) if kd_idx is not None else None
    print(
        f"track_subset {idx} -> {[bw_ids[i] for i in idx]} ({[track_assays[i] for i in idx]}); "
        f"teacher_num_tracks={teacher_num_tracks} index_select={kd_idx}",
        flush=True,
    )

    # All caching mechanics are shared with the cache-along-training hook (src.trainer.teacher_cache) so
    # the two never diverge: same FIXED window set, same compute loop, same on-disk format.
    teacher = load_finetuned_bigwig_teacher(
        args.teacher, args.teacher_base, teacher_num_tracks, device=device
    )
    ds = build_cache_dataset(
        fasta,
        [bw_paths[i] for i in idx],
        regions["train"],
        args.sequence_length,
        tokenizer,
        args.num_windows,
    )

    # MEMMAP branch: stream logits to disk (fp16) WITHOUT the full-RAM `compute_logits` array — the only
    # feasible path for the joint 34-track cache (~107GB fp32 in RAM otherwise). Returns before the RAM path.
    if args.memmap_out:
        from src.trainer.teacher_cache import compute_logits_memmap

        compute_logits_memmap(
            teacher,
            ds,
            args.memmap_out,
            kd_track_idx,
            device,
            args.amp,
            args.batch_size,
            args.num_workers,
            track_subset=stored_subset,
            track_ids=[bw_ids[i] for i in idx],
            seq_len=args.sequence_length,
            limit_num_samples=args.num_windows,
            teacher_tag=args.teacher,
            specialist=args.teacher_specialist,
        )
        print(
            f"=== wrote memmap cache to {args.memmap_out} (track_subset={stored_subset}) ===",
            flush=True,
        )
        return

    cache = compute_logits(
        teacher, ds, kd_track_idx, device, args.amp, args.batch_size, args.num_workers
    )
    coords = window_coords(ds)
    print(f"computed logits {cache.shape} over N={cache.shape[0]} windows", flush=True)

    if args.save_per_track:  # one file per track from this single teacher pass (no conflicts)
        w = save_per_track(
            cache,
            coords,
            idx,
            [bw_ids[i] for i in idx],
            args.out_dir,
            args.num_windows,
            args.sequence_length,
            args.teacher,
            args.teacher_specialist,
        )
        print(f"=== wrote {len(w)} per-track caches to {args.out_dir} ===", flush=True)
    else:  # single combined file (byte-identical format to the per-track files; same loader contract)
        torch.save(
            {
                "logits": torch.from_numpy(np.ascontiguousarray(cache)),
                "track_subset": idx,
                "num_windows": int(cache.shape[0]),
                "limit_num_samples": args.num_windows,
                "overlap": 0.0,
                "sequence_length": args.sequence_length,
                "teacher": args.teacher,
                "teacher_specialist": args.teacher_specialist,
                "track_ids": [bw_ids[i] for i in idx],
                "coords": coords,
            },
            args.out,
        )
        print(
            f"=== wrote {args.out}: logits {cache.shape} ({cache.nbytes / 1e6:.0f} MB) ===",
            flush=True,
        )


if __name__ == "__main__":
    main()
