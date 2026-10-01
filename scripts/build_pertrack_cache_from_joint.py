#!/usr/bin/env python
"""Build a single-track teacher-logit cache (gen/t{idx}.pt) by SLICING one channel out of the
joint-34 memmap cache (teacher_cache_joint34/{logits.npy,meta.pt}) — method (b) of the size-ladder
cache build.

Why this is provably identical to the canonical precompute
(`precompute_teacher_logits --teacher <ckpt> --track_subset idx`):
the joint-34 cache is that SAME generalist teacher's 34-track output over a FIXED window set
(coords in meta), and `--track_subset idx` on the generalist teacher simply index-selects channel
`idx` of that same 34-track forward. So joint34.logits[:, :, idx] are the exact same fp16 bits the
canonical path would recompute — no GPU pass, no teacher reload, no divergence risk. We reuse the
official `cache_record()` so the on-disk dict is byte-for-byte the format the cached-KD loader expects.

Safety: the joint logits.npy is read SEQUENTIALLY in row-blocks (C-contiguous [N,L,C] => a row-block
mm[i:j] is one contiguous read), NOT a strided full-array mmap fancy-index — sshfs-safe and light on
I/O so the running joint sweep is not starved.
"""

import argparse
import os
import sys

import numpy as np
import torch

# repo import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.trainer.teacher_cache import cache_record  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--joint_cache", required=True, help="dir with logits.npy + meta.pt (joint-34)")
    ap.add_argument(
        "--track_idx", type=int, required=True, help="dataset channel to slice (e.g. 12)"
    )
    ap.add_argument(
        "--expect_track_id", required=True, help="assert meta track_ids[track_idx]==this"
    )
    ap.add_argument("--out", required=True, help="output .pt path (e.g. .../gen/t12.pt)")
    ap.add_argument("--chunk_rows", type=int, default=2000, help="rows per sequential read block")
    args = ap.parse_args()

    logits_path = os.path.join(args.joint_cache, "logits.npy")
    meta_path = os.path.join(args.joint_cache, "meta.pt")
    meta = torch.load(meta_path, map_location="cpu")

    # --- Channel identity: joint cache track_subset must be a full/native ordering so channel c == dataset
    #     track c, and meta track_ids[idx] must be the track we expect (ENCSR325NFE). Fail loud otherwise. ---
    ts = meta["track_subset"]
    tids = meta["track_ids"]
    C = len(tids)
    if ts is not None and list(ts) != list(range(C)):
        raise SystemExit(
            f"FATAL: joint cache track_subset={ts} is not a full native ordering; "
            f"channel {args.track_idx} would not equal dataset track {args.track_idx}."
        )
    if tids[args.track_idx] != args.expect_track_id:
        raise SystemExit(
            f"FATAL: meta track_ids[{args.track_idx}]={tids[args.track_idx]!r} != "
            f"expected {args.expect_track_id!r}. Refusing to build a mis-targeted cache."
        )
    print(
        f"channel identity OK: joint track_subset covers 0..{C - 1}; "
        f"track_ids[{args.track_idx}]={tids[args.track_idx]!r} == {args.expect_track_id!r}",
        flush=True,
    )

    mm = np.load(logits_path, mmap_mode="r")  # [N, L, C] fp16, C-contiguous
    N, L, Cc = mm.shape
    assert Cc == C, f"logits C {Cc} != meta track_ids {C}"
    print(
        f"joint logits: shape={mm.shape} dtype={mm.dtype}; slicing channel {args.track_idx} "
        f"in row-blocks of {args.chunk_rows}",
        flush=True,
    )

    out = np.empty((N, L, 1), dtype=np.float16)
    for i in range(0, N, args.chunk_rows):
        j = min(i + args.chunk_rows, N)
        block = np.asarray(mm[i:j])  # contiguous sequential read of full rows
        out[i:j, :, 0] = block[:, :, args.track_idx]
        if (i // args.chunk_rows) % 5 == 0:
            print(f"  {j}/{N} rows", flush=True)
    del mm

    # --- Build the official on-disk record (identical format to the canonical precompute). coords /
    #     limit_num_samples / overlap / seq_len / teacher tag are reused VERBATIM from the joint meta so
    #     the trainer's coord-alignment + track_subset==idx asserts pass exactly. ---
    rec = cache_record(
        logits_NL1=out,
        coords=meta["coords"],
        track_idx=args.track_idx,
        track_id=tids[args.track_idx],
        n_windows=int(meta["limit_num_samples"]),
        seq_len=int(meta["sequence_length"]),
        teacher_tag=meta["teacher"],
        specialist=bool(meta.get("teacher_specialist", False)),
    )
    # sanity on the record before writing
    assert rec["logits"].shape == (N, L, 1), rec["logits"].shape
    assert rec["track_subset"] == [args.track_idx], rec["track_subset"]
    assert rec["overlap"] == float(meta.get("overlap", 0.0))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    tmp = args.out + ".tmp"
    torch.save(rec, tmp)
    os.replace(tmp, args.out)
    print(
        f"WROTE {args.out}  logits {tuple(rec['logits'].shape)} track_subset={rec['track_subset']} "
        f"n_windows={rec['limit_num_samples']} overlap={rec['overlap']} teacher={rec['teacher']}",
        flush=True,
    )

    # --- VERIFY: reload the written file and spot-check values against the joint memmap channel ---
    chk = torch.load(args.out, map_location="cpu")
    mm2 = np.load(logits_path, mmap_mode="r")
    lg = chk["logits"]
    assert lg.shape == (N, L, 1)
    assert chk["track_subset"] == [args.track_idx]
    assert chk["track_ids"] == [args.expect_track_id]
    assert len(chk["coords"]) == N and tuple(chk["coords"][0]) == tuple(meta["coords"][0])
    bad = 0
    for j in (0, N // 3, N // 2, 2 * N // 3, N - 1):
        a = lg[j, :, 0].numpy().astype(np.float16)
        b = np.asarray(mm2[j, :, args.track_idx]).astype(np.float16)
        if not np.array_equal(a, b):
            d = np.max(np.abs(a.astype(np.float32) - b.astype(np.float32)))
            print(f"  MISMATCH window {j}: max|d|={d}")
            bad += 1
    del mm2
    if bad:
        raise SystemExit(
            f"FATAL: {bad} sampled windows disagree with joint channel — cache is WRONG."
        )
    print(
        f"VERIFY OK: reload shape {tuple(lg.shape)}, track_subset {chk['track_subset']}, "
        f"track_ids {chk['track_ids']}; 5 sampled windows BYTE-IDENTICAL to joint channel "
        f"{args.track_idx}.",
        flush=True,
    )


if __name__ == "__main__":
    main()
