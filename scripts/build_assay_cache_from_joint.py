#!/usr/bin/env python
"""Build a MULTI-track (per-assay) teacher-logit cache by slicing K channels out of the joint-34 memmap
cache (teacher_cache_joint34/{logits.npy,meta.pt}) — the multi-track generalization of
build_pertrack_cache_from_joint.py. The resulting .pt is exactly what the cached-KD loader expects for a
specialist student trained with --track_subset "<assay indices>" (finetune_ntv3.py:263-272 asserts
cache["track_subset"] == the --track_subset list, then uses cached_logits[N,L,K] directly, no index-select).

Provably identical to `precompute_teacher_logits --track_subset i,j,k` on the generalist teacher: the joint
cache is that same teacher's 34-track output over the fixed window set, so joint.logits[:, :, [i,j,k]] are
the exact fp16 bits the canonical path would recompute. Reads the joint logits.npy SEQUENTIALLY in row
blocks (sshfs-safe, C-contiguous), never a strided fancy-index mmap.
"""

import argparse
import os

import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--joint_cache", required=True, help="dir with logits.npy + meta.pt (joint-34)")
    ap.add_argument(
        "--track_idxs",
        required=True,
        help="comma-separated dataset channels for the assay, e.g. '12,13,16,21,27'",
    )
    ap.add_argument(
        "--expect_track_ids",
        required=True,
        help="comma-separated expected meta track_ids for those idxs (order must match) — hard guard",
    )
    ap.add_argument("--out", required=True, help="output .pt path (e.g. .../assay/atac.pt)")
    ap.add_argument("--chunk_rows", type=int, default=2000)
    args = ap.parse_args()

    idxs = [int(x) for x in args.track_idxs.split(",")]
    expect = args.expect_track_ids.split(",")
    assert len(idxs) == len(expect), f"{len(idxs)} idxs vs {len(expect)} expected ids"

    logits_path = os.path.join(args.joint_cache, "logits.npy")
    meta = torch.load(
        os.path.join(args.joint_cache, "meta.pt"), map_location="cpu", weights_only=False
    )
    ts, tids, C = meta["track_subset"], meta["track_ids"], len(meta["track_ids"])

    # Channel identity: joint cache must be a full native 0..C-1 ordering so channel c == dataset track c.
    if ts is not None and list(ts) != list(range(C)):
        raise SystemExit(
            f"FATAL: joint track_subset={ts} not full native; channels would not equal dataset idx."
        )
    for i, e in zip(idxs, expect):
        if not (0 <= i < C):
            raise SystemExit(f"FATAL: idx {i} out of range 0..{C - 1}")
        if tids[i] != e:
            raise SystemExit(
                f"FATAL: meta track_ids[{i}]={tids[i]!r} != expected {e!r}; refusing mis-target."
            )
    print(
        f"channel identity OK: joint covers 0..{C - 1}; idxs {idxs} -> {[tids[i] for i in idxs]}",
        flush=True,
    )

    mm = np.load(logits_path, mmap_mode="r")  # [N, L, C] fp16 C-contiguous
    N, L, Cc = mm.shape
    assert Cc == C, f"logits C {Cc} != meta {C}"
    K = len(idxs)
    print(
        f"joint logits {mm.shape}; slicing {K} channels {idxs} in row-blocks of {args.chunk_rows}",
        flush=True,
    )
    out = np.empty((N, L, K), dtype=np.float16)
    for i in range(0, N, args.chunk_rows):
        j = min(i + args.chunk_rows, N)
        block = np.asarray(mm[i:j])  # contiguous sequential read of full rows
        for c, ti in enumerate(idxs):
            out[i:j, :, c] = block[:, :, ti]
        if (i // args.chunk_rows) % 5 == 0:
            print(f"  {j}/{N} rows", flush=True)
    del mm

    rec = {
        "logits": torch.from_numpy(np.ascontiguousarray(out)),  # [N, L, K] fp16, idx-aligned
        "track_subset": list(idxs),  # MUST equal --track_subset (assert @271)
        "num_windows": int(N),
        "limit_num_samples": int(meta["limit_num_samples"]),
        "overlap": float(meta.get("overlap", 0.0)),
        "sequence_length": int(meta["sequence_length"]),
        "teacher": meta["teacher"],
        "teacher_specialist": bool(meta.get("teacher_specialist", False)),
        "track_ids": [tids[i] for i in idxs],
        "coords": meta["coords"],
    }
    assert rec["logits"].shape == (N, L, K)
    assert rec["track_subset"] == list(idxs)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    tmp = args.out + ".tmp"
    torch.save(rec, tmp)
    os.replace(tmp, args.out)
    print(
        f"WROTE {args.out}  logits {tuple(rec['logits'].shape)} track_subset={rec['track_subset']} "
        f"n_windows={rec['limit_num_samples']} overlap={rec['overlap']}",
        flush=True,
    )

    # VERIFY: reload + byte-check sampled windows/channels against the joint memmap
    chk = torch.load(args.out, map_location="cpu", weights_only=False)
    mm2 = np.load(logits_path, mmap_mode="r")
    lg = chk["logits"]
    assert lg.shape == (N, L, K) and chk["track_subset"] == list(idxs)
    assert chk["track_ids"] == [tids[i] for i in idxs]
    assert len(chk["coords"]) == N and tuple(chk["coords"][0]) == tuple(meta["coords"][0])
    bad = 0
    for w in (0, N // 3, N // 2, 2 * N // 3, N - 1):
        for c, ti in enumerate(idxs):
            a = lg[w, :, c].numpy().astype(np.float16)
            b = np.asarray(mm2[w, :, ti]).astype(np.float16)
            if not np.array_equal(a, b):
                bad += 1
    del mm2
    if bad:
        raise SystemExit(
            f"FATAL: {bad} sampled (window,channel) disagree with joint — cache WRONG."
        )
    print(
        f"VERIFY OK: {tuple(lg.shape)}, track_subset {chk['track_subset']}; sampled windows/channels "
        f"BYTE-IDENTICAL to the joint cache.",
        flush=True,
    )


if __name__ == "__main__":
    main()
