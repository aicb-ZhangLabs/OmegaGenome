"""Generate NTv3 per-bp track targets for genomic windows (the GPU step of NTv3 distillation).

Reads the v1 track manifest, fetches hg38 windows (UCSC API), runs the NTv3 teacher on each, and
saves (sequences, targets[N, L_out, T]) per split for the student to distill. Train/test use
different chromosomes (leakage control).

  python -m src.train.ntv3_gen_targets --out <dir> [--model <repo|local>] [--window 16384] ...
"""

import argparse
import json
import os

import torch

from src.data.ntv3_windows import fetch_windows, tile_windows
from src.data.track_dataset import build_teacher_targets
from src.model.ntv3_teacher import NTV3_100M_POST_LOCAL, NTV3_650M_POST, NTv3Teacher, NTv3TeacherConfig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="config/distillation/ntv3_v1_tracks.json")
    ap.add_argument("--model", default=NTV3_650M_POST, help=f"HF repo or local (e.g. {NTV3_100M_POST_LOCAL})")
    ap.add_argument("--out", required=True, help="output dir (SSD), gets train.pt / test.pt")
    ap.add_argument("--window", type=int, default=16384, help="window bp (multiple of 128)")
    ap.add_argument("--n_train", type=int, default=2000)
    ap.add_argument("--n_test", type=int, default=400)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--train_chrom", default="chr1")
    ap.add_argument("--test_chrom", default="chr8")
    args = ap.parse_args()

    man = json.load(open(args.manifest))
    subset, labels = man["ntv3_track_subset"], [t["label"] for t in man["tracks"]]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"teacher={args.model} | tracks={labels} | device={device}")
    teacher = NTv3Teacher(
        NTv3TeacherConfig(model_name_or_path=args.model, species=man["species"], track_subset=subset),
        device=device,
    )

    os.makedirs(args.out, exist_ok=True)
    start = 1_000_000  # avoid the assembly gap near the start of each chromosome
    splits = [
        ("train", tile_windows(args.train_chrom, start, start + args.window * args.n_train * 2, args.window, n=args.n_train)),
        ("test", tile_windows(args.test_chrom, start, start + args.window * args.n_test * 2, args.window, n=args.n_test)),
    ]
    for name, coords in splits:
        print(f"[{name}] fetching {len(coords)} x {args.window}bp windows from UCSC ...")
        seqs = fetch_windows(coords)
        targets = build_teacher_targets(teacher, seqs, batch_size=args.batch_size)  # [N, L_out, T]
        path = os.path.join(args.out, f"{name}.pt")
        torch.save(
            {"sequences": seqs, "targets": targets, "labels": labels, "coords": coords, "window": args.window},
            path,
        )
        print(f"[{name}] saved {len(seqs)} windows, targets {tuple(targets.shape)} -> {path}")


if __name__ == "__main__":
    main()
