"""Evaluate an NTv3 teacher's per-bp track predictions against ENCODE ground-truth signal.

Paper-style per-track Pearson (NTv3 vs *measured* signal) on held-out windows, for a given model
size — this is the teacher ceiling each distillation student aims for, and the reproduction of the
paper's reported numbers (e.g. DNase base-res PCC ~0.75). Reuses the teacher loader, window fetcher,
the bigWig ground truth, and per_track_pearson.

  python -m src.train.eval_teacher --model InstaDeepAI/NTv3_650M_post --bigwig_dir <dir> --out <dir>
"""

import argparse
import json
import os

import numpy as np
import torch

from src.data.ground_truth import ground_truth_targets, prepare_bigwigs
from src.data.ntv3_windows import fetch_windows, tile_windows
from src.data.track_dataset import build_teacher_targets
from src.model.ntv3_teacher import NTV3_650M_POST, NTv3Teacher, NTv3TeacherConfig
from src.trainer.track_metrics import per_track_pearson


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="config/distillation/ntv3_v1_tracks.json")
    ap.add_argument("--model", default=NTV3_650M_POST, help="HF repo or local snapshot path")
    ap.add_argument("--bigwig_dir", required=True, help="SSD cache dir for ground-truth bigWigs")
    ap.add_argument("--out", required=True)
    ap.add_argument("--window", type=int, default=16384)
    ap.add_argument("--n_eval", type=int, default=200)
    ap.add_argument("--eval_chrom", default="chr10", help="held-out chromosome")
    ap.add_argument("--batch_size", type=int, default=2)
    args = ap.parse_args()

    man = json.load(open(args.manifest))
    subset, labels = man["ntv3_track_subset"], [t["label"] for t in man["tracks"]]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"teacher={args.model} | tracks={labels} | device={device}")
    teacher = NTv3Teacher(
        NTv3TeacherConfig(model_name_or_path=args.model, species=man["species"], track_subset=subset),
        device=device,
    )

    start = 1_000_000
    coords = tile_windows(args.eval_chrom, start, start + args.window * args.n_eval * 2, args.window, n=args.n_eval)
    print(f"fetching {len(coords)} x {args.window}bp held-out windows on {args.eval_chrom} ...")
    seqs = fetch_windows(coords)
    pred = build_teacher_targets(teacher, seqs, batch_size=args.batch_size)  # [N, L_out, T]
    L_out = pred.shape[1]

    print("preparing ground-truth bigWigs (download/cached) + reading signal ...")
    bw_paths = prepare_bigwigs(man, args.bigwig_dir)
    # NTv3-post CROPS its track outputs to the central 37.5% of the input at base resolution (per
    # the official docs), so L_out == 0.375*window bp at 1bp. Align ground truth to that SAME central
    # region at 1bp — otherwise predictions and truth are spatially misaligned (the 0.20-vs-0.75 bug).
    offset = (args.window - L_out) // 2
    crop_coords = [(c, s + offset, s + offset + L_out) for (c, s, _e) in coords]
    print(f"NTv3 crop: L_out={L_out} = central {100*L_out/args.window:.1f}% of {args.window}bp "
          f"(offset {offset}); ground truth read at 1bp over the crop")
    truth = ground_truth_targets(bw_paths, crop_coords, nbins=L_out)  # [N, L_out, T] @ 1bp

    # per-track Pearson: NTv3 prediction vs measured signal; [N, L, T] -> [N, T, L]
    mean_r, per = per_track_pearson(pred.permute(0, 2, 1).numpy(), np.transpose(truth, (0, 2, 1)))
    os.makedirs(args.out, exist_ok=True)
    result = {
        "model": args.model,
        "eval_chrom": args.eval_chrom,
        "n_eval": len(coords),
        "window": args.window,
        "mean_pearson_vs_truth": float(mean_r),
        "per_track_pearson": {labels[i]: (None if np.isnan(per[i]) else float(per[i])) for i in range(len(labels))},
    }
    tag = os.path.basename(args.model.rstrip("/"))
    with open(os.path.join(args.out, f"teacher_eval_{tag}.json"), "w") as f:
        json.dump(result, f, indent=2)
    print("\nNTv3 vs ground-truth per-track Pearson:")
    for i in range(len(labels)):
        print(f"  {labels[i]:12s} {per[i]:.4f}")
    print(f"mean vs truth: {mean_r:.4f} | model {args.model}")


if __name__ == "__main__":
    main()
