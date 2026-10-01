#!/usr/bin/env python
"""Signal-track figure — inference half (GPU/SLURM).

Runs the three JOINT-34-track NTv3 models (650M teacher | distilled 8M student | from-scratch 8M
baseline) over a handful of held-out TEST windows of the NTv3 benchmark (32,768 bp), and saves the
per-bp predicted regression signal + ground-truth target + per-track Pearson + window coordinates to
an .npz. The plotting half (`signal_track_plot.py`, CPU/enformer env) consumes that .npz.

Why joint-34: every one of the 5 assay families (ATAC / Histone ChIP / PRO-cap / eCLIP / RNA-seq) is
predicted by the SAME student, so one figure shows the distilled student recapitulating the teacher's
per-bp signal across all assay shapes, vs from-scratch. All three checkpoints exist on the SSD.

Predictions, targets and the metric all live in the OFFICIAL scaled space (x / track_mean, smooth-
clipped) — the exact space the models are trained and Pearson-scored in (`make_target_scaling_fn`).
We plot that space directly (no ad-hoc denorm), so the figure is apples-to-apples with the reported
Pearson numbers.

Reuses the production builders + loader verbatim (`build_bigwig_model` / `load_finetuned_bigwig_teacher`
/ `GenomeBigWigDataset`) so the graph evaluated here is the real fine-tune graph.

Run (galaxy, NTv3 policy):  python scripts/signal_track_inference.py --out <npz> [--n-windows 8]
"""

import argparse
import os
import sys

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from src.data.ntv3_ft_data import (  # noqa: E402
    GenomeBigWigDataset,
    load_benchmark_frames,
    make_target_scaling_fn,
)
from src.model.ntv3_finetune import (  # noqa: E402
    NTV3_CROP_FRAC,
    build_bigwig_model,
    load_finetuned_bigwig_teacher,
)

SSD = os.environ.get("SSD", os.environ.get("OG_SCRATCH", "output"))
DATA_DIR = os.environ.get("NTV3_DATA_DIR", f"{SSD}/ntv3_benchmark_data")
T_DIR = f"{SSD}/ntv3_targets"
M_DIR = f"{SSD}/ntv3_models"

# Joint-34 checkpoints (all three trained on ALL 34 tracks; score = mean over 34).
MODELS = [
    dict(
        name="teacher",
        label="NTv3-650M teacher",
        kind="teacher",
        model="InstaDeepAI/NTv3_650M_post",
        ckpt=f"{T_DIR}/ntv3_ft_faithful_s0/best_model.pth",
    ),
    dict(
        name="student_kd",
        label="Distilled 8M student",
        kind="ntv3",
        model=f"{M_DIR}/8m_pre",
        ckpt=f"{T_DIR}/ntv3_8m_kd_stdmse/best_model.pth",
    ),
    dict(
        name="baseline",
        label="From-scratch 8M baseline",
        kind="ntv3",
        model=f"{M_DIR}/8m_pre",
        ckpt=f"{T_DIR}/ntv3_8m_baseline/best_model.pth",
    ),
]


def build_model(spec, num_tracks, tokenizer, device):
    """Construct one model on its backbone and load its fine-tuned state_dict (mirrors
    benchmark_latency_ntv3._build_one_model)."""
    if spec["kind"] == "teacher":
        return load_finetuned_bigwig_teacher(spec["ckpt"], spec["model"], num_tracks, device=device)
    nuc_ids = {t: tokenizer.convert_tokens_to_ids(t) for t in "ACGT"}
    model = build_bigwig_model(
        spec["model"],
        num_tracks,
        keep_target_center_fraction=NTV3_CROP_FRAC,
        student_arch="ntv3",
        nuc_ids=nuc_ids,
        vocab_size=tokenizer.vocab_size,
    ).to(device)
    sd = torch.load(spec["ckpt"], map_location="cpu")
    sd = sd["model"] if isinstance(sd, dict) and "model" in sd else sd
    sd = {k[len("_orig_mod.") :] if k.startswith("_orig_mod.") else k: v for k, v in sd.items()}
    model.load_state_dict(sd, strict=False)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-windows", type=int, default=8, help="how many TEST windows to score")
    ap.add_argument("--seqlen", type=int, default=32768)
    ap.add_argument("--species", default="human")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device} data={DATA_DIR}", flush=True)

    fasta, bw_paths, bw_ids, regions_by_split, track_means, track_assays = load_benchmark_frames(
        DATA_DIR, args.species
    )
    num_tracks = len(bw_ids)
    print(f"native tracks={num_tracks} | assays={sorted(set(track_assays))}", flush=True)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODELS[1]["model"], trust_remote_code=True)
    transform_fn = make_target_scaling_fn(track_means)

    # Deterministic non-overlapping TEST windows; take the first n_windows.
    ds = GenomeBigWigDataset(
        fasta_path=fasta,
        bigwig_path_list=bw_paths,
        regions=regions_by_split["test"],
        sequence_length=args.seqlen,
        tokenizer=tokenizer,
        transform_fn=transform_fn,
        keep_target_center_fraction=NTV3_CROP_FRAC,
        overlap=0.0,
    )
    n_win = min(args.n_windows, len(ds))
    print(f"test windows available={len(ds)} | using first {n_win}", flush=True)

    # Recover each window's genomic coords (the dataset computes them in __getitem__).
    import bisect

    def coords(idx):
        ci = bisect.bisect_right(ds._cumulative_starts, idx) - 1
        chrom = ds.region_info[ci]["chr_name"]
        rs = ds.region_info[ci]["region_start_offset"]
        start = rs + (idx - ds._cumulative_starts[ci]) * ds.stride
        return chrom, start, start + args.seqlen

    tokens_list, targets_list, win_coords = [], [], []
    for i in range(n_win):
        item = ds[i]
        tokens_list.append(item["tokens"])
        targets_list.append(item["bigwig_targets"])
        win_coords.append(coords(i))
    tokens = torch.stack(tokens_list).to(device)  # [N, L]
    targets = torch.stack([torch.as_tensor(t) for t in targets_list]).cpu().numpy()  # [N, L_out, T]
    L_out = targets.shape[1]
    crop_off = int(
        (1.0 - NTV3_CROP_FRAC) / 2.0 * args.seqlen
    )  # bp offset of L_out start within the window
    print(
        f"L_out={L_out} crop_off={crop_off} targets {targets.shape} "
        f"range [{np.nanmin(targets):.3f},{np.nanmax(targets):.3f}]",
        flush=True,
    )

    preds = {}
    pearson = {}
    for spec in MODELS:
        model = build_model(spec, num_tracks, tokenizer, device)
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        nparam = sum(p.numel() for p in model.parameters())
        outs = []
        with torch.no_grad():
            for b in range(0, tokens.shape[0], 2):
                tb = tokens[b : b + 2]
                if device == "cuda":
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        o = model(tb)["bigwig_tracks_logits"]
                else:
                    o = model(tb)["bigwig_tracks_logits"]
                outs.append(o.float().cpu().numpy())
        pr = np.concatenate(outs, 0)  # [N, L_out, T]
        preds[spec["name"]] = pr
        # per-track Pearson over the pooled positions across these windows (sanity vs reported)
        pr_flat = pr.reshape(-1, num_tracks)
        tg_flat = targets.reshape(-1, num_tracks)
        rs = []
        for t in range(num_tracks):
            a, bvec = pr_flat[:, t], tg_flat[:, t]
            m = np.isfinite(a) & np.isfinite(bvec)
            rs.append(
                np.corrcoef(a[m], bvec[m])[0, 1] if m.sum() > 2 and a[m].std() > 0 else np.nan
            )
        pearson[spec["name"]] = np.array(rs)
        print(
            f"[{spec['name']:11s}] params={nparam / 1e6:.2f}M  pred shape={pr.shape}  "
            f"range [{pr.min():.3f},{pr.max():.3f}]  meanPearson(34,windowed)={np.nanmean(rs):.3f}",
            flush=True,
        )
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    np.savez_compressed(
        args.out,
        targets=targets,
        pred_teacher=preds["teacher"],
        pred_student_kd=preds["student_kd"],
        pred_baseline=preds["baseline"],
        pearson_teacher=pearson["teacher"],
        pearson_student_kd=pearson["student_kd"],
        pearson_baseline=pearson["baseline"],
        track_ids=np.array(bw_ids),
        track_assays=np.array(track_assays),
        track_means=np.asarray(track_means, dtype=np.float64),
        win_coords=np.array(win_coords, dtype=object),
        L_out=L_out,
        crop_off=crop_off,
        seqlen=args.seqlen,
    )
    print(f"saved -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
