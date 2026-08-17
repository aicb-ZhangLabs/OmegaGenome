#!/usr/bin/env python
"""Aggregate the R1.1c per-assay KD-vs-baseline runs into the distilled−baseline Pearson delta tables.

Scans ``<root>/<assay>/8m_<mode>_s<seed>/ntv3_finetune_result.json`` (mode in {kd, base}); a per-seed
"pair" exists when BOTH modes are present for the same (assay, seed). For each pair it computes the
per-track delta ``kd.per_track_pearson[t] - base.per_track_pearson[t]`` and the mean delta
``kd.test_mean_pearson - base.test_mean_pearson``, then aggregates across available seeds (mean±std).

Emits GitHub-flavored markdown to stdout: a per-assay summary table (distilled / baseline / Δ, 3-seed
mean±std) plus, for each assay, a per-track Δ table. Prints only what has actually landed, and lists the
still-missing (assay, seed, mode) cells so partial results are never mistaken for complete. Read-only.

Usage:  python scripts/aggregate_assay_results.py \
          --root /tmp/galaxy_srv_disk00/pengchx3/ntv3_targets/assay_experiment
"""
import argparse
import json
import os
from statistics import mean, pstdev

ASSAYS = ["atac", "histone", "rnaseq", "procap", "eclip"]
SEEDS = [0, 1, 2]


def _load(root, assay, mode, seed):
    """Return the parsed result.json dict for one run, or None if it hasn't finished."""
    p = os.path.join(root, assay, f"8m_{mode}_s{seed}", "ntv3_finetune_result.json")
    if not os.path.exists(p):
        return None
    with open(p) as fh:
        return json.load(fh)


def _fmt(vals):
    """Format a list of floats as 'mean±std' (std over seeds; blank std when a single seed)."""
    if not vals:
        return "—"
    m = mean(vals)
    s = pstdev(vals) if len(vals) > 1 else 0.0
    return f"{m:.4f}±{s:.4f}" if len(vals) > 1 else f"{m:.4f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="assay_experiment dir")
    args = ap.parse_args()

    summary_rows, per_track_blocks, missing = [], [], []
    for assay in ASSAYS:
        kd_means, base_means, delta_means = [], [], []
        track_deltas = {}  # track_id -> [delta per completed seed]
        track_order = None
        n_pairs = 0
        for seed in SEEDS:
            kd, base = _load(args.root, assay, "kd", seed), _load(args.root, assay, "base", seed)
            for mode, r in (("kd", kd), ("base", base)):
                if r is None:
                    missing.append(f"{assay}/8m_{mode}_s{seed}")
            if kd is None or base is None:
                continue
            n_pairs += 1
            kd_means.append(kd["test_mean_pearson"])
            base_means.append(base["test_mean_pearson"])
            delta_means.append(kd["test_mean_pearson"] - base["test_mean_pearson"])
            kpt, bpt = kd["per_track_pearson"], base["per_track_pearson"]
            track_order = track_order or list(kpt.keys())
            for t in kpt:
                track_deltas.setdefault(t, []).append(kpt[t] - bpt[t])
        if n_pairs == 0:
            summary_rows.append(f"| {assay} | (no complete seed pair yet) | | | {len(SEEDS)} |")
            continue
        summary_rows.append(
            f"| {assay} | {_fmt(kd_means)} | {_fmt(base_means)} | **{_fmt(delta_means)}** | {n_pairs} |")
        lines = [f"\n**{assay}** — per-track distilled−baseline Δ (mean over {n_pairs} seed pair(s)):",
                 "", "| track | Δ Pearson | distilled | baseline |", "|---|---|---|---|"]
        # reload seed-0 (or the first available pair) to show absolute levels alongside the Δ
        kd0 = next((_load(args.root, assay, "kd", s) for s in SEEDS
                    if _load(args.root, assay, "kd", s) and _load(args.root, assay, "base", s)), None)
        base0 = next((_load(args.root, assay, "base", s) for s in SEEDS
                      if _load(args.root, assay, "kd", s) and _load(args.root, assay, "base", s)), None)
        for t in track_order:
            d = _fmt(track_deltas[t])
            kv = f"{kd0['per_track_pearson'][t]:.4f}" if kd0 else "—"
            bv = f"{base0['per_track_pearson'][t]:.4f}" if base0 else "—"
            per_track_lines_mark = "**" if mean(track_deltas[t]) > 0 else ""
            lines.append(f"| {t} | {per_track_lines_mark}{d}{per_track_lines_mark} | {kv} | {bv} |")
        per_track_blocks.append("\n".join(lines))

    print("## Results — per-assay distilled vs from-scratch (8M student)\n")
    print("| assay | distilled (mean Pearson) | from-scratch | Δ (distilled−baseline) | seed pairs |")
    print("|---|---|---|---|---|")
    print("\n".join(summary_rows))
    print("\n".join(per_track_blocks))
    if missing:
        print(f"\n_Still running / not yet paired ({len(missing)}): " + ", ".join(sorted(missing)) + "_")


if __name__ == "__main__":
    main()
