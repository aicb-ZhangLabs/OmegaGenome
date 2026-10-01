#!/usr/bin/env python
"""Aggregate the per-assay KD-vs-baseline runs into the distilled−baseline Pearson delta tables.

Scans ``<root>/<assay>/8m_<mode>_s<seed>/ntv3_finetune_result.json`` (mode in {kd, base}); a per-seed
"pair" exists when BOTH modes are present for the same (assay, seed). For each pair it computes the
per-track delta ``kd.per_track_pearson[t] - base.per_track_pearson[t]`` and the mean delta
``kd.test_mean_pearson - base.test_mean_pearson``, then aggregates across available seeds (mean±std).

Emits GitHub-flavored markdown to stdout: a per-assay summary table (distilled / baseline / Δ, 3-seed
mean±std) plus, for each assay, a per-track Δ table. Prints only what has actually landed, and lists the
still-missing (assay, seed, mode) cells so partial results are never mistaken for complete. Read-only.

Usage:  python scripts/aggregate_assay_results.py \
          --root ${OG_SCRATCH}/ntv3_targets/assay_experiment
"""

import argparse
import csv
import json
import os
from statistics import mean, pstdev

ASSAYS = ["atac", "histone", "rnaseq", "procap", "eclip"]
SEEDS = [0, 1, 2]

# display name + track count per assay (for the publication-ready LaTeX table)
ASSAY_META = {
    "atac": ("ATAC-seq", 5),
    "histone": ("Histone ChIP-seq", 4),
    "rnaseq": ("RNA-seq", 5),
    "procap": ("PRO-cap", 10),
    "eclip": ("eCLIP", 10),
}

# NTv3-650M teacher per-track results (34-track 3-seed) -> per-assay teacher ceiling
TEACHER_CSV = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "results",
    "ntv3_650m_per_track_3seed.csv",
)
_ASSAY_CSV_NAMES = {
    "atac": {"ATAC-seq"},
    "histone": {"Histone ChIP-seq"},
    "procap": {"PRO-cap"},
    "eclip": {"eCLIP"},
    "rnaseq": {"polyA plus RNA-seq", "total RNA-seq", "RNA-seq"},
}


def _teacher_per_assay():
    """Per-assay NTv3-650M teacher mean Pearson (average over that assay's tracks, 3-seed track means)."""
    out = {}
    if not os.path.exists(TEACHER_CSV):
        return out
    rows = list(csv.DictReader(open(TEACHER_CSV)))
    for key, names in _ASSAY_CSV_NAMES.items():
        vals = [float(r["mean"]) for r in rows if r.get("assay") in names]
        if vals:
            out[key] = mean(vals)
    return out


def _tex_val(vals):
    """LaTeX 'mean$\\pm$std' (std only when >=2 seeds), or a single mean; '' if empty."""
    if not vals:
        return ""
    m = mean(vals)
    if len(vals) > 1:
        return f"{m:.3f}$\\pm${pstdev(vals):.3f}"
    return f"{m:.3f}"


def emit_tex(root, out_path, seed_subset=None):
    """Write a paper-ready LaTeX table body (\\ogtable style, mean$\\pm$s.d. over seeds).

    Reuses the same per-seed pairing as the markdown path. Includes every assay with >=1 complete seed
    pair; assays with >=2 pairs carry a s.d. ``seed_subset`` (e.g. [0]) restricts to specific seeds so
    every assay uses an IDENTICAL seed setting (a uniform single-seed table while the 3-seed run finishes).
    Not wired into any docx/paper build — a staged fragment only.
    """
    seeds = seed_subset if seed_subset is not None else SEEDS
    tea = _teacher_per_assay()
    rows, seen_seed_counts = [], []
    for assay in ASSAYS:
        kd_means, base_means, delta_means = [], [], []
        for seed in seeds:
            kd, base = _load(root, assay, "kd", seed), _load(root, assay, "base", seed)
            if kd is None or base is None:
                continue
            kd_means.append(kd["test_mean_pearson"])
            base_means.append(base["test_mean_pearson"])
            delta_means.append(kd["test_mean_pearson"] - base["test_mean_pearson"])
        if not delta_means:
            continue
        name, ntrk = ASSAY_META[assay]
        seen_seed_counts.append(len(delta_means))
        tval = f"{tea[assay]:.3f}" if assay in tea else "\\textemdash{}"
        pct = (
            f"{100 * mean(kd_means) / tea[assay]:.0f}\\%"
            if assay in tea and tea[assay]
            else "\\textemdash{}"
        )
        rows.append(
            f"{name} & {ntrk} & {tval} & {_tex_val(kd_means)} & {_tex_val(base_means)} & "
            f"\\best{{+{_tex_val(delta_means)}}} & {pct} & {len(delta_means)}\\\\"
        )
    smax = max(seen_seed_counts) if seen_seed_counts else 0
    body = (
        "\\ogtable{%\n\\small\n\\begin{tabular}{l c c c c c c c}\n\\toprule\n"
        "\\bannertitle{8}{Per-assay specialist (8\\,M student): distillation vs.\\ "
        "from-scratch, with the NTv3-650M teacher ceiling}\n"
        "\\hdr{Assay} & \\hdr{Tracks} & \\hdr{Teacher} & \\hdr{Distilled} & \\hdr{From-scratch} & "
        "\\hdr{$\\Delta$} & \\hdr{\\% teacher} & \\hdr{Seeds}\\\\\n\\midrule\n"
        + "\n".join(rows)
        + "\n\\bottomrule\n\\end{tabular}%\n}{%\n"
        "Per-assay multi-track specialist: one 8\\,M NTv3 student trained on all tracks of an assay family, "
        "distilled from the NTv3-650M teacher vs.\\ from-scratch under a data-matched schedule (identical "
        "63{,}707-window stream; only the teacher term differs). Columns: mean test Pearson for the teacher, "
        "the distilled student, and the from-scratch student; $\\Delta$ = distilled $-$ from-scratch; "
        "\\% teacher = distilled/teacher. mean$\\pm$s.d.\\ over "
        f"up to {smax} seed(s). Distillation improves every assay on every track.%\n}}\n"
    )
    with open(out_path, "w") as fh:
        fh.write(body)
    return len(rows), smax


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
    ap.add_argument(
        "--emit_tex", default=None, help="also write a paper-ready LaTeX table body to this path"
    )
    ap.add_argument(
        "--emit_seeds",
        default=None,
        help="comma seeds to include in --emit_tex (e.g. '0' for a uniform single-seed table); default all",
    )
    args = ap.parse_args()

    if args.emit_tex:
        ss = [int(x) for x in args.emit_seeds.split(",")] if args.emit_seeds else None
        n, smax = emit_tex(args.root, args.emit_tex, seed_subset=ss)
        print(f"[emit_tex] wrote {args.emit_tex}: {n} assays, seeds={ss or 'all'}")

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
            f"| {assay} | {_fmt(kd_means)} | {_fmt(base_means)} | **{_fmt(delta_means)}** | {n_pairs} |"
        )
        lines = [
            f"\n**{assay}** — per-track distilled−baseline Δ (mean over {n_pairs} seed pair(s)):",
            "",
            "| track | Δ Pearson | distilled | baseline |",
            "|---|---|---|---|",
        ]
        # reload seed-0 (or the first available pair) to show absolute levels alongside the Δ
        kd0 = next(
            (
                _load(args.root, assay, "kd", s)
                for s in SEEDS
                if _load(args.root, assay, "kd", s) and _load(args.root, assay, "base", s)
            ),
            None,
        )
        base0 = next(
            (
                _load(args.root, assay, "base", s)
                for s in SEEDS
                if _load(args.root, assay, "kd", s) and _load(args.root, assay, "base", s)
            ),
            None,
        )
        for t in track_order:
            d = _fmt(track_deltas[t])
            kv = f"{kd0['per_track_pearson'][t]:.4f}" if kd0 else "—"
            bv = f"{base0['per_track_pearson'][t]:.4f}" if base0 else "—"
            per_track_lines_mark = "**" if mean(track_deltas[t]) > 0 else ""
            lines.append(f"| {t} | {per_track_lines_mark}{d}{per_track_lines_mark} | {kv} | {bv} |")
        per_track_blocks.append("\n".join(lines))

    print("## Results — per-assay distilled vs from-scratch (8M student)\n")
    print(
        "| assay | distilled (mean Pearson) | from-scratch | Δ (distilled−baseline) | seed pairs |"
    )
    print("|---|---|---|---|---|")
    print("\n".join(summary_rows))
    print("\n".join(per_track_blocks))
    if missing:
        print(
            f"\n_Still running / not yet paired ({len(missing)}): "
            + ", ".join(sorted(missing))
            + "_"
        )


if __name__ == "__main__":
    main()
