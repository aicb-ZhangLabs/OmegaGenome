#!/usr/bin/env python
"""Render the R2.3 cross-task transfer heatmap + text summary from cross_task_matrix_<teacher>.csv.

Reads the MCC matrix CSV (rows=student task, cols=eval task; blank cell = N/A incompatible head),
draws a heatmap (diagonal = in-task strength, off-diagonal = transfer), and prints/saves the
mean-diagonal vs mean-off-diagonal summary plus any notable off-diagonal transfer. Pure plotting --
no GPU, no torch; run with a matplotlib-capable python.
"""
import argparse
import csv
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MULTICLASS = {"enhancers_types", "splice_sites_all"}  # 3-label tasks (separate sub-matrix)


def read_matrix(path):
    """Load the matrix CSV -> (tasks list, NxN float array with NaN for blank/N-A cells)."""
    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    tasks = rows[0][1:]
    mat = np.full((len(tasks), len(tasks)), np.nan)
    for i, r in enumerate(rows[1:]):
        for j, v in enumerate(r[1:]):
            if v != "":
                mat[i, j] = float(v)
    return tasks, mat


def summarize(tasks, mat, label, idx):
    """Print mean diagonal vs mean off-diagonal MCC for the sub-matrix over the given task indices."""
    diag = [mat[i, i] for i in idx if not np.isnan(mat[i, i])]
    off = [mat[i, j] for i in idx for j in idx if i != j and not np.isnan(mat[i, j])]
    lines = [f"--- {label} ({len(idx)} tasks) ---"]
    if diag and off:
        md, mo = np.mean(diag), np.mean(off)
        lines += [f"  mean DIAGONAL MCC     = {md:+.4f}",
                  f"  mean OFF-DIAGONAL MCC = {mo:+.4f}",
                  f"  drop (diag - off)     = {md - mo:+.4f}"]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", default="enformer")
    ap.add_argument("--dir", default=os.path.dirname(os.path.abspath(__file__)))
    args = ap.parse_args()
    csv_path = os.path.join(args.dir, f"cross_task_matrix_{args.teacher}.csv")
    tasks, mat = read_matrix(csv_path)
    n = len(tasks)

    # Heatmap (NaN -> shown as a distinct light-grey via masked array).
    masked = np.ma.masked_invalid(mat)
    fig, ax = plt.subplots(figsize=(max(8, 0.55 * n + 3), max(7, 0.55 * n + 2)))
    cmap = plt.cm.viridis.copy()
    cmap.set_bad(color="#dddddd")
    im = ax.imshow(masked, cmap=cmap, vmin=0.0, vmax=max(0.5, np.nanmax(mat)), aspect="auto")
    ax.set_xticks(range(n)); ax.set_xticklabels(tasks, rotation=90, fontsize=7)
    ax.set_yticks(range(n)); ax.set_yticklabels(tasks, fontsize=7)
    ax.set_xlabel("eval task (B)")
    ax.set_ylabel("student task (A)")
    ax.set_title(f"Cross-task transfer MCC -- {args.teacher} students (R2.3)\n"
                 "diagonal = in-task; grey = incompatible head (N/A)", fontsize=10)
    for i in range(n):
        for j in range(n):
            if not np.isnan(mat[i, j]):
                c = "white" if mat[i, j] < 0.5 * np.nanmax(mat) else "black"
                ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=5, color=c)
    fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02, label="MCC")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        p = os.path.join(args.dir, f"cross_task_matrix_{args.teacher}.{ext}")
        fig.savefig(p, dpi=150, bbox_inches="tight")
        print(f"SAVED {p}")

    # Text summary.
    binary_idx = [i for i, t in enumerate(tasks) if t not in MULTICLASS]
    multi_idx = [i for i, t in enumerate(tasks) if t in MULTICLASS]
    parts = ["", "========== R2.3 CROSS-TASK TRANSFER SUMMARY =========="]
    parts.append(summarize(tasks, mat, "BINARY sub-matrix", binary_idx))
    if multi_idx:
        parts.append(summarize(tasks, mat, "MULTICLASS (3-label) sub-matrix", multi_idx))
    # notable off-diagonal transfer: best off-diag per student
    parts.append("\n--- best OFF-DIAGONAL transfer per student (top by MCC) ---")
    notable = []
    for i, a in enumerate(tasks):
        offs = [(mat[i, j], tasks[j]) for j in range(n) if i != j and not np.isnan(mat[i, j])]
        if offs:
            best = max(offs)
            notable.append((best[0], a, best[1]))
    for v, a, b in sorted(notable, reverse=True)[:10]:
        parts.append(f"  {a:24s} -> {b:24s}  MCC={v:+.4f}  (diag {mat[tasks.index(a), tasks.index(a)]:+.4f})")
    summary = "\n".join(parts)
    print(summary)
    sp = os.path.join(args.dir, f"cross_task_matrix_{args.teacher}_summary.txt")
    with open(sp, "w") as f:
        f.write(summary + "\n")
    print(f"\nSAVED {sp}")


if __name__ == "__main__":
    main()
