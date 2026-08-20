#!/usr/bin/env python
"""R2.3 cross-task transfer — Nature-style renders (enlarged, print-legible fonts).

Reconstructs the two rebuttal figures directly from the saved MCC matrix CSVs (no GPU):

  * cross_task_matrix_enformer_nature.png  — the single annotated 18x18 heatmap (Enformer distilled
    students); every compatible cell shows its MCC number, blank cell = label-incompatible head.
  * cross_task_nature_3panel.png           — 3 panels: (a) Distilled (Enformer->BPNet),
    (b) From-scratch (pure-CE BPNet), (c) Delta = Distilled - From-scratch (diverging RdBu_r).

MCC panels use seaborn's "mako" colormap (dark-navy -> blue -> teal -> pale-green: low MCC = near-black
navy, high MCC 0.8+ = teal/pale-green) so both OOD figures read as one dark-blue/teal "nature" set; the
signed Delta panel keeps diverging RdBu_r (blue -> red, symmetric +/-0.2). Font sizes here are
deliberately large so both figures stay readable at print size. Run with a matplotlib-capable python
(e.g. the enformer conda env). Pure plotting; seaborn only for the mako colormap.
"""
import csv
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns  # noqa: F401  (registers mako/rocket/crest with matplotlib on import)

# Shared MCC colormap: seaborn "mako" (low = dark navy, mid = blue, high = teal/pale-green).
MCC_CMAP_OBJ = sns.color_palette("mako", as_cmap=True)
# Text-color threshold: mako luminance crosses ~0.6 near MCC 0.72, so cells at/above this are pale
# (teal/green) and need DARK numbers; darker cells below it need WHITE numbers.
MCC_TEXT_DARK_ABOVE = 0.72

HERE = os.path.dirname(os.path.abspath(__file__))

# Compact axis labels (long task names -> short, matching the finalized figures).
SHORT = {
    "promoter_all": "pr_all", "promoter_no_tata": "pr_no_tata", "promoter_tata": "pr_tata",
    "enhancers": "enh", "enhancers_types": "enh_types",
    "splice_sites_all": "ss_all", "splice_sites_acceptors": "ss_acceptors",
    "splice_sites_donors": "ss_donors",
}


def read_matrix(path):
    """Load the MCC matrix CSV -> (tasks, NxN float array with NaN for blank/incompatible cells)."""
    rows = list(csv.reader(open(path, newline="")))
    tasks = rows[0][1:]
    mat = np.full((len(tasks), len(tasks)), np.nan)
    for i, r in enumerate(rows[1:]):
        for j, v in enumerate(r[1:]):
            if v != "":
                mat[i, j] = float(v)
    return tasks, mat


def short_labels(tasks):
    return [SHORT.get(t, t) for t in tasks]


def _nolead(v):
    """Format to 2dp with the leading zero stripped: 0.50->.50, -0.08->-.08, 1.00->1.00."""
    s = f"{v:.2f}"
    if s.startswith("0."):
        return s[1:]
    if s.startswith("-0."):
        return "-" + s[2:]
    return s


def annotated_heatmap():
    """Single annotated 18x18 Enformer-distilled heatmap -> cross_task_matrix_enformer_nature.png."""
    tasks, mat = read_matrix(os.path.join(HERE, "cross_task_matrix_enformer.csv"))
    n = len(tasks)
    labels = short_labels(tasks)
    masked = np.ma.masked_invalid(mat)
    # mako MCC map: low = dark navy, mid = blue, high = teal/pale-green. White for incompatible cells.
    cmap = MCC_CMAP_OBJ.copy()
    cmap.set_bad(color="white")

    # Larger square figure so the enlarged per-cell numbers stay uncramped in an 18x18 grid.
    fig, ax = plt.subplots(figsize=(18.0, 16.0))
    im = ax.imshow(masked, cmap=cmap, vmin=0.0, vmax=1.0, aspect="equal")

    ax.set_xticks(range(n)); ax.set_xticklabels(labels, rotation=90, fontsize=16)
    ax.set_yticks(range(n)); ax.set_yticklabels(labels, fontsize=16)
    ax.set_xlabel("evaluation task", fontsize=22, labelpad=12)
    ax.set_ylabel("student trained on task", fontsize=22, labelpad=12)
    ax.set_title("Cross-task transfer matrix — Enformer distilled students (R2.3)\n"
                 "diagonal = in-task; off-diagonal = transfer; blank = label-incompatible (multiclass)",
                 fontsize=24, pad=18)

    # In-cell MCC annotations (the element most in need of enlarging).
    for i in range(n):
        for j in range(n):
            if not np.isnan(mat[i, j]):
                # mako is dark at LOW/MID MCC (white text) and pale teal/green at high MCC (dark text).
                c = "#222222" if mat[i, j] >= MCC_TEXT_DARK_ABOVE else "white"
                ax.text(j, i, _nolead(mat[i, j]), ha="center", va="center", fontsize=12.5, color=c)

    cb = fig.colorbar(im, ax=ax, fraction=0.030, pad=0.02)
    cb.set_label("transfer MCC", fontsize=22)
    cb.ax.tick_params(labelsize=16)
    ax.tick_params(length=4)
    fig.tight_layout()
    out = os.path.join(HERE, "cross_task_matrix_enformer_nature.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"SAVED {out}")


def three_panel():
    """3-panel distilled / from-scratch / delta figure -> cross_task_nature_3panel.png."""
    tasks, dist = read_matrix(os.path.join(HERE, "cross_task_matrix_enformer.csv"))
    tb, base = read_matrix(os.path.join(HERE, "cross_task_matrix_baseline.csv"))
    assert tasks == tb, "distilled and baseline task orders differ"
    n = len(tasks)
    labels = short_labels(tasks)
    delta = dist - base

    # SAME nature palette family as the 18x18 annotated heatmap: seaborn "mako" (0->1, dark navy ->
    # blue -> teal/pale-green) for the two MCC panels a/b, and a crisp diverging RdBu_r (symmetric
    # +/-0.2) for the signed Delta panel c. White for label-incompatible (masked) cells, matching the
    # 18x18. Fonts / colorbar / label styling are matched to the 18x18 so both OOD figures read as one.
    cmap = MCC_CMAP_OBJ.copy(); cmap.set_bad(color="white")
    dcmap = plt.cm.RdBu_r.copy(); dcmap.set_bad(color="white")

    fig, axes = plt.subplots(1, 3, figsize=(28.0, 9.6))
    titles = ["A  Distilled (Enformer→BPNet)", "B  From-scratch (pure-CE BPNet)",
              "C  Δ = Distilled − From-scratch"]
    fig.suptitle("Cross-task transfer: distillation vs from-scratch (R2.3)",
                 fontsize=32, fontweight="bold", y=0.99)

    for k, (ax, M, title) in enumerate(zip(axes, [dist, base, delta], titles)):
        masked = np.ma.masked_invalid(M)
        if k < 2:
            im = ax.imshow(masked, cmap=cmap, vmin=0.0, vmax=1.0, aspect="equal")
            cblabel = "MCC"
        else:
            im = ax.imshow(masked, cmap=dcmap, vmin=-0.2, vmax=0.2, aspect="equal")
            cblabel = "ΔMCC"
        # Font sizes / colorbar / label styling matched to the 18x18 annotated heatmap.
        ax.set_title(title, fontsize=25, fontweight="bold", loc="left", pad=14)
        ax.set_xticks(range(n)); ax.set_xticklabels(labels, rotation=90, fontsize=16)
        ax.set_yticks(range(n))
        ax.set_yticklabels(labels if k == 0 else [""] * n, fontsize=16)
        ax.set_xlabel("evaluation task", fontsize=22, labelpad=12)
        if k == 0:
            ax.set_ylabel("student trained on task", fontsize=22, labelpad=12)
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        cb.set_label(cblabel, fontsize=22)
        cb.ax.tick_params(labelsize=16)
        ax.tick_params(length=4)

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out = os.path.join(HERE, "cross_task_nature_3panel.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"SAVED {out}")


if __name__ == "__main__":
    annotated_heatmap()
    three_panel()
