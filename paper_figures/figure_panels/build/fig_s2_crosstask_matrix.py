#!/usr/bin/env python
"""Figure S2 — full cross-task transfer matrix (per-cell detail), restyled to match Figure S1.

Rebuilds the 18x18 Enformer-distilled cross-task MCC heatmap that appears as paper Fig. S2
(paper_src/figs/crosstask_matrix_detail.pdf) using the SAME palette as Fig. S1 (the 3-panel
crosstask_3panel.png), so the two OOD figures read as one set.

Style match to S1:
  * seaborn "mako" colormap (near-black navy -> dark blue -> blue -> teal/pale-green), vmin=0, vmax=1,
    the identical MCC colormap used by S1 panels A/B (plot_cross_task_nature.py).
  * per-cell MCC annotations with S1's dual text scheme: near-black (#222222) numbers on the pale/high
    cells, white numbers on the dark/low cells (threshold MCC 0.72, matching S1).
  * incompatible-head cells drawn grey (#dddddd) as the S2 caption states ("grey cells denoting
    incompatible output heads"); S1 uses white for these, otherwise identical styling.
  * clean Nature style: square cells, no top/right spines, leading-zero-stripped 2dp numbers.

DATA IS UNCHANGED: every value comes verbatim from cross_task_matrix_enformer.csv (the exact CSV the
original viridis Fig. S2 was rendered from). This script only recolors; it never edits the matrix.

Pure plotting (no GPU/torch). Run with a matplotlib+seaborn python, e.g.
  python fig_s2_crosstask_matrix.py
"""
import csv
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns  # registers mako with matplotlib on import

# --- Paths -------------------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))            # repository root
CSV_PATH = os.path.join(REPO, "analysis", "ood", "cross_task_matrix_enformer.csv")
OUT_DIR = os.path.abspath(os.path.join(HERE, "..", "..", "output"))     # plot_repo/output
PREVIEW_DIR = os.path.abspath(os.path.join(HERE, ".."))                 # paper_figures/figure_panels
STEM = "fig_s2_crosstask_matrix"

# --- Style constants (identical to Fig. S1 / plot_cross_task_nature.py) ------------------------
MCC_CMAP_OBJ = sns.color_palette("mako", as_cmap=True)  # low = dark navy, high = teal/pale-green
MCC_TEXT_DARK_ABOVE = 0.72   # cells at/above this are pale -> dark numbers; below -> white numbers
INCOMPATIBLE_COLOR = "#dddddd"  # grey for label-incompatible heads (per Fig. S2 caption)

# Compact axis labels (long task names -> short), matching Fig. S1.
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


def _nolead(v):
    """Format to 2dp with the leading zero stripped: 0.50->.50, -0.08->-.08, 1.00->1.00."""
    s = f"{v:.2f}"
    if s.startswith("0."):
        return s[1:]
    if s.startswith("-0."):
        return "-" + s[2:]
    return s


def build():
    """Render the mako-styled per-cell cross-task matrix -> PDF (vector) + PNG in plot_repo/output."""
    tasks, mat = read_matrix(CSV_PATH)
    n = len(tasks)
    labels = [SHORT.get(t, t) for t in tasks]

    masked = np.ma.masked_invalid(mat)
    cmap = MCC_CMAP_OBJ.copy()
    cmap.set_bad(color=INCOMPATIBLE_COLOR)

    fig, ax = plt.subplots(figsize=(15.5, 14.0))
    im = ax.imshow(masked, cmap=cmap, vmin=0.0, vmax=1.0, aspect="equal")

    ax.set_xticks(range(n)); ax.set_xticklabels(labels, rotation=90, fontsize=14)
    ax.set_yticks(range(n)); ax.set_yticklabels(labels, fontsize=14)
    ax.set_xlabel("evaluation task", fontsize=20, labelpad=12)
    ax.set_ylabel("student trained on task", fontsize=20, labelpad=12)
    ax.set_title("Cross-task transfer matrix — Enformer distilled students\n"
                 "diagonal = in-task; off-diagonal = transfer; grey = label-incompatible head",
                 fontsize=21, pad=16)

    # Per-cell MCC numbers, S1's dual text scheme (dark on pale/high, white on dark/low).
    for i in range(n):
        for j in range(n):
            if not np.isnan(mat[i, j]):
                c = "#222222" if mat[i, j] >= MCC_TEXT_DARK_ABOVE else "white"
                ax.text(j, i, _nolead(mat[i, j]), ha="center", va="center",
                        fontsize=11.5, color=c)

    # Clean Nature framing: thin outer border, no top/right spines, light minor grid between cells.
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.6)
    ax.tick_params(which="minor", length=0)
    ax.tick_params(which="major", length=4)

    cb = fig.colorbar(im, ax=ax, fraction=0.030, pad=0.02)
    cb.set_label("transfer MCC", fontsize=20)
    cb.ax.tick_params(labelsize=14)

    fig.tight_layout()
    os.makedirs(OUT_DIR, exist_ok=True)
    pdf = os.path.join(OUT_DIR, f"{STEM}.pdf")
    png = os.path.join(OUT_DIR, f"{STEM}.png")
    fig.savefig(pdf, bbox_inches="tight", dpi=600)   # vector PDF; dpi sets the
    # resolution of the heat map raster (matplotlib resamples imshow at figure dpi,
    # default 100, which is below the journal's 300 DPI floor)
    fig.savefig(png, dpi=300, bbox_inches="tight")   # raster preview
    # Preview copy alongside the other rebuttal previews.
    prev = os.path.join(PREVIEW_DIR, f"{STEM}.png")
    fig.savefig(prev, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"SAVED {pdf}")
    print(f"SAVED {png}")
    print(f"SAVED {prev}")


if __name__ == "__main__":
    build()
