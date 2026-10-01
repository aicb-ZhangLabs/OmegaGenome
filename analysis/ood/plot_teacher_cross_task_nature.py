#!/usr/bin/env python
"""TEACHER cross-task OOD — Nature-style renders, matching the student cross-task figures.

Reuses the exact style helpers from ``plot_cross_task_nature.py`` (mako MCC colormap, short task
labels, leading-zero-stripped annotations) so the teacher-OOD figures read as one set with the
existing student-OOD figures. Pure plotting from the saved MCC matrix CSVs (no GPU).

Outputs (in this dir):
  * teacher_cross_task_matrix_<model>_nature.png  — one annotated 18x18 heatmap per foundation model
    (each fine-tuned teacher-A evaluated on every compatible task-B; blank = label-incompatible head).
  * teacher_cross_task_5panel_nature.png/.pdf     — all five teachers side by side (shared mako 0..1
    scale, one colorbar): the headline figure showing every foundation model is strong in-task
    (bright diagonal) and drops sharply off-task.

Run: python plot_teacher_cross_task_nature.py
"""

import os
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Reuse the established style + IO helpers verbatim (same dir).
from plot_cross_task_nature import (
    read_matrix,
    short_labels,
    _nolead,
    MCC_CMAP_OBJ,
    MCC_TEXT_DARK_ABOVE,
)

HERE = os.path.dirname(os.path.abspath(__file__))
# Order + display names: two billion-scale gLMs first, then the smaller teachers.
MODELS = [
    ("nt", "NT-2.5B"),
    ("carbon", "Carbon-3B"),
    ("dnabert2", "DNABERT-2"),
    ("enformer", "Enformer"),
    ("caduceus", "Caduceus"),
]


def _csv(model):
    return os.path.join(HERE, f"teacher_cross_task_matrix_{model}.csv")


def annotated_heatmap(model, disp):
    """One annotated 18x18 teacher heatmap, styled identically to the student nature heatmap."""
    tasks, mat = read_matrix(_csv(model))
    n = len(tasks)
    labels = short_labels(tasks)
    masked = np.ma.masked_invalid(mat)
    cmap = MCC_CMAP_OBJ.copy()
    cmap.set_bad(color="white")

    fig, ax = plt.subplots(figsize=(18.0, 16.0))
    im = ax.imshow(masked, cmap=cmap, vmin=0.0, vmax=1.0, aspect="equal")
    ax.set_xticks(range(n))
    ax.set_xticklabels(labels, rotation=90, fontsize=16)
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels, fontsize=16)
    ax.set_xlabel("evaluation task", fontsize=22, labelpad=12)
    ax.set_ylabel("teacher fine-tuned on task", fontsize=22, labelpad=12)
    ax.set_title(
        f"Teacher cross-task transfer — {disp} fine-tuned teachers\n"
        "diagonal = in-task; off-diagonal = transfer; blank = label-incompatible (multiclass)",
        fontsize=24,
        pad=18,
    )
    for i in range(n):
        for j in range(n):
            if not np.isnan(mat[i, j]):
                c = "#222222" if mat[i, j] >= MCC_TEXT_DARK_ABOVE else "white"
                ax.text(j, i, _nolead(mat[i, j]), ha="center", va="center", fontsize=12.5, color=c)
    cb = fig.colorbar(im, ax=ax, fraction=0.030, pad=0.02)
    cb.set_label("transfer MCC", fontsize=22)
    cb.ax.tick_params(labelsize=16)
    ax.tick_params(length=4)
    fig.tight_layout()
    out = os.path.join(HERE, f"teacher_cross_task_matrix_{model}_nature.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"SAVED {out}")


def five_panel():
    """All five foundation models in a 2-per-row grid (3 rows x 2 cols; shared mako 0..1 scale, one
    colorbar). Two panels per row with large fonts and tight spacing so every 18x18 heatmap and its
    task labels stay legible when the figure is scaled to a page; the empty sixth slot holds the
    shared colorbar."""
    ncol = 2
    nrow = (len(MODELS) + ncol - 1) // ncol  # 3
    fig, axes = plt.subplots(nrow, ncol, figsize=(15.0, 6.9 * nrow))
    axflat = axes.ravel()
    labels = None
    im = None
    for idx, (model, disp) in enumerate(MODELS):
        ax = axflat[idx]
        tasks, mat = read_matrix(_csv(model))
        n = len(tasks)
        if labels is None:
            labels = short_labels(tasks)
        cmap = MCC_CMAP_OBJ.copy()
        cmap.set_bad(color="white")
        im = ax.imshow(np.ma.masked_invalid(mat), cmap=cmap, vmin=0.0, vmax=1.0, aspect="equal")
        ax.set_title(disp, fontsize=30, fontweight="bold", pad=6)
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        # y-tick labels only on the left column; x-tick labels only on the bottom panel of each
        # column (shared axes -> no title/label collisions between rows, and more compact)
        if idx % ncol == 0:
            ax.set_yticklabels(labels, fontsize=18)
            ax.set_ylabel("teacher fine-tuned on task", fontsize=22, labelpad=5)
        else:
            ax.set_yticklabels([""] * n)
        if (idx + ncol) >= len(MODELS):  # bottom-most panel in this column
            ax.set_xticklabels(labels, rotation=90, fontsize=18)
            ax.set_xlabel("evaluation task", fontsize=22, labelpad=5)
        else:
            ax.set_xticklabels([""] * n)
        ax.tick_params(length=3, pad=1.5)
    # hide unused trailing axes (the odd 6th slot)
    for k in range(len(MODELS), len(axflat)):
        axflat[k].axis("off")
    fig.suptitle(
        "Teacher cross-task transfer — five foundation models",
        fontsize=32,
        fontweight="bold",
        y=0.997,
    )
    # compact spacing; the empty bottom-right (6th) slot hosts the shared colorbar. A little extra
    # hspace gives the Enformer panel's x-tick labels room above that slot.
    fig.subplots_adjust(left=0.07, right=0.99, top=0.955, bottom=0.06, hspace=0.16, wspace=0.05)
    # Horizontal colorbar placed low in the empty bottom-right slot, well below the Enformer panel's
    # x-tick labels above it — fixes the legend/x-label overlap without stealing any panel width.
    cax = fig.add_axes([0.605, 0.090, 0.315, 0.020])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.set_label("transfer MCC", fontsize=24, labelpad=6)
    cb.ax.tick_params(labelsize=18)
    for ext in ("png", "pdf"):
        out = os.path.join(HERE, f"teacher_cross_task_5panel_nature.{ext}")
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print(f"SAVED {out}")
    plt.close(fig)


if __name__ == "__main__":
    for model, disp in MODELS:
        annotated_heatmap(model, disp)
    five_panel()
