#!/usr/bin/env python
"""Figure R11 (R2.3): teacher cross-task transfer, five foundation models, in a compact 2x3 grid.

Space-saving reformat requested in review (JZ comment: "change this figure to this direction so that
you can save space"): the five per-teacher 18x18 transfer heatmaps are laid out as 2 rows x 3 columns
(wider and shorter than the previous 3 rows x 2 columns portrait), and the empty 6th block (bottom-
right) holds the shared colorbar/legend. Reuses the exact style + IO helpers from
``plot_cross_task_nature.py`` (mako MCC colormap, short task labels) so it matches the other cross-task
figures. Pure plotting from the saved MCC matrix CSVs (no GPU).

Output: teacher_cross_task_5panel_2x3_nature.png / .pdf
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plot_cross_task_nature import read_matrix, short_labels, MCC_CMAP_OBJ  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = [
    ("nt", "NT-2.5B"), ("carbon", "Carbon-3B"), ("dnabert2", "DNABERT-2"),
    ("enformer", "Enformer"), ("caduceus", "Caduceus"),
]


def _csv(model):
    return os.path.join(HERE, f"teacher_cross_task_matrix_{model}.csv")


def five_panel_2x3():
    """Five teachers in a 2 rows x 3 cols grid; the empty 6th (bottom-right) slot holds the colorbar."""
    ncol, nrow = 3, 2  # 2 rows x 3 cols -> wide & short (saves vertical space vs. the old 3x2 portrait)
    fig, axes = plt.subplots(nrow, ncol, figsize=(24.0, 15.4))
    axflat = axes.ravel()
    labels = None
    im = None
    for idx, (model, disp) in enumerate(MODELS):
        ax = axflat[idx]
        tasks, mat = read_matrix(_csv(model))
        n = len(tasks)
        if labels is None:
            labels = short_labels(tasks)
        cmap = MCC_CMAP_OBJ.copy(); cmap.set_bad(color="white")
        im = ax.imshow(np.ma.masked_invalid(mat), cmap=cmap, vmin=0.0, vmax=1.0, aspect="equal")
        ax.set_title(disp, fontsize=30, fontweight="bold", pad=6)
        ax.set_xticks(range(n)); ax.set_yticks(range(n))
        # y-tick labels only on the left column; x-tick labels only on the bottom panel of each column
        if idx % ncol == 0:
            ax.set_yticklabels(labels, fontsize=17)
            ax.set_ylabel("teacher fine-tuned on task", fontsize=22, labelpad=5)
        else:
            ax.set_yticklabels([""] * n)
        if (idx + ncol) >= len(MODELS):  # bottom-most filled panel of this column
            ax.set_xticklabels(labels, rotation=90, fontsize=17)
            ax.set_xlabel("evaluation task", fontsize=22, labelpad=5)
        else:
            ax.set_xticklabels([""] * n)
        ax.tick_params(length=3, pad=1.5)
    # hide the unused 6th slot (bottom-right); the colorbar is drawn into that empty area
    for k in range(len(MODELS), len(axflat)):
        axflat[k].axis("off")
    fig.suptitle("Teacher cross-task transfer — five foundation models",
                 fontsize=34, fontweight="bold", y=0.995)
    fig.subplots_adjust(left=0.065, right=0.995, top=0.93, bottom=0.10, hspace=0.30, wspace=0.06)
    # Horizontal colorbar centered in the empty bottom-right (6th) block.
    cax = fig.add_axes([0.71, 0.26, 0.24, 0.028])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.set_label("transfer MCC", fontsize=26, labelpad=8); cb.ax.tick_params(labelsize=20)
    for ext in ("png", "pdf"):
        out = os.path.join(HERE, f"teacher_cross_task_5panel_2x3_nature.{ext}")
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print(f"SAVED {out}")
    plt.close(fig)


if __name__ == "__main__":
    five_panel_2x3()
