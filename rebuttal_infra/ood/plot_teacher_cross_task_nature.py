#!/usr/bin/env python
"""R2.3 TEACHER cross-task OOD — Nature-style renders, matching the student cross-task figures.

Reuses the exact style helpers from ``plot_cross_task_nature.py`` (mako MCC colormap, short task
labels, leading-zero-stripped annotations) so the teacher-OOD figures read as one set with the
existing student-OOD figures. Pure plotting from the saved MCC matrix CSVs (no GPU).

Outputs (in this dir):
  * teacher_cross_task_matrix_<model>_nature.png  — one annotated 18x18 heatmap per foundation model
    (each fine-tuned teacher-A evaluated on every compatible task-B; blank = label-incompatible head).
  * teacher_cross_task_5panel_nature.png/.pdf     — all five teachers side by side (shared mako 0..1
    scale, one colorbar): the headline figure showing every foundation model is strong in-task
    (bright diagonal) and drops sharply off-task.

Run: /home/pengchx3/.conda/envs/enformer/bin/python plot_teacher_cross_task_nature.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Reuse the established style + IO helpers verbatim (same dir).
from plot_cross_task_nature import (
    read_matrix, short_labels, _nolead, MCC_CMAP_OBJ, MCC_TEXT_DARK_ABOVE,
)

HERE = os.path.dirname(os.path.abspath(__file__))
# Order + display names: two billion-scale gLMs first, then the smaller teachers.
MODELS = [
    ("nt", "NT-2.5B"), ("carbon", "Carbon-3B"), ("dnabert2", "DNABERT-2"),
    ("enformer", "Enformer"), ("caduceus", "Caduceus"),
]


def _csv(model):
    return os.path.join(HERE, f"teacher_cross_task_matrix_{model}.csv")


def annotated_heatmap(model, disp):
    """One annotated 18x18 teacher heatmap, styled identically to the student nature heatmap."""
    tasks, mat = read_matrix(_csv(model))
    n = len(tasks); labels = short_labels(tasks)
    masked = np.ma.masked_invalid(mat)
    cmap = MCC_CMAP_OBJ.copy(); cmap.set_bad(color="white")

    fig, ax = plt.subplots(figsize=(18.0, 16.0))
    im = ax.imshow(masked, cmap=cmap, vmin=0.0, vmax=1.0, aspect="equal")
    ax.set_xticks(range(n)); ax.set_xticklabels(labels, rotation=90, fontsize=16)
    ax.set_yticks(range(n)); ax.set_yticklabels(labels, fontsize=16)
    ax.set_xlabel("evaluation task", fontsize=22, labelpad=12)
    ax.set_ylabel("teacher fine-tuned on task", fontsize=22, labelpad=12)
    ax.set_title(f"Teacher cross-task transfer — {disp} fine-tuned teachers (R2.3)\n"
                 "diagonal = in-task; off-diagonal = transfer; blank = label-incompatible (multiclass)",
                 fontsize=24, pad=18)
    for i in range(n):
        for j in range(n):
            if not np.isnan(mat[i, j]):
                c = "#222222" if mat[i, j] >= MCC_TEXT_DARK_ABOVE else "white"
                ax.text(j, i, _nolead(mat[i, j]), ha="center", va="center", fontsize=12.5, color=c)
    cb = fig.colorbar(im, ax=ax, fraction=0.030, pad=0.02)
    cb.set_label("transfer MCC", fontsize=22); cb.ax.tick_params(labelsize=16)
    ax.tick_params(length=4); fig.tight_layout()
    out = os.path.join(HERE, f"teacher_cross_task_matrix_{model}_nature.png")
    fig.savefig(out, dpi=300, bbox_inches="tight"); plt.close(fig)
    print(f"SAVED {out}")


def five_panel():
    """All five foundation models side by side (shared mako 0..1 scale, one colorbar)."""
    fig, axes = plt.subplots(1, 5, figsize=(34.0, 8.2))
    labels = None
    im = None
    for ax, (model, disp) in zip(axes, MODELS):
        tasks, mat = read_matrix(_csv(model))
        n = len(tasks)
        if labels is None:
            labels = short_labels(tasks)
        cmap = MCC_CMAP_OBJ.copy(); cmap.set_bad(color="white")
        im = ax.imshow(np.ma.masked_invalid(mat), cmap=cmap, vmin=0.0, vmax=1.0, aspect="equal")
        ax.set_title(disp, fontsize=24, fontweight="bold", pad=10)
        ax.set_xticks(range(n)); ax.set_xticklabels(labels, rotation=90, fontsize=10)
        ax.set_yticks(range(n))
        ax.set_yticklabels(labels if ax is axes[0] else [""] * n, fontsize=10)
        ax.set_xlabel("evaluation task", fontsize=15, labelpad=8)
        if ax is axes[0]:
            ax.set_ylabel("teacher fine-tuned on task", fontsize=15, labelpad=8)
        ax.tick_params(length=3)
    fig.suptitle("Teacher cross-task transfer across five foundation models (R2.3): "
                 "strong in-task (bright diagonal), sharp off-task drop",
                 fontsize=26, fontweight="bold", y=1.02)
    cb = fig.colorbar(im, ax=axes, fraction=0.012, pad=0.01)
    cb.set_label("transfer MCC", fontsize=18); cb.ax.tick_params(labelsize=13)
    for ext in ("png", "pdf"):
        out = os.path.join(HERE, f"teacher_cross_task_5panel_nature.{ext}")
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print(f"SAVED {out}")
    plt.close(fig)


if __name__ == "__main__":
    for model, disp in MODELS:
        annotated_heatmap(model, disp)
    five_panel()
