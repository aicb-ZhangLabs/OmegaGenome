"""Render the R1.9/R1.13 feature-space t-SNE figure from the npz produced by
feature_space_compare.py. Lets the PNG be drawn with a matplotlib-equipped python
(e.g. /usr/bin/python3) when the GPU venv lacks matplotlib."""
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

FIGS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figs")


def render_figure(Z_dkd, Z_omega, Z_students, no, nd,
                  cka_omega, cka_dkd, cos_omega_t, cos_dkd_t, cos_students, out_png):
    """3-panel t-SNE: (DKD vs teacher) | (OmegaGenome vs teacher) | (students together)."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    axes[0].scatter(Z_dkd[:nd, 0], Z_dkd[:nd, 1], s=14, c="#d62728", alpha=0.7, label="DKD student")
    axes[0].scatter(Z_dkd[nd:, 0], Z_dkd[nd:, 1], s=14, c="#000000", alpha=0.7, label="teacher (proj)")
    axes[0].set_title(f"DKD vs teacher\nCKA={cka_dkd:.3f}  cos={cos_dkd_t:.3f}")
    axes[0].legend(fontsize=9)
    axes[1].scatter(Z_omega[:no, 0], Z_omega[:no, 1], s=14, c="#2ca02c", alpha=0.7, label="OmegaGenome student")
    axes[1].scatter(Z_omega[no:, 0], Z_omega[no:, 1], s=14, c="#000000", alpha=0.7, label="teacher (proj)")
    axes[1].set_title(f"OmegaGenome vs teacher\nCKA={cka_omega:.3f}  cos={cos_omega_t:.3f}")
    axes[1].legend(fontsize=9)
    axes[2].scatter(Z_students[:no, 0], Z_students[:no, 1], s=14, c="#2ca02c", alpha=0.7, label="OmegaGenome")
    axes[2].scatter(Z_students[no:, 0], Z_students[no:, 1], s=14, c="#d62728", alpha=0.7, label="DKD")
    axes[2].set_title(f"students (64-d)\ncos(O,D)={cos_students:.3f}")
    axes[2].legend(fontsize=9)
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(
        "Feature-space preservation on splice_sites_all (penultimate features)\n"
        "OmegaGenome (feature-aligned mse=0.2) vs DKD vs Enformer teacher  —  R1.9 / R1.13",
        fontsize=13,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out_png, dpi=160, bbox_inches="tight")


def main():
    d = np.load(os.path.join(FIGS, "feature_space_data.npz"))
    no, nd = len(d["F_omega"]), len(d["F_dkd"])
    out_png = os.path.join(FIGS, "feature_space_splice_dkd_vs_omega_vs_teacher.png")
    render_figure(
        d["Z_dkd"], d["Z_omega"], d["Z_students"], no, nd,
        float(d["cka_omega"]), float(d["cka_dkd"]),
        float(d["cos_omega_t"]), float(d["cos_dkd_t"]), float(d["cos_students"]),
        out_png,
    )
    print(f"SAVED {out_png}")


if __name__ == "__main__":
    main()
