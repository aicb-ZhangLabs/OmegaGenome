"""
Build the compact 3-panel feature-space figure (Fig. S: DKD vs OmegaGenome vs teacher)
in the paper's Nature / muted-blue palette, as a VECTOR PDF.

Layout (1x3), penultimate-feature t-SNE on splice_sites_all:
  A = DKD student vs Enformer teacher (projected)      -> detaches, CKA=0.412
  B = OmegaGenome student vs teacher (projected)       -> overlays,  CKA=0.539
  C = both students in the shared 64-d space           -> cos(O,D)=0.912

Coordinates and metrics come straight from the npz produced by
analysis/feature_space_compare.py (Z_dkd / Z_omega / Z_students already t-SNE'd, plus the
CKA / cosine scalars) -- no re-embedding, no fabricated coordinates.

Palette matches Fig 2/4/6 (NAT dict): teacher grey-blue, OmegaGenome blue #356491,
DKD amber. Run with the biolaysum env python.
Saves output/fig_s_dkd_feature_space.pdf (vector) and copies it to
../../paper_src/figs/fig_dkd_feature_space.pdf .
"""
import os
import shutil
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "axes.linewidth": 1.1,
})

# Nature palette (identical to Fig 2/4/6 NAT dict).
NAT = {
    "blue_dark":  "#356491",   # OmegaGenome hero
    "blue_light": "#A9C6E0",   # teacher
    "amber_dark": "#CE8B3B",   # DKD
    "grey_teach": "#8A9099",   # teacher (projected) neutral
    "ink":        "#222222",
}
OMEGA = NAT["blue_dark"]
DKD   = NAT["amber_dark"]
TEACH = NAT["grey_teach"]
SPINE = "#444444"

HERE = os.path.dirname(os.path.abspath(__file__))
# Produced by ../../../analysis/feature_space_compare.py (GPU; needs the trained checkpoints).
NPZ = os.path.join(HERE, "..", "..", "..", "analysis", "figs", "feature_space_data.npz")
OUT_DIR = os.path.join(HERE, "..", "..", "output")
PAPER_FIG = os.path.join(OUT_DIR, "fig_dkd_feature_space.pdf")


def style_ax(ax):
    for s in ax.spines.values():
        s.set_edgecolor(SPINE)
        s.set_linewidth(1.1)
    ax.set_xticks([])
    ax.set_yticks([])


def main():
    d = np.load(NPZ)
    no, nd = len(d["F_omega"]), len(d["F_dkd"])
    Z_dkd, Z_omega, Z_students = d["Z_dkd"], d["Z_omega"], d["Z_students"]
    cka_omega, cka_dkd = float(d["cka_omega"]), float(d["cka_dkd"])
    cos_omega_t, cos_dkd_t = float(d["cos_omega_t"]), float(d["cos_dkd_t"])
    cos_students = float(d["cos_students"])

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4))
    S, A = 15, 0.72

    # Panel A: DKD student vs teacher (projected).
    axes[0].scatter(Z_dkd[nd:, 0], Z_dkd[nd:, 1], s=S, c=TEACH, alpha=A,
                    edgecolors="none", label="Enformer teacher (proj.)")
    axes[0].scatter(Z_dkd[:nd, 0], Z_dkd[:nd, 1], s=S, c=DKD, alpha=A,
                    edgecolors="none", label="DKD student")
    axes[0].set_title(f"DKD vs teacher\nCKA = {cka_dkd:.3f}",
                      fontsize=12.5, color=NAT["ink"])

    # Panel B: OmegaGenome student vs teacher (projected).
    axes[1].scatter(Z_omega[no:, 0], Z_omega[no:, 1], s=S, c=TEACH, alpha=A,
                    edgecolors="none", label="Enformer teacher (proj.)")
    axes[1].scatter(Z_omega[:no, 0], Z_omega[:no, 1], s=S, c=OMEGA, alpha=A,
                    edgecolors="none", label="OmegaGenome student")
    axes[1].set_title(f"OmegaGenome vs teacher\nCKA = {cka_omega:.3f}",
                      fontsize=12.5, color=NAT["ink"])

    # Panel C: both students in the shared 64-d space.
    axes[2].scatter(Z_students[:no, 0], Z_students[:no, 1], s=S, c=OMEGA, alpha=A,
                    edgecolors="none", label="OmegaGenome")
    axes[2].scatter(Z_students[no:, 0], Z_students[no:, 1], s=S, c=DKD, alpha=A,
                    edgecolors="none", label="DKD")
    axes[2].set_title(f"students (shared 64-d)\ncos(O, D) = {cos_students:.3f}",
                      fontsize=12.5, color=NAT["ink"])

    for ax in axes:
        style_ax(ax)

    handles = [
        Line2D([0], [0], marker="o", ls="", mfc=TEACH, mec="none", ms=8,
               label="Enformer teacher"),
        Line2D([0], [0], marker="o", ls="", mfc=OMEGA, mec="none", ms=8,
               label="OmegaGenome student"),
        Line2D([0], [0], marker="o", ls="", mfc=DKD, mec="none", ms=8,
               label="DKD student"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               fontsize=11, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=[0, 0.06, 1, 1])

    os.makedirs(OUT_DIR, exist_ok=True)
    out_pdf = os.path.join(OUT_DIR, "fig_s_dkd_feature_space.pdf")
    fig.savefig(out_pdf, bbox_inches="tight")
    shutil.copyfile(out_pdf, PAPER_FIG)
    print(f"SAVED {out_pdf}")
    print(f"COPIED -> {PAPER_FIG}")
    print(f"metrics: CKA_omega={cka_omega:.4f} CKA_dkd={cka_dkd:.4f} "
          f"cos_omega_t={cos_omega_t:.4f} cos_dkd_t={cos_dkd_t:.4f} "
          f"cos_students={cos_students:.4f}")


if __name__ == "__main__":
    main()
