"""
Appendix figure: student-size scaling of regression distillation (random-init ladder).

Three-panel horizontal figure (1 row x 3 cols), log-x parameter axis, house Nature
style (matches fig6b_size_18task.py / fig6_method_size_hp.py).

  Panel A -- Joint 34-track mean Pearson vs student size, with (+KD, filled) and
             without (no-KD, hollow) the OmegaGenome distillation objective; the
             NTv3-650M teacher (0.606) is a dashed reference line, and the two
             pretrained-init from-scratch baselines of Table S6 are open triangles.
  Panel B -- Per-track ATAC specialist (ENCODE ENCSR325NFE): +KD / no-KD, plus the
             joint 34-track model's prediction of the SAME track (joint slice), and
             the teacher (0.832) dashed line.
  Panel C -- Distillation contribution Delta (= +KD - no-KD) vs size, for the joint
             task and the single-track specialist.

Data are seed-0 values verified from ntv3_finetune_result.json; hard-coded below
(single source of truth for this figure). Saves:
  output/fig_sizeladder.pdf         (clean vector, bbox_inches='tight')
  output/fig_sizeladder.png @300dpi (bbox_inches='tight')
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------- #
# House Nature style (identical rcParams / palette to fig6b_size_18task.py)    #
# --------------------------------------------------------------------------- #
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "axes.linewidth": 1.1,
})

NAT = {
    "grey_dot": "#8A9099", "grey_base": "#C3C8CE",
    "blue_dark": "#356491", "blue_mid": "#6E97BC", "blue_light": "#A9C6E0",
    "teal": "#37917F", "slate": "#9AA4AE", "slate_deep": "#7C8894",
    "ink": "#222222", "grid": "#DBDEE2",
}
HERO = NAT["blue_dark"]          # +KD / hero
SEC = NAT["grey_dot"]            # no-KD / secondary
SLICE = NAT["blue_mid"]          # joint-model slice (Panel B)
TEAL = NAT["teal"]              # per-track second line (Panel C)
TEACH = "#444444"               # teacher dashed line (neutral dark)
SPINE = "#444444"
TITLE_FS, LAB_FS, TICK_FS, ANN_FS = 15.5, 13, 11.5, 11

# --------------------------------------------------------------------------- #
# Data (seed 0, verified from result.json -- single source of truth)          #
# --------------------------------------------------------------------------- #
params_M = np.array([4.34, 7.69, 29.87, 106.46, 303.05])   # x-axis (log)
XLABELS = ["4M", "8M", "30M", "100M", "300M"]

# Panel A -- Joint 34-track (mean Pearson)
jointKD = np.array([0.4606, 0.4769, 0.4958, 0.5188, 0.5309])
jointNoKD = np.array([0.4349, 0.4483, 0.4681, 0.4848, 0.4969])
joint_teacher = 0.606
pretrained_baselines = {7.69: 0.475, 106.46: 0.518}         # Table S6 open triangles

# Panel B -- Per-track ATAC specialist (ENCSR325NFE)
specKD = np.array([0.7322, 0.7362, 0.7521, 0.7631, 0.7681])
specNoKD = np.array([0.7160, 0.7183, 0.7396, 0.7448, 0.7562])
joint_slice = np.array([0.6882, 0.7060, 0.7255, 0.7472, 0.7585])
spec_teacher = 0.832

# Panel C -- KD contribution Delta (= +KD - no-KD)
jointDelta = np.array([0.0257, 0.0286, 0.0277, 0.0341, 0.0341])
specDelta = np.array([0.0162, 0.0179, 0.0126, 0.0183, 0.0119])


def style_axis(ax, title, ylabel):
    """Apply the shared house style to one panel (log-x, y-grid, spines, ticks).

    Sets the log parameter axis with size-labelled ticks ("4M".."300M"),
    the horizontal-only grid, the #444444 spines, and the title / axis labels
    at the house font sizes. Returns nothing; mutates ``ax`` in place.
    """
    ax.set_xscale("log")
    ax.set_xlim(params_M[0] * 0.72, params_M[-1] * 1.55)
    ax.set_xticks(params_M)
    ax.set_xticklabels(XLABELS, fontsize=TICK_FS, color=NAT["ink"])
    ax.minorticks_off()
    ax.set_xlabel("Student parameters", fontsize=LAB_FS, color=NAT["ink"])
    ax.set_ylabel(ylabel, fontsize=LAB_FS, color=NAT["ink"])
    ax.set_title(title, fontsize=TITLE_FS, pad=10, color=NAT["ink"])
    ax.tick_params(axis="both", labelsize=TICK_FS, length=4, width=1.0)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color=NAT["grid"], lw=0.8, zorder=0)
    for sp in ax.spines.values():
        sp.set_edgecolor(SPINE)
        sp.set_linewidth(1.1)


def hero_line(ax, x, y, color, label, ms=9, lw=2.2, zorder=5):
    """Plot a filled-marker 'hero' series (solid line, colored fill = colored edge)."""
    ax.plot(x, y, "-o", color=color, mfc=color, mec=color, ms=ms, mew=1.6,
            lw=lw, label=label, zorder=zorder)


def open_line(ax, x, y, color, label, marker="o", ms=8, lw=2.0, zorder=4):
    """Plot a hollow-marker series (white face, colored edge) -- the 'no-KD' idiom."""
    ax.plot(x, y, "-", color=color, lw=lw, zorder=zorder, label=label)
    ax.plot(x, y, marker, mfc="white", mec=color, ms=ms, mew=1.7,
            lw=0, zorder=zorder + 0.5)


def tag(ax, letter):
    """Bold per-panel letter tag (A/B/C) at the top-left corner."""
    ax.text(0.018, 0.975, letter, transform=ax.transAxes, ha="left", va="top",
            fontsize=17, fontweight="bold", color=NAT["ink"])


# --------------------------------------------------------------------------- #
# Figure                                                                      #
# --------------------------------------------------------------------------- #
fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(15.5, 5.2))

# ---- Panel A: Joint 34-track ---------------------------------------------- #
hero_line(axA, params_M, jointKD, HERO, r"$+$KD (distilled)")
open_line(axA, params_M, jointNoKD, SEC, "no-KD (from scratch)")
# teacher dashed reference
axA.axhline(joint_teacher, ls="--", lw=1.6, color=TEACH, zorder=2)
axA.text(params_M[-1] * 1.02, joint_teacher + 0.003, "NTv3-650M teacher (0.606)",
         ha="right", va="bottom", fontsize=10, color=TEACH)
# pretrained-init from-scratch baselines (Table S6) as open triangles
pk = sorted(pretrained_baselines.keys())
pv = [pretrained_baselines[k] for k in pk]
axA.plot(pk, pv, "^", mfc="white", mec=NAT["ink"], ms=11, mew=1.6, lw=0,
         zorder=6, label="Pretrained-init (Table S6)")
axA.annotate("Table S6", xy=(pk[1], pv[1]), xytext=(pk[1] * 0.60, pv[1] - 0.028),
             ha="center", va="top", fontsize=ANN_FS, color=NAT["ink"],
             arrowprops=dict(arrowstyle="-", color=NAT["ink"], lw=1.0))
axA.set_ylim(0.425, 0.628)
style_axis(axA, "Joint 34-track", "Test Pearson r")
axA.legend(loc="lower right", frameon=False, fontsize=10)
tag(axA, "A")

# ---- Panel B: Per-track ATAC specialist ----------------------------------- #
# joint-model slice first (reference, sits behind the specialist)
axB.plot(params_M, joint_slice, "--s", color=SLICE, mfc=SLICE, mec=SLICE,
         ms=7, mew=1.2, lw=1.8, zorder=3, label="Joint-model slice")
hero_line(axB, params_M, specKD, HERO, r"$+$KD (distilled)")
open_line(axB, params_M, specNoKD, SEC, "no-KD (from scratch)")
axB.axhline(spec_teacher, ls="--", lw=1.6, color=TEACH, zorder=2)
axB.text(params_M[-1] * 1.02, spec_teacher + 0.003, "NTv3-650M teacher (0.832)",
         ha="right", va="bottom", fontsize=10, color=TEACH)
axB.set_ylim(0.660, 0.855)
style_axis(axB, "Per-track ATAC specialist (ENCSR325NFE)", "Test Pearson r")
axB.legend(loc="lower right", frameon=False, fontsize=10)
tag(axB, "B")

# ---- Panel C: Distillation gain Delta ------------------------------------- #
hero_line(axC, params_M, jointDelta, HERO, "Joint 34-track")
hero_line(axC, params_M, specDelta, TEAL, "Per-track ATAC")
axC.set_ylim(0.005, 0.039)
style_axis(axC, r"Distillation gain $\Delta$ vs. size",
           r"$\Delta$ Pearson ($+$KD $-$ no-KD)")
axC.legend(loc="center left", frameon=False, fontsize=10)
tag(axC, "C")

fig.tight_layout()

# --------------------------------------------------------------------------- #
# Save into paper_figures/output (derived from this file's location)           #
# --------------------------------------------------------------------------- #
HERE = os.path.dirname(os.path.abspath(__file__))
FIGS = os.path.abspath(os.path.join(HERE, "..", "..", "output"))
os.makedirs(FIGS, exist_ok=True)
PDF = os.path.join(FIGS, "fig_sizeladder.pdf")
PNG = os.path.join(FIGS, "fig_sizeladder.png")
fig.savefig(PDF, bbox_inches="tight")
fig.savefig(PNG, dpi=300, bbox_inches="tight")
print("saved:", PDF)
print("saved:", PNG, "(300 dpi)")

# monotonicity sanity check (console only)
for name, arr in [("jointKD", jointKD), ("jointNoKD", jointNoKD),
                  ("specKD", specKD), ("specNoKD", specNoKD)]:
    mono = bool(np.all(np.diff(arr) >= -1e-9))
    print("  monotonic non-decreasing %-10s %s" % (name, mono))
