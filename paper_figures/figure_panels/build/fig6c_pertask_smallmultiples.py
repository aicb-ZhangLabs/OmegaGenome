"""
Fig 6C -- per-task student-size scaling small multiples (seed0).

18-panel grid (3 rows x 6 cols), one mini size-scaling curve per task:
MCC vs student parameter count (log-x, shared), per-panel y with clean ticks so
each task's own scaling shape is legible.  Tasks are grouped/coloured by family
(histone / promoter / enhancer / splice).  The deployed 0.1M operating point is
marked (filled hero marker) on every panel.

Data = _size18_data.assemble() (identical provenance to Fig 6B).
Single-seed (seed0).
Saves output/fig6c_pertask_smallmultiples.{pdf,png} @300dpi + previews/ copy.
"""
import os
import shutil
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import _size18_data as D

os.makedirs("output", exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "axes.linewidth": 1.0,
})

NAT = {"blue_dark": "#356491", "green_dark": "#4F8C6B", "amber_dark": "#CE8B3B",
       "purple_dark": "#7C619E", "ink": "#222222", "grid": "#DBDEE2"}
SPINE = "#555555"
FAM_COLOR = {"histone": NAT["blue_dark"], "promoter": NAT["green_dark"],
             "enhancer": NAT["amber_dark"], "splice": NAT["purple_dark"]}
FAM_TINT = {"histone": "#F2F6FA", "promoter": "#F1F7F3",
            "enhancer": "#FBF5EC", "splice": "#F6F3F9"}
FAM_NICE = {"histone": "Histone marks", "promoter": "Promoter",
            "enhancer": "Enhancer", "splice": "Splice site"}

HERE = os.path.dirname(os.path.abspath(__file__))
os.makedirs(os.path.join(HERE, "output"), exist_ok=True)

M, TASKS, SIZES = D.assemble(write_csv=False)
x = np.array([D.PARAMS[s] for s in SIZES])
di = SIZES.index("original")   # deployed 0.1M index

NROW, NCOL = 3, 6
fig, axes = plt.subplots(NROW, NCOL, figsize=(15.6, 8.0), sharex=True)

for k, t in enumerate(TASKS):
    r, c = divmod(k, NCOL)
    ax = axes[r, c]
    fam = D.TASK_FAMILY[t]
    col = FAM_COLOR[fam]
    ax.set_facecolor(FAM_TINT[fam])
    y = np.array([M[t][s] for s in SIZES], dtype=float)

    ax.plot(x, y, "-", color=col, lw=2.0, zorder=3)
    # non-deployed points: open circles
    for i, s in enumerate(SIZES):
        if i == di:
            continue
        ax.plot(x[i], y[i], "o", ms=5.0, mfc="white", mec=col, mew=1.4, zorder=4)
    # deployed 0.1M hero point
    ax.plot(x[di], y[di], "o", ms=8.5, mfc=col, mec="white", mew=1.4, zorder=6)

    # per-panel y-range with padding + 3 clean ticks
    lo, hi = float(np.nanmin(y)), float(np.nanmax(y))
    pad = max(0.02, 0.16 * (hi - lo))
    y0, y1 = lo - pad, hi + pad
    ax.set_ylim(y0, y1)
    ticks = np.round(np.linspace(lo, hi, 3), 2)
    ax.set_yticks(ticks)
    ax.tick_params(axis="y", labelsize=8.5, length=2.5, width=0.8, pad=1.5)
    ax.tick_params(axis="x", length=2.5, width=0.8)

    ax.set_title(D.TASK_LABEL[t], fontsize=11, color=NAT["ink"], pad=3.5)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color=NAT["grid"], lw=0.6, zorder=0)
    for sp in ax.spines.values():
        sp.set_edgecolor(SPINE); sp.set_linewidth(0.9)

# shared log-x: set scale on every panel FIRST, then custom ticks last so the
# FixedFormatter is not reset by a later set_xscale (sharex -> set once).
for a in axes.flat:
    a.set_xscale("log")
    a.minorticks_off()
axes[0, 0].set_xticks(x)
axes[0, 0].set_xticklabels([D.LABEL[s] for s in SIZES])
for c in range(NCOL):
    for lab in axes[NROW - 1, c].get_xticklabels():
        lab.set(fontsize=8.0, color=NAT["ink"], rotation=45,
                ha="right", rotation_mode="anchor")

# figure-level labels
fig.supxlabel("Student model size (parameters, log scale)", fontsize=13,
              color=NAT["ink"], y=0.045)
fig.supylabel("Best-test MCC", fontsize=13, color=NAT["ink"], x=0.006)
fig.suptitle("Per-task student-size scaling  (18 tasks, seed0)", fontsize=17,
             color=NAT["ink"], y=0.995)

# family legend + deployed marker legend
fam_handles = [Line2D([0], [0], color=FAM_COLOR[f], lw=2.6, label=FAM_NICE[f])
               for f in D.FAMILY_ORDER]
fam_handles.append(Line2D([0], [0], marker="o", color="none", mfc="#555555",
                          mec="white", mew=1.2, ms=9, label="deployed (0.1M)"))
fig.legend(handles=fam_handles, loc="upper center", ncol=5, frameon=False,
           fontsize=11, bbox_to_anchor=(0.5, 0.955),
           columnspacing=1.8, handletextpad=0.6)

# single-seed note
fig.text(0.994, 0.012,
         "single-seed (seed 0); per-panel y-scale",
         ha="right", va="bottom", fontsize=9.0, style="italic", color="#7A808A")

fig.tight_layout(rect=[0.018, 0.055, 1, 0.925])
pdf = os.path.join(HERE, "output", "fig6c_pertask_smallmultiples.pdf")
png = os.path.join(HERE, "output", "fig6c_pertask_smallmultiples.png")
fig.savefig(pdf, bbox_inches="tight")
fig.savefig(png, dpi=300, bbox_inches="tight")
shutil.copy(png, os.path.join(HERE, "..", "fig6c_pertask_smallmultiples.png"))
print("saved:", png)
