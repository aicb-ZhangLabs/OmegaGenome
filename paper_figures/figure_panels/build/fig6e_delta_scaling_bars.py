"""
Fig 6E -- per-task Delta(3.6M - 0.1M): who benefits from scaling past deployed size (seed0).

Horizontal bars, one per task, Delta MCC = best-test MCC at the largest student
(3.6M) minus at the deployed student (0.1M).  Bars coloured by family, sorted by
Delta.  A vertical zero line marks "no gain past the deployed size".  The story:
most tasks are ~flat (small |Delta|) -- only the harder splice tasks keep gaining --
so the deployed 0.1M student already captures most of the attainable quality.

Data = _size18_data.assemble() (identical provenance to Fig 6B).
Single-seed (seed0).
Saves output/fig6e_delta_scaling_bars.{pdf,png} @300dpi + previews/ copy.
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
FAM_COLOR = {"histone": NAT["blue_dark"], "promoter": NAT["green_dark"],
             "enhancer": NAT["amber_dark"], "splice": NAT["purple_dark"]}
FAM_NICE = {"histone": "Histone", "promoter": "Promoter",
            "enhancer": "Enhancer", "splice": "Splice"}
SPINE = "#555555"

HERE = os.path.dirname(os.path.abspath(__file__))
os.makedirs(os.path.join(HERE, "output"), exist_ok=True)

M, TASKS, SIZES = D.assemble(write_csv=False)

# Delta = MCC(3.6M) - MCC(0.1M) per task
deltas = [(t, float(M[t]["xxlarge"] - M[t]["original"])) for t in TASKS]
deltas.sort(key=lambda kv: kv[1])          # ascending -> largest gain at top
tasks = [t for t, _ in deltas]
dvals = np.array([d for _, d in deltas])
y = np.arange(len(tasks))
colors = [FAM_COLOR[D.TASK_FAMILY[t]] for t in tasks]

fig, ax = plt.subplots(figsize=(7.6, 8.2))
ax.barh(y, dvals, color=colors, height=0.68, zorder=3,
        edgecolor="white", linewidth=0.6)
ax.axvline(0, color=SPINE, lw=1.2, zorder=4)

# value labels at bar ends
for yi, d in zip(y, dvals):
    ax.text(d + (0.004 if d >= 0 else -0.004), yi, f"{d:+.3f}",
            ha="left" if d >= 0 else "right", va="center", fontsize=9.2,
            color=NAT["ink"])

ax.set_yticks(y)
ylabs = ax.set_yticklabels([D.TASK_LABEL[t] for t in tasks], fontsize=10.5)
for lab, t in zip(ylabs, tasks):
    lab.set_color(FAM_COLOR[D.TASK_FAMILY[t]])
ax.set_ylim(-0.7, len(tasks) - 0.3)

ax.set_xlabel("$\\Delta$ MCC  (3.6M $-$ deployed 0.1M)", fontsize=12.5,
              color=NAT["ink"])
ax.set_title("Does scaling past the deployed size help?  (per task, seed0)",
             fontsize=14.5, color=NAT["ink"], pad=10)
xpad = 0.03
ax.set_xlim(min(dvals.min(), 0) - xpad, max(dvals.max(), 0) + xpad + 0.015)
ax.tick_params(axis="x", labelsize=10, length=4)
ax.grid(axis="x", color=NAT["grid"], lw=0.8, zorder=0)
ax.set_axisbelow(True)
for sp in ax.spines.values():
    sp.set_edgecolor(SPINE); sp.set_linewidth(1.0)

# annotate the "flat" story
ax.text(0.006, 0.5, "deployed 0.1M\nalready sufficient", transform=ax.transData,
        rotation=90, ha="left", va="center", fontsize=9.0, style="italic",
        color="#7A808A", clip_on=False)

# family legend
handles = [Line2D([0], [0], marker="s", color="none", mfc=FAM_COLOR[f],
                  mec="none", ms=11, label=FAM_NICE[f]) for f in D.FAMILY_ORDER]
ax.legend(handles=handles, loc="lower right", frameon=False, fontsize=10.5,
          title="Task family", title_fontsize=10.5, handletextpad=0.4)

fig.text(0.99, 0.008,
         "single-seed (seed 0)",
         ha="right", va="bottom", fontsize=8.8, style="italic", color="#7A808A")

fig.tight_layout(rect=[0, 0.02, 1, 1])
pdf = os.path.join(HERE, "output", "fig6e_delta_scaling_bars.pdf")
png = os.path.join(HERE, "output", "fig6e_delta_scaling_bars.png")
fig.savefig(pdf, bbox_inches="tight")
fig.savefig(png, dpi=300, bbox_inches="tight")
shutil.copy(png, os.path.join(HERE, "..", "fig6e_delta_scaling_bars.png"))
print("saved:", png)
# quick console read
print("median |Delta| =", float(np.median(np.abs(dvals))),
      " tasks with |Delta|<0.02:", int(np.sum(np.abs(dvals) < 0.02)), "/", len(dvals))
