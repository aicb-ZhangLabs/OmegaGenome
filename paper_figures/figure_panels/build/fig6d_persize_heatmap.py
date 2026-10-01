"""
Fig 6D -- per-size view: 18-task x 7-size MCC heatmap (seed0).

Rows = 18 tasks (grouped by family; within family sorted by mean MCC), cols = 7
student sizes (2K..3.6M).  Cell = best-test MCC, sequential mako colormap,
annotated.  A separated bottom row gives the 18-task mean per size, and a top
marginal line panel plots that same per-size mean curve (0.521 -> 0.629 -> 0.638)
so "each size" is legible at a glance.  Family blocks are separated and the task
labels are coloured by family.

Data = _size18_data.assemble() (identical provenance to Fig 6B).
Single-seed (seed0); band/spread = task-to-task s.d. over the 18 tasks.
Saves output/fig6d_persize_heatmap.{pdf,png} @300dpi + previews/ copy.
"""
import os
import shutil
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

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
MAKO = sns.color_palette("mako", as_cmap=True)
SPINE = "#555555"

HERE = os.path.dirname(os.path.abspath(__file__))
os.makedirs(os.path.join(HERE, "output"), exist_ok=True)

M, TASKS, SIZES = D.assemble(write_csv=False)

# within-family sort by mean MCC (desc) for a clean gradient; keep family blocks
def task_mean(t):
    return float(np.nanmean([M[t][s] for s in SIZES]))
ordered = []
fam_bounds = []   # (family, start_row, end_row) in display order (top->bottom)
row = 0
for fam in D.FAMILY_ORDER:
    fts = sorted([t for t in D.FAMILY_TASKS[fam] if t in M],
                 key=task_mean, reverse=True)
    fam_bounds.append((fam, row, row + len(fts)))
    ordered += fts
    row += len(fts)

mc, sd = D.mean_curve(M, TASKS)     # per-size 18-task mean + task s.d.

# data matrix (tasks rows) + mean row appended
Z = np.array([[M[t][s] for s in SIZES] for t in ordered], dtype=float)
mean_row = mc.reshape(1, -1)
nT, nS = Z.shape

vmin = float(np.nanmin([Z.min(), mean_row.min()]))
vmax = float(np.nanmax([Z.max(), mean_row.max()]))

def txt_color(v):
    # white text on dark (low-normalized mako) cells, dark on light cells
    return "white" if (v - vmin) / (vmax - vmin) < 0.55 else NAT["ink"]

# ---- layout: top marginal mean-curve panel + main heatmap ------------------
fig = plt.figure(figsize=(8.6, 11.8))
gs = gridspec.GridSpec(3, 1, height_ratios=[1.15, nT, 1.25], hspace=0.07,
                       left=0.255, right=0.87, top=0.935, bottom=0.085)
ax_top = fig.add_subplot(gs[0])
ax_hm = fig.add_subplot(gs[1])
ax_mean = fig.add_subplot(gs[2])

xc = np.arange(nS) + 0.5     # column centres

# ---- top marginal: per-size 18-task mean curve -----------------------------
ax_top.plot(xc, mc, "-", color=NAT["blue_dark"], lw=2.2, zorder=3)
di = SIZES.index("original")
for i in range(nS):
    hero = (i == di)
    ax_top.plot(xc[i], mc[i], "o", ms=10 if hero else 6.5,
                mfc=NAT["blue_dark"] if hero else "white",
                mec=NAT["blue_dark"], mew=1.8 if hero else 1.5, zorder=5)
    ax_top.text(xc[i], mc[i] + 0.012, f"{mc[i]:.3f}", ha="center", va="bottom",
                fontsize=8.6, fontweight="bold" if hero else "normal",
                color=NAT["ink"])
ax_top.set_xlim(0, nS)
ax_top.set_ylim(mc.min() - 0.03, mc.max() + 0.055)
ax_top.set_ylabel("18-task\nmean MCC", fontsize=10.5, color=NAT["ink"])
ax_top.tick_params(axis="y", labelsize=8.5, length=3)
ax_top.set_xticks([])
ax_top.set_title("Per-size 18-task mean  (marks deployed 0.1M)",
                 fontsize=11, color=NAT["ink"], pad=4)
ax_top.grid(axis="y", color=NAT["grid"], lw=0.7)
ax_top.set_axisbelow(True)
for sp in ax_top.spines.values():
    sp.set_edgecolor(SPINE); sp.set_linewidth(0.9)

# ---- main heatmap ----------------------------------------------------------
im = ax_hm.imshow(Z, aspect="auto", cmap=MAKO, vmin=vmin, vmax=vmax,
                  extent=[0, nS, nT, 0])
for i in range(nT):
    for j in range(nS):
        v = Z[i, j]
        ax_hm.text(j + 0.5, i + 0.5, f"{v:.2f}", ha="center", va="center",
                   fontsize=8.2, color=txt_color(v))
# family block separators
for _, _, end in fam_bounds[:-1]:
    ax_hm.axhline(end, color="white", lw=2.4)
# deployed-column highlight (0.1M)
ax_hm.add_patch(plt.Rectangle((di, 0), 1, nT, fill=False, ec="#E8B04B",
                              lw=2.4, zorder=6))

ax_hm.set_yticks(np.arange(nT) + 0.5)
ylabs = ax_hm.set_yticklabels([D.TASK_LABEL[t] for t in ordered], fontsize=9.2)
for lab, t in zip(ylabs, ordered):
    lab.set_color(FAM_COLOR[D.TASK_FAMILY[t]])
ax_hm.set_xticks([])
ax_hm.tick_params(length=0)
# family bracket labels on far left (axes-fraction x, data-y)
for fam, a, b in fam_bounds:
    ax_hm.text(-0.235, (a + b) / 2, FAM_NICE[fam], rotation=90,
               ha="center", va="center", fontsize=11, fontweight="bold",
               color=FAM_COLOR[fam], transform=ax_hm.get_yaxis_transform(),
               clip_on=False)
for sp in ax_hm.spines.values():
    sp.set_edgecolor(SPINE); sp.set_linewidth(0.9)

# ---- bottom: 18-task mean row as its own heatmap strip ----------------------
ax_mean.imshow(mean_row, aspect="auto", cmap=MAKO, vmin=vmin, vmax=vmax,
               extent=[0, nS, 1, 0])
for j in range(nS):
    ax_mean.text(j + 0.5, 0.5, f"{mc[j]:.3f}", ha="center", va="center",
                 fontsize=9.0, fontweight="bold", color=txt_color(mc[j]))
ax_mean.add_patch(plt.Rectangle((di, 0), 1, 1, fill=False, ec="#E8B04B",
                                lw=2.4, zorder=6))
ax_mean.set_yticks([0.5])
ax_mean.set_yticklabels(["18-task mean"], fontsize=9.8, fontweight="bold")
ax_mean.set_xticks(xc)
ax_mean.set_xticklabels([D.LABEL[s] for s in SIZES], fontsize=10, color=NAT["ink"])
ax_mean.tick_params(length=0)
ax_mean.set_xlabel("Student model size (parameters)", fontsize=12,
                   color=NAT["ink"], labelpad=8)
for sp in ax_mean.spines.values():
    sp.set_edgecolor(SPINE); sp.set_linewidth(0.9)

# colorbar
cax = fig.add_axes([0.885, 0.22, 0.022, 0.46])
cb = fig.colorbar(im, cax=cax)
cb.set_label("Best-test MCC", fontsize=11, color=NAT["ink"])
cb.ax.tick_params(labelsize=8.5)
cb.outline.set_edgecolor(SPINE)

fig.suptitle("Per-size student scaling across 18 tasks  (seed0)",
             fontsize=16.5, color=NAT["ink"], y=0.99)
fig.text(0.255, 0.018,
         "single-seed (seed 0); gold outline = deployed 0.1M",
         ha="left", va="bottom", fontsize=8.8, style="italic", color="#7A808A")

pdf = os.path.join(HERE, "output", "fig6d_persize_heatmap.pdf")
png = os.path.join(HERE, "output", "fig6d_persize_heatmap.png")
fig.savefig(pdf, dpi=600)        # keep manual margins; dpi sets the heat map raster resolution
fig.savefig(png, dpi=300)
shutil.copy(png, os.path.join(HERE, "..", "fig6d_persize_heatmap.png"))
print("saved:", png)
