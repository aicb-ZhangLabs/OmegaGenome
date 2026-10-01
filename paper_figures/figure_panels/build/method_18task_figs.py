"""
Nature-style figures for the 4-method x 18-task distillation comparison.

Methods: OmegaGenome (vanilla, NT distilled student), DKD, DIST, LS -- all share
the NT-2.5B teacher + vanilla-best HP. Per-task values from
data/method_comparison_18task.csv (DKD/DIST/LS = 3-seed mean of best_test_mcc;
OmegaGenome = NT distilled-student column of the 5-teacher benchmark).

House style reused verbatim from fig6_method_size_hp.py (NAT palette, DejaVu
Sans, pdf.fonttype 42, despine, value labels, hero highlight, PDF+PNG 300 dpi).

Produces (each .pdf + .png, copies .png to previews/):
  1 method_18task_mean       -- 4-method bar of 18-task-mean MCC, across-task s.d.
  2 method_pertask_heatmap   -- 18 tasks x 4 methods mako + Delta(OG - best-other)
  3 method_pertask_grouped   -- grouped bars, 2 rows of 9 tasks, 4 methods/group
  4 method_winrate           -- per-method count of tasks where it is best
"""
import os
import shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "axes.linewidth": 1.1,
})

NAT = {
    "grey_base": "#C3C8CE", "blue_dark": "#356491", "blue_light": "#A9C6E0",
    "amber_dark": "#CE8B3B", "green_dark": "#4F8C6B", "purple_dark": "#7C619E",
    "rose_dark": "#B0477E", "slate": "#9AA4AE", "slate_deep": "#7C8894",
    "ink": "#222222", "grid": "#DBDEE2",
}
SPINE = "#444444"
TITLE_FS, LAB_FS, TICK_FS, ANN_FS = 15.5, 13, 11.5, 11

# 4-method categorical palette; OmegaGenome = hero blue
METHODS = ["OmegaGenome", "DKD", "DIST", "LS"]
MCOL = {"OmegaGenome": NAT["blue_dark"], "DKD": NAT["green_dark"],
        "DIST": NAT["amber_dark"], "LS": NAT["purple_dark"]}
MEC = {"OmegaGenome": "#22475f", "DKD": "#37624b", "DIST": "#94622a",
       "LS": "#57436f", "OTHER": NAT["slate_deep"]}

# family-sensible task order (histone marks, then regulatory, then splice)
HISTONE = ["H2AFZ", "H3K27ac", "H3K27me3", "H3K36me3", "H3K4me1", "H3K4me2",
           "H3K4me3", "H3K9ac", "H3K9me3", "H4K20me1"]
REGUL = ["enhancers", "enhancers_types", "promoter_all", "promoter_no_tata",
         "promoter_tata"]
SPLICE = ["splice_sites_acceptors", "splice_sites_all", "splice_sites_donors"]
TASK_ORDER = HISTONE + REGUL + SPLICE
TASK_LABEL = {t: t.replace("splice_sites_", "splice_").replace("_", " ") for t in TASK_ORDER}

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
CSV = os.path.join(REPO, "data", "method_comparison_18task.csv")
OUTD = os.path.join(HERE, "output")
PREV = os.path.join(REPO, "previews")
os.makedirs(OUTD, exist_ok=True)

# Fairness note: where the OmegaGenome-optimal HP degenerates, DIST/LS use a
# method-appropriate kl config (kl=0.5, and kl=0.25 on the lowest-signal H3K9me3)
# so the comparison is fair. Every collapse was an HP-degeneracy artifact, fully
# recovered under the method-appropriate kl. See hp_note column of the source CSV.
KL_NOTE = ("DIST/LS use a method-appropriate kl config where the OmegaGenome-optimal HP "
           "degenerated (kl=0.5, and kl=0.25 on the lowest-signal H3K9me3); all collapses "
           "were HP-degeneracy artifacts, fully recovered")
KL_NOTE2 = ("DIST/LS use a method-appropriate kl config where the OmegaGenome-optimal HP degenerated\n"
            "(kl=0.5, and kl=0.25 on the lowest-signal H3K9me3); all collapses were HP-degeneracy artifacts, fully recovered")

df = pd.read_csv(CSV)
mean_w = df.pivot(index="task", columns="method", values="mcc_mean").reindex(
    TASK_ORDER)[METHODS]
std_w = df.pivot(index="task", columns="method", values="mcc_std").reindex(
    TASK_ORDER)[METHODS]

# 18-task mean per method (mean over 18 per-task means) + across-task s.d.
task_mean = {m: mean_w[m].mean() for m in METHODS}
task_sd = {m: mean_w[m].std(ddof=1) for m in METHODS}


def style_spines(ax):
    for s in ax.spines.values():
        s.set_edgecolor(SPINE); s.set_linewidth(1.1)


def save(fig, name):
    fig.savefig(os.path.join(OUTD, name + ".pdf"), bbox_inches="tight", dpi=600)  # heat map raster >= 300 DPI
    fig.savefig(os.path.join(OUTD, name + ".png"), dpi=300, bbox_inches="tight")
    shutil.copy(os.path.join(OUTD, name + ".png"), os.path.join(PREV, name + ".png"))
    plt.close(fig)
    print("saved", name, "(pdf+png+preview)")


# ===========================================================================
# FIG 1 — 4-method 18-task-mean bar
# ===========================================================================
fig, ax = plt.subplots(figsize=(7.4, 5.6))
x = np.arange(len(METHODS))
LO = 0.0
for i, m in enumerate(METHODS):
    hero = (m == "OmegaGenome")
    if hero:
        ax.axvspan(i - 0.46, i + 0.46, color=NAT["blue_dark"], alpha=0.06, zorder=0)
    ax.bar(i, task_mean[m] - LO, bottom=LO, width=0.64, facecolor=MCOL[m],
           edgecolor=MEC[m], linewidth=1.6 if hero else 1.1, zorder=3,
           alpha=0.96 if hero else 0.85)
    ax.errorbar(i, task_mean[m], yerr=task_sd[m], fmt="none", ecolor=NAT["ink"],
                elinewidth=1.7, capsize=5, capthick=1.5, zorder=4)
    ax.text(i, task_mean[m] + task_sd[m] + 0.012, f"{task_mean[m]:.3f}",
            ha="center", va="bottom", fontsize=14.0,
            fontweight="bold" if hero else "normal", color=NAT["ink"], zorder=5)
ax.annotate("vanilla (deployed)", xy=(0, task_mean["OmegaGenome"]),
            xytext=(0, task_mean["OmegaGenome"] + task_sd["OmegaGenome"] + 0.055),
            ha="center", va="bottom", fontsize=13.0, fontweight="bold",
            color=NAT["blue_dark"])
ax.set_xticks(x)
ax.set_xticklabels(["Omega\nGenome", "DKD", "DIST", "LS"], fontsize=14.5,
                   color=NAT["ink"])
ax.set_ylim(0, 0.9)
ax.set_ylabel("18-task mean best-test MCC", fontsize=16.0, color=NAT["ink"])
ax.set_title("Distillation-method comparison (18-task mean)", fontsize=18.5,
             pad=10, color=NAT["ink"])
ax.tick_params(axis="y", labelsize=14.0, length=4, width=1.0)
ax.set_axisbelow(True)
ax.grid(axis="y", color=NAT["grid"], lw=0.8, zorder=0)
# The error-bar-definition note and the KL-degeneracy note that used to sit under
# this panel (sub-9pt) are stated in full in the LaTeX caption instead, so the
# panel carries no small print and stays compact.
style_spines(ax)
fig.tight_layout()
save(fig, "method_18task_mean")


# ===========================================================================
# FIG 2 — per-task heatmap (18 x 4) + Delta(OG - best-other) column
# ===========================================================================
mako = sns.color_palette("mako", as_cmap=True)
best_other = mean_w[["DKD", "DIST", "LS"]].max(axis=1)
delta = mean_w["OmegaGenome"] - best_other  # >0 OG wins

fig = plt.figure(figsize=(9.2, 11.0))
gs = fig.add_gridspec(1, 2, width_ratios=[4, 1.15], wspace=0.08,
                      left=0.24, right=0.9, top=0.945, bottom=0.06)
axH = fig.add_subplot(gs[0, 0])
axD = fig.add_subplot(gs[0, 1])

M = mean_w.values  # rows tasks, cols methods
im = axH.imshow(M, cmap=mako, aspect="auto", vmin=0.0, vmax=1.0)
axH.set_xticks(range(len(METHODS)))
axH.set_xticklabels(METHODS, fontsize=11.5, color=NAT["ink"])
axH.set_yticks(range(len(TASK_ORDER)))
axH.set_yticklabels([TASK_LABEL[t] for t in TASK_ORDER], fontsize=10, color=NAT["ink"])
axH.set_title("Per-task best-test MCC", fontsize=13.5, pad=10, color=NAT["ink"])
for i in range(len(TASK_ORDER)):
    for j in range(len(METHODS)):
        v = M[i, j]
        axH.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8.8,
                 color="white" if v < 0.55 else NAT["ink"])
# family separators
for k in (len(HISTONE) - 0.5, len(HISTONE) + len(REGUL) - 0.5):
    axH.axhline(k, color="white", lw=2.2)
axH.tick_params(length=0)
cb = fig.colorbar(im, ax=axH, fraction=0.05, pad=0.02, location="bottom")
cb.set_label("best-test MCC", fontsize=10.5, color=NAT["ink"])
cb.ax.tick_params(labelsize=9)
cb.outline.set_edgecolor(SPINE)

# Delta panel
dmax = float(np.nanmax(np.abs(delta.values)))
div = sns.color_palette("vlag", as_cmap=True)
imd = axD.imshow(delta.values.reshape(-1, 1), cmap=div, aspect="auto",
                 vmin=-dmax, vmax=dmax)
axD.set_xticks([0]); axD.set_xticklabels(["Δ"], fontsize=13, color=NAT["ink"])
axD.set_yticks([])
axD.set_title("OmegaGenome\n− best other", fontsize=11, pad=10, color=NAT["ink"])
for i in range(len(TASK_ORDER)):
    v = delta.values[i]
    axD.text(0, i, f"{v:+.2f}", ha="center", va="center", fontsize=8.6,
             color="white" if abs(v) > 0.55 * dmax else NAT["ink"])
for k in (len(HISTONE) - 0.5, len(HISTONE) + len(REGUL) - 0.5):
    axD.axhline(k, color="white", lw=2.2)
axD.tick_params(length=0)
for s in axH.spines.values():
    s.set_edgecolor(SPINE); s.set_linewidth(1.1)
for s in axD.spines.values():
    s.set_edgecolor(SPINE); s.set_linewidth(1.1)
fig.text(0.5, 0.024, "red = OmegaGenome wins the task, blue = another method wins   |   "
         "all cells: 3-seed mean of best_test MCC (best-val selection)",
         ha="center", va="bottom", fontsize=8.2, style="italic", color="#7A808A")
fig.text(0.5, 0.006, KL_NOTE, ha="center", va="bottom", fontsize=8.0,
         style="italic", color="#7A808A")
save(fig, "method_pertask_heatmap")


# ===========================================================================
# FIG 3 — per-task grouped bars, 2 rows of 9 tasks
# ===========================================================================
half = 9
groups = [TASK_ORDER[:half], TASK_ORDER[half:]]
fig, axes = plt.subplots(2, 1, figsize=(13.6, 9.2))
bw = 0.2
for r, (ax, tasks) in enumerate(zip(axes, groups)):
    xpos = np.arange(len(tasks))
    for mi, m in enumerate(METHODS):
        vals = mean_w.loc[tasks, m].values
        errs = std_w.loc[tasks, m].values
        off = (mi - 1.5) * bw
        ax.bar(xpos + off, vals, width=bw, facecolor=MCOL[m], edgecolor=MEC[m],
               linewidth=0.9, label=m if r == 0 else None, zorder=3,
               alpha=0.96 if m == "OmegaGenome" else 0.85)
        ax.errorbar(xpos + off, vals, yerr=errs, fmt="none", ecolor=NAT["ink"],
                    elinewidth=0.9, capsize=2.2, capthick=0.8, zorder=4)
    ax.set_xticks(xpos)
    ax.set_xticklabels([TASK_LABEL[t] for t in tasks], fontsize=10.5,
                       rotation=25, ha="right", color=NAT["ink"])
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("best-test MCC", fontsize=LAB_FS, color=NAT["ink"])
    ax.tick_params(axis="y", labelsize=TICK_FS, length=4, width=1.0)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color=NAT["grid"], lw=0.8, zorder=0)
    style_spines(ax)
handles = [Patch(facecolor=MCOL[m], edgecolor=MEC[m], label=m) for m in METHODS]
axes[0].legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.20),
               ncol=4, frameon=False, fontsize=12, handlelength=1.4, columnspacing=1.8)
axes[0].set_title("Per-task distillation-method comparison (3-seed mean ± s.d.)",
                  fontsize=TITLE_FS, pad=42, color=NAT["ink"])
fig.text(0.5, 0.005, KL_NOTE, ha="center", va="bottom", fontsize=8.5,
         style="italic", color="#7A808A")
fig.tight_layout(rect=(0, 0.02, 1, 0.97))
save(fig, "method_pertask_grouped")


# ===========================================================================
# FIG 4 — win-rate (tasks each method is best on)
# ===========================================================================
winner = mean_w.idxmax(axis=1)
wins = {m: int((winner == m).sum()) for m in METHODS}
fig, ax = plt.subplots(figsize=(6.6, 4.8))
x = np.arange(len(METHODS))
for i, m in enumerate(METHODS):
    hero = (m == "OmegaGenome")
    ax.bar(i, wins[m], width=0.64, facecolor=MCOL[m], edgecolor=MEC[m],
           linewidth=1.6 if hero else 1.1, zorder=3,
           alpha=0.96 if hero else 0.85)
    ax.text(i, wins[m] + 0.15, str(wins[m]), ha="center", va="bottom",
            fontsize=ANN_FS + 2, fontweight="bold" if hero else "normal",
            color=NAT["ink"])
ax.set_xticks(x)
ax.set_xticklabels(["Omega\nGenome", "DKD", "DIST", "LS"], fontsize=12, color=NAT["ink"])
ax.set_ylim(0, max(wins.values()) + 1.5)
ax.set_ylabel("# tasks where method is best", fontsize=LAB_FS, color=NAT["ink"])
ax.set_title(f"Per-method win count ({sum(wins.values())} tasks)", fontsize=TITLE_FS,
             pad=10, color=NAT["ink"])
ax.tick_params(axis="y", labelsize=TICK_FS, length=4, width=1.0)
ax.set_axisbelow(True)
ax.grid(axis="y", color=NAT["grid"], lw=0.8, zorder=0)
style_spines(ax)
fig.tight_layout()
fig.text(0.5, -0.02, KL_NOTE2, ha="center", va="top", fontsize=8.0,
         style="italic", color="#7A808A")
save(fig, "method_winrate")

print("\n18-task means:", {m: (round(task_mean[m], 4), round(task_sd[m], 4)) for m in METHODS})
print("win counts:", wins)
print("Delta(OG-best_other) per task:")
for t in TASK_ORDER:
    print(f"  {t:24s} {delta[t]:+.4f}")
