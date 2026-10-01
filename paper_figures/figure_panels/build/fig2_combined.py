"""
Build the combined Figure 2 for OmegaGenome (Nature-style, single figure).

Layout:
  A = 3x3 lollipop grid (Histone / CREs / Splicing MCC), LEFT ~62% width.
  B = per-model MCC scatter, TOP-RIGHT.
  C = relative-improvement bars, BOTTOM-RIGHT.
  Shared bottom legend spanning full width (applies to all panels).

Run with the biolaysum env python. Saves output/fig2_combined.pdf + .png.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, FormatStrFormatter

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "axes.linewidth": 1.2,
})

DATA = "data/model_comparison_long_final.csv"
os.makedirs("output", exist_ok=True)

# ---------------------------------------------------------------------------
# NATURE PALETTE  (same hue identity, refined to muted editorial tones)
# ---------------------------------------------------------------------------
NAT = {
    "grey_dot":     "#8A9099",   # BPNet marker (slate grey)
    "grey_base":    "#C3C8CE",   # BPNet baseline line
    "blue_dark":    "#356491",   # NT student (muted steel blue)
    "blue_light":   "#A9C6E0",   # NT teacher
    "amber_dark":   "#CE8B3B",   # DNABERT-2 student (warm amber)
    "amber_light":  "#EED0A3",   # DNABERT-2 teacher
    "green_dark":   "#4F8C6B",   # Enformer student (sage/forest)
    "green_light":  "#A9CDB7",   # Enformer teacher
    "purple_dark":  "#7C619E",   # Caduceus student (dusty purple)
    "purple_light": "#C7B7DA",   # Caduceus teacher
    "red":          "#C24B44",   # Best distilled (muted brick red)
    "red_face":     "#FBEDEC",   # best-distilled highlight fill
    "ink":          "#222222",
}

# palette keyed by the Model column (for panel A)
MODEL_PAL = {
    "BPNet": NAT["grey_dot"],
    "Distilled Nucleotide Transformer": NAT["blue_dark"],
    "Nucleotide Transformer": NAT["blue_light"],
    "Distilled DNABERT-2": NAT["amber_dark"],
    "DNABERT-2": NAT["amber_light"],
    "Enformer Distilled": NAT["green_dark"],
    "Enformer": NAT["green_light"],
    "Distilled Caduceus": NAT["purple_dark"],
    "Caduceus": NAT["purple_light"],
}

# family -> (light teacher color, dark student color)
FAM_COLORS = {
    "Nucleotide Transformer": (NAT["blue_light"],  NAT["blue_dark"]),
    "DNABERT-2":              (NAT["amber_light"], NAT["amber_dark"]),
    "Enformer":               (NAT["green_light"], NAT["green_dark"]),
    "Caduceus":               (NAT["purple_light"],NAT["purple_dark"]),
}
FAM_SHORT = {
    "Nucleotide Transformer": "NT",
    "DNABERT-2": "DNABERT-2",
    "Enformer": "Enformer",
    "Caduceus": "Caduceus",
}

# ---------------------------------------------------------------------------
# DATA
# ---------------------------------------------------------------------------
df = pd.read_csv(DATA)
df["Score"] = pd.to_numeric(df["Score"], errors="coerce")

family_order = ["BPNet", "Nucleotide Transformer", "DNABERT-2", "Enformer", "Caduceus"]

epi   = ["H3K27ac", "H3K4me3", "H3K27me3"]
cres  = ["enhancers_types", "promoter_tata", "promoter_no_tata"]
splic = ["splice_sites_donors", "splice_sites_all", "splice_sites_acceptors"]
task_rows = [epi, cres, splic]
row_titles = ["Histone Modifications  MCC", "CREs  MCC", "Splicing Sites  MCC"]


def best_distilled_family(task_data):
    best_score, best_family = -1, None
    for fam in family_order[1:]:
        sd = task_data[(task_data["Model Family"] == fam) & (task_data["Type"] == "Student")]
        if not sd.empty and sd["Score"].iloc[0] > best_score:
            best_score, best_family = sd["Score"].iloc[0], fam
    return best_family


# ===========================================================================
# FIGURE SCAFFOLD
# ===========================================================================
fig = plt.figure(figsize=(21.5, 14.8))
outer = fig.add_gridspec(
    2, 2, width_ratios=[0.615, 0.385], height_ratios=[1.0, 0.115],
    wspace=0.085, hspace=0.055,
    left=0.052, right=0.986, top=0.945, bottom=0.015,
)
gsA  = outer[0, 0].subgridspec(3, 4, width_ratios=[0.11, 1, 1, 1], hspace=0.34, wspace=0.30)
gsBC = outer[0, 1].subgridspec(2, 1, height_ratios=[1.28, 1.0], hspace=0.30)
axB  = fig.add_subplot(gsBC[0])
axC  = fig.add_subplot(gsBC[1])
axL  = fig.add_subplot(outer[1, :]); axL.axis("off")

TITLE_FS, TICK_FS, ANN_FS = 15.5, 11.5, 11
MARK = 430

# ===========================================================================
# PANEL A  (3x3 lollipop)
# ===========================================================================
axA_first = None
for r, tasks in enumerate(task_rows):
    axlab = fig.add_subplot(gsA[r, 0]); axlab.axis("off")
    axlab.text(0.75, 0.5, row_titles[r], fontsize=15, rotation=90,
               va="center", ha="center", transform=axlab.transAxes, color=NAT["ink"])
    for c, task in enumerate(tasks):
        ax = fig.add_subplot(gsA[r, c + 1])
        if axA_first is None:
            axA_first = ax
        td = df[df["Task"] == task].copy()
        bestfam = best_distilled_family(td)
        sp = 0.07
        centers = np.arange(len(family_order)) * sp

        for i, fam in enumerate(family_order):
            x = centers[i]
            fd = td[td["Model Family"] == fam]
            if fam == bestfam:
                ax.add_patch(mpatches.FancyBboxPatch(
                    (x - sp * 0.27, 0), sp * 0.54, 1.0,
                    boxstyle="round,pad=0.01,rounding_size=0.015",
                    facecolor=NAT["red_face"], alpha=0.8,
                    edgecolor=NAT["red"], linestyle="--", linewidth=1.6,
                    zorder=0, transform=ax.get_xaxis_transform()))
            if fam == "BPNet":
                if fd.empty:
                    continue
                s = fd["Score"].iloc[0]
                ax.plot([x, x], [0, s], color=NAT["grey_dot"], lw=2.6, zorder=1)
                ax.scatter(x, s, s=MARK, color=NAT["grey_dot"], edgecolor="black",
                           lw=1.4, zorder=3, marker="o")
                ax.annotate(f"{s:.2f}", (x, s), textcoords="offset points",
                            xytext=(0, 12), ha="center", va="bottom",
                            fontsize=ANN_FS, rotation=90, color=NAT["ink"])
            else:
                sd = fd[fd["Type"] == "Student"]
                tc = fd[fd["Type"] == "Teacher"]
                if sd.empty or tc.empty:
                    continue
                ss, tt = sd["Score"].iloc[0], tc["Score"].iloc[0]
                light, dark = FAM_COLORS[fam]
                ax.plot([x, x], [0, max(ss, tt)], color="#CFCFCF", lw=2.6, zorder=1, alpha=0.7)
                ax.plot([x, x], [ss, tt], color=light, lw=2.0, linestyle=":", zorder=2, alpha=0.85)
                sct = ax.scatter(x, tt, s=MARK, facecolors="white", edgecolors=light,
                                 lw=2.6, zorder=3, marker="o")
                sct.set_linestyle("--")
                ax.scatter(x, ss, s=MARK, color=dark, edgecolor="black", lw=1.4,
                           zorder=3, marker="o")
                hi, lo = max(ss, tt), min(ss, tt)
                ax.annotate(f"{hi:.2f}", (x, hi), textcoords="offset points",
                            xytext=(0, 12), ha="center", va="bottom",
                            fontsize=ANN_FS, rotation=90, color=NAT["ink"])
                ax.annotate(f"{lo:.2f}", (x, lo), textcoords="offset points",
                            xytext=(0, -16), ha="center", va="top",
                            fontsize=ANN_FS, rotation=90, color=NAT["ink"])

        base = td[td["Model"] == "BPNet"]["Score"].values
        if len(base):
            ax.axhline(base[0], color=NAT["grey_base"], linestyle=":", lw=3.0, alpha=0.9, zorder=0)

        sc = td["Score"].dropna().values
        if len(sc):
            lo, hi = sc.min(), sc.max()
            pad = (hi - lo) * 0.32
            ax.set_ylim(max(0, lo - pad), hi + pad)
        ax.set_title(task.replace("_", " ").title(), fontsize=TITLE_FS, pad=9, color=NAT["ink"])
        ax.tick_params(axis="y", labelsize=TICK_FS, length=4, width=1.1, labelrotation=90)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4, prune="both"))
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        ax.set_xticks(centers); ax.set_xticklabels([])
        m = sp * 0.85
        ax.set_xlim(centers[0] - m, centers[-1] + m)
        for s_ in ax.spines.values():
            s_.set_edgecolor("#444444"); s_.set_linewidth(1.1)
        ax.set_axisbelow(True)

# ===========================================================================
# PANEL B  (per-model MCC scatter)
# ===========================================================================
groups = ["BPNet", "NT", "DNABERT-2", "Enformer", "Caduceus", "Best Distilled"]
gc = {n: i for i, n in enumerate(groups)}
off = 0.2
np.random.seed(0)


def jitter(n, w=0.05):
    return (np.random.rand(n) - 0.5) * 2 * w


baseline_mean = df[df["Type"] == "Baseline"]["Score"].mean()
LBOX = dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor="none", alpha=0.82)

for i in range(len(groups) - 1):
    axB.axvline(i + 0.5, color="#B9BEC4", linestyle="--", lw=1.1, alpha=0.7, zorder=0)

# BPNet
bp = df[df["Model Family"] == "BPNet"]["Score"].dropna().values
axB.scatter(gc["BPNet"] + jitter(len(bp)), bp, s=190, color=NAT["grey_dot"],
            edgecolor="black", lw=0.9, zorder=2)
axB.hlines(bp.mean(), gc["BPNet"] - 0.16, gc["BPNet"] + 0.16, color=NAT["grey_dot"], lw=4, zorder=5)
axB.text(gc["BPNet"], bp.mean() + 0.016, f"{bp.mean():.3f}", ha="center", va="bottom",
         fontsize=9.5, fontweight="bold", color=NAT["ink"], zorder=7, bbox=LBOX)

for fam, short in FAM_SHORT.items():
    light, dark = FAM_COLORS[fam]
    ctr = gc[short]
    tv = df[(df["Model Family"] == fam) & (df["Type"] == "Teacher")]["Score"].dropna().values
    sv = df[(df["Model Family"] == fam) & (df["Type"] == "Student")]["Score"].dropna().values
    sct = axB.scatter((ctr - off) + jitter(len(tv)), tv, s=190, facecolors="white",
                      edgecolors=light, lw=2.0, zorder=2)
    sct.set_linestyle("--")
    axB.scatter((ctr + off) + jitter(len(sv)), sv, s=190, color=dark,
                edgecolor="black", lw=0.9, zorder=2)
    axB.plot([ctr - off, ctr + off], [tv.mean(), sv.mean()], color="#666666",
             linestyle="--", lw=1.6, alpha=0.6, zorder=1)
    for xx, vv, cc in [(ctr - off, tv.mean(), light), (ctr + off, sv.mean(), dark)]:
        axB.hlines(vv, xx - 0.13, xx + 0.13, color=cc, lw=4, zorder=5)
    # teacher label below its dash, student label above -> never collide
    axB.text(ctr - off, tv.mean() - 0.018, f"{tv.mean():.3f}", ha="center", va="top",
             fontsize=9.5, fontweight="bold", color=NAT["ink"], zorder=7, bbox=LBOX)
    axB.text(ctr + off, sv.mean() + 0.016, f"{sv.mean():.3f}", ha="center", va="bottom",
             fontsize=9.5, fontweight="bold", color=NAT["ink"], zorder=7, bbox=LBOX)

# Best distilled = per-task max student
best_vals = df[df["Type"] == "Student"].groupby("Task")["Score"].max().values
axB.scatter(gc["Best Distilled"] + jitter(len(best_vals)), best_vals, s=190,
            color=NAT["red"], edgecolor="black", lw=0.9, zorder=2)
axB.hlines(best_vals.mean(), gc["Best Distilled"] - 0.16, gc["Best Distilled"] + 0.16,
           color=NAT["red"], lw=4, zorder=5)
axB.text(gc["Best Distilled"], best_vals.mean() + 0.016, f"{best_vals.mean():.3f}",
         ha="center", va="bottom", fontsize=9.5, fontweight="bold", color=NAT["ink"], zorder=7, bbox=LBOX)

axB.axhline(baseline_mean, color=NAT["grey_base"], linestyle=":", lw=2.6, alpha=0.95, zorder=0)
axB.set_xlim(-0.5, len(groups) - 0.5)
axB.set_ylabel("MCC Score", fontsize=15, color=NAT["ink"])
axB.set_xticks(list(gc.values()))
axB.set_xticklabels(list(gc.keys()), fontsize=12.5, color=NAT["ink"])
axB.tick_params(axis="y", labelsize=TICK_FS, length=4, width=1.1, labelrotation=90)
axB.grid(False)
for s_ in axB.spines.values():
    s_.set_edgecolor("#444444"); s_.set_linewidth(1.1)

# ===========================================================================
# PANEL C  (relative-improvement bars, tightened)
# ===========================================================================
bar_families = ["Nucleotide Transformer", "DNABERT-2", "Enformer", "Caduceus"]
bar_centers = {f: i for i, f in enumerate(bar_families)}
for i in range(len(bar_families) - 1):
    axC.axvline(i + 0.5, color="#B9BEC4", linestyle="--", lw=1.1, alpha=0.7, zorder=0)

heights = []
for fam in bar_families:
    light, dark = FAM_COLORS[fam]
    ctr = bar_centers[fam]
    t = df[(df["Model Family"] == fam) & (df["Type"] == "Teacher")]["Score"].mean()
    s = df[(df["Model Family"] == fam) & (df["Type"] == "Student")]["Score"].mean()
    ti = (t - baseline_mean) / baseline_mean * 100
    si = (s - baseline_mean) / baseline_mean * 100
    heights += [ti, si]
    for xx, hh, cc in [(ctr - off, ti, light), (ctr + off, si, dark)]:
        axC.bar(xx, hh, width=0.33, facecolor="white", edgecolor=cc, hatch="////",
                linewidth=2.0, zorder=3)
        axC.text(xx, hh + max(heights) * 0.015 + 0.05, f"{hh:.1f}%", ha="center",
                 va="bottom", fontsize=12, fontweight="bold", color=NAT["ink"], zorder=4)

axC.axhline(0, color="#444444", lw=1.2)
axC.set_ylim(0, max(heights) * 1.13)
axC.set_xlim(-0.5, len(bar_families) - 0.5)
axC.set_ylabel("Relative Improvement (%)", fontsize=15, color=NAT["ink"])
axC.set_xticks(list(bar_centers.values()))
axC.set_xticklabels([FAM_SHORT[f] for f in bar_families], fontsize=12.5, color=NAT["ink"])
axC.tick_params(axis="y", labelsize=TICK_FS, length=4, width=1.1, labelrotation=90)
axC.grid(False)
for s_ in axC.spines.values():
    s_.set_edgecolor("#444444"); s_.set_linewidth(1.1)

# ===========================================================================
# SHARED BOTTOM LEGEND  (manual, applies to all panels)
# ===========================================================================
def draw_legend(ax):
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.add_patch(mpatches.FancyBboxPatch(
        (0.006, 0.10), 0.988, 0.82, boxstyle="round,pad=0.006,rounding_size=0.02",
        facecolor="#FAFAFB", edgecolor="#D8DCE0", linewidth=1.2,
        transform=ax.transAxes, zorder=0))
    ax.text(0.5, 0.72,
            "Legend  (applies to all panels  A · B · C)      "
            "Teacher = open dashed ring         Distilled / Student = solid filled dot",
            ha="center", va="center", fontsize=13.5, fontweight="bold",
            color=NAT["ink"], transform=ax.transAxes)

    ms = 15
    yg, yt = 0.34, 0.34   # glyph row / text baseline
    # slot x-centers for a single row of entries
    def dot(x, y, fc, ec, lw, dashed=False):
        h = ax.scatter([x], [y], s=ms ** 2, facecolors=fc, edgecolors=ec,
                       linewidths=lw, transform=ax.transAxes, zorder=3, clip_on=False)
        if dashed:
            h.set_linestyle("--")

    def txt(x, s, weight="normal", size=12.5):
        ax.text(x, yt, s, ha="left", va="center", fontsize=size, fontweight=weight,
                color=NAT["ink"], transform=ax.transAxes, zorder=3)

    # ---- entry layout (left -> right) ----
    x = 0.028
    # BPNet
    dot(x, yg, NAT["grey_dot"], "black", 1.4); txt(x + 0.016, "BPNet")
    x += 0.105
    # BPNet baseline
    ax.plot([x - 0.006, x + 0.020], [yg, yg], color=NAT["grey_base"], lw=3,
            linestyle=":", transform=ax.transAxes, zorder=3, clip_on=False)
    txt(x + 0.030, "BPNet Baseline"); x += 0.165
    # four families: open dashed + filled dot, colored
    for fam in bar_families:
        light, dark = FAM_COLORS[fam]
        dot(x, yg, "white", light, 2.4, dashed=True)
        dot(x + 0.026, yg, dark, "black", 1.4)
        txt(x + 0.044, FAM_SHORT[fam]);
        x += 0.075 + 0.010 * len(FAM_SHORT[fam])
    # Best distilled: red dashed box + red filled dot
    ax.add_patch(mpatches.FancyBboxPatch(
        (x - 0.012, yg - 0.17), 0.030, 0.34,
        boxstyle="round,pad=0.002,rounding_size=0.01",
        facecolor=NAT["red_face"], edgecolor=NAT["red"], linestyle="--",
        linewidth=1.6, transform=ax.transAxes, zorder=2, clip_on=False))
    dot(x + 0.003, yg, NAT["red"], "black", 1.4)
    txt(x + 0.026, "Best Distilled")


draw_legend(axL)

# ===========================================================================
# PANEL LABELS  A / B / C  (top-left of each panel)
# ===========================================================================
fig.canvas.draw()
def label(ax, s, dx=0.0, dy=0.0):
    bb = ax.get_position()
    fig.text(bb.x0 + dx, bb.y1 + dy, s, fontsize=30, fontweight="bold",
             ha="left", va="bottom", color=NAT["ink"])

label(axA_first, "A", dx=-0.040, dy=0.006)
label(axB, "B", dx=-0.042, dy=0.006)
label(axC, "C", dx=-0.030, dy=0.030)

fig.savefig("output/fig2_combined.pdf", bbox_inches="tight")
fig.savefig("output/fig2_combined.png", dpi=200, bbox_inches="tight")
print("saved fig2_combined")
