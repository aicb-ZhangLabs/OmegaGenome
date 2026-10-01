"""
Build the APPENDIX hyperparameter figure for OmegaGenome (Nature-style).

Two parts, single figure:
  TOP    : 5 faceted KL x MSE heatmaps (one tile per teacher) of mean best-test
           MCC over temperature, on a shared reference task (enhancers_types,
           matching main Fig 6 Panel D). Each tile uses the SAME diverging
           colormap but is centred on that teacher's own grid mean, so colour
           encodes within-teacher relative performance (below <-> above the
           teacher's own average config). Cells annotated with absolute MCC;
           best cell starred + boxed. Per-tile colourbar carries absolute MCC.
  BOTTOM : temperature response curves, one line per teacher, mean +/- 1 s.d.
           test MCC vs temperature in {0.5,1,1.5,2,4}, marginalised over
           (KL, MSE). Ribbon = +/-1 s.d. Teacher colours from the shared palette.

Data are read ONLY from the real HP sweep CSVs (4 wandb exports + carbon grid).
No values are fabricated. Caduceus has no MSE=0 runs on this task, so the top
row of its tile is genuinely empty (rendered as hatched "n/a").

Run with the biolaysum env python.
Saves output/fig_appendix_hp.pdf (vector) + .png and copies the PNG to
previews/fig_appendix_hp.png .
"""
import os
import shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "axes.linewidth": 1.1,
})

# ---------------------------------------------------------------------------
# NATURE PALETTE  (identical hue identity to Fig 2/4/6)
# ---------------------------------------------------------------------------
NAT = {
    "blue_dark":   "#356491",   # NT
    "amber_dark":  "#CE8B3B",   # DNABERT-2
    "green_dark":  "#4F8C6B",   # Enformer
    "purple_dark": "#7C619E",   # Caduceus
    "rose_dark":   "#B0477E",   # Carbon-3B
    "ink":         "#222222",
    "grid":        "#DBDEE2",
}
SPINE = "#444444"

# shared diverging colormap (magenta = below teacher mean, blue = above)
NAT_DIV = LinearSegmentedColormap.from_list(
    "nat_div",
    ["#8E2F63", "#B0477E", "#D8A9BE", "#F2EEF0",
     "#F4F0E6", "#AFC8D9", "#5B84A8", "#2C557E"],
)
NAT_DIV.set_bad("#E9EBEE")   # NaN cells -> light grey

# teacher order + display + colour
TEACHERS = [
    ("caduceus", "Caduceus",               NAT["purple_dark"]),
    ("dnabert2", "DNABERT-2",              NAT["amber_dark"]),
    ("enformer", "Enformer",               NAT["green_dark"]),
    ("nt",       "Nucleotide Transformer", NAT["blue_dark"]),
    ("carbon",   "Carbon-3B",              NAT["rose_dark"]),
]

TASK = "enhancers_types"          # shared reference task (matches Fig 6 D)
KL_VALS  = [0.0, 0.25, 0.5, 1.0]  # x-axis
MSE_VALS = [0.0, 1.0, 2.0, 5.0]   # y-axis (clean shared 4x4)
TEMPS    = [0.5, 1.0, 1.5, 2.0, 4.0]

os.makedirs("output", exist_ok=True)

# ---------------------------------------------------------------------------
# DATA LOADERS  (real CSVs only)
# ---------------------------------------------------------------------------
WANDB_CSV = {
    "nt":       "hyperparam_csvs/nt_hyperparam_flat_clean.csv",
    "dnabert2": "hyperparam_csvs/dnabert2_hyperparam_flat_clean.csv",
    "enformer": "hyperparam_csvs/enformer_hyperparam_flat_clean.csv",
    "caduceus": "hyperparam_csvs/caduceus_hyperparam_1-15_flat_clean.csv",
}
CARBON_CSV = "../results/carbon_grid_results.csv"


def load_long(key):
    """Return long-form df with columns kl, mse, temp, mcc for `key` on TASK."""
    if key == "carbon":
        df = pd.read_csv(CARBON_CSV, low_memory=False)
        df = df[(df["task"] == TASK) & (df["variant"] == "raw")
                & df["best_test_mcc"].notna()]
        df = df.rename(columns={"weight_kl": "kl", "weight_mse": "mse",
                                "temperature": "temp", "best_test_mcc": "mcc"})
    else:
        df = pd.read_csv(WANDB_CSV[key], low_memory=False)
        TC = "config.dataset_config.task_name"
        df = df[(df[TC] == TASK) & df["summary.best_test/mcc"].notna()]
        df = df.rename(columns={
            "config.distillation_config.weight_kl": "kl",
            "config.distillation_config.weight_mse": "mse",
            "config.distillation_config.temperature": "temp",
            "summary.best_test/mcc": "mcc"})
    df = df[["kl", "mse", "temp", "mcc"]].copy()
    return df[df["kl"].isin(KL_VALS) & df["mse"].isin(MSE_VALS)]


def grid_of(long):
    """4x4 grid (rows=MSE, cols=KL) of mean MCC over temperature."""
    G = np.full((len(MSE_VALS), len(KL_VALS)), np.nan)
    N = np.zeros_like(G)
    for i, mv in enumerate(MSE_VALS):
        for j, kv in enumerate(KL_VALS):
            s = long[(long["mse"] == mv) & (long["kl"] == kv)]["mcc"]
            if len(s):
                G[i, j] = s.mean()
                N[i, j] = len(s)
    return G, N


def temp_curve(long):
    """mean +/- sd MCC vs temperature, marginalised over KL, MSE."""
    m, sd = [], []
    for t in TEMPS:
        v = long[long["temp"] == t]["mcc"]
        m.append(v.mean() if len(v) else np.nan)
        sd.append(v.std() if len(v) > 1 else 0.0)
    return np.array(m), np.array(sd)


LONG = {k: load_long(k) for k, _, _ in TEACHERS}
GRID = {k: grid_of(LONG[k]) for k, _, _ in TEACHERS}
TCUR = {k: temp_curve(LONG[k]) for k, _, _ in TEACHERS}

# ===========================================================================
# FIGURE SCAFFOLD
# ===========================================================================
fig = plt.figure(figsize=(19.4, 10.6))
outer = fig.add_gridspec(
    2, 1, height_ratios=[1.0, 0.90], hspace=0.40,
    left=0.052, right=0.975, top=0.865, bottom=0.085,
)
# top: 5 heatmap tiles, each with its own slim colourbar
gsTop = outer[0].subgridspec(1, 5, wspace=0.42)
# bottom: a compact, ~square temperature panel centred in the figure with the
# legend beside it. Symmetric empty spacer columns on the outside keep the
# curve+legend unit centred; set_box_aspect(1) below forces the panel square so
# it reads as a balanced tile rather than a long horizontal band.
gsBot = outer[1].subgridspec(
    1, 4, width_ratios=[0.315, 0.185, 0.185, 0.315], wspace=0.05)
axCur = fig.add_subplot(gsBot[1])
axLeg = fig.add_subplot(gsBot[2]); axLeg.axis("off")

TITLE_FS, LAB_FS, TICK_FS, ANN_FS = 15.0, 12.5, 11.0, 10.5

# ===========================================================================
# PART 1  — faceted KL x MSE heatmaps
# ===========================================================================
best_info = {}
for c, (key, disp, col) in enumerate(TEACHERS):
    ax = fig.add_subplot(gsTop[c])
    G, N = GRID[key]
    gmean = np.nanmean(G)
    dev = np.nanmax(np.abs(G - gmean))
    dev = dev if dev > 1e-6 else 1e-3
    norm = TwoSlopeNorm(vmin=gmean - dev, vcenter=gmean, vmax=gmean + dev)

    Gm = np.ma.masked_invalid(G)
    im = ax.imshow(Gm, cmap=NAT_DIV, norm=norm, aspect="auto", origin="upper")

    bi, bj = np.unravel_index(np.nanargmax(G), G.shape)
    best_info[key] = (KL_VALS[bj], MSE_VALS[bi], G[bi, bj])

    for i in range(len(MSE_VALS)):
        for j in range(len(KL_VALS)):
            v = G[i, j]
            if np.isnan(v):
                ax.add_patch(mpatches.Rectangle(
                    (j - 0.5, i - 0.5), 1, 1, fill=True, facecolor="#E9EBEE",
                    hatch="////", edgecolor="#C4C8CD", lw=0.0, zorder=2))
                ax.text(j, i, "n/a", ha="center", va="center",
                        fontsize=9.0, color="#9198A0", zorder=3)
                continue
            is_best = (i == bi and j == bj)
            # choose readable text colour against cell shade
            rel = (v - (gmean - dev)) / (2 * dev)
            tcol = "white" if (rel < 0.14 or rel > 0.86) else NAT["ink"]
            ax.text(j, i - (0.13 if is_best else 0.0), f"{v:.3f}",
                    ha="center", va="center", fontsize=ANN_FS,
                    fontweight="bold" if is_best else "normal",
                    color=tcol, zorder=4)
            if is_best:
                ax.scatter(j, i + 0.235, marker="*", s=190, color="#F0C441",
                           edgecolor=NAT["ink"], lw=0.7, zorder=6)
                ax.add_patch(mpatches.Rectangle(
                    (j - 0.5, i - 0.5), 1, 1, fill=False,
                    edgecolor=NAT["ink"], lw=2.2, zorder=5))

    ax.set_xticks(range(len(KL_VALS)))
    ax.set_xticklabels([f"{v:g}" for v in KL_VALS], fontsize=TICK_FS)
    ax.set_yticks(range(len(MSE_VALS)))
    ax.set_yticklabels([f"{v:g}" for v in MSE_VALS], fontsize=TICK_FS)
    ax.set_xlabel("KL weight", fontsize=LAB_FS, color=NAT["ink"])
    if c == 0:
        ax.set_ylabel("MSE weight", fontsize=LAB_FS, color=NAT["ink"])
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_edgecolor(SPINE); s.set_linewidth(1.1)

    bk, bm, bv = best_info[key]
    # teacher name + best-config subtitle placed with an explicit vertical gap
    # so they never collide with each other, the suptitle, or the colourbar.
    ax.text(0.5, 1.150, disp, transform=ax.transAxes, ha="center", va="bottom",
            fontsize=TITLE_FS, color=col, fontweight="bold")
    ax.text(0.5, 1.040, f"best {bv:.3f}  @ KL={bk:g}, MSE={bm:g}",
            transform=ax.transAxes, ha="center", va="bottom",
            fontsize=9.0, color="#4A5058")

    # slim per-tile colourbar with absolute MCC ticks
    cb = fig.colorbar(im, ax=ax, fraction=0.070, pad=0.045)
    cb.ax.tick_params(labelsize=8.0, length=2)
    cb.set_ticks([gmean - dev, gmean, gmean + dev])
    cb.set_ticklabels([f"{gmean - dev:.3f}", f"{gmean:.3f}", f"{gmean + dev:.3f}"])
    cb.ax.axhline(gmean, color=NAT["ink"], lw=0.9, alpha=0.55)
    cb.outline.set_edgecolor(SPINE); cb.outline.set_linewidth(0.9)

# ===========================================================================
# PART 2  — temperature response curves
# ===========================================================================
xt = np.arange(len(TEMPS))
# small horizontal offset per teacher so overlapping error bars stay legible
X_OFF = np.linspace(-0.10, 0.10, len(TEACHERS))
for oi, (key, disp, col) in enumerate(TEACHERS):
    m, sd = TCUR[key]
    xx = xt + X_OFF[oi]
    axCur.plot(xx, m, "-", color=col, lw=2.3, zorder=3, alpha=0.95)
    axCur.errorbar(xx, m, yerr=sd, fmt="none", ecolor=col, elinewidth=1.4,
                   capsize=3.4, capthick=1.3, alpha=0.85, zorder=3)
    axCur.scatter(xx, m, s=70, color=col, edgecolor="white", lw=1.1, zorder=4)
    # mark each teacher's peak temperature
    pk = int(np.nanargmax(m))
    axCur.scatter(xx[pk], m[pk], s=235, facecolors="none", edgecolors=col,
                  lw=1.9, zorder=5)

axCur.set_box_aspect(1.0)   # force a square plot box (balanced, not a wide band)
axCur.set_xticks(xt)
axCur.set_xticklabels([f"{t:g}" for t in TEMPS], fontsize=TICK_FS)
axCur.set_xlim(-0.30, len(TEMPS) - 0.70)
axCur.margins(y=0.10)
axCur.set_xlabel("Distillation temperature  $\\tau$", fontsize=LAB_FS, color=NAT["ink"])
axCur.set_ylabel("Mean test MCC  (enhancers_types)", fontsize=LAB_FS, color=NAT["ink"])
axCur.set_title("Temperature response  (mean ± 1 s.d., marginalised over KL & MSE)",
                fontsize=TITLE_FS, pad=10, color=NAT["ink"])
axCur.tick_params(axis="y", labelsize=TICK_FS, length=4, width=1.0)
axCur.grid(axis="y", color=NAT["grid"], lw=0.8, zorder=0)
axCur.set_axisbelow(True)
for s in axCur.spines.values():
    s.set_edgecolor(SPINE); s.set_linewidth(1.1)

# legend (right of curves)
handles = [Line2D([0], [0], color=col, lw=2.6, marker="o", markersize=7,
                  markeredgecolor="white",
                  label=disp) for _, disp, col in TEACHERS]
handles.append(Line2D([0], [0], color="#888888", lw=0, marker="o",
                      markersize=10, markerfacecolor="none",
                      markeredgewidth=1.8, label="peak $\\tau$ (per teacher)"))
axLeg.legend(handles=handles, loc="center left", frameon=True, fontsize=11.5,
             handlelength=1.8, borderpad=0.9, labelspacing=0.9,
             edgecolor="#D8DCE0")

# ===========================================================================
# PANEL LABELS  a / b  +  colour-key note
# ===========================================================================
fig.canvas.draw()
fig.text(0.010, 0.945, "a", fontsize=26, fontweight="bold", color=NAT["ink"])
fig.text(0.010, 0.455, "b", fontsize=26, fontweight="bold", color=NAT["ink"])

fig.text(0.512, 0.965,
         "Hyperparameter accuracy landscape across all five distillation teachers  "
         "(reference task: enhancers_types)",
         ha="center", va="center", fontsize=13.0, color=NAT["ink"])
fig.text(0.512, 0.020,
         "Heatmap colour encodes mean best-test MCC relative to each teacher's own grid mean "
         "(magenta = below ↔ blue = above); ★ marks the best (KL, MSE) cell.  "
         "Caduceus was not swept at MSE = 0 (hatched n/a).",
         ha="center", va="center", fontsize=9.6, style="italic", color="#6A707A")

fig.savefig("output/fig_appendix_hp.pdf", bbox_inches="tight")
fig.savefig("output/fig_appendix_hp.png", dpi=200, bbox_inches="tight")
shutil.copy("output/fig_appendix_hp.png", "previews/fig_appendix_hp.png")

print("saved fig_appendix_hp (pdf + png + preview copy)")
for key, disp, _ in TEACHERS:
    bk, bm, bv = best_info[key]
    G, N = GRID[key]
    filled = int(np.sum(~np.isnan(G)))
    print(f"  {disp:24s} cells={filled}/16  best MCC={bv:.4f} @ KL={bk:g},MSE={bm:g}  "
          f"n={int(np.nansum(N))}")
