"""
SUPPLEMENTARY "hyperparameter atlas" for OmegaGenome (Nature-style).

A dense small-multiples grid: 18 tasks x 5 distillation teachers.  Each cell of
the big grid is a mini 4x4 KL(x) x MSE(y) heatmap of mean best-test MCC over
temperature for that (task, teacher).  Absolute MCC varies enormously across
tasks, so every mini-heatmap is normalised to ITS OWN (task,teacher) range:
    rel = (MCC - tile_mean) / max|MCC - tile_mean|   in [-1, +1]
so colour encodes WHERE the hyperparameter optimum sits, not absolute accuracy.
A single shared diverging colormap (magenta = below tile-mean <-> blue = above)
with ONE shared colorbar.  The best (KL, MSE) cell in each tile is starred.

To keep 18x5 legible on one page it is split into two side-by-side blocks of
9 tasks each (still one figure, one file).  Rows are grouped by task family
(histone marks -> promoters/enhancers/CREs -> splice), shown by a coloured
family rail on the left of each block.  Teacher names sit atop each block in
their palette colour.  Missing / sparsely-sampled cells (Caduceus MSE=0 row,
outside its standard search grid for most tasks; one extra splice_acceptors
cell) are left at a subtle neutral "no-data" grey -- never fabricated and never
painted with a data-colormap z-value.

Data are read ONLY from the real HP sweep CSVs (4 wandb exports + carbon grid).

Run with the biolaysum env python.  Saves output/fig_hp_atlas_18task.pdf
(vector) + .png and copies the PNG to previews/fig_hp_atlas_18task.png .
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
from matplotlib.colors import LinearSegmentedColormap, Normalize

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "axes.linewidth": 0.8,
})

# ---------------------------------------------------------------------------
# NATURE PALETTE  (identical hue identity to the main figures)
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
SPINE = "#5A6069"
TILE_EDGE = "#9AA0A8"

# shared diverging colormap (magenta = below tile-mean, blue = above)
NAT_DIV = LinearSegmentedColormap.from_list(
    "nat_div",
    ["#8E2F63", "#B0477E", "#D8A9BE", "#F2EEF0",
     "#F4F0E6", "#AFC8D9", "#5B84A8", "#2C557E"],
)
NAT_DIV.set_bad("#E6E8EB")   # NaN cells -> light grey
DIVNORM = Normalize(vmin=-1.0, vmax=1.0)

# teacher order + display + colour
TEACHERS = [
    ("caduceus", "Caduceus",                NAT["purple_dark"]),
    ("dnabert2", "DNABERT-2",               NAT["amber_dark"]),
    ("enformer", "Enformer",                NAT["green_dark"]),
    ("nt",       "Nucleotide\nTransformer", NAT["blue_dark"]),
    ("carbon",   "Carbon-3B",               NAT["rose_dark"]),
]
TKEYS = [t[0] for t in TEACHERS]

# task families -> ordered task list + display label + rail colour
FAMILIES = [
    ("Histone marks", "#7A8794", [
        "H2AFZ", "H3K27ac", "H3K27me3", "H3K36me3", "H3K4me1",
        "H3K4me2", "H3K4me3", "H3K9ac", "H3K9me3", "H4K20me1"]),
    ("Promoters / enhancers", "#B58A3C", [
        "enhancers", "enhancers_types", "promoter_all",
        "promoter_no_tata", "promoter_tata"]),
    ("Splice sites", "#4F8C6B", [
        "splice_sites_acceptors", "splice_sites_all", "splice_sites_donors"]),
]
# nicer row labels
DISP = {
    "H2AFZ": "H2A.Z", "H4K20me1": "H4K20me1",
    "enhancers": "enhancers", "enhancers_types": "enhancers (types)",
    "promoter_all": "promoter (all)", "promoter_no_tata": "promoter (no TATA)",
    "promoter_tata": "promoter (TATA)",
    "splice_sites_acceptors": "splice acceptor",
    "splice_sites_all": "splice (all)", "splice_sites_donors": "splice donor",
}
# flat task order + per-task family colour
TASK_ORDER, TASK_FAM = [], {}
for fname, fcol, ts in FAMILIES:
    for t in ts:
        TASK_ORDER.append(t)
        TASK_FAM[t] = (fname, fcol)

KL_VALS  = [0.0, 0.25, 0.5, 1.0]   # x-axis of every mini-heatmap
MSE_VALS = [0.0, 1.0, 2.0, 5.0]    # y-axis of every mini-heatmap (0 at bottom)

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


def load_all(key):
    """Long-form df (task, kl, mse, temp, mcc) for one teacher, all tasks."""
    if key == "carbon":
        df = pd.read_csv(CARBON_CSV, low_memory=False)
        df = df[(df["variant"] == "raw") & df["best_test_mcc"].notna()]
        df = df.rename(columns={"weight_kl": "kl", "weight_mse": "mse",
                                "temperature": "temp", "best_test_mcc": "mcc",
                                "task": "task"})
    else:
        df = pd.read_csv(WANDB_CSV[key], low_memory=False)
        TC = "config.dataset_config.task_name"
        df = df[df["summary.best_test/mcc"].notna()]
        df = df.rename(columns={
            "config.distillation_config.weight_kl": "kl",
            "config.distillation_config.weight_mse": "mse",
            "config.distillation_config.temperature": "temp",
            "summary.best_test/mcc": "mcc", TC: "task"})
    df = df[["task", "kl", "mse", "temp", "mcc"]].copy()
    return df[df["kl"].isin(KL_VALS) & df["mse"].isin(MSE_VALS)]


def grid_of(long, task):
    """4x4 grid (rows=MSE ascending upward, cols=KL) of mean MCC over temp."""
    d = long[long["task"] == task]
    G = np.full((len(MSE_VALS), len(KL_VALS)), np.nan)
    N = np.zeros_like(G)
    for i, mv in enumerate(MSE_VALS):
        for j, kv in enumerate(KL_VALS):
            s = d[(d["mse"] == mv) & (d["kl"] == kv)]["mcc"]
            if len(s):
                G[i, j] = s.mean()
                N[i, j] = len(s)
    return G, N


LONG = {k: load_all(k) for k in TKEYS}

# ---------------------------------------------------------------------------
# FIGURE SCAFFOLD  -- two side-by-side blocks of 9 tasks each
# ---------------------------------------------------------------------------
BLOCKS = [TASK_ORDER[:9], TASK_ORDER[9:]]

FIGW, FIGH = 15.6, 11.0
fig = plt.figure(figsize=(FIGW, FIGH))

# global layout box (figure fraction)
TOP    = 0.895          # below suptitle
BOT    = 0.105          # above colorbar + caption
GAPX   = 0.052          # gap between the two blocks
LMARG  = 0.012
RMARG  = 0.010
RAILW  = 0.011          # family rail width (fig frac)
LABW   = 0.086          # task-label strip width (fig frac)
HDR    = 0.058          # teacher-header band height (fig frac)
TILEPAD_X = 0.24        # fraction of a column cell used as inter-tile gap
TILEPAD_Y = 0.16

block_w = (1.0 - LMARG - RMARG - GAPX) / 2.0
NR = 9                                   # tasks per block
tiles_top = TOP - HDR
row_h = (tiles_top - BOT) / NR

best_records = []   # (task, teacher, kl, mse, mcc, ncells)


def draw_block(bx0, tasks):
    """Render one 9-task x 5-teacher block starting at figure-x = bx0."""
    tiles_x0 = bx0 + RAILW + LABW
    tiles_w = (bx0 + block_w) - tiles_x0
    col_w = tiles_w / len(TEACHERS)

    # teacher column headers
    for c, (key, disp, col) in enumerate(TEACHERS):
        cx = tiles_x0 + (c + 0.5) * col_w
        fig.text(cx, tiles_top + 0.010, disp, ha="center", va="bottom",
                 fontsize=9.6, color=col, fontweight="bold", rotation=0,
                 linespacing=0.95)

    # family rails + task rows
    fam_runs = []   # (fname, fcol, y_lo, y_hi)
    for r, task in enumerate(tasks):
        # row band spans [ry0, ry1] top-to-bottom (row 0 at top)
        ry_top = tiles_top - r * row_h
        ry_bot = ry_top - row_h
        cy = (ry_top + ry_bot) / 2.0
        fname, fcol = TASK_FAM[task]

        # task label (right-aligned in label strip)
        fig.text(tiles_x0 - 0.006, cy, DISP.get(task, task),
                 ha="right", va="center", fontsize=8.7, color=NAT["ink"])

        # accumulate family runs for rail
        if fam_runs and fam_runs[-1][0] == fname:
            fam_runs[-1] = (fname, fcol, ry_bot, fam_runs[-1][3])
        else:
            fam_runs.append((fname, fcol, ry_bot, ry_top))

        # the five teacher tiles
        for c, key in enumerate(TKEYS):
            G, N = grid_of(LONG[key], task)
            gmean = np.nanmean(G)
            dev = np.nanmax(np.abs(G - gmean)) if np.isfinite(gmean) else np.nan
            if not np.isfinite(dev) or dev < 1e-9:
                REL = np.where(np.isnan(G), np.nan, 0.0)
            else:
                REL = (G - gmean) / dev
            RELm = np.ma.masked_invalid(REL)

            th = row_h * (1 - TILEPAD_Y)
            tw = th * FIGH / FIGW        # square physical tiles (=> square cells)
            tw = min(tw, col_w * (1 - TILEPAD_X))
            tx = tiles_x0 + c * col_w + (col_w - tw) * 0.5
            ty = ry_bot + (row_h - th) * 0.5
            ax = fig.add_axes([tx, ty, tw, th])
            ax.imshow(RELm, cmap=NAT_DIV, norm=DIVNORM, aspect="auto",
                      origin="lower", interpolation="nearest")

            # Missing / sparsely-sampled cells (e.g. Caduceus MSE=0, which was not
            # part of its standard search grid for most tasks) are left at the
            # colormap's subtle "no-data" grey (NAT_DIV.set_bad) -- an unobtrusive
            # neutral shade that reads as "not sampled" without a loud hatch and
            # WITHOUT painting a fabricated data-colour z-value into the tile.

            # thin interior gridlines
            for gx in np.arange(0.5, len(KL_VALS) - 0.5):
                ax.axvline(gx, color="white", lw=0.6, zorder=3)
            for gy in np.arange(0.5, len(MSE_VALS) - 0.5):
                ax.axhline(gy, color="white", lw=0.6, zorder=3)

            # star the best cell
            ncells = int(np.sum(~np.isnan(G)))
            if ncells:
                bi, bj = np.unravel_index(np.nanargmax(G), G.shape)
                ax.scatter(bj, bi, marker="*", s=46, color="#F5CE3A",
                           edgecolor=NAT["ink"], lw=0.55, zorder=6)
                best_records.append((task, key, KL_VALS[bj], MSE_VALS[bi],
                                     G[bi, bj], ncells))

            ax.set_xticks([]); ax.set_yticks([])
            ax.set_xlim(-0.5, len(KL_VALS) - 0.5)
            ax.set_ylim(-0.5, len(MSE_VALS) - 0.5)
            for s in ax.spines.values():
                s.set_edgecolor(TILE_EDGE); s.set_linewidth(0.7)

    # draw family rails + rotated family labels
    for fname, fcol, y_lo, y_hi in fam_runs:
        rx = bx0 + 0.001
        fig.patches.append(mpatches.Rectangle(
            (rx, y_lo + 0.002), RAILW * 0.5, (y_hi - y_lo) - 0.004,
            transform=fig.transFigure, facecolor=fcol, edgecolor="none",
            alpha=0.85, zorder=5))
        fig.text(rx + RAILW * 0.5 + 0.004, (y_lo + y_hi) / 2.0, fname,
                 ha="center", va="center", rotation=90, fontsize=8.4,
                 color=fcol, fontweight="bold")


draw_block(LMARG, BLOCKS[0])
draw_block(LMARG + block_w + GAPX, BLOCKS[1])

# ---------------------------------------------------------------------------
# AXIS-KEY INSET  (explains the KL x MSE layout of every tile, once)
# ---------------------------------------------------------------------------
keyw, keyh = 0.050, 0.070
kx = LMARG + 0.010
ky = 0.028
axk = fig.add_axes([kx, ky, keyw, keyh])
axk.imshow(np.zeros((4, 4)), cmap=LinearSegmentedColormap.from_list(
    "g", ["#EDEFF2", "#EDEFF2"]), aspect="auto", origin="lower")
for gx in np.arange(0.5, 3.5):
    axk.axvline(gx, color="white", lw=0.6)
for gy in np.arange(0.5, 3.5):
    axk.axhline(gy, color="white", lw=0.6)
axk.set_xticks(range(4)); axk.set_yticks(range(4))
axk.set_xticklabels([f"{v:g}" for v in KL_VALS], fontsize=6.6)
axk.set_yticklabels([f"{v:g}" for v in MSE_VALS], fontsize=6.6)
axk.tick_params(length=0, pad=1.5)
axk.set_xlabel("KL weight", fontsize=7.6, labelpad=1.5)
axk.set_ylabel("MSE weight", fontsize=7.6, labelpad=1.5)
for s in axk.spines.values():
    s.set_edgecolor(TILE_EDGE); s.set_linewidth(0.7)
fig.text(kx + keyw / 2.0, ky + keyh + 0.010, "each tile:", ha="center",
         va="bottom", fontsize=7.8, color=NAT["ink"], fontweight="bold")

# ---------------------------------------------------------------------------
# SHARED COLORBAR  (within-tile relative MCC)
# ---------------------------------------------------------------------------
cbx0, cbw = 0.360, 0.30
cbax = fig.add_axes([cbx0, 0.062, cbw, 0.016])
sm = plt.cm.ScalarMappable(cmap=NAT_DIV, norm=DIVNORM)
cb = fig.colorbar(sm, cax=cbax, orientation="horizontal")
cb.set_ticks([-1, 0, 1])
cb.set_ticklabels(["min\n(worst in tile)", "tile mean", "max\n(best in tile)"])
cb.ax.tick_params(labelsize=8.2, length=2.5, pad=2)
cb.set_label("within-tile relative MCC   (magenta = below tile mean   ↔   blue = above)",
             fontsize=9.4, labelpad=4)
cb.ax.xaxis.set_label_position("top")
cb.outline.set_edgecolor(SPINE); cb.outline.set_linewidth(0.8)

# star legend, right of colorbar (missing tiles are shown as subtle neutral grey
# and are intentionally not called out with a loud marker)
lx = cbx0 + cbw + 0.030
handles = [
    Line2D([0], [0], marker="*", color="none", markerfacecolor="#F5CE3A",
           markeredgecolor=NAT["ink"], markeredgewidth=0.55, markersize=12,
           label="best (KL, MSE) cell"),
]
axL = fig.add_axes([lx, 0.040, 0.16, 0.055]); axL.axis("off")
axL.legend(handles=handles, loc="center left", frameon=False, fontsize=9.2,
           handlelength=1.6, labelspacing=0.9, borderpad=0.2)

# ---------------------------------------------------------------------------
# TITLES + NOTE
# ---------------------------------------------------------------------------
fig.text(0.512, 0.963,
         "Hyperparameter atlas: KL x MSE distillation-weight landscape for all "
         "18 tasks across five teachers",
         ha="center", va="center", fontsize=13.5, color=NAT["ink"],
         fontweight="bold")
fig.text(0.512, 0.934,
         "Each mini-heatmap = mean best-test MCC over temperature; colour is "
         "normalised within each (task, teacher) tile so it shows WHERE the "
         "optimum sits, not absolute accuracy.",
         ha="center", va="center", fontsize=9.6, color="#5A6069")

fig.savefig("output/fig_hp_atlas_18task.pdf", dpi=600)  # heat map raster >= 300 DPI
fig.savefig("output/fig_hp_atlas_18task.png", dpi=200)
shutil.copy("output/fig_hp_atlas_18task.png",
            "previews/fig_hp_atlas_18task.png")

# ---------------------------------------------------------------------------
# COVERAGE REPORT
# ---------------------------------------------------------------------------
full = sum(1 for r in best_records if r[5] == 16)
partial = [(t, k, n) for (t, k, _, _, _, n) in best_records if n < 16]
print("saved fig_hp_atlas_18task (pdf + png + preview copy)")
print(f"tiles rendered: {len(best_records)}/90   full 16/16: {full}   "
      f"partial: {len(partial)}")
for t, k, n in partial:
    print(f"  partial  {t:24s} {k:10s} {n}/16")
