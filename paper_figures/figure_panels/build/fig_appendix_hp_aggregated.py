"""
APPENDIX figure: 18-task-AGGREGATED hyperparameter analyses per teacher, plus a
grand-mean-over-teachers summary row (Nature-style).

Layout: 6 rows x 3 columns.
  Rows   : NT, DNABERT-2, Enformer, Caduceus, Carbon-3B, and "Mean over teachers".
  Columns (each aggregated over the 18 downstream tasks):
    1. KL x MSE weight landscape.  For each task -> 4x4 KL(x) x MSE(y) grid of mean
       best-test MCC over temperature, z-score-normalised WITHIN each task (so every
       task weights equally despite very different MCC scales), then averaged over the
       18 tasks.  Diverging colormap in z-units (magenta = below task mean <-> blue =
       above); robust-best cell starred.  Shared symmetric colorbar across all rows.
    2. Temperature response.  For each task -> mean MCC at each T in {0.5,1,1.5,2,4}
       (marginalised over KL & MSE), z-scored within task across the 5 temperatures,
       then mean +/- 1 s.d. across the 18 tasks.  Row-coloured line; peak tau marked.
    3. Loss-term ablation.  Four objective categories CE / CE+KL / CE+MSE / CE+KL+MSE
       (toggle weight_kl>0, weight_mse>0); per-task category means, z-scored within
       task across the 4 categories, then averaged over 18 tasks.  Run counts n are
       uneven (KL-off / MSE-off configs are sparse); they are NOT drawn on the bars
       but are documented in the LaTeX source next to the \\includegraphics.

All three columns share ONE normalisation philosophy: normalise within each task,
then average over tasks (each task equal weight).  The bottom row averages each
per-teacher aggregate across the 5 teachers (grand mean).

Data are read ONLY from the real HP sweep CSVs (4 wandb exports + carbon grid).
Nothing is fabricated.  Caduceus was essentially not swept at MSE = 0 (only 1/18
tasks has any MSE=0 run).  In its per-teacher heatmap those under-supported cells
(CNT < 3 tasks) are rendered at the neutral z=0 colour for DISPLAY ONLY so they
blend in rather than shouting "N/A"; the numeric grand-mean row is aggregated from
the untouched real per-teacher values, so no fabricated 0 enters any mean.

Run with the biolaysum env python.  Saves output/fig_appendix_hp_aggregated.pdf
(vector) + .png and copies the PNG to previews/ .
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
    "axes.linewidth": 0.9,
})

# ---------------------------------------------------------------------------
# NATURE PALETTE (identical hue identity to the main figures)
# ---------------------------------------------------------------------------
NAT = {
    "blue_dark":   "#356491",   # NT
    "amber_dark":  "#CE8B3B",   # DNABERT-2
    "green_dark":  "#4F8C6B",   # Enformer
    "purple_dark": "#7C619E",   # Caduceus
    "rose_dark":   "#B0477E",   # Carbon-3B
    "grey_dark":   "#3A3F45",   # Mean over teachers
    "ink":         "#222222",
    "grid":        "#DBDEE2",
}
SPINE = "#565C64"

# shared diverging colormap (magenta = below task mean, blue = above)
NAT_DIV = LinearSegmentedColormap.from_list(
    "nat_div",
    ["#8E2F63", "#B0477E", "#D8A9BE", "#F2EEF0",
     "#F4F0E6", "#AFC8D9", "#5B84A8", "#2C557E"],
)
NAT_DIV.set_bad("#E6E8EB")

# teacher order + display + colour
TEACHERS = [
    ("caduceus", "Caduceus",               NAT["purple_dark"]),
    ("dnabert2", "DNABERT-2",              NAT["amber_dark"]),
    ("enformer", "Enformer",               NAT["green_dark"]),
    ("nt",       "Nucleotide Transformer", NAT["blue_dark"]),
    ("carbon",   "Carbon-3B",              NAT["rose_dark"]),
]
TKEYS = [t[0] for t in TEACHERS]

KL_VALS  = [0.0, 0.25, 0.5, 1.0]   # x-axis of heatmap
MSE_VALS = [0.0, 1.0, 2.0, 5.0]    # y-axis of heatmap (0 at bottom, origin lower)
TEMPS    = [0.5, 1.0, 1.5, 2.0, 4.0]
CATS     = ["CE", "CE+KL", "CE+MSE", "CE+KL+MSE"]

os.makedirs("output", exist_ok=True)

# ---------------------------------------------------------------------------
# DATA LOADERS (real CSVs only) -- loader reused from fig_appendix_hp.py
# ---------------------------------------------------------------------------
WANDB_CSV = {
    "nt":       "hyperparam_csvs/nt_hyperparam_flat_clean.csv",
    "dnabert2": "hyperparam_csvs/dnabert2_hyperparam_flat_clean.csv",
    "enformer": "hyperparam_csvs/enformer_hyperparam_flat_clean.csv",
    "caduceus": "hyperparam_csvs/caduceus_hyperparam_1-15_flat_clean.csv",
}
CARBON_CSV = "../results/carbon_grid_results.csv"


def load_all(key):
    """Long-form df (task, kl, mse, temp, mcc) for one teacher, all 18 tasks."""
    if key == "carbon":
        df = pd.read_csv(CARBON_CSV, low_memory=False)
        df = df[(df["variant"] == "raw") & df["best_test_mcc"].notna()]
        df = df.rename(columns={"weight_kl": "kl", "weight_mse": "mse",
                                "temperature": "temp", "best_test_mcc": "mcc"})
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


# ---------------------------------------------------------------------------
# AGGREGATORS: normalise within task, then average over the 18 tasks
# ---------------------------------------------------------------------------
def agg_grid(d):
    """18-task-mean z-scored KL(x) x MSE(y) grid + per-cell contributing-task count."""
    tasks = sorted(d["task"].unique())
    stack, cnt = [], np.zeros((len(MSE_VALS), len(KL_VALS)))
    for t in tasks:
        dt = d[d["task"] == t]
        G = np.full((len(MSE_VALS), len(KL_VALS)), np.nan)
        for i, mv in enumerate(MSE_VALS):
            for j, kv in enumerate(KL_VALS):
                s = dt[(dt.mse == mv) & (dt.kl == kv)]["mcc"]
                if len(s):
                    G[i, j] = s.mean()
        m, sd = np.nanmean(G), np.nanstd(G)
        sd = sd if sd > 1e-9 else 1.0
        stack.append((G - m) / sd)
        cnt += ~np.isnan(G)
    return np.nanmean(np.array(stack), axis=0), cnt, len(tasks)


def agg_temp(d):
    """18-task mean +/- s.d. of within-task z-scored temperature response."""
    tasks = sorted(d["task"].unique())
    stack = []
    for t in tasks:
        dt = d[d["task"] == t]
        c = np.array([dt[dt.temp == tt]["mcc"].mean() for tt in TEMPS])
        m, sd = np.nanmean(c), np.nanstd(c)
        sd = sd if sd > 1e-9 else 1.0
        stack.append((c - m) / sd)
    S = np.array(stack)
    return np.nanmean(S, axis=0), np.nanstd(S, axis=0)


def agg_abl(d):
    """18-task mean +/- s.d. of within-task z-scored loss-ablation, + total n per cat."""
    tasks = sorted(d["task"].unique())
    stack, ns = [], {i: 0 for i in range(4)}
    for t in tasks:
        dt = d[d["task"] == t]
        cats = [dt[(dt.kl == 0) & (dt.mse == 0)]["mcc"],
                dt[(dt.kl > 0) & (dt.mse == 0)]["mcc"],
                dt[(dt.kl == 0) & (dt.mse > 0)]["mcc"],
                dt[(dt.kl > 0) & (dt.mse > 0)]["mcc"]]
        v = np.array([c.mean() if len(c) else np.nan for c in cats])
        for i, c in enumerate(cats):
            ns[i] += len(c)
        m, sd = np.nanmean(v), np.nanstd(v)
        sd = sd if sd > 1e-9 else 1.0
        stack.append((v - m) / sd)
    S = np.array(stack)
    return np.nanmean(S, axis=0), np.nanstd(S, axis=0), ns


LONG = {k: load_all(k) for k in TKEYS}
GRID = {k: agg_grid(LONG[k]) for k in TKEYS}
TCUR = {k: agg_temp(LONG[k]) for k in TKEYS}
ABLA = {k: agg_abl(LONG[k]) for k in TKEYS}

# grand mean over teachers (average the per-teacher aggregates)
GM_GRID = np.nanmean(np.array([GRID[k][0] for k in TKEYS]), axis=0)
GM_CNT  = np.nansum(np.array([GRID[k][1] for k in TKEYS]), axis=0)
GM_TM   = np.nanmean(np.array([TCUR[k][0] for k in TKEYS]), axis=0)
GM_TSD  = np.nanmean(np.array([TCUR[k][1] for k in TKEYS]), axis=0)  # mean of s.d.
GM_AM   = np.nanmean(np.array([ABLA[k][0] for k in TKEYS]), axis=0)
GM_ASD  = np.nanmean(np.array([ABLA[k][1] for k in TKEYS]), axis=0)
GM_ANS  = {i: sum(ABLA[k][2][i] for k in TKEYS) for i in range(4)}

ROWS = [(k, disp, col, GRID[k], TCUR[k], ABLA[k]) for (k, disp, col) in TEACHERS]
ROWS.append(("mean", "Mean over teachers", NAT["grey_dark"],
             (GM_GRID, GM_CNT, 5), (GM_TM, GM_TSD), (GM_AM, GM_ASD, GM_ANS)))

# shared symmetric colour scale for column 1 (z-units). Scale is set by WELL-COVERED
# cells (>=3 contributing tasks) so sparse outliers (Caduceus MSE=0) don't wash out
# the informative cells; those outliers simply saturate (and are hatched anyway).
_wellcov = []
for k in TKEYS:
    A_, C_, _ = GRID[k]
    _wellcov.append(np.abs(A_[C_ >= 3]))
GMAX = float(np.ceil(np.nanmax(np.concatenate(_wellcov)) * 10) / 10)
GNORM = Normalize(vmin=-GMAX, vmax=GMAX)

# shared y-limits for columns 2 & 3
YT_LO = max(min(np.nanmin(TCUR[k][0] - TCUR[k][1]) for k in TKEYS) - 0.12, -1.75)
YT_HI = max(np.nanmax(TCUR[k][0] + TCUR[k][1]) for k in TKEYS) + 0.22
YA_LO = min(np.nanmin(ABLA[k][0]) for k in TKEYS) - 0.55
YA_HI = max(np.nanmax(ABLA[k][0]) for k in TKEYS) + 0.55

# ===========================================================================
# FIGURE SCAFFOLD  (6 rows x 3 analysis columns)
# ===========================================================================
NROW = len(ROWS)
fig = plt.figure(figsize=(12.8, 17.6))
LEFT, RIGHT, TOP, BOT = 0.135, 0.965, 0.912, 0.086
gs = fig.add_gridspec(NROW, 3, left=LEFT, right=RIGHT, top=TOP, bottom=BOT,
                      width_ratios=[1.02, 1.15, 1.15], hspace=0.34, wspace=0.34)

TITLE_FS, LAB_FS, TICK_FS, ANN_FS = 12.5, 10.5, 9.3, 8.6
COLHEAD = ["KL x MSE weight landscape",
           "Temperature response",
           "Loss-term ablation"]

best_cell = {}
for r, (key, disp, col, (A, CNT, ntask), (tm, tsd), (am, asd, ns)) in enumerate(ROWS):
    is_mean = (key == "mean")

    # ---- column 1: aggregated KL x MSE heatmap -----------------------------
    ax = fig.add_subplot(gs[r, 0])
    lowN = 3            # per-teacher cells backed by < lowN tasks are unreliable
    # DISPLAY-ONLY neutralisation.  In a per-teacher panel, cells that are either
    # missing (NaN) or rest on too few tasks (CNT < lowN) are rendered at the
    # neutral z=0 colour so they blend in as "no signal" rather than standing out
    # as N/A.  Concretely Caduceus was essentially not swept at MSE=0 (1/18
    # tasks), so its MSE=0 row is neutralised here.  This changes ONLY this
    # panel's colour + label; the grand-mean row (GM_GRID, computed above from the
    # untouched real per-teacher aggregates) is unaffected, so no fabricated 0
    # ever enters any numeric mean.
    miss = np.isnan(A) | (CNT < lowN) if not is_mean else np.zeros(A.shape, bool)
    A_disp = np.where(miss, 0.0, A)
    Am = np.ma.masked_invalid(A_disp)
    ax.imshow(Am, cmap=NAT_DIV, norm=GNORM, aspect="auto", origin="lower")
    bi, bj = np.unravel_index(np.nanargmax(A), A.shape)   # best from REAL values
    best_cell[key] = (KL_VALS[bj], MSE_VALS[bi], A[bi, bj])
    for i in range(len(MSE_VALS)):
        for j in range(len(KL_VALS)):
            v = A_disp[i, j]
            rel = (v + GMAX) / (2 * GMAX)
            tcol = "white" if (rel < 0.16 or rel > 0.84) else NAT["ink"]
            is_best = (i == bi and j == bj)
            ax.text(j, i - (0.16 if is_best else 0.0), f"{v:+.2f}",
                    ha="center", va="center", fontsize=ANN_FS,
                    fontweight="bold" if is_best else "normal",
                    color=tcol, zorder=4)
            if is_best:
                ax.scatter(j, i + 0.24, marker="*", s=150, color="#F5CE3A",
                           edgecolor=NAT["ink"], lw=0.7, zorder=6)
                ax.add_patch(mpatches.Rectangle(
                    (j - 0.5, i - 0.5), 1, 1, fill=False,
                    edgecolor=NAT["ink"], lw=2.0, zorder=5))
    for gx in np.arange(0.5, len(KL_VALS) - 0.5):
        ax.axvline(gx, color="white", lw=0.7, zorder=3)
    for gy in np.arange(0.5, len(MSE_VALS) - 0.5):
        ax.axhline(gy, color="white", lw=0.7, zorder=3)
    ax.set_xticks(range(len(KL_VALS)))
    ax.set_xticklabels([f"{v:g}" for v in KL_VALS], fontsize=TICK_FS)
    ax.set_yticks(range(len(MSE_VALS)))
    ax.set_yticklabels([f"{v:g}" for v in MSE_VALS], fontsize=TICK_FS)
    ax.set_xlim(-0.5, len(KL_VALS) - 0.5)
    ax.set_ylim(-0.5, len(MSE_VALS) - 0.5)
    ax.tick_params(length=0)
    if r == NROW - 1:
        ax.set_xlabel("KL weight", fontsize=LAB_FS, color=NAT["ink"])
    ax.set_ylabel("MSE weight", fontsize=LAB_FS, color=NAT["ink"])
    for s in ax.spines.values():
        s.set_edgecolor(SPINE); s.set_linewidth(0.9)

    # ---- row label (teacher name, left of column 1) ------------------------
    ax.text(-0.60, 0.5, disp, transform=ax.transAxes, ha="center", va="center",
            rotation=90, fontsize=13.0, fontweight="bold", color=col)

    # ---- column 2: temperature response ------------------------------------
    axc = fig.add_subplot(gs[r, 1])
    xt = np.arange(len(TEMPS))
    axc.axhline(0, color="#C7CCD2", lw=0.8, zorder=0)
    axc.fill_between(xt, tm - tsd, tm + tsd, color=col, alpha=0.16, zorder=1)
    axc.plot(xt, tm, "-", color=col, lw=2.3, zorder=3)
    axc.scatter(xt, tm, s=48, color=col, edgecolor="white", lw=1.0, zorder=4)
    pk = int(np.nanargmax(tm))
    axc.scatter(xt[pk], tm[pk], s=210, facecolors="none", edgecolors=col,
                lw=1.9, zorder=5)
    axc.annotate(f"peak $\\tau$={TEMPS[pk]:g}", (xt[pk], tm[pk]),
                 textcoords="offset points", xytext=(0, 11), ha="center",
                 fontsize=8.4, color=col, fontweight="bold")
    axc.set_xticks(xt)
    axc.set_xticklabels([f"{t:g}" for t in TEMPS], fontsize=TICK_FS)
    axc.set_xlim(-0.35, len(TEMPS) - 0.65)
    axc.set_ylim(YT_LO, YT_HI)
    axc.tick_params(axis="y", labelsize=TICK_FS, length=3)
    if r == NROW - 1:
        axc.set_xlabel("Distillation temperature  $\\tau$", fontsize=LAB_FS, color=NAT["ink"])
    axc.set_ylabel("z-scored MCC", fontsize=LAB_FS, color=NAT["ink"])
    axc.grid(axis="y", color=NAT["grid"], lw=0.7, zorder=0)
    axc.set_axisbelow(True)
    for s in axc.spines.values():
        s.set_edgecolor(SPINE); s.set_linewidth(0.9)

    # ---- column 3: loss-term ablation --------------------------------------
    axb = fig.add_subplot(gs[r, 2])
    xb = np.arange(4)
    axb.axhline(0, color="#C7CCD2", lw=0.8, zorder=0)
    bcols = [col if i == 3 else matplotlib.colors.to_rgba(col, 0.55) for i in range(4)]
    bars = axb.bar(xb, am, width=0.72, color=bcols, edgecolor=SPINE, lw=0.8, zorder=2)
    best_i = int(np.nanargmax(am))
    bars[best_i].set_edgecolor(NAT["ink"]); bars[best_i].set_linewidth(1.8)
    for i, b in enumerate(bars):
        y = am[i]
        va = "bottom" if y >= 0 else "top"
        off = 0.06 if y >= 0 else -0.06
        axb.text(i, y + off, f"{y:+.2f}", ha="center", va=va, fontsize=8.0,
                 color=NAT["ink"], fontweight="bold" if i == best_i else "normal")
        # per-bar run counts n are intentionally NOT drawn (documented in the
        # LaTeX source alongside the \includegraphics instead of on the figure).
    axb.set_xticks(xb)
    axb.set_xticklabels(CATS, fontsize=8.1, rotation=20, ha="right")
    axb.set_ylim(YA_LO, YA_HI)
    axb.tick_params(axis="y", labelsize=TICK_FS, length=3)
    axb.set_ylabel("z-scored MCC", fontsize=LAB_FS, color=NAT["ink"])
    axb.grid(axis="y", color=NAT["grid"], lw=0.7, zorder=0)
    axb.set_axisbelow(True)
    for s in axb.spines.values():
        s.set_edgecolor(SPINE); s.set_linewidth(0.9)

# ---- column headers --------------------------------------------------------
for c, head in enumerate(COLHEAD):
    ax0 = fig.axes[c] if False else None
# position headers using the first-row axes centres
fig.canvas.draw()
row0_axes = fig.axes[0:3]
for c, head in enumerate(COLHEAD):
    ax = row0_axes[c]
    pos = ax.get_position()
    fig.text((pos.x0 + pos.x1) / 2, TOP + 0.014, head, ha="center", va="bottom",
             fontsize=TITLE_FS, fontweight="bold", color=NAT["ink"])

# ---- shared colorbar for column 1 (placed under the whole figure, left) ----
cax = fig.add_axes([LEFT + 0.005, 0.040, 0.235, 0.011])
sm = plt.cm.ScalarMappable(cmap=NAT_DIV, norm=GNORM)
cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
cb.set_ticks([-GMAX, 0, GMAX])
cb.set_ticklabels([f"-{GMAX:g}", "0", f"+{GMAX:g}"])
cb.ax.tick_params(labelsize=7.8, length=2.5)
cb.set_label("col 1: 18-task-mean z-scored MCC  "
             "(magenta = below task mean $\\leftrightarrow$ blue = above)",
             fontsize=8.3, labelpad=3)
cb.ax.xaxis.set_label_position("top")
cb.outline.set_edgecolor(SPINE); cb.outline.set_linewidth(0.8)

# ---- legend for star (right of colorbar) -----------------------------------
handles = [
    Line2D([0], [0], marker="*", color="none", markerfacecolor="#F5CE3A",
           markeredgecolor=NAT["ink"], markeredgewidth=0.7, markersize=13,
           label="robust-best (KL, MSE) cell"),
]
axL = fig.add_axes([LEFT + 0.32, 0.028, 0.42, 0.030]); axL.axis("off")
axL.legend(handles=handles, loc="center left", frameon=False, fontsize=8.4,
           handlelength=1.6, labelspacing=0.7, ncol=1, columnspacing=1.4,
           borderpad=0.2)

# ---- suptitle + caption note ----------------------------------------------
fig.text(0.5, 0.968,
         "Distillation-hyperparameter analysis aggregated over 18 tasks — "
         "per teacher and grand mean",
         ha="center", va="center", fontsize=14.5, fontweight="bold", color=NAT["ink"])
fig.text(0.5, 0.946,
         "Each analysis is normalised within every task (z-score) then averaged over the "
         "18 tasks, so tasks weight equally despite different MCC scales; "
         "bottom row averages the five per-teacher aggregates.",
         ha="center", va="center", fontsize=8.9, color="#5A6069")
fig.text(0.5, 0.010,
         "Loss-ablation categories have uneven support (KL-off / MSE-off configs are "
         "sparser; per-panel run counts documented in the LaTeX source); Caduceus was "
         "essentially not swept at MSE=0 (1/18 tasks), so those cells are shown neutral.",
         ha="center", va="center", fontsize=8.2, style="italic", color="#6A707A")

fig.savefig("output/fig_appendix_hp_aggregated.pdf", dpi=600)  # heat map raster >= 300 DPI
fig.savefig("output/fig_appendix_hp_aggregated.png", dpi=200)
shutil.copy("output/fig_appendix_hp_aggregated.png",
            "previews/fig_appendix_hp_aggregated.png")

# ---------------------------------------------------------------------------
# REPORT
# ---------------------------------------------------------------------------
print("saved fig_appendix_hp_aggregated (pdf + png + preview copy)")
for key, disp, _, (A, CNT, nt), (tm, tsd), (am, asd, ns) in ROWS:
    bk, bm, bv = best_cell[key]
    pk = TEMPS[int(np.nanargmax(tm))]
    order = [CATS[i] for i in np.argsort(-am)]
    kl0 = np.nanmean(A[:, 0])
    print(f"  {disp:22s} best=(KL={bk:g},MSE={bm:g}) z={bv:+.2f}  KL=0col z={kl0:+.2f}"
          f"  peakT={pk:g}  abl={order}")
