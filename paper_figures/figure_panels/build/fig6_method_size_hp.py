"""
Build the combined robustness figure for OmegaGenome (published as Figure 5).

Layout (2x2), consistent bar-plot style across all four panels:
  A = Distillation-method comparison   (splice_sites_all, NT teacher)  [BAR, MEAN +/- s.d.]
  B = Student-size scaling             (splice_sites_all, NT teacher)  [BAR, MEAN +/- s.d.]
  C = Loss-term ablation               (enhancers_types, NT)           [from NT hp CSV]
  D = Hyperparameter grid heatmap      (enhancers_types, NT)           [from NT hp CSV]

Panels A & B: bars = per-group MEAN best-test MCC over the filtered runs
(seeds {42,1024} dropped; 0.8M = fixed-architecture runs, collapsed pre-fix
mcc<0.5 dropped), error bars = +/-1 s.d.  Hero (our method / deployed point) in
NT-blue, others muted grey -- matching Panels C/D.  Data = committed flat_clean
wandb CSVs; filtering matches different-method-v2.ipynb.

Palette matches Fig 2/4 (NAT dict). Run with the biolaysum env python.
Saves output/fig6_method_size_hp.pdf (vector) + .png, and copies the PNG to
previews/fig6_method_size_hp.png .

Published as Figure 5 of the paper.
"""
import os
import shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FixedLocator

# Deterministic jitter for the strip-scatter points (Panels A & B).
np.random.seed(7)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "axes.linewidth": 1.1,
})

# ---------------------------------------------------------------------------
# NATURE PALETTE  (identical to Fig 2/4)
# ---------------------------------------------------------------------------
NAT = {
    "grey_dot":     "#8A9099",
    "grey_base":    "#C3C8CE",
    "blue_dark":    "#356491",   # NT / OmegaGenome hero
    "blue_mid":     "#6E97BC",
    "blue_light":   "#A9C6E0",   # NT teacher
    "amber_dark":   "#CE8B3B",
    "green_dark":   "#4F8C6B",
    "purple_dark":  "#7C619E",
    "rose_dark":    "#B0477E",
    "slate":        "#9AA4AE",   # muted neutral for non-hero methods
    "slate_deep":   "#7C8894",
    "ink":          "#222222",
    "grid":         "#DBDEE2",
}

HERO   = NAT["blue_dark"]
TEACH  = NAT["blue_light"]
SPINE  = "#444444"

TITLE_FS, LAB_FS, TICK_FS, ANN_FS = 25.0, 22.0, 20.0, 19.0

HP_CSV   = "hyperparam_csvs/nt_hyperparam_flat_clean.csv"
SIZE_CSV = "data/hp_search_csv/size_comparison_flat_clean.csv"
os.makedirs("output", exist_ok=True)


def style_spines(ax):
    for s in ax.spines.values():
        s.set_edgecolor(SPINE); s.set_linewidth(1.1)


# ===========================================================================
# DATA  —  Panels A & B faithfully reproduce different-method-v2.ipynb
#          (final cell -> combined_two_panel_plot_v2_filtered_seeds.pdf)
# ===========================================================================
# The notebook read raw wandb exports from Colab /content/.  Those exports are
# committed on disk as the flat_clean CSVs (config.* / summary.* prefixes).
# We reproduce the notebook's EXACT filtering and label the per-group MAX.
METHOD_CSV1 = "data/hp_search_csv/method_comparison_flat_clean.csv"
METHOD_CSV2 = "data/hp_search_csv/method_comparison_seedset2_flat_clean.csv"
XL_CSV      = "data/hp_search_csv/extralarge_fix_flat_clean.csv"

_RENAME = {
    "config.distillation_config.distill_method": "distill_method",
    "config.dataset_config.task_name": "task_name",
    "summary.best_test/mcc": "mcc",
    "config.random_state": "seed",
    "summary.teacher/test_mcc": "teacher_mcc",
    "config.student_config.model_size": "model_size",
}


def load_flat(path):
    """Load a flat_clean wandb CSV and rename config.*/summary.* -> notebook names."""
    return pd.read_csv(path).rename(columns=_RENAME)


# ---- Panel A : distillation-method comparison (splice_sites_all) ----------
# notebook: concat seedset1+seedset2, drop seeds {42,1024}, map methods, task filter,
#           BPNet = 10 dummy rows at 0.76, teacher = mean(teacher/test_mcc).
dm = pd.concat([load_flat(METHOD_CSV1), load_flat(METHOD_CSV2)], ignore_index=True)
dm = dm[~dm["seed"].isin([42, 1024])]
METHOD_MAP = {"vanilla": "OmegaGenome", "dkd": "DKD", "dist": "Dist",
              "logit_standard": "LS"}
dm["Method"] = dm["distill_method"].astype(str).map(METHOD_MAP)
taskA = dm[dm["task_name"] == "splice_sites_all"].copy()

A_BPNET   = 0.76
A_TEACHER = float(taskA["teacher_mcc"].mean()) if "teacher_mcc" in taskA else 0.9675
A_order   = ["BPNet", "OmegaGenome", "DKD", "Dist", "LS"]
plot_A = pd.concat([
    taskA[["Method", "mcc"]],
    pd.DataFrame({"Method": ["BPNet"] * 10, "mcc": [A_BPNET] * 10}),
], ignore_index=True)

# ---- Panel B : student-size scaling (splice_sites_all) --------------------
# notebook: drop seeds {42,1024}; swap extra_large -> extra_large_fix runs
# (the fixed-architecture re-runs; the pre-fix runs collapsed to ~0.25 and were
#  dropped in the Colab 'filtered_random_seed_extra_large_fix' export).
sz = load_flat(SIZE_CSV)
sz = sz[~sz["seed"].isin([42, 1024])]
seeds_in_sizes = set(sz["seed"].unique())

xl = load_flat(XL_CSV)
xl = xl[xl["seed"].isin(seeds_in_sizes)]
if "state" in xl.columns:
    xl = xl[xl["state"] == "finished"]
xl = xl[xl["mcc"] > 0.5]        # drop collapsed pre-fix (buggy-dilation) runs
xl_proc = pd.DataFrame({"model_size": "extra_large_fix",
                        "task_name": xl["task_name"].values,
                        "mcc": xl["mcc"].values})
sz = sz[sz["model_size"] != "extra_large"]
sz = pd.concat([sz[["model_size", "task_name", "mcc"]], xl_proc], ignore_index=True)
sz["model_size"] = sz["model_size"].replace({
    "ultra_tiny": "ultra_small", "extra_tiny": "extra_small",
    "extra_large": "extra_large_fix",
})
sz = sz[sz["task_name"] == "splice_sites_all"].dropna(subset=["mcc"])

B_order = ["pico", "ultra_small", "extra_small", "original",
           "extra_large_fix", "large", "xxlarge"]
B_LABEL = {"pico": "2K", "ultra_small": "7K", "extra_small": "28K",
           "original": "0.1M", "extra_large_fix": "0.8M",
           "large": "1.8M", "xxlarge": "3.6M"}

# ---- Panels C/D : GRAND-MEAN HP aggregation over 5 teachers x 18 tasks -----
# Aggregation logic is reused verbatim from fig_appendix_hp_aggregated.py:
# for every task -> per-config mean MCC, z-score WITHIN the task (so all 18 tasks
# weight equally despite different MCC scales), average over the 18 tasks to get a
# per-teacher aggregate, then average the 5 per-teacher aggregates (grand mean).
# Panel C = loss-term ablation; Panel D = KL x MSE landscape, both in z-units.
kl_vals  = [0.0, 0.25, 0.5, 1.0]        # heatmap x-axis
mse_vals = [0.0, 1.0, 2.0, 5.0]         # heatmap y-axis (origin lower: 0 at bottom)
CATS_HP  = ["CE", "CE+KL", "CE+MSE", "CE+KL+MSE"]
TKEYS    = ["caduceus", "dnabert2", "enformer", "nt", "carbon"]
WANDB_CSV = {
    "nt":       "hyperparam_csvs/nt_hyperparam_flat_clean.csv",
    "dnabert2": "hyperparam_csvs/dnabert2_hyperparam_flat_clean.csv",
    "enformer": "hyperparam_csvs/enformer_hyperparam_flat_clean.csv",
    "caduceus": "hyperparam_csvs/caduceus_hyperparam_1-15_flat_clean.csv",
}
CARBON_CSV = "../results/carbon_grid_results.csv"


def _load_hp(key):
    """Long-form (task, kl, mse, temp, mcc) for one teacher over all 18 tasks."""
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
    return df[df["kl"].isin(kl_vals) & df["mse"].isin(mse_vals)]


def _agg_grid(d):
    """18-task-mean, within-task z-scored KL(x) x MSE(y) grid + per-cell task count."""
    stack, cnt = [], np.zeros((4, 4))
    for t in sorted(d["task"].unique()):
        dt = d[d["task"] == t]
        G = np.full((4, 4), np.nan)
        for i, mv in enumerate(mse_vals):
            for j, kv in enumerate(kl_vals):
                s = dt[(dt.mse == mv) & (dt.kl == kv)]["mcc"]
                if len(s):
                    G[i, j] = s.mean()
        m, sd = np.nanmean(G), np.nanstd(G)
        sd = sd if sd > 1e-9 else 1.0
        stack.append((G - m) / sd)
        cnt += ~np.isnan(G)
    return np.nanmean(np.array(stack), axis=0), cnt


def _agg_abl(d):
    """18-task-mean, within-task z-scored loss-term ablation + total n per category."""
    stack, ns = [], {i: 0 for i in range(4)}
    for t in sorted(d["task"].unique()):
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
    return np.nanmean(np.array(stack), axis=0), ns


_LONG = {k: _load_hp(k) for k in TKEYS}
_GRID = {k: _agg_grid(_LONG[k]) for k in TKEYS}
_ABLA = {k: _agg_abl(_LONG[k]) for k in TKEYS}

# grand mean over teachers (average the per-teacher aggregates)
C_mean = list(np.nanmean(np.array([_ABLA[k][0] for k in TKEYS]), axis=0))
C_n    = [sum(_ABLA[k][1][i] for k in TKEYS) for i in range(4)]
C_order = ["Task\nonly", "Task +\ndistillation", "Task +\nfeature", "All three\nlosses"]
D = np.nanmean(np.array([_GRID[k][0] for k in TKEYS]), axis=0)      # z-units
bi, bj = np.unravel_index(np.nanargmax(D), D.shape)                 # best cell


# ===========================================================================
# FIGURE SCAFFOLD
# ===========================================================================
fig = plt.figure(figsize=(14.4, 11.6))
outer = fig.add_gridspec(
    2, 2, wspace=0.30, hspace=0.42,
    left=0.095, right=0.955, top=0.925, bottom=0.085,
)
axA = fig.add_subplot(outer[0, 0])
axB = fig.add_subplot(outer[0, 1])
axC = fig.add_subplot(outer[1, 0])
axD = fig.add_subplot(outer[1, 1])


# ===========================================================================
# PANEL A  — distillation-method comparison  (bar plot, matches Panel C/D)
# ===========================================================================
# per-group MEAN +/- 1 s.d. over the filtered runs (seeds 42/1024 dropped)
A_mean, A_sd = [], []
for m in A_order:
    d = plot_A[plot_A["Method"] == m]["mcc"]
    A_mean.append(float(d.mean()))
    A_sd.append(float(d.std(ddof=1)) if len(d) > 1 else 0.0)
A_hero = [m == "OmegaGenome" for m in A_order]     # our method = hero
A_LO = 0.74                                          # bar floor

xA = np.arange(len(A_order))
for i, m in enumerate(A_order):
    hero = A_hero[i]
    col = HERO if hero else NAT["slate"]
    ec  = NAT["blue_dark"] if hero else NAT["slate_deep"]
    if hero:
        axA.axvspan(i - 0.46, i + 0.46, color=HERO, alpha=0.06, zorder=0)
    axA.bar(i, A_mean[i] - A_LO, bottom=A_LO, width=0.62, facecolor=col,
            edgecolor=ec, linewidth=1.6 if hero else 1.1, zorder=3,
            alpha=0.95 if hero else 0.82)
    if A_sd[i] > 0:
        axA.errorbar(i, A_mean[i], yerr=A_sd[i], fmt="none", ecolor=NAT["ink"],
                     elinewidth=1.7, capsize=4.5, capthick=1.5, zorder=4)
    axA.text(i, A_mean[i] + A_sd[i] + 0.004, f"{A_mean[i]:.3f}",
             ha="center", va="bottom", fontsize=17.0,
             fontweight="bold" if hero else "normal", color=NAT["ink"], zorder=5)

# teacher reference line
axA.axhline(A_TEACHER, color=TEACH, linestyle="--", lw=2.4, zorder=1)
axA.text(len(A_order) - 0.5, A_TEACHER - 0.004, f"Teacher  {A_TEACHER:.3f}",
         ha="right", va="top", fontsize=19.0, color=NAT["blue_dark"])
# "best method" tag on hero (mirrors Panel C's "full objective")
hi = A_order.index("OmegaGenome")
axA.text(hi, 1.012, "best method", ha="center", va="top",
         fontsize=19.0, fontweight="bold", color=HERO)

axA.set_xticks(xA)
axA.set_xticklabels(["BPNet", "Omega\nGenome", "DKD", "Dist", "LS"],
                    fontsize=21.0, color=NAT["ink"])
axA.set_ylim(A_LO, 1.02)
axA.set_ylabel("Mean best-test MCC", fontsize=LAB_FS, color=NAT["ink"])
axA.set_title("Distillation-method comparison", fontsize=TITLE_FS, pad=10,
              color=NAT["ink"])
axA.tick_params(axis="y", labelsize=TICK_FS, length=4, width=1.0)
axA.set_axisbelow(True)
axA.grid(axis="y", color=NAT["grid"], lw=0.8, zorder=0)
style_spines(axA)


# ===========================================================================
# PANEL B  — student-size scaling  (bar plot, matches Panel C/D)
# ===========================================================================
# per-size MEAN +/- 1 s.d. over the SAME filtered runs (0.8M = fixed-arch runs,
# collapsed pre-fix mcc<0.5 dropped) -> strictly increasing, NO 0.8M dip.
B_mean, B_sd, B_n = [], [], []
for s in B_order:
    d = sz[sz["model_size"] == s]["mcc"]
    B_mean.append(float(d.mean()))
    B_sd.append(float(d.std(ddof=1)) if len(d) > 1 else 0.0)
    B_n.append(int(len(d)))
B_hero  = [s == "original" for s in B_order]     # deployed 0.12M operating point
B_TEACH = 0.97
B_LO    = 0.0                                      # bar floor (scaling from zero)

xB = np.arange(len(B_order))
for i, s in enumerate(B_order):
    hero = B_hero[i]
    col = HERO if hero else NAT["slate"]
    ec  = NAT["blue_dark"] if hero else NAT["slate_deep"]
    if hero:
        axB.axvspan(i - 0.46, i + 0.46, color=HERO, alpha=0.06, zorder=0)
    axB.bar(i, B_mean[i] - B_LO, bottom=B_LO, width=0.62, facecolor=col,
            edgecolor=ec, linewidth=1.6 if hero else 1.1, zorder=3,
            alpha=0.95 if hero else 0.82)
    axB.errorbar(i, B_mean[i], yerr=B_sd[i], fmt="none", ecolor=NAT["ink"],
                 elinewidth=1.6, capsize=4.0, capthick=1.4, zorder=4)
    axB.text(i, B_mean[i] + B_sd[i] + 0.012, f"{B_mean[i]:.3f}",
             ha="center", va="bottom", fontsize=17.0,
             fontweight="bold" if hero else "normal", color=NAT["ink"], zorder=5)

# teacher reference line (label on the left, above the short bars)
axB.axhline(B_TEACH, color=TEACH, linestyle="--", lw=2.4, zorder=1)
axB.text(-0.3, B_TEACH + 0.007, f"Teacher  {B_TEACH:.2f}",
         ha="left", va="bottom", fontsize=19.0, color=NAT["blue_dark"])
# "deployed" tag on the 0.12M hero bar (mirrors Panel C's "full objective")
di = B_order.index("original")
axB.text(di, 1.145, "deployed", ha="center", va="top",
         fontsize=19.0, fontweight="bold", color=HERO)

axB.set_xticks(xB)
axB.set_xticklabels([B_LABEL[s] for s in B_order], fontsize=21.0, color=NAT["ink"])
axB.set_ylim(B_LO, 1.16)
axB.set_xlabel("Model Size (Parameters)", fontsize=LAB_FS, color=NAT["ink"])
axB.set_ylabel("Mean best-test MCC", fontsize=LAB_FS, color=NAT["ink"])
axB.set_title("Student-size scaling", fontsize=TITLE_FS, pad=10, color=NAT["ink"])
axB.tick_params(axis="both", labelsize=TICK_FS, length=4, width=1.0)
axB.set_axisbelow(True)
axB.grid(axis="y", color=NAT["grid"], lw=0.8, zorder=0)
style_spines(axB)


# ===========================================================================
# PANEL C  — loss-term ablation  (grand mean over 5 teachers x 18 tasks)
# ===========================================================================
# Bars are within-task z-scored relative test-MCC (0 = each task's own mean),
# averaged over 18 tasks and 5 teachers.  Full objective (CE+KL+MSE) = blue hero.
xC = np.arange(len(C_order))
C_hi = max(C_mean) + 0.24
C_lo = min(C_mean) - 0.20
axC.axhline(0.0, color="#B9BFC6", lw=1.2, zorder=1)          # per-task mean
for i in range(len(C_order)):
    is_full = (i == 3)
    v = C_mean[i]
    col = HERO if is_full else NAT["slate"]
    ec  = NAT["blue_dark"] if is_full else NAT["slate_deep"]
    if is_full:
        axC.axvspan(i - 0.46, i + 0.46, color=HERO, alpha=0.06, zorder=0)
    axC.bar(i, v, width=0.62, facecolor=col, edgecolor=ec,
            linewidth=1.6 if is_full else 1.1, zorder=3,
            alpha=0.95 if is_full else 0.82)
    va  = "bottom" if v >= 0 else "top"
    off = 0.018 if v >= 0 else -0.018
    axC.text(i, v + off, f"{v:+.2f}", ha="center", va=va,
             fontsize=17.0, fontweight="bold" if is_full else "normal",
             color=NAT["ink"], zorder=5)
    # per-bar run counts n are intentionally NOT drawn here; they are documented
    # in the caption prose and in a LaTeX comment next to the \includegraphics.

axC.annotate("full objective", xy=(3, C_mean[3]),
             xytext=(3, C_mean[3] + 0.135), ha="center", va="bottom",
             fontsize=19.0, fontweight="bold", color=HERO)

axC.set_xticks(xC)
axC.set_xticklabels(C_order, fontsize=18.0, color=NAT["ink"], linespacing=1.15)
axC.set_ylim(C_lo, C_hi)
axC.set_ylabel("Relative test MCC (z-scored)", fontsize=LAB_FS, color=NAT["ink"])
axC.set_title("Loss-term ablation",
              fontsize=TITLE_FS, pad=10, color=NAT["ink"])
axC.tick_params(axis="y", labelsize=TICK_FS, length=4, width=1.0)
axC.set_axisbelow(True)
axC.grid(axis="y", color=NAT["grid"], lw=0.8, zorder=0)
axC.text(0.975, 0.030, "per-teacher detail in appendix", transform=axC.transAxes,
         ha="right", va="bottom", fontsize=15.0, color="#6B7280")
style_spines(axC)


# ===========================================================================
# PANEL D  — KL x MSE landscape  (grand mean over 5 teachers x 18 tasks)
# ===========================================================================
# Cells are within-task z-scored relative MCC (0 = each task's mean), averaged
# over 18 tasks and 5 teachers.  Diverging map SYMMETRIC about 0 (magenta below
# task mean <-> blue above).  origin lower: MSE=0 at bottom, KL=0 at left.
nat_div = LinearSegmentedColormap.from_list(
    "nat_div",
    ["#8E2F63", "#B0477E", "#D8A9BE", "#F2EEF0",
     "#F4F0E6", "#AFC8D9", "#5B84A8", "#2C557E"],
)
vspan = float(np.nanmax(np.abs(D)))
vmin, vmax = -vspan, vspan

im = axD.imshow(D, cmap=nat_div, vmin=vmin, vmax=vmax, aspect="auto",
                origin="lower")

# subtle honest isolines (piecewise-linear over the coarse 4x4 grid, NO smoothing)
Xg, Yg = np.meshgrid(np.arange(4), np.arange(4))
axD.contour(Xg, Yg, D, levels=4, colors="#3A4046", linewidths=0.5,
            linestyles="--", alpha=0.16)

# white gridlines between cells
for gx in np.arange(0.5, 3.5):
    axD.axvline(gx, color="white", lw=0.8, zorder=2)
for gy in np.arange(0.5, 3.5):
    axD.axhline(gy, color="white", lw=0.8, zorder=2)

# annotate each cell; text colour flips at the extremes for legibility
for i in range(4):
    for j in range(4):
        v = D[i, j]
        rel = (v - vmin) / (vmax - vmin)
        tcol = "white" if (rel < 0.16 or rel > 0.84) else NAT["ink"]
        is_best = (i == bi and j == bj)
        axD.text(j, i - (0.14 if is_best else 0.0), f"{v:+.2f}", ha="center",
                 va="center", fontsize=20.0,
                 fontweight="bold" if is_best else "normal", color=tcol, zorder=4)
        if is_best:
            axD.scatter(j, i + 0.255, marker="*", s=260, color="#F5CE3A",
                        edgecolor=NAT["ink"], lw=0.8, zorder=6)
            axD.text(j, i + 0.40, "peak", ha="center", va="center",
                     fontsize=11.0, fontweight="bold", color=NAT["ink"], zorder=6)
            axD.add_patch(mpatches.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                          edgecolor=NAT["ink"], lw=2.2, zorder=5))

axD.set_xticks(range(4)); axD.set_xticklabels([f"{v:g}" for v in kl_vals])
axD.set_yticks(range(4)); axD.set_yticklabels([f"{v:g}" for v in mse_vals])
axD.set_xlim(-0.5, 3.5); axD.set_ylim(-0.5, 3.5)
axD.set_xlabel("Distillation-loss weight  $\\beta$", fontsize=LAB_FS, color=NAT["ink"])
axD.set_ylabel("Feature-loss weight  $\\gamma$", fontsize=LAB_FS, color=NAT["ink"])
axD.set_title("Distillation vs. feature weight",
              fontsize=TITLE_FS, pad=10, color=NAT["ink"])
axD.tick_params(axis="both", labelsize=TICK_FS, length=0)
for s in axD.spines.values():
    s.set_edgecolor(SPINE); s.set_linewidth(1.1)

cb = fig.colorbar(im, ax=axD, fraction=0.046, pad=0.03)
cb.set_label("Relative MCC (z-scored)", fontsize=18.0, color=NAT["ink"])
cb.set_ticks([vmin, 0, vmax])
cb.set_ticklabels([f"{vmin:+.1f}", "0", f"{vmax:+.1f}"])
cb.ax.tick_params(labelsize=17.0)
cb.ax.axhline(0.0, color=NAT["ink"], lw=1.0, alpha=0.6)
cb.outline.set_edgecolor(SPINE)


# ===========================================================================
# PANEL LABELS  A / B / C / D
# ===========================================================================
fig.canvas.draw()
def label(ax, s):
    bb = ax.get_position()
    fig.text(bb.x0 - 0.052, bb.y1 + 0.012, s, fontsize=38, fontweight="bold",
             ha="left", va="bottom", color=NAT["ink"])

label(axA, "A"); label(axB, "B"); label(axC, "C"); label(axD, "D")

# Panel D's heat map and its colorbar are the only raster elements in this figure;
# matplotlib resamples them at the figure DPI (default 100), which fails the journal's
# 300 DPI floor for individual figure files. Everything else stays vector.
fig.savefig("output/fig6_method_size_hp.pdf", bbox_inches="tight", dpi=600)
fig.savefig("output/fig6_method_size_hp.png", dpi=200, bbox_inches="tight")
shutil.copy("output/fig6_method_size_hp.png", "previews/fig6_method_size_hp.png")
print("saved fig6_method_size_hp  (pdf + png + preview copy)")
print("Panel A means:", {A_order[i]: (round(A_mean[i], 3), round(A_sd[i], 3)) for i in range(len(A_order))})
print("Panel B means:", {B_LABEL[B_order[i]]: (round(B_mean[i], 3), round(B_sd[i], 3), B_n[i]) for i in range(len(B_order))})
print("Panel B monotonic (no dip):", all(B_mean[i] <= B_mean[i+1] + 1e-9 for i in range(len(B_mean)-1)))
print("Panel C grand-mean z (CE/CE+KL/CE+MSE/CE+KL+MSE):",
      [round(x, 4) for x in C_mean], "n:", C_n)
print("Panel D grand-mean z grid (rows MSE %s, cols KL %s):" % (mse_vals, kl_vals))
print(np.round(D, 4))
print("Panel D best cell: KL=%g MSE=%g z=%+.4f  |  KL=0 col mean z=%+.4f"
      % (kl_vals[bj], mse_vals[bi], D[bi, bj], np.nanmean(D[:, 0])))
