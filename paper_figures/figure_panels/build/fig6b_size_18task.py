"""
Fig 6B (standalone): 18-task-mean best-test MCC vs student size (scaling curve).

SINGLE-SEED curve (seed0 sweep).  The shaded band is the TASK-TO-TASK spread
(s.d. over the 18 tasks) at each size -- NOT a seed s.d.

Data provenance (per task x size, single MCC each):
  * 17 tasks x 6 sizes (pico..xxlarge): new seed0 size sweep, best_test_mcc read
    from ${OG_SCRATCH}/size18task/<task>/<size>/seed0/.../final_summary.json
  * `original` (0.12M, deployed student), 17 tasks: reused NT distilled-student
    column ("Distilled Nucleotide Transformer") of
    data/model_comparison_5teacher_formal.csv (that student IS the 0.12M/original).
  * splice_sites_all, ALL 7 sizes: kept from the pre-existing size data
    (analysis size_comparison_flat_clean.csv + extralarge_fix), matching
    the current Fig 6 Panel B; this is the one task with multi-seed data, so its
    per-size value is the mean over its available seeds.

x-axis = student parameter count (log): 2K/7K/28K/0.1M/0.8M/1.8M/3.6M.
Deployed 0.1M operating point is marked.  House Nature style (matches fig6).
Saves output/fig6b_size_18task.{pdf,png} @300dpi and copies PNG to previews/.
"""
import os
import shutil
import json
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.makedirs("output", exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "axes.linewidth": 1.1,
})

# ---- house Nature palette (identical to fig6_method_size_hp.py) -------------
NAT = {
    "grey_dot": "#8A9099", "grey_base": "#C3C8CE",
    "blue_dark": "#356491", "blue_mid": "#6E97BC", "blue_light": "#A9C6E0",
    "slate": "#9AA4AE", "slate_deep": "#7C8894", "ink": "#222222", "grid": "#DBDEE2",
}
HERO = NAT["blue_dark"]
SPINE = "#444444"
TITLE_FS, LAB_FS, TICK_FS, ANN_FS = 15.5, 13, 11.5, 11

HERE = os.path.dirname(os.path.abspath(__file__))
os.makedirs(os.path.join(HERE, "output"), exist_ok=True)

# size roster + labels + representative parameter counts (matches Panel B) -----
SIZES = ["pico", "ultra_tiny", "extra_tiny", "original",
         "extra_large_fix", "large", "xxlarge"]
LABEL = {"pico": "2K", "ultra_tiny": "7K", "extra_tiny": "28K", "original": "0.1M",
         "extra_large_fix": "0.8M", "large": "1.8M", "xxlarge": "3.6M"}
PARAMS = {"pico": 2e3, "ultra_tiny": 7e3, "extra_tiny": 28e3, "original": 1.2e5,
          "extra_large_fix": 8e5, "large": 1.8e6, "xxlarge": 3.6e6}

SWEEP_SIZES = ["pico", "ultra_tiny", "extra_tiny",
               "extra_large_fix", "large", "xxlarge"]   # new seed0 sweep (no original)


# =========================================================================== #
# 1. New seed0 sweep: 17 tasks x 6 sizes  (read over sshfs on galaxy)          #
# =========================================================================== #
def load_sweep_seed0():
    """Return {task: {size: best_test_mcc}} from the seed0 size sweep on galaxy SSD.

    Reads best_test_mcc from each .../seed0/**/final_summary.json via a remote
    python one-liner over ssh; exactly one summary per (task,size) cell.
    """
    # A reader of the archive has neither the galaxy SSD nor ssh access, so the extracted
    # matrix ships in plot_repo/data/; the ssh read below is the authors' regeneration path.
    bundled = os.path.join(HERE, "..", "..", "data", "size18task_seed0_best_test_mcc.json")
    if os.path.exists(bundled):
        with open(bundled) as fh:
            return json.load(fh)
    raise FileNotFoundError(
        f"{bundled} is missing; it holds the extracted seed-0 size-sweep matrix this figure reads."
    )


# =========================================================================== #
# 2. original (0.12M) NT distilled-student per task (17 tasks used here)       #
# =========================================================================== #
def load_original_nt():
    """{task: MCC} for the deployed 0.12M student = Distilled NT column."""
    df = pd.read_csv(os.path.join(HERE, "..", "..", "data",
                                  "model_comparison_5teacher_formal.csv"))
    nt = df[df["Model"] == "Distilled Nucleotide Transformer"]
    return dict(zip(nt["Task"], nt["Score"]))


# =========================================================================== #
# 3. splice_sites_all across all 7 sizes (pre-existing size data)             #
# =========================================================================== #
def load_splice_all():
    """{size: mean MCC} for splice_sites_all from the existing size CSVs.

    Reproduces the current Fig 6 Panel B extraction: drop seeds {42,1024}, swap
    extra_large -> extra_large_fix (finished, mcc>0.5). Returns per-size mean over
    the available seeds (splice_sites_all is the multi-seed task).
    """
    _R = {"config.dataset_config.task_name": "task_name",
          "summary.best_test/mcc": "mcc", "config.random_state": "seed",
          "config.student_config.model_size": "model_size"}
    base = os.path.join(HERE, "..", "..", "data", "hp_search_csv") + os.sep
    load = lambda p: pd.read_csv(base + p).rename(columns=_R)
    sz = load("size_comparison_flat_clean.csv")
    sz = sz[~sz["seed"].isin([42, 1024])]
    seeds_in = set(sz["seed"].unique())
    xl = load("extralarge_fix_flat_clean.csv")
    xl = xl[xl["seed"].isin(seeds_in)]
    if "state" in xl.columns:
        xl = xl[xl["state"] == "finished"]
    xl = xl[xl["mcc"] > 0.5]
    xl_proc = pd.DataFrame({"model_size": "extra_large_fix",
                            "task_name": xl["task_name"].values, "mcc": xl["mcc"].values})
    sz = sz[sz["model_size"] != "extra_large"]
    sz = pd.concat([sz[["model_size", "task_name", "mcc"]], xl_proc], ignore_index=True)
    # note: sweep already uses ultra_tiny / extra_tiny names, so keep them as-is
    ssa = sz[sz["task_name"] == "splice_sites_all"].dropna(subset=["mcc"])
    out = {}
    for s in SIZES:
        d = ssa[ssa["model_size"] == s]["mcc"]
        out[s] = float(d.mean())
    return out


# =========================================================================== #
# ASSEMBLE 18 tasks x 7 sizes                                                  #
# =========================================================================== #
sweep = load_sweep_seed0()                       # 17 tasks x 6 sizes
nt_orig = load_original_nt()                     # original per task
splice = load_splice_all()                       # splice_sites_all all sizes

TASKS = sorted(sweep.keys()) + ["splice_sites_all"]     # 18 tasks
assert len(TASKS) == 18, TASKS

# matrix[task][size]
M = {}
missing = []
for t in sorted(sweep.keys()):                   # the 17 swept tasks
    M[t] = {}
    for s in SIZES:
        if s == "original":
            v = nt_orig.get(t)
        else:
            v = sweep[t].get(s)
        if v is None:
            missing.append((t, s))
        M[t][s] = v
M["splice_sites_all"] = {s: splice[s] for s in SIZES}     # all 7 from existing

# 18-task mean + task-to-task s.d. at each size
mean_curve, sd_curve = [], []
for s in SIZES:
    vals = np.array([M[t][s] for t in TASKS], dtype=float)
    mean_curve.append(float(np.nanmean(vals)))
    sd_curve.append(float(np.nanstd(vals, ddof=1)))
mean_curve = np.array(mean_curve)
sd_curve = np.array(sd_curve)
x = np.array([PARAMS[s] for s in SIZES])

# ---- console report --------------------------------------------------------
print("=== Fig 6B: 18-task-mean MCC vs student size (seed0) ===")
for i, s in enumerate(SIZES):
    print(f"  {LABEL[s]:>6s}  ({s:16s})  mean={mean_curve[i]:.4f}  sd(task)={sd_curve[i]:.4f}")
mono = all(mean_curve[i] <= mean_curve[i + 1] + 1e-9 for i in range(len(mean_curve) - 1))
print("strictly monotonic non-decreasing:", mono)
print("missing cells:", missing if missing else "NONE (102/102)")

# =========================================================================== #
# PLOT                                                                         #
# =========================================================================== #
fig, ax = plt.subplots(figsize=(7.6, 5.6))

# task-to-task spread band
ax.fill_between(x, mean_curve - sd_curve, mean_curve + sd_curve,
                color=NAT["blue_mid"], alpha=0.16, lw=0, zorder=1,
                label="task-to-task s.d. (n=18)")
ax.plot(x, mean_curve, "-", color=HERO, lw=2.4, zorder=3)

# points; deployed 0.1M marked as hero
di = SIZES.index("original")
for i, s in enumerate(SIZES):
    hero = (i == di)
    ax.plot(x[i], mean_curve[i], "o",
            ms=12 if hero else 8,
            mfc=HERO if hero else "white",
            mec=NAT["blue_dark"], mew=2.0 if hero else 1.6, zorder=5)
    off = 0.011 if not hero else 0.016
    ax.text(x[i], mean_curve[i] + sd_curve[i] * 0 + off, f"{mean_curve[i]:.3f}",
            ha="center", va="bottom", fontsize=ANN_FS,
            fontweight="bold" if hero else "normal", color=NAT["ink"], zorder=6)

# "deployed" annotation on the 0.1M hero point
ax.annotate("deployed\n(0.1M)", xy=(x[di], mean_curve[di]),
            xytext=(x[di] * 0.62, mean_curve[di] - 0.075),
            ha="center", va="top", fontsize=10.5, fontweight="bold", color=HERO,
            arrowprops=dict(arrowstyle="-", color=HERO, lw=1.4))

ax.set_xscale("log")
ax.set_xticks([PARAMS[s] for s in SIZES])
ax.set_xticklabels([LABEL[s] for s in SIZES], fontsize=TICK_FS, color=NAT["ink"])
ax.minorticks_off()
ax.set_xlabel("Student model size (parameters)", fontsize=LAB_FS, color=NAT["ink"])
ax.set_ylabel("18-task-mean best-test MCC", fontsize=LAB_FS, color=NAT["ink"])
ax.set_title("Student-size scaling  (mean over 18 tasks)", fontsize=TITLE_FS,
             pad=10, color=NAT["ink"])

lo = float(np.min(mean_curve - sd_curve))
hi = float(np.max(mean_curve + sd_curve))
ax.set_ylim(lo - 0.02, hi + 0.05)
ax.tick_params(axis="both", labelsize=TICK_FS, length=4, width=1.0)
ax.set_axisbelow(True)
ax.grid(axis="y", color=NAT["grid"], lw=0.8, zorder=0)
for sp in ax.spines.values():
    sp.set_edgecolor(SPINE); sp.set_linewidth(1.1)

# band descriptor (across-task spread, not a seed s.d.)
ax.text(0.985, 0.045,
        "single-seed (seed 0); band = task-to-task s.d. (n=18)",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=9.0,
        style="italic", color="#7A808A")
ax.legend(loc="upper left", frameon=False, fontsize=10.0)

fig.tight_layout()
pdf = os.path.join(HERE, "output", "fig6b_size_18task.pdf")
png = os.path.join(HERE, "output", "fig6b_size_18task.png")
fig.savefig(pdf, bbox_inches="tight")
fig.savefig(png, dpi=300, bbox_inches="tight")
shutil.copy(png, os.path.join(HERE, "..", "fig6b_size_18task.png"))
print("saved:", pdf)
print("saved:", png, "(300 dpi) + preview copy in previews/")
