"""
Efficiency figure for OmegaGenome (published as Figure 3) -- clean 2x3 grid.

A symmetric 2x3 grid (every panel in a row shares top/bottom; every panel in a
column shares left/right; panel letters A-F at identical relative positions).

Row 1  (18 classification-task efficiency):
  A  Accuracy vs. model size (bubble scatter): teachers = open rings/bubbles sized
     by log(params); 5 distilled students = solid diamonds at 0.12 M; BPNet = grey
     square. A zoomed accuracy-retention inset resolves the student cluster.
  B  All-18-task total GPU time (log bars, fold vs student). All six models are
     18/18 measured (NT 3651.88 s; Enformer 51.04 s, fp32, 18/18 measured on the
     RTX 3090 -- no longer provisional). Enformer is flagged fp32 (others fp16).
  C  GPU peak memory (log bars, fold vs student) -- measured GPU VRAM, not host RSS.

Row 2  (multitrack base-resolution regression benchmark; 34 tracks, N=10,531
        32-kb windows, batch 4, RTX 3090, AMP):
  D  Accuracy vs. model size (parallel to A): x = params (log), y = mean 34-track
     Pearson. NTv3-650M teacher = open ring (0.606); distilled 8 M / 100 M students
     = solid diamonds (0.505 / 0.545); from-scratch baselines = grey open squares
     (0.475 / 0.518). Values from Table tab:app:regression (science_template.tex).
  E  Multitrack whole-test GPU time (log bars, fold vs NTv3-8M distilled student).
  F  Multitrack GPU peak memory (log bars, fold vs NTv3-8M distilled student).

Data provenance (NO fabricated numbers):
  * data/model_comparison_5teacher_formal.csv  -> mean MCC (accuracy, panel A).
  * data/efficiency_18task_authoritative.csv    -> ALL_18_TOTAL gpu_time_s (panel B).
  * data/efficiency_18task_authoritative.csv  gpu_peak_mem_mb (panel C) -- measured GPU
      VRAM, MB: student 17, Caduceus 489, DNABERT-2 3074, Enformer 1184, NT 11553, Carbon-3B 8004.
  * data/efficiency_multitrack.csv               -> whole-test time + GPU peak mem (E/F).
  * paper_src/science_template.tex tab:app:regression -> panel D Pearson values.

Honesty footnotes (rendered under the grid):
  * Enformer all-18 time is fp32, 18/18 measured on the RTX 3090 (all others fp16); † flags fp32.
  * Caduceus measured on RTX A6000 (GLIBC constraint), not the RTX 3090.
  * all-18 total = sum over tasks of (per-batch latency x ceil(n_test/16)), 38,822 samples.
  * multitrack = 34-track mean Pearson, N = 10,531 x 32-kb windows.

Run with the biolaysum env python from plot_repo/. Saves output/fig4_efficiency_carbon.{pdf,png}.

Published as Figure 3 of the paper.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.ticker import NullFormatter

WHITE_HALO = [pe.withStroke(linewidth=2.6, foreground="white")]

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "axes.linewidth": 1.0,
})
os.makedirs("output", exist_ok=True)
os.makedirs("previews", exist_ok=True)

# ---------------------------------------------------------------------------
# NATURE PALETTE (identical to fig2_combined_carbon.py)
# ---------------------------------------------------------------------------
NAT = {
    "grey_dot":  "#8A9099", "grey_base": "#C3C8CE",
    "blue_dark": "#356491", "blue_light": "#A9C6E0",   # NT
    "amber_dark":"#CE8B3B", "amber_light":"#EED0A3",   # DNABERT-2
    "green_dark":"#4F8C6B", "green_light":"#A9CDB7",   # Enformer
    "purple_dark":"#7C619E","purple_light":"#C7B7DA",  # Caduceus
    "rose_dark": "#B0477E", "rose_light": "#E3AEC9",   # Carbon-3B
    "ink": "#222222",
}
FAM = {  # accuracy family -> (short, light, dark)
    "Nucleotide Transformer": ("NT",        NAT["blue_light"],   NAT["blue_dark"]),
    "DNABERT-2":              ("DNABERT-2", NAT["amber_light"],  NAT["amber_dark"]),
    "Enformer":               ("Enformer",  NAT["green_light"],  NAT["green_dark"]),
    "Caduceus":               ("Caduceus",  NAT["purple_light"], NAT["purple_dark"]),
    "Carbon-3B":              ("Carbon-3B", NAT["rose_light"],   NAT["rose_dark"]),
}
# Row-2 multitrack: BPNet grey + teal sequential ramp for NTv3 sizes.
TEAL = {"NTv3-8M": "#7FC6BB", "NTv3-100M": "#37917F", "NTv3-650M-teacher": "#1C6157"}

# ---------------------------------------------------------------------------
# DATA
# ---------------------------------------------------------------------------
acc = pd.read_csv("data/model_comparison_5teacher_formal.csv")
acc["Score"] = pd.to_numeric(acc["Score"], errors="coerce")
BPNET_MCC = acc[acc["Type"] == "Baseline"]["Score"].mean()
def mcc(fam, typ):
    return acc[(acc["Model Family"] == fam) & (acc["Type"] == typ)]["Score"].mean()

def parse_params(s):
    s = str(s).strip()
    if s.endswith("B"): return float(s[:-1]) * 1000.0
    if s.endswith("M"): return float(s[:-1])
    return float(s)

eff = pd.read_csv("data/efficiency_18task_authoritative.csv")
tot = eff[eff["task"] == "ALL_18_TOTAL"].copy()
tot["params_M"] = tot["params"].map(parse_params)
NAME2FAM = {"NT-2.5B": "Nucleotide Transformer", "Carbon-3B": "Carbon-3B",
            "DNABERT-2": "DNABERT-2", "Enformer": "Enformer", "Caduceus": "Caduceus",
            "BPNet-student": None}
E = {r["model"]: r for _, r in tot.iterrows()}
PARAMS_M = {NAME2FAM[m]: E[m]["params_M"] for m in NAME2FAM if NAME2FAM[m]}
STUDENT_M = float(E["BPNet-student"]["params_M"])

# Measured GPU VRAM (MB) -- read from the authoritative CSV column gpu_peak_mem_mb
# (replaces the earlier host-RSS numbers).
GPU_VRAM_MB = {m: float(E[m]["gpu_peak_mem_mb"]) for m in E}

mt = pd.read_csv("data/efficiency_multitrack.csv")
MT_ROW = {r["model"]: r for _, r in mt.iterrows()}

# Panel D -- multitrack regression accuracy (Table tab:app:regression).
# (params_M, distilled Pearson, from-scratch Pearson)
MT_ACC = {
    "NTv3-8M":   (7.702,   0.505, 0.475),
    "NTv3-100M": (106.491, 0.545, 0.518),
}
MT_TEACHER = (655.326, 0.606)   # NTv3-650M teacher, 34-track mean Pearson

def bubble_size(pm):
    return 80 + 820 * np.log10(pm + 1)
def param_text(pm):
    if pm >= 1000: return (f"{pm/1000:.2f}".rstrip("0").rstrip(".")) + " B"
    if pm >= 10:   return f"{pm:.0f} M"
    return f"{pm:.1f} M"
def fold_txt(f):
    if abs(f - 1) < 1e-6: return "1x"
    return f"{f:.0f}x" if f >= 10 else f"{f:.1f}x"
def fmt_time(v):
    if v >= 3600: return f"{v/3600:.1f} h"
    if v >= 90:   return f"{v/60:.1f} min"
    if v >= 1:    return f"{v:.0f} s"
    return f"{v:.1f} s"
def fmt_mem(v):
    return f"{v/1024:.1f} GB" if v >= 1024 else f"{v:.0f} MB"

# ---------------------------------------------------------------------------
# FIGURE SCAFFOLD -- hero left scatter column + two narrow bar columns
#   Row 1:  A (large) | B (small) | C (small)
#   Row 2:  D (large) | E (small) | F (small)
# The two accuracy-vs-size scatters (A, D) are the hero panels; the four
# latency/memory bar panels (B, C, E, F) are deliberately secondary/smaller.
# ---------------------------------------------------------------------------
# Taller per-row height (A/D breathe) with tightened inter-row + outer margins so
# the reclaimed space goes into the panels, not into emptiness. Equal (default)
# height_ratios keep the two hero scatters A and D the same shape (matched pair).
fig = plt.figure(figsize=(13.0, 8.4))
gs = fig.add_gridspec(2, 3, width_ratios=[2.45, 1.0, 1.0],
                      left=0.056, right=0.982, top=0.950, bottom=0.088,
                      wspace=0.235, hspace=0.38)
axA = fig.add_subplot(gs[0, 0]); axB = fig.add_subplot(gs[0, 1]); axC = fig.add_subplot(gs[0, 2])
axD = fig.add_subplot(gs[1, 0]); axE = fig.add_subplot(gs[1, 1]); axF = fig.add_subplot(gs[1, 2])

TITLE_FS, LAB_FS, TICK_FS = 20.0, 19.0, 16.0
ANN_FS = 18.0

def style_spines(ax):
    for s_ in ax.spines.values():
        s_.set_edgecolor("#444444"); s_.set_linewidth(1.0)

# ===========================================================================
# PANEL A : bubble scatter (accuracy vs size) -- 18 classification tasks
# ===========================================================================
axA.set_xscale("log")
XLIM = (0.055, 26000); YLIM = (0.576, 0.726)
axA.set_xlim(*XLIM); axA.set_ylim(*YLIM)
axA.axvspan(XLIM[0], 1.0, color="#EEF1F4", zorder=0)
axA.text(0.070, 0.722, "deployable\nregime", fontsize=16.0, style="italic",
         color="#8A929B", va="top", ha="left", linespacing=1.35)
axA.axhline(BPNET_MCC, color=NAT["grey_base"], linestyle=":", lw=1.8, zorder=1)

# teacher -> student connectors
for fam, (short, light, dark) in FAM.items():
    axA.plot([PARAMS_M[fam], STUDENT_M], [mcc(fam, "Teacher"), mcc(fam, "Student")],
             color=dark, lw=1.2, linestyle="--", alpha=0.38, zorder=2)

# teacher bubbles (largest first so the two ~3 B bubbles nest cleanly)
torder = sorted(FAM, key=lambda f: -PARAMS_M[f])
OVERLAP = {"Nucleotide Transformer", "Carbon-3B"}
# external labels (name + MCC) placed in open bands; leader lines to bubbles.
EXTLAB = {  # (x, y, ha, va)
    # NT label moved directly above its bubble (clear of the top frame and of the
    # Carbon-3B label/bubbles) with a clean vertical leader -- previously it sat on
    # the top spine and crowded the Carbon-3B bubble.
    "Nucleotide Transformer": (5200.0,  0.706, "center", "bottom"),
    "Carbon-3B":              (16000.0, 0.664, "right",  "center"),
    "Enformer":               (70.0,    0.700, "center", "bottom"),
    "DNABERT-2":              (20.0,    0.628, "center", "top"),
    "Caduceus":               (3.2,     0.598, "center", "top"),
}
for fam in torder:
    short, light, dark = FAM[fam]
    x, y = PARAMS_M[fam], mcc(fam, "Teacher")
    b_alpha, b_lw = (0.50, 1.6) if fam in OVERLAP else (0.90, 1.9)
    axA.scatter([x], [y], s=bubble_size(x), facecolors=light, edgecolors=dark,
                linewidths=b_lw, alpha=b_alpha, zorder=4)
    lx, ly, ha, va = EXTLAB[fam]
    axA.annotate(f"{short} {y:.3f}", (x, y), textcoords="data", xytext=(lx, ly),
                 ha=ha, va=va, fontsize=15.0, fontweight="bold", color=dark, zorder=7,
                 path_effects=WHITE_HALO,
                 arrowprops=dict(arrowstyle="-", color=dark, lw=1.0, alpha=0.7,
                                 shrinkA=3, shrinkB=8))

# distilled students (diamonds) + BPNet square, all at 0.12 M
for fam, (short, light, dark) in FAM.items():
    axA.scatter([STUDENT_M], [mcc(fam, "Student")], s=95, marker="D", color=dark,
                edgecolor="white", lw=0.8, zorder=5)
axA.scatter([STUDENT_M], [BPNET_MCC], s=105, marker="s", color=NAT["grey_dot"],
            edgecolor="white", lw=0.8, zorder=5)
axA.annotate("BPNet", (STUDENT_M, BPNET_MCC), textcoords="offset points",
             xytext=(13, -9), ha="left", va="center", fontsize=13.0,
             fontweight="bold", color=NAT["grey_dot"], zorder=7)
axA.annotate("distilled\nstudents\n0.12 M", (STUDENT_M, 0.640), textcoords="offset points",
             xytext=(14, 2), ha="left", va="center", fontsize=13.0, fontweight="bold",
             color=NAT["ink"], zorder=7, linespacing=1.35)

axA.set_xlabel("Model size (million parameters)", fontsize=LAB_FS, color=NAT["ink"])
axA.set_ylabel("Mean MCC", fontsize=LAB_FS, color=NAT["ink"])
axA.set_xticks([0.1, 1, 10, 100, 1000, 10000]); axA.set_xticklabels(["0.1 M", "1 M", "10 M", "100 M", "1 B", "10 B"])
axA.tick_params(axis="both", labelsize=TICK_FS, length=5, width=1.0)
axA.grid(True, which="major", alpha=0.20, linestyle="--")
style_spines(axA)

# NOTE: the zoomed retention inset was removed -- at legible type it crowded panel A.
# The 93--99%% retention range is stated in the caption and resolved per teacher in Table S3.

# ===========================================================================
# PANELS B / C : all-18 total GPU time + GPU peak memory (log bars, borderless)
# ===========================================================================
bar_order = ["BPNet-student", "Caduceus", "DNABERT-2", "Enformer", "NT-2.5B", "Carbon-3B"]
BAR_SHORT = {"BPNet-student": "Stud.", "Caduceus": "Cad.", "DNABERT-2": "DB-2",
             "Enformer": "Enf.", "NT-2.5B": "NT", "Carbon-3B": "C-3B"}
def barcol(m):
    return NAT["grey_dot"] if m == "BPNet-student" else FAM[NAME2FAM[m]][2]
CAVEAT = {"Enformer": "†", "Caduceus": "‡"}

def draw_bars(ax, values, title, ylabel, kind, order, colors, ref, shorts,
              caveats=None, hatch_models=None, ann_fs=13.0):
    # hatch_models: {model: (face, edge)} -> light fill + same-hue darker hatch (provisional)
    caveats = caveats or {}; hatch_models = hatch_models or {}
    x = np.arange(len(order))
    bars = ax.bar(x, values, width=0.52, color=colors, edgecolor="none", zorder=3)
    for xi, m in enumerate(order):
        if m in hatch_models:
            face, edge = hatch_models[m]
            plt.rcParams["hatch.linewidth"] = 1.4
            bars[xi].set_facecolor(face)
            bars[xi].set_hatch("///")
            bars[xi].set_edgecolor(edge); bars[xi].set_linewidth(0.9)
    ax.set_yscale("log")
    ax.set_ylim(ref * 0.92, max(values) * 9.0)   # floor near the student -> teacher bars read long
    ax.set_xlim(-0.75, len(order) - 0.25)        # pad so end bars clear the y-axis / right frame
    for xi, v, m in zip(x, values, order):
        a = fmt_time(v) if kind == "time" else fmt_mem(v)
        f = fold_txt(v / ref)
        cav = caveats.get(m, "")
        ax.annotate(cav + r"$\bf{" + f.replace("x", r"\times") + r"}$",
                    (xi, v), textcoords="offset points", xytext=(0, 6),
                    ha="left", va="bottom", rotation=90, fontsize=ann_fs, color=NAT["ink"])
    ax.set_xticks(x)
    ax.set_xticklabels([shorts[m] for m in order], rotation=90, ha="center", va="top", fontsize=12.0)
    ax.set_ylabel(ylabel, fontsize=LAB_FS, color=NAT["ink"])
    ax.set_title(title, fontsize=TITLE_FS, pad=8, color=NAT["ink"])
    ax.tick_params(axis="y", labelsize=TICK_FS - 1, length=4, width=1.0)
    ax.grid(axis="y", which="major", alpha=0.28, linestyle="--")
    ax.yaxis.set_minor_formatter(NullFormatter())
    style_spines(ax)

gpu_vals = [float(E[m]["gpu_time_s"]) for m in bar_order]
mem_vals = [float(GPU_VRAM_MB[m]) for m in bar_order]
gpu_ref = float(E["BPNet-student"]["gpu_time_s"])
mem_ref = float(GPU_VRAM_MB["BPNet-student"])
# Panels B & C (row 1): clarify the student bar is the BPNet student on its x-axis label. Two-line
# "Student / (BPNet)" so the widened tick label doesn't collide with the neighboring subplot letter.
BPNET_TICK = {**BAR_SHORT}
draw_bars(axB, gpu_vals, "Total GPU time", "Total GPU time (s)", "time",
          bar_order, [barcol(m) for m in bar_order], gpu_ref,
          BPNET_TICK,
          caveats=CAVEAT)  # Enformer now 18/18 measured -> no hatch; † flags fp32 only
draw_bars(axC, mem_vals, "Peak GPU memory", "Peak memory (MB)", "mem",
          bar_order, [barcol(m) for m in bar_order], mem_ref, BPNET_TICK)

# ===========================================================================
# PANEL D : multitrack regression accuracy vs size -- SAME GRAMMAR AS PANEL A
#   * bubble AREA encodes model size (identical log-scaling: bubble_size()).
#   * NTv3-650M teacher = large open teal ring at (655 M, 0.606).
#   * distilled students = filled teal bubbles sized by params (8 M / 100 M).
#   * from-scratch baselines = open grey squares, size-matched to their student.
#   * dashed leader from teacher -> each distilled student (as A: teacher->student);
#     thin vertical connector between each student and its size-matched scratch run
#     so the distilled-vs-scratch gap reads at a glance. Retention %% kept.
# ===========================================================================
axD.set_xscale("log")
DXLIM = (2.6, 4200); DYLIM = (0.435, 0.655)
axD.set_xlim(*DXLIM); axD.set_ylim(*DYLIM)
axD.grid(True, which="major", alpha=0.20, linestyle="--")
tpm, tp = MT_TEACHER
tcol = TEAL["NTv3-650M-teacher"]

# --- connectors first (under the bubbles) ---
# teacher -> distilled student (dashed leader, teal-of-student)
for name, (pm, dist, scr) in MT_ACC.items():
    axD.plot([tpm, pm], [tp, dist], color=TEAL[name], lw=1.3, linestyle="--",
             alpha=0.42, zorder=2)
# distilled <-> size-matched from-scratch (thin solid connector; the gap of interest)
for name, (pm, dist, scr) in MT_ACC.items():
    axD.plot([pm, pm], [scr, dist], color=NAT["grey_dot"], lw=1.4, linestyle="-",
             alpha=0.55, zorder=2)

# Panel-D marker scaling: shrink all bubbles/squares (relative log-areas preserved
# so "bubble area = model size" still holds) so the distilled bubble and its size-
# matched from-scratch square at each x are clearly separated (no marker overlap).
D_SFAC = 1.0           # bubble-area scale for teacher ring + distilled bubbles
                       # (== Panel A: same bubble_size(pm) area-per-param, matched pair)
D_SQ_FAC = 0.30        # from-scratch square area = bubble_size(pm) * D_SQ_FAC
                       # (squares kept smaller -- different marker -- to avoid overlap)

# --- teacher: elegant open ring sized by params ---
# Crisp solid teal outline over a faint teal wash (no dashes): reads as a clean
# "teacher" marker, clearly an outline vs. the opaque filled distilled bubbles.
TEACHER_FILL = (0.11, 0.38, 0.34, 0.16)  # #1C6157 @ alpha 0.16
ring = axD.scatter([tpm], [tp], s=bubble_size(tpm) * D_SFAC, facecolors=[TEACHER_FILL],
                   edgecolors=tcol, lw=2.3, zorder=5)
axD.annotate(f"NTv3-650M teacher {tp:.3f}", (tpm, tp), textcoords="data",
             xytext=(60.0, 0.632), ha="center", va="bottom", fontsize=15.0,
             fontweight="bold", color=tcol, zorder=8, path_effects=WHITE_HALO,
             arrowprops=dict(arrowstyle="-", color=tcol, lw=1.0, alpha=0.7,
                             shrinkA=3, shrinkB=9))

# --- distilled bubbles (sized by params) + size-matched from-scratch squares ---
# per-model label placement (student label anchor, from-scratch label anchor)
D_STU_LAB = {"NTv3-8M":   dict(xytext=(0, 20),   ha="center"),   # above its own bubble
             "NTv3-100M": dict(xytext=(0, 22),   ha="center")}   # directly above
D_SCR_LAB = {"NTv3-8M":   dict(xytext=(15, -1),  ha="left"),     # right of square
             "NTv3-100M": dict(xytext=(16, -1),  ha="left")}     # right of square
for name, (pm, dist, scr) in MT_ACC.items():
    c = TEAL[name]
    # from-scratch open grey square, size-matched to the student bubble
    axD.scatter([pm], [scr], s=bubble_size(pm) * D_SQ_FAC, marker="s", facecolors="white",
                edgecolors=NAT["grey_dot"], lw=1.7, zorder=5)
    # distilled filled bubble sized by params
    axD.scatter([pm], [dist], s=bubble_size(pm) * D_SFAC, facecolors=c, edgecolors="white",
                lw=1.3, alpha=0.92, zorder=6)
    ret = dist / tp * 100
    # student label (name + Pearson + retention), halo'd, clear of the left spine
    axD.annotate(f"NTv3-{name.split('-')[1]}  {dist:.3f}\n({ret:.0f}% of teacher)",
                 (pm, dist), textcoords="offset points",
                 va="bottom", fontsize=14.0, fontweight="bold",
                 color=c, zorder=8, linespacing=1.1, path_effects=WHITE_HALO,
                 **D_STU_LAB[name])
    # baseline value, placed to the right of its square (clear of the connector)
    axD.annotate(f"baseline {scr:.3f}", (pm, scr), textcoords="offset points",
                 va="center", fontsize=13.0, color=NAT["grey_dot"], zorder=8,
                 path_effects=WHITE_HALO, **D_SCR_LAB[name])

# proxy legend (in-axes)
# NOTE: panel-D marker legend removed; the caption defines ring = teacher, bubble = distilled
# student, square = size-matched baseline.
axD.set_xlabel("Model size (million parameters)", fontsize=LAB_FS, color=NAT["ink"])
axD.set_ylabel("Mean Pearson r", fontsize=LAB_FS, color=NAT["ink"])
axD.set_xticks([10, 100, 1000]); axD.set_xticklabels(["10 M", "100 M", "1 B"])
axD.tick_params(axis="both", labelsize=TICK_FS, length=5, width=1.0)
style_spines(axD)

# ===========================================================================
# PANELS E / F : multitrack whole-test time + GPU peak memory (log bars)
# ===========================================================================
# Multitrack row shows the SAME roster as Panel D: distilled student NTv3-8M,
# NTv3-100M, and the NTv3-650M teacher (BPNet dropped). Fold-change reference is
# NTv3-8M (the deployable distilled student), matching Panel D's story.
mt_order = ["NTv3-8M", "NTv3-100M", "NTv3-650M-teacher"]
# compact two-line tick labels (narrow bar panels): short id on top, params below
MT_SHORT = {"BPNet-0.12M": "BPNet\n0.12 M", "NTv3-8M": "8M\n7.7 M",
            "NTv3-100M": "100M\n106 M", "NTv3-650M-teacher": "650M\n655 M"}
def mtcol(m):
    return NAT["grey_dot"] if m == "BPNet-0.12M" else TEAL[m]

def draw_mt(ax, valcol, title, ylabel, kind):
    x = np.arange(len(mt_order))
    vals = [float(MT_ROW[m][valcol]) for m in mt_order]
    cols = [mtcol(m) for m in mt_order]
    ref = vals[0]
    ax.bar(x, vals, width=0.60, color=cols, edgecolor="none", zorder=3)
    ax.set_yscale("log")
    ax.set_ylim(ref * 0.92, max(vals) * 2.6)      # same emphasis for the multitrack row
    ax.set_xlim(-0.7, len(mt_order) - 0.3)        # pad so end bars clear the y-axis / right frame
    for xi, v in zip(x, vals):
        a = fmt_time(v) if kind == "time" else fmt_mem(v)
        f = fold_txt(v / ref)
        ax.annotate(a + "\n" + r"$\bf{" + f.replace("x", r"\times") + r"}$",
                    (xi, v), textcoords="offset points", xytext=(0, 4),
                    ha="center", va="bottom", fontsize=15.0, color=NAT["ink"])
    ax.set_xticks(x)
    ax.set_xticklabels([MT_SHORT[m] for m in mt_order], fontsize=14.0, linespacing=1.35)
    ax.set_ylabel(ylabel, fontsize=LAB_FS, color=NAT["ink"])
    ax.set_title(title, fontsize=TITLE_FS, pad=8, color=NAT["ink"])
    ax.tick_params(axis="y", labelsize=TICK_FS - 1, length=4, width=1.0)
    ax.tick_params(axis="x", length=0)
    ax.grid(axis="y", which="major", alpha=0.28, linestyle="--")
    ax.yaxis.set_minor_formatter(NullFormatter())
    style_spines(ax)

draw_mt(axE, "gpu_whole_test_s", "Whole-test GPU time", "Whole-test time (s)", "time")
draw_mt(axF, "gpu_peak_mem_mb", "Peak GPU memory", "Peak memory (MB)", "mem")

# ===========================================================================
# PANEL LETTERS A..F (identical relative positions) + titles + footnotes
# ===========================================================================
fig.canvas.draw()
for ax, lb in [(axA, "A"), (axB, "B"), (axC, "C"), (axD, "D"), (axE, "E"), (axF, "F")]:
    bb = ax.get_position()
    fig.text(bb.x0 - 0.044, bb.y1 + 0.020, lb, fontsize=30, fontweight="bold",
             ha="left", va="bottom", color=NAT["ink"])

# NOTE: the figure-level title was removed -- at legible type it was wider than the panel grid,
# forcing bbox_inches='tight' to pad the canvas and shrink everything else. The caption
# carries the same message.
bbA = axA.get_position(); bbD = axD.get_position()
# Row-group titles: large, upright, black (moved off the grey-italic style so they read as
# the primary "18-task" / "multi-track" section headers requested in review).
fig.text(bbA.x0, bbA.y1 + 0.012, "Classification (18 tasks)", ha="left", va="center",
         fontsize=21.0, fontweight="bold", color=NAT["ink"])
fig.text(bbD.x0, bbD.y1 + 0.012, "Multitrack regression", ha="left", va="center",
         fontsize=21.0, fontweight="bold", color=NAT["ink"])
# NOTE: the per-figure caveat footnotes (fp32/A6000 hardware, all-18 sample count, marker legend)
# were moved into the LaTeX figure caption so the figure carries no sub-9pt text.

fig.savefig("output/fig4_efficiency_carbon.pdf", bbox_inches="tight")
fig.savefig("output/fig4_efficiency_carbon.png", dpi=200, bbox_inches="tight")
import shutil
shutil.copy("output/fig4_efficiency_carbon.png", "previews/fig4_efficiency_carbon.png")
print("saved fig4_efficiency_carbon (2x3 grid)")
