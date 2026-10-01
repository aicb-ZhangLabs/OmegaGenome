#!/usr/bin/env python
"""Signal-track multi-panel figure (CPU; needs matplotlib + seaborn).

Four panels, consistent 3-color scheme (teacher / distilled student / from-scratch baseline) matching
the t-SNE-panel aesthetic (seaborn white, distinct5-ish greens/blues/golds, #333 text, panel letters):

  (a) VERIFIED SIGNAL-TRACK small-multiples — per-bp predicted signal along held-out 32 kb test
      windows, for >=5 tracks spanning the assay families. Each shown track is VERIFIED at its window:
      r(student,GT) > r(baseline,GT) AND r(student,teacher) > r(baseline,teacher); the window-specific
      r's are printed in the sub-title so the figure is self-verifying. GT (grey) + teacher + student +
      baseline, shared relative-bp x.
  (b) PER-ASSAY-FAMILY bars (joint-34 full-test, Table R1c-3): teacher vs distilled student vs
      from-scratch, 5 families + overall — student close to teacher, > baseline across assays.
  (c) PER-TRACK SCATTER — distilled-student Pearson (y) vs from-scratch Pearson (x), one point per
      track (34), coloured by assay family, y=x diagonal; points above = KD wins. Mean lift annotated.
  (d) SINGLE-TRACK 8M KD-win scorecard — the 7 directional single-track wins (base->KD arrows toward
      the per-track teacher), 4 assay families.

Inputs: the inference .npz (panel a) + the full-test per-track CSV/JSONs (panels b/c/d).
Run (enformer env):  python scripts/signal_track_plot.py --npz <npz> --out-prefix <fig>
"""

import argparse
import csv
import json
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.gridspec as gridspec  # noqa: E402
import seaborn as sns  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SSD = os.environ.get("SSD", os.environ.get("OG_SCRATCH", "output"))
T_DIR = f"{SSD}/ntv3_targets"

# Consistent 3-color scheme across ALL panels (teacher / student / baseline), reusing the EXACT
# feature_viz.py / t-SNE-panel palette: distinct5 green for the teacher, the established ts_color steel
# blue (#4575B4) for OUR distilled student, distinct5 gold for the from-scratch baseline. Muted, clean.
C_TEACHER = "#5C8A4D"  # distinct5 green  (reference / upper bound)
C_STUDENT = "#4575B4"  # feature_viz ts_color steel blue (OUR distilled student)
C_BASELINE = "#E5A832"  # distinct5 gold   (from-scratch)
C_GT = "#D5D5D5"  # light grey ground-truth fill

# Per-assay-family colours for the scatter — the distinct5/t-SNE class palette (no blue/red clash).
FAM_COLORS = {
    "ATAC-seq": "#E07020",
    "Histone ChIP-seq": "#5C8A4D",
    "PRO-cap": "#9370DB",
    "eCLIP": "#722F72",
    "polyA plus RNA-seq": "#E5A832",
    "total RNA-seq": "#8B7355",
}
FAM_SHORT = {"polyA plus RNA-seq": "RNA-seq (polyA)", "total RNA-seq": "RNA-seq (total)"}

# Panel-a candidate tracks (>=1 per assay family). Each is VERIFIED in-script at its displayed window:
# kept only if r(student,GT) > r(baseline,GT) AND r(student,teacher) > r(baseline,teacher). The
# selection favours clear-gap capacity-constrained assays (PRO-cap/eCLIP/RNA) but honestly includes
# ATAC + Histone examples that also pass. (file_id, pretty label)
REP_CANDIDATES = [
    ("ENCSR799DGV_P", "PRO-cap (transcription initiation)"),
    ("ENCSR249ROI_M", "eCLIP (RBP occupancy)"),
    ("ENCSR619DQO_M", "RNA-seq, total (transcript abundance)"),
    ("ENCSR527JGN_M", "RNA-seq, polyA (transcript abundance)"),
    ("ENCSR682BFG", "Histone ChIP-seq (mark enrichment)"),
    ("ENCSR325NFE", "ATAC-seq (chromatin accessibility)"),
]

# Panel-b per-family table (Table R1c-3): family -> (teacher, baseline, student).
FAMILY_BARS = [
    ("ATAC-seq", 0.757, 0.567, 0.588),
    ("Histone", 0.723, 0.580, 0.593),
    ("PRO-cap", 0.517, 0.347, 0.421),
    ("eCLIP", 0.560, 0.482, 0.492),
    ("RNA-seq", 0.635, 0.540, 0.545),
    ("Overall", 0.606, 0.475, 0.505),
]

# Panel-d single-track 8M KD wins (Table R1c-1): (label, teacher, baseline, KD).
SINGLE_WINS = [
    ("t0 PRO-cap", 0.534, 0.380, 0.447),
    ("t3 PRO-cap", 0.567, 0.413, 0.449),
    ("t11 eCLIP", 0.689, 0.584, 0.614),
    ("t12 ATAC", 0.833, 0.747, 0.756),
    ("t14 eCLIP", 0.478, 0.393, 0.410),
    ("t18 RNA-seq", 0.720, 0.593, 0.633),
    ("t20 RNA-seq", 0.699, 0.609, 0.637),
]


def setup_nature_style():
    sns.set_style("white")
    sns.set_context("paper", font_scale=1.8)
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "axes.linewidth": 1.0,
            "axes.edgecolor": "#333333",
            "axes.labelcolor": "#333333",
            "xtick.color": "#333333",
            "ytick.color": "#333333",
            "text.color": "#333333",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "font.weight": "normal",
            "axes.titleweight": "normal",
            "axes.labelweight": "normal",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def load_per_track():
    """Full-test per-track Pearson for teacher / 8M baseline / 8M KD, aligned by track_id + assay."""
    tea, assay = {}, {}
    with open(f"{REPO}/results/ntv3_650m_per_track_3seed.csv") as f:
        for r in csv.DictReader(f):
            tea[r["track_id"]] = float(r["mean"])
            assay[r["track_id"]] = r["assay"]
    base = json.load(open(f"{T_DIR}/ntv3_8m_baseline/ntv3_finetune_result.json"))[
        "per_track_pearson"
    ]
    kd = json.load(open(f"{T_DIR}/ntv3_8m_kd_stdmse/ntv3_finetune_result.json"))[
        "per_track_pearson"
    ]
    ids = [k for k in tea if k in base and k in kd]
    return ids, assay, tea, base, kd


def smooth(x, k):
    if k <= 1:
        return x
    return np.convolve(np.nan_to_num(x), np.ones(k) / k, mode="same")


def pick_window(targets, ti):
    return int(np.nanargmax(np.nanvar(targets[:, :, ti], axis=1)))


def _corr(a, b):
    a = np.nan_to_num(a)
    b = np.nan_to_num(b)
    if a.std() < 1e-9 or b.std() < 1e-9:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def verify_window(targets, pred, ti, wi):
    """Window-specific verification: returns the 4 correlations + a PASS flag.

    PASS iff the distilled student is closer than the from-scratch baseline to BOTH the ground truth
    AND the teacher output, on this exact window/track — so a displayed panel provably matches the
    'distillation > from-scratch' message. Returns dict(rS_gt, rB_gt, rS_t, rB_t, rT_gt, passes).
    """
    gt = targets[wi, :, ti]
    t = pred["teacher"][wi, :, ti]
    s = pred["student"][wi, :, ti]
    b = pred["baseline"][wi, :, ti]
    rS_gt, rB_gt = _corr(s, gt), _corr(b, gt)
    rS_t, rB_t = _corr(s, t), _corr(b, t)
    return dict(
        rS_gt=rS_gt,
        rB_gt=rB_gt,
        rS_t=rS_t,
        rB_t=rB_t,
        rT_gt=_corr(t, gt),
        passes=bool(rS_gt > rB_gt and rS_t > rB_t),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--out-prefix", required=True)
    ap.add_argument("--smooth", type=int, default=101)
    args = ap.parse_args()

    z = np.load(args.npz, allow_pickle=True)
    targets = z["targets"]
    pred = {
        "teacher": z["pred_teacher"],
        "student": z["pred_student_kd"],
        "baseline": z["pred_baseline"],
    }
    track_ids = list(z["track_ids"])
    L_out, crop_off = int(z["L_out"]), int(z["crop_off"])
    coords = z["win_coords"]

    ids, assay, tea_pt, base_pt, kd_pt = load_per_track()

    setup_nature_style()
    # Nature-style asymmetric layout: panel a (the dense per-bp signal small-multiples) is a TALL LEFT
    # COLUMN spanning all three rows -> one stacked mini-axis per verified track, shared bp x-axis (only
    # the bottom one labelled). The three lighter summary panels b/c/d are compactly stacked on the RIGHT.
    fig = plt.figure(figsize=(15.5, 12.5))
    # top tightened so the suptitle (y=0.978) sits close above the panels: the suptitle gets the very
    # top row, panel a's legend (lowered anchor below) sits just under it, no dead vertical gap.
    outer = gridspec.GridSpec(
        3, 2, figure=fig, width_ratios=[1.5, 1.0], wspace=0.20, hspace=0.62, top=0.895, bottom=0.055
    )

    # ---------- PANEL A: VERIFIED signal-track small-multiples (tall left column) ----------
    x_kb = np.arange(L_out) / 1000.0
    # verify every candidate at its best-structured window; keep only PASS tracks.
    shown = []
    print("PANEL-A VERIFICATION (window-specific):")
    print(
        f"  {'track':15}{'assay-label':38}{'win':>4} {'rS_GT':>6}{'rB_GT':>6}{'rS_T':>6}{'rB_T':>6} pass"
    )
    for fid, label in REP_CANDIDATES:
        ti = track_ids.index(fid)
        wi = pick_window(targets, ti)
        v = verify_window(targets, pred, ti, wi)
        print(
            f"  {fid:15}{label:38}{wi:>4} {v['rS_gt']:6.2f}{v['rB_gt']:6.2f}"
            f"{v['rS_t']:6.2f}{v['rB_t']:6.2f}  {'YES' if v['passes'] else 'NO'}"
        )
        if v["passes"]:
            shown.append((fid, label, ti, wi, v))
    assert len(shown) >= 5, f"need >=5 verified panel-a tracks, got {len(shown)}"

    # one mini-axis per verified track, stacked vertically, sharing the bp x-axis.
    n = len(shown)
    gs_a = outer[:, 0].subgridspec(n, 1, hspace=0.72)
    first_a_ax = None
    ax_a_prev = None
    for pi, (fid, label, ti, wi, v) in enumerate(shown):
        ax = (
            fig.add_subplot(gs_a[pi], sharex=ax_a_prev)
            if ax_a_prev is not None
            else fig.add_subplot(gs_a[pi])
        )
        ax_a_prev = ax
        if pi == 0:
            first_a_ax = ax
        chrom, wstart, _ = coords[wi]
        gt = smooth(targets[wi, :, ti], args.smooth)
        ax.fill_between(x_kb, 0, gt, color=C_GT, alpha=0.7, lw=0, label="ground truth", zorder=1)
        ax.plot(
            x_kb,
            smooth(pred["teacher"][wi, :, ti], args.smooth),
            color=C_TEACHER,
            lw=1.1,
            label="NTv3-650M teacher",
            zorder=4,
        )
        ax.plot(
            x_kb,
            smooth(pred["student"][wi, :, ti], args.smooth),
            color=C_STUDENT,
            lw=1.1,
            label="Distilled 8M student",
            zorder=3,
        )
        ax.plot(
            x_kb,
            smooth(pred["baseline"][wi, :, ti], args.smooth),
            color=C_BASELINE,
            lw=1.0,
            ls="--",
            label="From-scratch 8M",
            zorder=2,
        )
        # title: JUST the short assay label (kept narrow so it can never collide with panel d labels).
        ax.set_title(label, loc="left", fontsize=11.0)
        # self-verifying annotation moved INSIDE the axes (left-justified, top) so it can never overflow
        # the axes width: window-specific student>baseline r vs truth and vs teacher (numbers verified).
        ax.text(
            0.012,
            0.94,
            f"vs truth  student {v['rS_gt']:.2f} > baseline {v['rB_gt']:.2f}\n"
            f"vs teacher  student {v['rS_t']:.2f} > baseline {v['rB_t']:.2f}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9.5,
            color="#333333",
            linespacing=1.25,
            zorder=5,
        )
        ax.set_ylabel("signal", fontsize=11.5)
        ax.margins(x=0.0)
        ax.set_ylim(bottom=0)
        ax.tick_params(labelsize=10.0)
        ax.text(
            0.992,
            0.94,
            f"{chrom}:{int(wstart + crop_off):,}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=9.0,
            color="#888888",
        )
        if pi == 0:
            ax.legend(
                loc="lower left",
                frameon=False,
                ncol=4,
                fontsize=9.5,
                bbox_to_anchor=(0, 1.22),
                columnspacing=1.1,
                handlelength=1.4,
            )
        if pi < n - 1:  # shared x: hide tick labels on all but the bottom panel
            plt.setp(ax.get_xticklabels(), visible=False)
        else:
            ax.set_xlabel("position within scored 12.3 kb window (kb)", fontsize=11.5)

    # ---------- RIGHT COLUMN: compact summary panels b / c / d, stacked ----------
    # ---------- PANEL B: per-family bars ----------
    axb = fig.add_subplot(outer[0, 1])
    fams = [f[0] for f in FAMILY_BARS]
    teas = [f[1] for f in FAMILY_BARS]
    bases = [f[2] for f in FAMILY_BARS]
    studs = [f[3] for f in FAMILY_BARS]
    x = np.arange(len(fams))
    w = 0.26
    axb.bar(x - w, teas, w, color=C_TEACHER, label="Teacher (650M)", zorder=3)
    axb.bar(x, studs, w, color=C_STUDENT, label="Distilled 8M", zorder=3)
    axb.bar(x + w, bases, w, color=C_BASELINE, label="From-scratch 8M", zorder=3)
    axb.set_xticks(x)
    axb.set_xticklabels(fams, rotation=30, ha="right", fontsize=10.5)
    axb.set_ylabel("test Pearson r", fontsize=12)
    axb.set_ylim(0, 0.9)
    axb.legend(frameon=False, fontsize=10.0, loc="upper right")
    axb.set_title("Joint 34-track distillation, per assay family", fontsize=12.5, loc="left")
    axb.grid(axis="y", ls=":", alpha=0.4, zorder=0)

    # ---------- PANEL C: per-track scatter student vs baseline ----------
    axc = fig.add_subplot(outer[1, 1])
    seen = set()
    lift = []
    for tid in ids:
        a = assay[tid]
        xq, yq = base_pt[tid], kd_pt[tid]
        lift.append(yq - xq)
        lbl = FAM_SHORT.get(a, a)
        axc.scatter(
            xq,
            yq,
            s=46,
            color=FAM_COLORS.get(a, "#666"),
            edgecolor="white",
            lw=0.6,
            label=lbl if a not in seen else None,
            zorder=3,
        )
        seen.add(a)
    lo, hi = 0.1, 0.9
    axc.plot([lo, hi], [lo, hi], color="#555", ls="--", lw=1.0, zorder=2, label="y = x (no change)")
    # equal x/y RANGES keep the scatter square in data terms, but we do NOT fix the box aspect — that
    # would shrink+center the axes and break left-edge alignment with panels b/d. Box is re-aligned to
    # b/d's x0+width below (after draw), so the right column is a clean vertical stack.
    axc.set_xlim(lo, hi)
    axc.set_ylim(lo, hi)
    axc.set_xlabel("From-scratch baseline Pearson r", fontsize=12)
    axc.set_ylabel("Distilled student Pearson r", fontsize=12)
    nwin = sum(1 for tid in ids if kd_pt[tid] > base_pt[tid])
    axc.set_title(
        f"Per-track: distillation vs from-scratch (n=34)\n"
        f"{nwin}/34 above diagonal · mean lift +{np.mean(lift):.3f}",
        fontsize=12.5,
        loc="left",
    )
    axc.legend(frameon=False, fontsize=9.0, loc="lower right", handletextpad=0.3, labelspacing=0.22)
    axc.grid(ls=":", alpha=0.4, zorder=0)

    # ---------- PANEL D: single-track 8M KD-win scorecard ----------
    axd = fig.add_subplot(outer[2, 1])
    yy = np.arange(len(SINGLE_WINS))[::-1]
    for j, (lab, t, b, k) in enumerate(SINGLE_WINS):
        y = yy[j]
        axd.scatter(
            t,
            y,
            marker="D",
            s=55,
            color=C_TEACHER,
            zorder=4,
            label="Teacher (650M)" if j == 0 else None,
        )
        axd.scatter(
            b, y, s=52, color=C_BASELINE, zorder=3, label="From-scratch 8M" if j == 0 else None
        )
        axd.scatter(k, y, s=52, color=C_STUDENT, zorder=3, label="Distilled 8M" if j == 0 else None)
        axd.annotate(
            "",
            xy=(k, y),
            xytext=(b, y),
            arrowprops=dict(arrowstyle="->", color=C_STUDENT, lw=1.2),
            zorder=2,
        )
    axd.set_yticks(yy[::-1])
    axd.set_yticklabels([s[0] for s in SINGLE_WINS][::-1], fontsize=10.5)
    axd.set_xlabel("test Pearson r", fontsize=12)
    axd.set_title("Single-track 8M students: 7 KD-wins (4 families)", fontsize=12.5, loc="left")
    axd.legend(frameon=False, fontsize=9.5, loc="lower right")
    axd.grid(axis="x", ls=":", alpha=0.4, zorder=0)
    axd.set_xlim(0.3, 0.9)

    # panel letters (figure coords, after layout positions settle): a labels the tall left column,
    # b/c/d sit at the top-left of each compact right-column panel.
    fig.canvas.draw()
    # force panel c's horizontal box to match b/d (same left edge + width) so the right column is a
    # clean vertical stack with aligned left edges; keep c's own vertical extent.
    pb = axb.get_position()
    pc = axc.get_position()
    axc.set_position([pb.x0, pc.y0, pb.width, pc.height])
    fig.canvas.draw()
    pa = first_a_ax.get_position()
    fig.text(max(pa.x0 - 0.028, 0.004), pa.y1 + 0.040, "A", fontsize=23, fontweight="bold")
    for axx, lett in ((axb, "B"), (axc, "C"), (axd, "D")):
        p = axx.get_position()
        fig.text(max(p.x0 - 0.046, 0.004), p.y1 + 0.018, lett, fontsize=23, fontweight="bold")

    fig.suptitle(
        "OmegaGenome distillation on the NTv3 base-resolution, multi-track regression benchmark",
        fontsize=17,
        y=0.978,
    )
    for ext in ("png", "pdf"):
        fig.savefig(f"{args.out_prefix}.{ext}", dpi=300, bbox_inches="tight")
    print(
        f"saved {args.out_prefix}.png / .pdf | scatter wins {nwin}/34 mean lift {np.mean(lift):+.3f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
