#!/usr/bin/env python3
"""Generate NTv3 size-ladder scaling figures.

Reproducible plotting for the 20-tier NTv3 size sweep.
All numbers below were read from the per-run ``ntv3_finetune_result.json``
``test_mean_pearson`` (34-track joint / 1-track specialist) and the joint
runs' ``per_track_pearson['ENCSR325NFE']`` slice, verified on the galaxy SSD
at ``${OG_SCRATCH}/ntv3_targets/{size_sweep,size_sweep_pertrack}``.

Run with the fast system python (matplotlib), per project convention:
    /usr/bin/python3 plot_size_ladder.py

Produces (150 dpi PNG):
    fig1_joint34_scaling.png
    fig2_pertrack_t12_scaling.png
    fig3_kd_delta.png
"""

import os
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# Verified data (source: ntv3_finetune_result.json on galaxy SSD)
# ---------------------------------------------------------------------------
# Student parameter counts (x-axis, log scale).
PARAMS = [4.34e6, 7.69e6, 29.87e6, 106.46e6, 303.05e6]
TEACHER_PARAMS = 651.83e6
SIZE_LABELS = ["4M", "8M", "30M", "100M", "300M"]

# Fig 1 -- JOINT 34-track test_mean_pearson.
JOINT_KD = [
    0.4606471669063459,
    0.4768708367942698,
    0.49582419754634266,
    0.5188377911260873,
    0.5309141744530618,
]
JOINT_NOKD = [
    0.4349191411066653,
    0.4483069102471927,
    0.46807474264756777,
    0.48478547398375205,
    0.4968581091005864,
]
# Pretrained-init reference (only 8M and 100M were run).
PRETRAINED_INIT = {7.69e6: 0.4747732478058837, 106.46e6: 0.5178211217864834}
TEACHER_JOINT = 0.6058604871998875  # 34-track mean

# Fig 2 -- PER-TRACK t12 (ATAC ENCSR325NFE), single-track pearson.
PT_KD = [
    0.7322410774399275,
    0.7361891537787985,
    0.7521458578345879,
    0.7631253286436455,
    0.768078039219381,
]
PT_NOKD = [
    0.7160343804293897,
    0.7182608479411978,
    0.7395822453215777,
    0.7448439142014617,
    0.7562202260856036,
]
# Joint 34-track model's ENCSR325NFE slice (from the joint KD result.jsons).
JOINT_SLICE = [
    0.6882138349565987,
    0.705974003088749,
    0.7254992718402737,
    0.7471823661275808,
    0.7585421291297799,
]
TEACHER_SLICE = 0.8325899966792608  # teacher ENCSR325NFE slice

# Fig 3 -- KD delta (KD - no-KD), computed from the verified values above.
JOINT_DELTA = [kd - nk for kd, nk in zip(JOINT_KD, JOINT_NOKD)]
PT_DELTA = [kd - nk for kd, nk in zip(PT_KD, PT_NOKD)]

# ---------------------------------------------------------------------------
# Okabe-Ito colorblind-safe palette; colors are consistent per arm across figs.
# ---------------------------------------------------------------------------
C_KD = "#0072B2"  # blue    -- scratch + KD
C_NOKD = "#E69F00"  # orange  -- scratch no-KD (matched)
C_PRETRAIN = "#009E73"  # green   -- pretrained-init reference
C_SLICE = "#CC79A7"  # purple  -- joint-model single-track slice
C_TEACHER = "#000000"  # black   -- teacher (horizontal reference)
C_PT_DELTA = "#D55E00"  # vermillion -- per-track KD delta

plt.rcParams.update(
    {
        "font.size": 11,
        "axes.titlesize": 12.5,
        "axes.labelsize": 11.5,
        "legend.fontsize": 9.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 150,
    }
)


def _style_logx(ax):
    """Apply shared log-x ticks/grid/labels to a size-ladder axis."""
    ax.set_xscale("log")
    ax.set_xticks(PARAMS + [TEACHER_PARAMS])
    ax.set_xticklabels(SIZE_LABELS + ["650M\n(teacher)"])
    ax.set_xlabel("Student params (log scale)")
    ax.grid(True, which="major", axis="both", ls="-", lw=0.5, alpha=0.35)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=9.5)


def fig1():
    """Fig 1: joint 34-track scaling (KD vs no-KD vs pretrained-init vs teacher)."""
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    ax.plot(PARAMS, JOINT_KD, "-o", color=C_KD, lw=2, ms=7, label="scratch + KD (34-trk)")
    ax.plot(PARAMS, JOINT_NOKD, "-s", color=C_NOKD, lw=2, ms=7, label="scratch, no-KD (matched)")
    px = sorted(PRETRAINED_INIT)
    ax.plot(
        px,
        [PRETRAINED_INIT[p] for p in px],
        "D",
        color=C_PRETRAIN,
        ms=8,
        ls="none",
        label="pretrained-init (ref)",
    )
    ax.axhline(
        TEACHER_JOINT, ls="--", lw=1.6, color=C_TEACHER, label="650M teacher = %.3f" % TEACHER_JOINT
    )

    # Subtle KD-gap annotation at the largest size.
    x = PARAMS[-1]
    ax.annotate(
        "",
        xy=(x, JOINT_KD[-1]),
        xytext=(x, JOINT_NOKD[-1]),
        arrowprops=dict(arrowstyle="<->", color="0.35", lw=1.1),
    )
    ax.annotate(
        "KD +%.3f" % (JOINT_KD[-1] - JOINT_NOKD[-1]),
        xy=(x, (JOINT_KD[-1] + JOINT_NOKD[-1]) / 2),
        xytext=(-8, 0),
        textcoords="offset points",
        ha="right",
        va="center",
        fontsize=8.5,
        color="0.30",
    )

    _style_logx(ax)
    ax.set_ylabel("Test mean Pearson (34 tracks)")
    ax.set_title("NTv3 size-ladder: joint 34-track multitrack scaling")
    ax.set_ylim(0.42, 0.62)
    ax.legend(loc="lower right", frameon=False)
    fig.tight_layout()
    out = os.path.join(HERE, "fig1_joint34_scaling.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def fig2():
    """Fig 2: per-track ATAC ENCSR325NFE scaling (specialist vs joint slice)."""
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    ax.plot(PARAMS, PT_KD, "-o", color=C_KD, lw=2, ms=7, label="specialist + KD (1-trk)")
    ax.plot(PARAMS, PT_NOKD, "-s", color=C_NOKD, lw=2, ms=7, label="specialist, no-KD (matched)")
    ax.plot(
        PARAMS,
        JOINT_SLICE,
        "-^",
        color=C_SLICE,
        lw=2,
        ms=7,
        label="joint 34-trk model, this-track slice",
    )
    ax.axhline(
        TEACHER_SLICE,
        ls="--",
        lw=1.6,
        color=C_TEACHER,
        label="650M teacher slice = %.3f" % TEACHER_SLICE,
    )

    _style_logx(ax)
    ax.set_ylabel("Pearson on ENCSR325NFE (single track)")
    ax.set_title(
        "Per-track scaling: ATAC ENCSR325NFE (ONE track,\nnot comparable to Fig 1's 34-track mean)"
    )
    ax.set_ylim(0.66, 0.85)
    ax.legend(loc="lower right", frameon=False)
    fig.tight_layout()
    out = os.path.join(HERE, "fig2_pertrack_t12_scaling.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def fig3():
    """Fig 3: KD delta (KD - no-KD) vs size, joint 34-trk vs per-track."""
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    ax.axhline(0.0, ls="-", lw=1.0, color="0.55")
    ax.plot(PARAMS, JOINT_DELTA, "-o", color=C_KD, lw=2, ms=7, label="joint 34-track  KD $\\Delta$")
    ax.plot(
        PARAMS,
        PT_DELTA,
        "-^",
        color=C_PT_DELTA,
        lw=2,
        ms=7,
        label="per-track (ENCSR325NFE)  KD $\\Delta$",
    )

    _style_logx(ax)
    ax.set_ylabel("KD gain  $\\Delta$ = Pearson(KD) - Pearson(no-KD)")
    ax.set_title("KD benefit vs model size:\njoint arm gains ~2x more from KD")
    ax.set_ylim(0.0, 0.040)
    ax.legend(loc="upper left", frameon=False)
    fig.tight_layout()
    out = os.path.join(HERE, "fig3_kd_delta.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


if __name__ == "__main__":
    for f in (fig1(), fig2(), fig3()):
        print("wrote", f)
    print("joint KD delta:", ["%.4f" % d for d in JOINT_DELTA])
    print("per-track KD delta:", ["%.4f" % d for d in PT_DELTA])
