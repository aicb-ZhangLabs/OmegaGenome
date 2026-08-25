#!/usr/bin/env python
"""Revision feature panel: Teacher / OmegaGenome / DKD (paper `Combined_KD` style).

Companion to ``feature_viz.py`` that reuses its verbatim Nature style, the
``distinct5`` ("Sunset Meadow") class palette, the t-SNE + silhouette computation,
the PCA-aligned sample-wise cosine, and the logit-scatter logic -- but arranges the
panels as requested for the revision so the figure directly contrasts the
feature-aligned student (OmegaGenome) against the output-only student (DKD):

  Row 1 (t-SNE, ``distinct5`` class colours + silhouette box):
      A  Enformer teacher      B  OmegaGenome       C  DKD
  Row 2 (logit scatters first, cosine histogram LAST):
      D  Logit Scatter (T-OmegaGenome)   E  Logit Scatter (T-DKD)
      F  Sample-wise Cosine Similarity   (narrow-bar HISTOGRAM, Teacher-OmegaGenome
                                          vs Teacher-DKD, mean lines)

OmegaGenome recovers teacher-like class separability and output agreement, while the
output-only DKD student does not. Consumes the cached ``feature_logit_data.npz``
(teacher/omega/dkd features+logits) produced by ``extract_feats_logits.py``; runs in
the same matplotlib+seaborn+sklearn env as ``feature_viz.py`` (no GPU).
"""
import argparse
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.manifold import TSNE  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.metrics import silhouette_score  # noqa: E402
from scipy.spatial.distance import cosine  # noqa: E402
from scipy.stats import pearsonr  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
# Reuse the exact style/palette from the paper-panel script (no duplication of config).
from feature_viz import setup_nature_style, ROW1_COLOR_OPTIONS, ROW2_COLORS, TEACHER_NAME  # noqa: E402


def build(npz, teacher, scheme, out):
    """Render the Teacher/OmegaGenome/DKD 2x3 panel and return the summary stats.

    Args:
        npz: path to feature_logit_data.npz (keys F_teacher/F_omega/F_dkd,
             L_teacher/L_omega/L_dkd, labels).
        teacher: teacher key into TEACHER_NAME (e.g. "enformer").
        scheme: Row-1 class-colour scheme key (e.g. "distinct5").
        out: output .png path (a sibling .pdf is written too).

    Returns:
        dict of silhouettes, class-0 logit correlations, and mean cosines.
    """
    d = np.load(npz)
    labels = d["labels"]
    F_t, F_o, F_k = d["F_teacher"], d["F_omega"], d["F_dkd"]
    L_t, L_o, L_k = d["L_teacher"], d["L_omega"], d["L_dkd"]

    class_colors = ROW1_COLOR_OPTIONS[scheme]["class_colors"][:len(np.unique(labels))]
    o_color, k_color = ROW2_COLORS["ts_color"], ROW2_COLORS["tb_color"]  # OmegaGenome=blue, DKD=grey
    accent, box = ROW2_COLORS["accent"], ROW2_COLORS["text_box_color"]
    legend_fs, textbox_fs, corr_fs = 12.6, 15.4, 15.4

    setup_nature_style()
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    t_s = StandardScaler().fit_transform(F_t)
    o_s = StandardScaler().fit_transform(F_o)
    k_s = StandardScaler().fit_transform(F_k)

    # ----- ROW 1: t-SNE (Teacher | OmegaGenome | DKD) -----
    t_2 = TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(t_s)
    o_2 = TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(o_s)
    k_2 = TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(k_s)
    sil_t = silhouette_score(t_2, labels)
    sil_o = silhouette_score(o_2, labels)
    sil_k = silhouette_score(k_2, labels)

    def tsne_panel(ax, e2, title, sil):
        for i, lab in enumerate(np.unique(labels)):
            m = labels == lab
            ax.scatter(e2[m, 0], e2[m, 1], c=class_colors[i], s=25, alpha=0.7,
                       label=f"Class {lab}", edgecolors="white", linewidth=0.3)
        ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2", labelpad=2)
        ax.set_title(title, fontweight="normal")
        ax.legend(fontsize=legend_fs, frameon=True, fancybox=False, edgecolor="#CCCCCC", loc="upper right")
        ax.text(0.05, 0.95, f"Silhouette = {sil:.3f}", transform=ax.transAxes, fontsize=textbox_fs,
                va="top", ha="left",
                bbox=dict(boxstyle="round,pad=0.3", facecolor=box, edgecolor="#CCCCCC", alpha=0.9))
        ax.yaxis.set_major_locator(plt.MaxNLocator(5))
        ax.tick_params(axis="y", which="both", left=True, labelleft=True, rotation=90)

    tsne_panel(axes[0, 0], t_2, TEACHER_NAME[teacher], sil_t)
    tsne_panel(axes[0, 1], o_2, "OmegaGenome", sil_o)
    tsne_panel(axes[0, 2], k_2, "DKD", sil_k)

    # ----- ROW 2: D,E logit scatters (teacher vs OmegaGenome / DKD) -----
    o_r, _ = pearsonr(L_t[:, 0], L_o[:, 0])
    k_r, _ = pearsonr(L_t[:, 0], L_k[:, 0])

    def logit_panel(ax, x, y, color, ylab, title, r):
        ax.scatter(x, y, c=color, alpha=0.4, s=15, edgecolors="none")
        lims = [min(x.min(), y.min()), max(x.max(), y.max())]
        ax.plot(lims, lims, "--", color=accent, alpha=0.7, linewidth=1.5, label="y=x")
        ax.set_xlabel("Teacher Logit (Class 0)"); ax.set_ylabel(ylab, labelpad=2)
        ax.set_title(title, fontweight="normal")
        ax.legend(fontsize=legend_fs, frameon=True, fancybox=False, edgecolor="#CCCCCC")
        ax.text(0.95, 0.05, f"r = {r:.3f}", transform=ax.transAxes, fontsize=corr_fs,
                va="bottom", ha="right",
                bbox=dict(boxstyle="round,pad=0.3", facecolor=box, edgecolor="#CCCCCC", alpha=0.9))
        ax.yaxis.set_major_locator(plt.MaxNLocator(5))
        ax.tick_params(axis="y", which="both", left=True, labelleft=True, rotation=90)

    logit_panel(axes[1, 0], L_t[:, 0], L_o[:, 0], o_color,
                "OmegaGenome Logit (Class 0)", "Logit Scatter (T-OmegaGenome)", o_r)
    logit_panel(axes[1, 1], L_t[:, 0], L_k[:, 0], k_color,
                "DKD Logit (Class 0)", "Logit Scatter (T-DKD)", k_r)

    # ----- ROW 2: F sample-wise cosine HISTOGRAM (LAST) -----
    ax = axes[1, 2]
    cd = min(50, t_s.shape[1], o_s.shape[1], k_s.shape[1])
    t_al = PCA(n_components=cd, random_state=42).fit_transform(t_s)
    o_al = PCA(n_components=cd, random_state=42).fit_transform(o_s)
    k_al = PCA(n_components=cd, random_state=42).fit_transform(k_s)
    to = np.array([1 - cosine(t_al[i], o_al[i]) for i in range(len(t_al))])
    tk = np.array([1 - cosine(t_al[i], k_al[i]) for i in range(len(t_al))])
    all_s = np.concatenate([to, tk]); rng = (all_s.min() - 0.02, all_s.max() + 0.02)
    ax.hist(to, bins=80, range=rng, alpha=0.5, color=o_color, edgecolor="white", linewidth=0.3,
            label=f"Teacher-OmegaGenome\n(mean={to.mean():.3f})")
    ax.hist(tk, bins=80, range=rng, alpha=0.5, color=k_color, edgecolor="white", linewidth=0.3,
            label=f"Teacher-DKD\n(mean={tk.mean():.3f})")
    ax.axvline(to.mean(), color=o_color, linestyle="--", linewidth=1.5, alpha=0.8)
    ax.axvline(tk.mean(), color=k_color, linestyle="--", linewidth=1.5, alpha=0.8)
    ax.set_xlabel("Cosine Similarity"); ax.set_ylabel("Count", labelpad=2)
    ax.set_title("Sample-wise Cosine Similarity", fontweight="normal")
    ax.legend(fontsize=legend_fs, frameon=True, fancybox=False, edgecolor="#CCCCCC")
    ax.yaxis.set_major_locator(plt.MaxNLocator(5))
    ax.tick_params(axis="y", which="both", left=True, labelleft=True, rotation=90)

    for a, lab in zip(axes.flat, ["A", "B", "C", "D", "E", "F"]):
        a.text(-0.12, 1.08, lab, transform=a.transAxes, fontsize=22.4, fontweight="normal", va="top", ha="left")

    plt.tight_layout(); plt.subplots_adjust(hspace=0.25, wspace=0.22)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    plt.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.savefig(out.replace(".png", ".pdf"), dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print("SAVED", out)
    print(f"silhouette teacher={sil_t:.3f} omega={sil_o:.3f} dkd={sil_k:.3f} | "
          f"logit r T-O={o_r:.3f} T-D={k_r:.3f} | cos T-O={to.mean():.3f} T-D={tk.mean():.3f}")
    return {"sil_t": sil_t, "sil_o": sil_o, "sil_k": sil_k,
            "o_r": o_r, "k_r": k_r, "cos_to": float(to.mean()), "cos_tk": float(tk.mean())}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", default="enformer", choices=list(TEACHER_NAME))
    ap.add_argument("--scheme", default="distinct5", choices=list(ROW1_COLOR_OPTIONS))
    ap.add_argument("--npz", default=os.path.join(HERE, "figs/feature_logit_data.npz"))
    ap.add_argument("--out", default=os.path.join(
        HERE, "figs/panel_omega_dkd_vs_teacher_enformer_distinct5_hist.png"))
    a = ap.parse_args()
    build(a.npz, a.teacher, a.scheme, a.out)
