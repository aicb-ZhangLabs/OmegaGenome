#!/usr/bin/env python
"""PI-faithful 2x3 panel figure for R1.9 / R1.13 (replaces the prior ad-hoc CKA plot).

Clean, config-driven port of the PI's ACTUAL paper-panel script
`/extra/zhanglab0/INDV/pengchx3/bpnet-pytorch/umap-tsne-260127_panel_v12_omegagenome-title.py`
into our rebuttal infra. REUSES verbatim:

  * setup_nature_style()  -- Nature style: font_scale 2.1, sizes 16.8/18.9/21, #333333 text/edges,
                             white facecolor.
  * ROW1_COLOR_OPTIONS / ROW2_COLORS -- the per-task class-colour schemes (greens/purples/oranges/
                             browns/golds, NO blue/red) selectable via `row1_scheme_key`; Row-2 fixed
                             colours Teacher-Student = blue #4575B4 (ts_color), Teacher-BPNet/baseline
                             = grey #7F7F7F (tb_color).
  * create_combined_figure() 2x3 layout -- Row 1 = three t-SNE scatters (Teacher | Student | BPNet
                             from scratch) coloured by CLASS with silhouette boxes; Row 2 = logit
                             scatter T-S (D), logit scatter T-B (E), sample-wise cosine-similarity
                             distribution (F, KDE+hist) with PCA-align to common_dim and 1-cosine().
  * panel labels A-F, MaxNLocator(5), y-tick rotation 90, PNG+PDF at 300 dpi.

The 2nd row's "logit-vs-feature" distinction from the PI script is preserved: logit scatter uses
LOGITS (class 0); the cosine distribution uses the FEATURE (second_to_last) embeddings PCA-aligned.

Config-driven: `--teacher {enformer,nt,dnabert2,caduceus}`, `--student {omega,dkd}` (so we get the
panel for the OmegaGenome-best student AND for the DKD-best student vs the same teacher + from-scratch
BPNet baseline), `--scheme` to pick the Row-1 colour scheme.

No GPU here -- consumes the cached `feature_logit_data.npz` (teacher/omega/dkd/baseline feats+logits)
produced by `extract_feats_logits.py` (SLURM). Runs in the enformer conda env (matplotlib+seaborn).
Also emits a cosine-similarity TABLE (mean+-std/median) for teacher-vs-student & teacher-vs-baseline,
for DKD and OmegaGenome, into a CSV for the rebuttal docx.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402
from sklearn.manifold import TSNE  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.metrics import silhouette_score  # noqa: E402
from scipy.spatial.distance import cosine  # noqa: E402
from scipy.stats import pearsonr  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ==================== COLOR SCHEMES (verbatim from PI v12_panel) ====================
ROW2_COLORS = {
    "ts_color": "#4575B4",   # Steel blue for Teacher-Student
    "tb_color": "#7F7F7F",   # Grey for Teacher-BPNet (baseline)
    "accent": "#808080",     # Gray for identity line
    "text_box_color": "#F5F5F5",
}
ROW1_COLOR_OPTIONS = {
    "distinct1": {"name": "Forest & Gold", "class_colors": ["#2D5016", "#8B7355", "#DAA520", "#6B4C7A", "#4A7C59"]},
    "distinct5": {"name": "Sunset Meadow", "class_colors": ["#E07020", "#5C8A4D", "#E5A832", "#9370DB", "#6B8E4E"]},
    "distinct7": {"name": "Vineyard", "class_colors": ["#722F72", "#808000", "#D4A656", "#4E3D4E", "#3D8B3D"]},
    "distinct10": {"name": "Japanese Garden", "class_colors": ["#7BA05B", "#696969", "#FFA500", "#8B008B", "#2E5D3D"]},
    "distinct12": {"name": "Mountain Meadow", "class_colors": ["#ED9121", "#355E3B", "#808069", "#7851A9", "#4D7D4D"]},
}


def setup_nature_style():
    """Nature journal style (verbatim from the PI v12_panel script): font_scale 2.1, 40%-larger
    fonts, #333333 text/edges, white facecolor, all spines, vertical-friendly ticks."""
    sns.set_style("white")
    sns.set_context("paper", font_scale=2.1)
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 16.8, "axes.labelsize": 18.9, "axes.titlesize": 21,
        "xtick.labelsize": 16.8, "ytick.labelsize": 16.8, "legend.fontsize": 14.7,
        "axes.linewidth": 1.0, "axes.edgecolor": "#333333", "axes.labelcolor": "#333333",
        "xtick.color": "#333333", "ytick.color": "#333333", "text.color": "#333333",
        "axes.spines.top": True, "axes.spines.right": True, "axes.spines.bottom": True,
        "axes.spines.left": True, "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.facecolor": "white", "savefig.edgecolor": "white",
        "font.weight": "normal", "axes.titleweight": "normal", "axes.labelweight": "normal",
        "ytick.left": True, "ytick.right": False, "xtick.bottom": True, "xtick.top": False,
    })


# ==================== CONFIG: checkpoints/data per teacher ====================
# Only the data source matters here (the GPU extract already produced the npz); kept for provenance.
TEACHER_NAME = {"enformer": "Enformer Teacher", "nt": "NT Teacher",
                "dnabert2": "DNABERT-2 Teacher", "caduceus": "Caduceus Teacher",
                "carbon": "Carbon-3B Teacher"}
STUDENT_TITLE = {"omega": "OmegaGenome", "dkd": "DKD"}


def create_combined_figure(teacher_emb, student_emb, bpnet_emb, labels,
                           teacher_logits, student_logits, bpnet_logits,
                           task_name, teacher_label, student_title, save_path,
                           row1_scheme_key, use_histogram=False):
    """Verbatim-structure port of the PI v12_panel `create_combined_figure`: 2x3 Nature panel.
    Row 1 = t-SNE(class-coloured) for Teacher|Student|BPNet + silhouette; Row 2 (matching main-text
    Fig. 5) = sample-wise cosine-similarity distribution (D), logit scatter T-S (E), logit scatter T-B (F)."""
    row1_scheme = ROW1_COLOR_OPTIONS[row1_scheme_key]
    setup_nature_style()
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    n_classes = len(np.unique(labels))
    class_colors = row1_scheme["class_colors"][:n_classes]
    ts_color, tb_color = ROW2_COLORS["ts_color"], ROW2_COLORS["tb_color"]
    accent_color, text_box_color = ROW2_COLORS["accent"], ROW2_COLORS["text_box_color"]

    teacher_scaled = StandardScaler().fit_transform(teacher_emb)
    student_scaled = StandardScaler().fit_transform(student_emb)
    bpnet_scaled = StandardScaler().fit_transform(bpnet_emb)

    # ----- ROW 1: three t-SNE plots -----
    teacher_tsne = TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(teacher_scaled)
    student_tsne = TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(student_scaled)
    bpnet_tsne = TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(bpnet_scaled)
    sil_t = silhouette_score(teacher_tsne, labels)
    sil_s = silhouette_score(student_tsne, labels)
    sil_b = silhouette_score(bpnet_tsne, labels)

    legend_fs, textbox_fs, corr_fs = 12.6, 15.4, 15.4

    def _tsne_panel(ax, emb2d, title, sil):
        for i, lab in enumerate(np.unique(labels)):
            m = labels == lab
            ax.scatter(emb2d[m, 0], emb2d[m, 1], c=class_colors[i], s=25, alpha=0.7,
                       label=f"Class {lab}", edgecolors="white", linewidth=0.3)
        ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2", labelpad=2)
        ax.set_title(title, fontweight="normal")
        ax.legend(fontsize=legend_fs, frameon=True, fancybox=False, edgecolor="#CCCCCC", loc="upper right")
        ax.text(0.05, 0.95, f"Silhouette = {sil:.3f}", transform=ax.transAxes, fontsize=textbox_fs,
                va="top", ha="left",
                bbox=dict(boxstyle="round,pad=0.3", facecolor=text_box_color, edgecolor="#CCCCCC", alpha=0.9))
        ax.yaxis.set_major_locator(plt.MaxNLocator(5))
        ax.tick_params(axis="y", which="both", left=True, labelleft=True, rotation=90)

    _tsne_panel(axes[0, 0], teacher_tsne, teacher_label, sil_t)
    _tsne_panel(axes[0, 1], student_tsne, student_title, sil_s)
    _tsne_panel(axes[0, 2], bpnet_tsne, "BPNet baseline", sil_b)

    # ----- ROW 2: logit scatters (D,E) + cosine-similarity distribution (F) -----
    ts_corr, _ = pearsonr(teacher_logits[:, 0], student_logits[:, 0])
    tb_corr, _ = pearsonr(teacher_logits[:, 0], bpnet_logits[:, 0])

    def _logit_panel(ax, x, y, color, ylab, title, r):
        ax.scatter(x, y, c=color, alpha=0.4, s=15, edgecolors="none")
        lims = [min(x.min(), y.min()), max(x.max(), y.max())]
        ax.plot(lims, lims, "--", color=accent_color, alpha=0.7, linewidth=1.5, label="y=x")
        ax.set_xlabel("Teacher Logit (Class 0)"); ax.set_ylabel(ylab, labelpad=2)
        ax.set_title(title, fontweight="normal")
        ax.legend(fontsize=legend_fs, frameon=True, fancybox=False, edgecolor="#CCCCCC")
        ax.text(0.95, 0.05, f"r = {r:.3f}", transform=ax.transAxes, fontsize=corr_fs,
                va="bottom", ha="right",
                bbox=dict(boxstyle="round,pad=0.3", facecolor=text_box_color, edgecolor="#CCCCCC", alpha=0.9))
        ax.yaxis.set_major_locator(plt.MaxNLocator(5))
        ax.tick_params(axis="y", which="both", left=True, labelleft=True, rotation=90)

    # Row 2 order (matches the main-text Fig. 5): D = cosine similarity, E = logit scatter T-S,
    # F = logit scatter T-B. Cosine occupies the first cell of the second row.
    _logit_panel(axes[1, 1], teacher_logits[:, 0], student_logits[:, 0], ts_color,
                 "Student Logit (Class 0)", "Logit Scatter (T-S)", ts_corr)
    _logit_panel(axes[1, 2], teacher_logits[:, 0], bpnet_logits[:, 0], tb_color,
                 "BPNet Logit (Class 0)", "Logit Scatter (T-B)", tb_corr)

    # D: sample-wise cosine similarity, PCA-align all to common_dim, 1 - cosine() per sample
    ax = axes[1, 0]
    common_dim = min(50, teacher_scaled.shape[1], student_scaled.shape[1], bpnet_scaled.shape[1])
    t_al = PCA(n_components=common_dim, random_state=42).fit_transform(teacher_scaled)
    s_al = PCA(n_components=common_dim, random_state=42).fit_transform(student_scaled)
    b_al = PCA(n_components=common_dim, random_state=42).fit_transform(bpnet_scaled)
    ts_sim = np.array([1 - cosine(t_al[i], s_al[i]) for i in range(len(t_al))])
    tb_sim = np.array([1 - cosine(t_al[i], b_al[i]) for i in range(len(t_al))])

    if use_histogram:
        all_s = np.concatenate([ts_sim, tb_sim])
        rng = (all_s.min() - 0.02, all_s.max() + 0.02)
        ax.hist(ts_sim, bins=80, range=rng, alpha=0.5, color=ts_color, edgecolor="white", linewidth=0.3,
                label=f"Teacher-Student\n(mean={ts_sim.mean():.3f})")
        ax.hist(tb_sim, bins=80, range=rng, alpha=0.5, color=tb_color, edgecolor="white", linewidth=0.3,
                label=f"Teacher-BPNet\n(mean={tb_sim.mean():.3f})")
        ax.set_ylabel("Count", labelpad=2)
    else:
        sns.kdeplot(ts_sim, ax=ax, color=ts_color, fill=True, alpha=0.3, linewidth=2,
                    label=f"Teacher-Student\n(mean={ts_sim.mean():.3f})")
        sns.kdeplot(tb_sim, ax=ax, color=tb_color, fill=True, alpha=0.3, linewidth=2,
                    label=f"Teacher-BPNet\n(mean={tb_sim.mean():.3f})")
        ax.set_ylabel("Density", labelpad=2)
    ax.axvline(ts_sim.mean(), color=ts_color, linestyle="--", linewidth=1.5, alpha=0.8)
    ax.axvline(tb_sim.mean(), color=tb_color, linestyle="--", linewidth=1.5, alpha=0.8)
    ax.set_xlabel("Cosine Similarity"); ax.set_title("Sample-wise Cosine Similarity", fontweight="normal")
    ax.legend(fontsize=legend_fs, frameon=True, fancybox=False, edgecolor="#CCCCCC")
    ax.yaxis.set_major_locator(plt.MaxNLocator(5))
    ax.tick_params(axis="y", which="both", left=True, labelleft=True, rotation=90)

    # panel labels A-F
    for ax, lab in zip(axes.flat, ["A", "B", "C", "D", "E", "F"]):
        ax.text(-0.12, 1.08, lab, transform=ax.transAxes, fontsize=22.4, fontweight="normal", va="top", ha="left")

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.25, wspace=0.22)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.savefig(save_path.replace(".png", ".pdf"), dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  SAVED {save_path}")
    # return the per-sample cosine arrays + silhouettes for the table
    return {"ts_sim": ts_sim, "tb_sim": tb_sim, "sil_teacher": sil_t,
            "sil_student": sil_s, "sil_bpnet": sil_b, "ts_corr": ts_corr, "tb_corr": tb_corr}


def cosine_row(name, sims):
    return {"comparison": name, "cosine_mean": float(sims.mean()), "cosine_std": float(sims.std()),
            "cosine_median": float(np.median(sims))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", default="enformer", choices=list(TEACHER_NAME))
    ap.add_argument("--scheme", default="distinct10", choices=list(ROW1_COLOR_OPTIONS),
                    help="Row-1 class-colour scheme (PI preferred = distinct10 Japanese Garden)")
    ap.add_argument("--npz", default=os.path.join(REPO, "rebuttal_infra/figs/feature_logit_data.npz"))
    ap.add_argument("--out", default=os.path.join(REPO, "rebuttal_infra/figs"))
    ap.add_argument("--layer", default="secondlast", choices=["secondlast", "last"],
                    help="Which teacher hidden layer to use for the t-SNE/cosine: 'secondlast' "
                         "(hidden_states[-2], paper-method, F_teacher) or 'last' (hidden_states[-1], "
                         "the distillation-MSE-aligned layer, F_teacher_last). 'last' suffixes the "
                         "output filenames + cosine table with '_last' so both panels coexist.")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    d = np.load(args.npz)
    labels = d["labels"]
    # Teacher feature layer selector: 'last' uses F_teacher_last (hidden_states[-1], MSE-aligned); the
    # default 'secondlast' uses F_teacher (hidden_states[-2], paper-method). Output names get a suffix.
    if args.layer == "last":
        if "F_teacher_last" not in d.files:
            raise KeyError(f"--layer last requested but '{args.npz}' has no F_teacher_last "
                           f"(re-extract with the both-layer extract_feats_logits_multi).")
        F_teacher = d["F_teacher_last"]
        lay_suffix = "_last"
    else:
        F_teacher = d["F_teacher"]
        lay_suffix = ""
    F_omega, F_dkd, F_base = d["F_omega"], d["F_dkd"], d["F_baseline"]
    L_teacher, L_omega, L_dkd, L_base = d["L_teacher"], d["L_omega"], d["L_dkd"], d["L_base"]
    tname = TEACHER_NAME[args.teacher]
    task = "splice_sites_all"

    table_rows = []
    arrays = {}
    # one 2x3 panel per student (OmegaGenome, DKD); BPNet = from-scratch baseline in both.
    for skey, Sf, Sl in [("omega", F_omega, L_omega), ("dkd", F_dkd, L_dkd)]:
        for hist in (False, True):
            suffix = "hist" if hist else "kde"
            save_path = os.path.join(
                args.out, f"panel_{skey}_vs_teacher_{args.teacher}{lay_suffix}_{args.scheme}_{suffix}.png")
            res = create_combined_figure(
                F_teacher, Sf, F_base, labels, L_teacher, Sl, L_base,
                task, tname, STUDENT_TITLE[skey], save_path, args.scheme, use_histogram=hist)
        # record once per student (kde and hist share the same sims)
        sname = "OmegaGenome" if skey == "omega" else "DKD"
        # Two-sided independent-samples t-test, Teacher-Student vs Teacher-baseline cosine (PI v14
        # ttest script): tests whether the distilled student aligns to the teacher significantly more
        # than the from-scratch BPNet does.
        from scipy.stats import ttest_ind  # noqa: E402
        _t, p_ts_tb = ttest_ind(res["ts_sim"], res["tb_sim"], equal_var=False)
        table_rows.append({**cosine_row(f"teacher-vs-{sname} (student)", res["ts_sim"]),
                           "silhouette": res["sil_student"], "logit_r_class0": res["ts_corr"],
                           "p_value_ts_vs_tb": float(p_ts_tb),
                           "note": f"{sname} distilled student"})
        table_rows.append({**cosine_row(f"teacher-vs-baseline ({sname}-panel)", res["tb_sim"]),
                           "silhouette": res["sil_bpnet"], "logit_r_class0": res["tb_corr"],
                           "p_value_ts_vs_tb": float(p_ts_tb),
                           "note": "from-scratch BPNet (no distillation)"})
        arrays[f"ts_{skey}"] = res["ts_sim"]
        arrays[f"tb_{skey}"] = res["tb_sim"]
        print(f"  [{sname}] silhouette teacher={res['sil_teacher']:.3f} student={res['sil_student']:.3f} "
              f"bpnet={res['sil_bpnet']:.3f} | logit r T-S={res['ts_corr']:.3f} T-B={res['tb_corr']:.3f}")
    # teacher silhouette (constant across panels) as a reference row
    table_rows.insert(0, {"comparison": "teacher (reference)", "cosine_mean": np.nan,
                          "cosine_std": np.nan, "cosine_median": np.nan,
                          "silhouette": res["sil_teacher"], "logit_r_class0": np.nan,
                          "note": f"{tname} t-SNE silhouette"})

    table = pd.DataFrame(table_rows)
    csv = os.path.join(args.out, f"cosine_table_panel_{args.teacher}{lay_suffix}.csv")
    table.to_csv(csv, index=False)
    print("\n=== COSINE TABLE (PCA-aligned sample-wise, PI panel F) ===")
    print(table.to_string(index=False))
    print(f"SAVED {csv}")

    np.savez_compressed(os.path.join(args.out, f"cosine_arrays_panel_{args.teacher}{lay_suffix}.npz"),
                        labels=labels, **arrays)
    print("DONE")


if __name__ == "__main__":
    main()
