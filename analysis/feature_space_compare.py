"""Feature-space comparison: DKD vs OmegaGenome student vs Enformer teacher (splice_sites_all).

Asks whether OmegaGenome's edge is feature preservation (cosine/CKA panel).

For a few hundred splice_sites_all test sequences this script extracts the PENULTIMATE
(pooled) features from:
  - teacher       : Enformer (epoch17_mcc0.8740.pt, hidden_dim=3072), via the repo's
                    load_enformer_model + EnformerTokenizer (the exact distillation recipe).
  - omega student : the OmegaGenome feature-aligned (mse=0.2) best splice student (VKD-style,
                    best val MCC 0.8545) — BPNet original, pooled dim 64.
  - dkd student   : the DKD splice student (best val MCC 0.6952) — BPNet original, pooled dim 64.

Quantitative outputs:
  (a) Linear CKA(student_feats, teacher_feats) for each student. Student dim (64) != teacher
      dim (3072), so a per-sample COSINE-to-teacher is undefined; CKA is the dim-agnostic
      teacher-similarity used instead (reported and labelled as such).
  (b) Per-sample mean COSINE similarity between the two students' own 64-d features
      (DKD vs OmegaGenome), a same-space sanity number.
  (c) For an apples-to-apples teacher cosine we ALSO project the teacher's 3072-d features
      through each student's frozen learned teacher_proj (3072->64, trained during distillation)
      and report mean per-sample cosine(student_feats, proj(teacher_feats)). This is the closest
      thing to "feature preservation in the student's own space".

Figure: a 2D t-SNE of the three pooled feature sets, coloured by model. Teacher (3072-d) and
students (64-d) live in different spaces, so we t-SNE EACH student's 64-d feats together with the
teacher projected into that same 64-d student space (proj(teacher)); two panels (DKD | OmegaGenome)
plus a combined student-vs-student panel. Saved to analysis/figs/.

GPU job, SLURM only. Smoke with --n 8 first.
"""

import argparse
import os
import sys

import numpy as np
import torch

REPO = os.environ.get("OG_ROOT", os.getcwd())
sys.path.insert(0, REPO)

from src.model.bpnet_classifier import BPNetClassifier, BPNetClassifierConfig  # noqa: E402
from src.model.enformer import load_enformer_model, EnformerTokenizer  # noqa: E402
from src.data.dataset import encode_seq  # noqa: E402

# ---- fixed paths (the checkpoints identified on disk; all enformer-teacher, splice_sites_all) ----
TEACHER_CKPT = (
    os.environ.get("OG_STORE", "data") + "/splice_sites_all_checkpoints/epoch17_mcc0.8740.pt"
)
OMEGA_CKPT = (
    os.environ.get("OG_STORE", "data") + "/enformer_distill_to_bpnet_enhanced-different-200epoch/"
    "splice_sites_all/vanilla/lr0.0001_ce0.5_kl0.3_temp4.0_mse0.2/epoch95_valmcc_0.8545_vanilla.pt"
)
DKD_CKPT = (
    os.environ.get("OG_STORE", "data") + "/enformer_distill_to_bpnet_enhanced-different-200epoch/"
    "splice_sites_all/dkd/lr0.0001_ce0.5_kl0.3_alpha1.0_beta8.0_mse0.2/epoch157_valmcc_0.6952_dkd.pt"
)
NUM_LABELS = 3
TEACHER_HIDDEN = 3072
MAX_LEN = 1000


def load_student(ckpt_path, device):
    """Load a BPNet OmegaGenome/DKD student. Remaps the legacy `bpnet.` prefix in the saved
    state_dict to the current `backbone.` attribute name; the unused profile/total_count heads
    are left uninitialised (features come from the backbone trunk only)."""
    cfg = BPNetClassifierConfig(
        num_labels=NUM_LABELS,
        model_size="original",
        teacher_hidden_size=TEACHER_HIDDEN,
        teacher_projection_opt="down",
    )
    model = BPNetClassifier(cfg)
    sd = torch.load(ckpt_path, map_location="cpu")
    sd = {
        (k.replace("bpnet.", "backbone.", 1) if k.startswith("bpnet.") else k): v
        for k, v in sd.items()
    }
    res = model.load_state_dict(sd, strict=False)
    leftover = [k for k in res.missing_keys if "profile" not in k and "total_count" not in k]
    assert not leftover, f"unexpected missing keys: {leftover}"
    assert not res.unexpected_keys, f"unexpected keys: {res.unexpected_keys}"
    return model.to(device).eval()


def linear_cka(X, Y):
    """Linear Centered Kernel Alignment between feature matrices X[N,Dx] and Y[N,Dy].
    Dimension-agnostic similarity in [0,1]; 1 = identical representational geometry."""
    X = X - X.mean(0, keepdims=True)
    Y = Y - Y.mean(0, keepdims=True)
    # HSIC_linear(X,Y) = ||X^T Y||_F^2 ; CKA = HSIC(X,Y)/sqrt(HSIC(X,X) HSIC(Y,Y))
    xty = X.T @ Y
    hsic_xy = (xty * xty).sum()
    xtx = X.T @ X
    yty = Y.T @ Y
    hsic_xx = (xtx * xtx).sum()
    hsic_yy = (yty * yty).sum()
    return float(hsic_xy / (np.sqrt(hsic_xx * hsic_yy) + 1e-12))


def mean_cosine(A, B):
    """Mean per-sample cosine similarity between rows of A[N,D] and B[N,D] (same D)."""
    a = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)
    b = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-12)
    return float((a * b).sum(1).mean())


def load_splice_test_seqs(n, hf_home):
    """Load splice_sites_all test sequences+labels from the NT-revised HF dataset (task filter)."""
    from datasets import load_dataset

    ds = load_dataset(
        "InstaDeepAI/nucleotide_transformer_downstream_tasks_revised",
        split="test",
        trust_remote_code=True,
    )
    ds = ds.filter(lambda ex: ex["task"] == "splice_sites_all")
    seqs = ds["sequence"]
    labs = ds["label"]
    if n and n < len(seqs):
        rng = np.random.RandomState(42)
        idx = rng.choice(len(seqs), size=n, replace=False)
        seqs = [seqs[i] for i in idx]
        labs = [labs[i] for i in idx]
    return list(seqs), list(labs)


@torch.no_grad()
def student_feats(model, seqs, device, bs=32):
    """Pooled penultimate features [N,64] from a BPNet student over A/C/G/T int-encoded seqs."""
    out = []
    for i in range(0, len(seqs), bs):
        ids = torch.tensor(
            [encode_seq(s, MAX_LEN) for s in seqs[i : i + bs]], dtype=torch.long, device=device
        )
        _, feats = model(ids, return_feats=True)
        out.append(feats.float().cpu().numpy())
    return np.concatenate(out, 0)


@torch.no_grad()
def teacher_feats(model, tok, seqs, device, bs=8):
    """Pooled penultimate features [N,3072] from the Enformer teacher (distillation recipe)."""
    out = []
    for i in range(0, len(seqs), bs):
        enc = tok(seqs[i : i + bs], padding="max_length", truncation=True, max_length=MAX_LEN)
        ids = enc["input_ids"].to(device)
        mask = enc["attention_mask"].to(device)
        _, feats = model(input_ids=ids, attention_mask=mask, return_features=True)
        out.append(feats.float().cpu().numpy())
    return np.concatenate(out, 0)


def project_teacher(student_model, T):
    """Project teacher feats T[N,3072] through the student's frozen learned teacher_proj -> [N,64].
    This is the 'down' projection trained during distillation to align teacher->student space."""
    with torch.no_grad():
        w = student_model.teacher_proj.weight.detach().cpu().numpy()  # [64,3072]
        b = student_model.teacher_proj.bias.detach().cpu().numpy()  # [64]
    return T @ w.T + b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300, help="# test seqs (use 8 for smoke)")
    ap.add_argument("--out", default=os.path.join(REPO, "analysis/figs"))
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}", flush=True)

    seqs, labs = load_splice_test_seqs(args.n, os.environ.get("HF_HOME", ""))
    print(f"loaded {len(seqs)} splice_sites_all test seqs", flush=True)

    print("loading students...", flush=True)
    omega = load_student(OMEGA_CKPT, device)
    dkd = load_student(DKD_CKPT, device)

    print("extracting student features...", flush=True)
    F_omega = student_feats(omega, seqs, device)
    F_dkd = student_feats(dkd, seqs, device)

    print("loading enformer teacher (1GB, may stall on NFS)...", flush=True)
    wrapped, _, _ = load_enformer_model(TEACHER_CKPT, NUM_LABELS, device)
    tok = EnformerTokenizer()
    print("extracting teacher features...", flush=True)
    F_teacher = teacher_feats(wrapped, tok, seqs, device)
    print(f"shapes: teacher={F_teacher.shape} omega={F_omega.shape} dkd={F_dkd.shape}", flush=True)

    # ---- quantitative ----
    cka_omega = linear_cka(F_omega, F_teacher)
    cka_dkd = linear_cka(F_dkd, F_teacher)

    Tp_omega = project_teacher(omega, F_teacher)  # teacher in omega's 64-d space
    Tp_dkd = project_teacher(dkd, F_teacher)
    cos_omega_t = mean_cosine(F_omega, Tp_omega)
    cos_dkd_t = mean_cosine(F_dkd, Tp_dkd)
    cos_students = mean_cosine(F_omega, F_dkd)

    summary = (
        "=== feature-space comparison (splice_sites_all) ===\n"
        f"n_seqs = {len(seqs)}\n"
        f"teacher  = Enformer epoch17_mcc0.8740.pt (hidden 3072)\n"
        f"omega    = VKD feature-aligned mse=0.2 epoch95_valmcc_0.8545_vanilla.pt (dim 64)\n"
        f"dkd      = DKD epoch157_valmcc_0.6952_dkd.pt (dim 64)\n"
        "--- teacher similarity (dims differ -> linear CKA) ---\n"
        f"CKA(OmegaGenome, teacher) = {cka_omega:.4f}\n"
        f"CKA(DKD,        teacher) = {cka_dkd:.4f}\n"
        "--- cosine in student space (teacher projected via student's learned teacher_proj) ---\n"
        f"mean cos(OmegaGenome, proj(teacher)) = {cos_omega_t:.4f}\n"
        f"mean cos(DKD,        proj(teacher)) = {cos_dkd_t:.4f}\n"
        "--- student-vs-student (same 64-d space) ---\n"
        f"mean cos(OmegaGenome, DKD) = {cos_students:.4f}\n"
    )
    print(summary, flush=True)
    with open(os.path.join(args.out, "feature_space_metrics.txt"), "w") as f:
        f.write(summary)

    # ---- precompute t-SNE coords (sklearn is in the GPU venv; matplotlib may not be) ----
    from sklearn.manifold import TSNE

    def tsne_of(blocks, perplexity):
        X = np.concatenate(blocks, 0)
        per = min(perplexity, max(5, (X.shape[0] - 1) // 3))
        return TSNE(n_components=2, init="pca", perplexity=per, random_state=0).fit_transform(X)

    Z_dkd = tsne_of([F_dkd, Tp_dkd], 30)
    Z_omega = tsne_of([F_omega, Tp_omega], 30)
    Z_students = tsne_of([F_omega, F_dkd], 30)
    # Persist everything so the PNG can be (re)rendered anywhere, even without matplotlib here.
    npz_path = os.path.join(args.out, "feature_space_data.npz")
    np.savez_compressed(
        npz_path,
        F_omega=F_omega,
        F_dkd=F_dkd,
        F_teacher=F_teacher,
        Tp_omega=Tp_omega,
        Tp_dkd=Tp_dkd,
        Z_dkd=Z_dkd,
        Z_omega=Z_omega,
        Z_students=Z_students,
        cka_omega=cka_omega,
        cka_dkd=cka_dkd,
        cos_omega_t=cos_omega_t,
        cos_dkd_t=cos_dkd_t,
        cos_students=cos_students,
        labels=np.array(labs),
    )
    print(f"SAVED {npz_path}", flush=True)

    # ---- t-SNE figure (skip gracefully if matplotlib unavailable; render from npz instead) ----
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # noqa: F401
    except ModuleNotFoundError:
        print(
            "matplotlib not installed in this env; figure NOT rendered here. "
            "Render from the npz with: python analysis/render_feature_space_fig.py",
            flush=True,
        )
        return

    out_png = os.path.join(args.out, "feature_space_splice_dkd_vs_omega_vs_teacher.png")
    render_figure(
        plt,
        Z_dkd,
        Z_omega,
        Z_students,
        len(F_omega),
        len(F_dkd),
        cka_omega,
        cka_dkd,
        cos_omega_t,
        cos_dkd_t,
        cos_students,
        out_png,
    )
    print(f"SAVED {out_png}", flush=True)


def render_figure(
    plt,
    Z_dkd,
    Z_omega,
    Z_students,
    no,
    nd,
    cka_omega,
    cka_dkd,
    cos_omega_t,
    cos_dkd_t,
    cos_students,
    out_png,
):
    """Render the 3-panel t-SNE figure from precomputed 2D coords + metrics."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    # panel A: DKD student vs teacher (projected into DKD's 64-d space)
    axes[0].scatter(Z_dkd[:nd, 0], Z_dkd[:nd, 1], s=14, c="#d62728", alpha=0.7, label="DKD student")
    axes[0].scatter(
        Z_dkd[nd:, 0], Z_dkd[nd:, 1], s=14, c="#000000", alpha=0.7, label="teacher (proj)"
    )
    axes[0].set_title(f"DKD vs teacher\nCKA={cka_dkd:.3f}  cos={cos_dkd_t:.3f}")
    axes[0].legend(fontsize=9)

    # panel B: OmegaGenome student vs teacher (projected into OmegaGenome's 64-d space)
    axes[1].scatter(
        Z_omega[:no, 0], Z_omega[:no, 1], s=14, c="#2ca02c", alpha=0.7, label="OmegaGenome student"
    )
    axes[1].scatter(
        Z_omega[no:, 0], Z_omega[no:, 1], s=14, c="#000000", alpha=0.7, label="teacher (proj)"
    )
    axes[1].set_title(f"OmegaGenome vs teacher\nCKA={cka_omega:.3f}  cos={cos_omega_t:.3f}")
    axes[1].legend(fontsize=9)

    # panel C: both students together (same 64-d space)
    axes[2].scatter(
        Z_students[:no, 0], Z_students[:no, 1], s=14, c="#2ca02c", alpha=0.7, label="OmegaGenome"
    )
    axes[2].scatter(
        Z_students[no:, 0], Z_students[no:, 1], s=14, c="#d62728", alpha=0.7, label="DKD"
    )
    axes[2].set_title(f"students (64-d)\ncos(O,D)={cos_students:.3f}")
    axes[2].legend(fontsize=9)

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(
        "Feature-space preservation on splice_sites_all (penultimate features)\n"
        "OmegaGenome (feature-aligned mse=0.2) vs DKD vs Enformer teacher",
        fontsize=13,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out_png, dpi=160, bbox_inches="tight")


if __name__ == "__main__":
    main()
