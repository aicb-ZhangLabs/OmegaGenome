#!/usr/bin/env python
"""R2.3 OOD / cross-task transfer matrix (INFERENCE ONLY -- no training).

Reviewer R2.3 asks: "each distilled student is a task-specific expert -- how bad is cross-task
transfer?" This script answers it by building the full A x B matrix where each cell (A, B) is the
MCC (and accuracy) of *student trained on task A* evaluated on *task B's TEST set*.

For each student A and eval task B (with matching head size, see below):
  1. load student A's best-HP BPNet checkpoint (original-size, one-hot input, 0.12M),
  2. run its forward on task B's HF test sequences (same loader the distillation used:
     ``build_data_splits_from_huggingface``),
  3. argmax the student-A classifier logits, score MCC + accuracy against task B's true labels.

LABEL-SPACE COMPATIBILITY (critical, no silent fudging):
  A student's classifier head has ``num_labels`` fixed to its training task. We ONLY evaluate
  (A, B) pairs where ``get_num_labels(A) == get_num_labels(B)``. Binary tasks (2-label) form one
  sub-matrix; the 3-label multiclass tasks (``enhancers_types``, ``splice_sites_all``) form a
  separate sub-matrix among themselves. Incompatible (A, B) cells are recorded as N/A (NaN) -- we
  never force a head onto a mismatched label space.

DIAGONAL SANITY: cell (A, A) must reproduce student A's own best test MCC (from the manifest CSV
``best_hp_<teacher>.csv``'s ``best_test_mcc``) within ``--diag-tol`` (default 0.02). A mismatch means
the student or the data loading is wrong, so we FLAG it loudly (and optionally hard-fail).

Outputs (written under rebuttal_infra/ood/):
  * cross_task_matrix_<teacher>.csv      -- rows = student task, cols = eval task, values = MCC.
  * cross_task_matrix_<teacher>_acc.csv  -- same shape, accuracy (companion).
  * cross_task_matrix_<teacher>_diag.csv -- per-task diagonal MCC vs manifest best_test_mcc + delta.
The heatmap + summary are produced by a separate (matplotlib) step from these CSVs.

SLURM only (GPU). Checkpoints load with map_location="cpu" then move to GPU. Smoke with --n 64.
"""
import argparse
import csv
import os
import sys

import numpy as np
import torch
from sklearn.metrics import matthews_corrcoef, accuracy_score

REPO = "/home/pengchx3/text-dna/OmegaGenome_Revise_202606/code_carbon"
sys.path.insert(0, REPO)

from src.data.dataset import (  # noqa: E402
    DatasetConfig, build_data_splits_from_huggingface, get_num_labels, encode_seq,
)
from src.model.bpnet_classifier import BPNetClassifier, BPNetClassifierConfig  # noqa: E402

# The 18 NT-revised downstream tasks (== the distilled_students/<teacher>/ subdirs and the manifest).
TASKS = [
    "H2AFZ", "H3K27ac", "H3K27me3", "H3K36me3", "H3K4me1", "H3K4me2", "H3K4me3",
    "H3K9ac", "H3K9me3", "H4K20me1",
    "promoter_all", "promoter_no_tata", "promoter_tata",
    "enhancers", "enhancers_types",
    "splice_sites_all", "splice_sites_acceptors", "splice_sites_donors",
]
# Enformer trunk hidden dim -> the student's teacher_proj in_features (vanilla MSE feature align).
# Students distilled with weight_mse==0 were saved WITHOUT a teacher_proj; load_student handles both.
TEACHER_HIDDEN = {"enformer": 3072, "caduceus": 512}

# From-scratch (no-distillation) BPNet baseline checkpoints: per-task ``model_final.pt`` under this
# tree (one-hot input, original-size BPNet, trained WITHOUT any teacher -> no teacher_proj). Used for
# the R2.3 baseline cross-task matrix that asks: does distillation improve cross-task TRANSFER vs a
# from-scratch expert? NOTE: H3K27ac's baseline is absent (17/18) -> it is skipped from both axes.
FROM_SCRATCH_ROOT = (
    "/extra/zhanglab0/INDV/pengchx3/NT/nucleotide-transformer-student-results-3-11-v4-"
    "cnn-layer16-kernel-5-bpnet-4-21-onehot-laniakea"
)
# MUST match the distillation training/eval max_len EXACTLY. The training path that produced these
# students is src/trainer/distill_trainer.py, whose DistillConfig.max_len default is 1024 (NOT the 1000
# used by some older standalone entrypoints). Every SeqDataset pads/truncates to it, so the student's
# AdaptiveAvgPool denominator (the pooled feature) is defined over a length-1024 zero-padded input. The
# 18 tasks have native lengths 300 (promoters) / 400 (enhancers) / 600 (splice) / 1000 (histone marks);
# encoding at 1024 reproduces the training-time padding for ALL of them. VERIFIED: at MAX_LEN=1024 every
# diagonal cell reproduces the manifest best_test_mcc to <=1e-4 (splice_donors 0.9007, splice_all 0.8400,
# splice_acceptors 0.8775, etc.); MAX_LEN=600 underran by up to 0.19 and 1000 by up to 0.09 (the
# diagonal sanity check correctly flagged both).
MAX_LEN = 1024


def load_student(ckpt_path, num_labels, teacher_hidden, device):
    """Load a best-HP distilled BPNet student (original-size, one-hot input) for the given num_labels.

    teacher_proj is a TRAINING-ONLY head (the deep feature-MSE alignment projection) and is never
    touched at inference -- only backbone+pool+classifier run. But its ``in_features`` varies per
    student: Enformer-trunk-feature alignment saved [64, 3072], while configs that aligned on a
    smaller "feature" (e.g. logit/class-dim) saved [64, num_labels], and weight_mse==0 students saved
    NO teacher_proj at all. So we INTROSPECT the saved ``teacher_proj.weight`` shape from the
    state_dict and size the config's teacher_hidden_size to match, guaranteeing a clean load for any
    variant (``teacher_hidden`` is only the fallback when no teacher_proj key is present). Legacy
    ``bpnet.`` keys remap to ``backbone.``. Returns an eval()-mode model on ``device``."""
    sd = torch.load(ckpt_path, map_location="cpu")
    if isinstance(sd, dict) and "model_state_dict" in sd:
        sd = sd["model_state_dict"]
    elif isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    sd = {(k.replace("bpnet.", "backbone.", 1) if k.startswith("bpnet.") else k): v
          for k, v in sd.items()}
    # Introspect the saved teacher_proj in_features (down-projection: weight is [C, teacher_hidden]).
    tp = sd.get("teacher_proj.weight")
    th = int(tp.shape[1]) if tp is not None else teacher_hidden
    cfg = BPNetClassifierConfig(
        num_labels=num_labels, model_size="original",
        teacher_hidden_size=th, teacher_projection_opt="down",
    )
    model = BPNetClassifier(cfg)
    res = model.load_state_dict(sd, strict=False)
    # teacher_proj/profile/total_count are training-only heads -> allowed to be missing.
    leftover = [k for k in res.missing_keys
                if "profile" not in k and "total_count" not in k and "teacher_proj" not in k]
    assert not leftover, f"unexpected missing keys loading {ckpt_path}: {leftover}"
    assert not res.unexpected_keys, f"unexpected keys loading {ckpt_path}: {res.unexpected_keys}"
    return model.to(device).eval()


@torch.no_grad()
def predict(model, seqs, device, bs=256):
    """Argmax class predictions [N] from a BPNet student over a list of sequences (one-hot path)."""
    preds = []
    for i in range(0, len(seqs), bs):
        ids = torch.tensor([encode_seq(s, MAX_LEN) for s in seqs[i:i + bs]],
                           dtype=torch.long, device=device)
        logits = model(ids)
        preds.append(logits.argmax(-1).cpu().numpy())
    return np.concatenate(preds, 0)


def load_test_sets(tasks, data_path, n):
    """Load each task's HF TEST split once: {task: (seqs, labels)}.

    Uses the exact loader the distillation used (build_data_splits_from_huggingface). ``n`` caps the
    test set for smoke runs (n<=0 => full)."""
    out = {}
    for t in tasks:
        cfg = DatasetConfig(task_name=t, data_path=data_path)
        _, _, _, _, X_test, y_test = build_data_splits_from_huggingface(cfg)
        X_test, y_test = list(X_test), list(y_test)
        if n and n > 0:
            X_test, y_test = X_test[:n], y_test[:n]
        out[t] = (X_test, np.asarray(y_test))
        print(f"  [{t}] test={len(X_test)} num_labels={get_num_labels(t)} "
              f"label_set={sorted(set(out[t][1].tolist()))}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", default="enformer", choices=["enformer", "caduceus"])
    ap.add_argument("--students-mode", default="distilled",
                    choices=["distilled", "baseline", "baseline-grid"],
                    help="'distilled': per-teacher distilled students (default). 'baseline': the "
                         "from-scratch (no-teacher) BPNet model_final.pt baselines. 'baseline-grid': "
                         "the PURE-CE (weight_kl0.0_weight_mse0.0) students from the HP grid, read from "
                         "--baseline-manifest (task->student.pt). Both baseline modes tag output "
                         "'baseline' and self-report the diagonal (no best_test_mcc reference).")
    ap.add_argument("--baseline-manifest", default=None,
                    help="CSV with columns task,student_path for --students-mode baseline-grid (the "
                         "best original-size pure-CE student per task selected from the HP grid).")
    ap.add_argument("--students-root",
                    default=os.path.join(REPO, "data/distilled_students"))
    ap.add_argument("--manifest", default=None,
                    help="best_hp_<teacher>.csv with best_test_mcc (default: best_hp_all_teachers/...). "
                         "Ignored in --students-mode baseline (no manifest ref for from-scratch).")
    ap.add_argument("--n", type=int, default=0, help="cap test seqs per task (0=full; smoke e.g. 64).")
    ap.add_argument("--diag-tol", type=float, default=0.02)
    ap.add_argument("--strict-diag", action="store_true",
                    help="hard-fail if any diagonal cell deviates from manifest by > diag-tol.")
    ap.add_argument("--out", default=os.path.join(REPO, "rebuttal_infra/ood"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    teacher_hidden = TEACHER_HIDDEN[args.teacher]
    print(f"device={device} teacher={args.teacher} teacher_hidden={teacher_hidden} n={args.n}", flush=True)

    data_path = os.environ.get("HF_DATASETS_CACHE", "")  # DatasetConfig only uses dataset_name for HF

    # Output tag: distilled runs are tagged by teacher; both from-scratch baseline sources are tagged
    # 'baseline' (they answer the same R2.3 question: distillation vs from-scratch transfer).
    is_baseline = args.students_mode in ("baseline", "baseline-grid")
    out_tag = "baseline" if is_baseline else args.teacher

    # Manifest best_test_mcc per task (diagonal reference). The from-scratch baselines have no
    # best_test_mcc manifest, so in baseline mode the reference is empty -> the diagonal is
    # self-reported and never flagged as a "mismatch" (no ground-truth value to compare against).
    best_test = {}
    grid_paths = {}
    if args.students_mode == "distilled":
        manifest = args.manifest or os.path.join(
            REPO, "best_hp_all_teachers/best_hp", f"best_hp_{args.teacher}.csv")
        with open(manifest, newline="") as f:
            for row in csv.DictReader(f):
                best_test[row["task"]] = float(row["best_test_mcc"])
        print(f"manifest={manifest} ({len(best_test)} tasks)", flush=True)
    elif args.students_mode == "baseline-grid":
        if not args.baseline_manifest:
            sys.exit("--students-mode baseline-grid requires --baseline-manifest task,student_path CSV")
        with open(args.baseline_manifest, newline="") as f:
            for row in csv.DictReader(f):
                if row.get("student_path"):
                    grid_paths[row["task"]] = row["student_path"]
        print(f"baseline-grid: {len(grid_paths)} pure-CE student paths from {args.baseline_manifest} "
              "(diagonal self-reported)", flush=True)
    else:
        print("baseline mode: no manifest reference (diagonal self-reported)", flush=True)

    # Resolve student ckpts; only keep tasks whose checkpoint exists.
    students = {}
    for t in TASKS:
        if args.students_mode == "baseline":
            p = os.path.join(FROM_SCRATCH_ROOT, t, "model_final.pt")
        elif args.students_mode == "baseline-grid":
            p = grid_paths.get(t, "")
        else:
            p = os.path.join(args.students_root, args.teacher, t, "student_bestHP.pt")
        if p and os.path.exists(p):
            students[t] = p
        else:
            print(f"  WARNING: missing student for {t}: {p}", flush=True)
    print(f"found {len(students)}/{len(TASKS)} students (mode={args.students_mode})", flush=True)

    print("loading test sets...", flush=True)
    test_sets = load_test_sets(list(students.keys()), data_path, args.n)

    tasks = list(students.keys())
    mcc_mat = {a: {b: np.nan for b in tasks} for a in tasks}
    acc_mat = {a: {b: np.nan for b in tasks} for a in tasks}
    diag_rows = []

    for a in tasks:
        nla = get_num_labels(a)
        print(f"\n=== student A={a} (num_labels={nla}) ===", flush=True)
        model = load_student(students[a], nla, teacher_hidden, device)
        for b in tasks:
            if get_num_labels(b) != nla:
                continue  # incompatible head -> N/A (left as NaN)
            Xb, yb = test_sets[b]
            preds = predict(model, Xb, device)
            mcc = float(matthews_corrcoef(yb, preds))
            acc = float(accuracy_score(yb, preds))
            mcc_mat[a][b] = mcc
            acc_mat[a][b] = acc
            tag = " (DIAG)" if a == b else ""
            print(f"   eval B={b:24s} mcc={mcc:+.4f} acc={acc:.4f}{tag}", flush=True)
        # diagonal sanity. With no manifest reference (baseline mode) there is nothing to mismatch
        # against -> mark OK and report the self-MCC.
        d = mcc_mat[a][a]
        ref = best_test.get(a, np.nan)
        has_ref = not np.isnan(ref)
        delta = (d - ref) if has_ref else np.nan
        ok = (abs(delta) <= args.diag_tol) if has_ref else True
        diag_rows.append((a, d, ref, delta, ok))
        if has_ref:
            flag = "OK" if ok else "*** MISMATCH ***"
            print(f"   DIAGONAL {a}: matrix={d:+.4f} manifest={ref:+.4f} delta={delta:+.4f} [{flag}]",
                  flush=True)
        else:
            print(f"   DIAGONAL {a}: matrix={d:+.4f} (self-reported, no manifest ref)", flush=True)
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    # --- write matrices --------------------------------------------------------------------------
    def write_matrix(path, mat):
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["student_task"] + tasks)
            for a in tasks:
                w.writerow([a] + ["" if np.isnan(mat[a][b]) else f"{mat[a][b]:.4f}" for b in tasks])
        print(f"SAVED {path}", flush=True)

    mcc_path = os.path.join(args.out, f"cross_task_matrix_{out_tag}.csv")
    acc_path = os.path.join(args.out, f"cross_task_matrix_{out_tag}_acc.csv")
    diag_path = os.path.join(args.out, f"cross_task_matrix_{out_tag}_diag.csv")
    write_matrix(mcc_path, mcc_mat)
    write_matrix(acc_path, acc_mat)
    with open(diag_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["task", "matrix_diag_mcc", "manifest_best_test_mcc", "delta", "within_tol"])
        for a, d, ref, delta, ok in diag_rows:
            refs = "" if np.isnan(ref) else f"{ref:.4f}"
            ds = "" if np.isnan(delta) else f"{delta:+.4f}"
            w.writerow([a, f"{d:.4f}", refs, ds, int(ok)])
    print(f"SAVED {diag_path}", flush=True)

    # --- summary ---------------------------------------------------------------------------------
    bad = [r for r in diag_rows if not r[4]]
    print("\n========== DIAGONAL SANITY ==========", flush=True)
    print(f"diag within tol ({args.diag_tol}): {len(diag_rows) - len(bad)}/{len(diag_rows)}", flush=True)
    for a, d, ref, delta, ok in bad:
        ds = "n/a" if np.isnan(delta) else f"{delta:+.4f}"
        rs = "n/a" if np.isnan(ref) else f"{ref:+.4f}"
        print(f"  MISMATCH {a}: matrix={d:+.4f} manifest={rs} delta={ds}", flush=True)

    def offdiag_summary(label, sub):
        diag = [mcc_mat[a][a] for a in sub if not np.isnan(mcc_mat[a][a])]
        off = [mcc_mat[a][b] for a in sub for b in sub
               if a != b and not np.isnan(mcc_mat[a][b])]
        if diag and off:
            print(f"\n--- {label} ({len(sub)} tasks) ---", flush=True)
            print(f"  mean DIAGONAL MCC     = {np.mean(diag):+.4f}", flush=True)
            print(f"  mean OFF-DIAGONAL MCC = {np.mean(off):+.4f}", flush=True)
            print(f"  drop (diag - off)     = {np.mean(diag) - np.mean(off):+.4f}", flush=True)
        return diag, off

    binary = [t for t in tasks if get_num_labels(t) == 2]
    multi = [t for t in tasks if get_num_labels(t) == 3]
    offdiag_summary("BINARY sub-matrix", binary)
    offdiag_summary("MULTICLASS (3-label) sub-matrix", multi)

    if args.strict_diag and bad:
        sys.exit(f"STRICT: {len(bad)} diagonal mismatch(es) > {args.diag_tol}")
    print("\nDONE", flush=True)


if __name__ == "__main__":
    main()
