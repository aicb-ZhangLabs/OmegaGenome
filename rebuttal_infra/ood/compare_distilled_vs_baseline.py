#!/usr/bin/env python
"""R2.3: compare distilled-student vs from-scratch-baseline cross-task transfer (does distillation
improve cross-task transfer?). Reads both MCC matrices, restricts to the COMMON task set present in
both (the baseline is missing H3K27ac -> 17 tasks), and reports mean diagonal vs mean off-diagonal
for each, plus the distilled-minus-baseline deltas. Pure CSV math -- no GPU/torch.
"""
import csv
import os

import numpy as np

MULTICLASS = {"enhancers_types", "splice_sites_all"}
HERE = os.path.dirname(os.path.abspath(__file__))


def read_matrix(path):
    """Load a cross_task_matrix CSV -> (tasks, NxN float array with NaN for blank/N-A cells)."""
    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    tasks = rows[0][1:]
    mat = np.full((len(tasks), len(tasks)), np.nan)
    for i, r in enumerate(rows[1:]):
        for j, v in enumerate(r[1:]):
            if v != "":
                mat[i, j] = float(v)
    return tasks, mat


def submatrix(tasks, mat, keep):
    """Reindex (tasks, mat) onto the ordered ``keep`` task list -> aligned NxN array."""
    idx = [tasks.index(t) for t in keep]
    return mat[np.ix_(idx, idx)]


def diag_off(mat, keep, group):
    """(mean diagonal, mean off-diagonal) MCC over the rows/cols of ``keep`` that are in ``group``."""
    pos = [i for i, t in enumerate(keep) if t in group]
    diag = [mat[i, i] for i in pos if not np.isnan(mat[i, i])]
    off = [mat[i, j] for i in pos for j in pos if i != j and not np.isnan(mat[i, j])]
    return (np.mean(diag) if diag else np.nan), (np.mean(off) if off else np.nan)


def main():
    dpath = os.path.join(HERE, "cross_task_matrix_enformer.csv")
    bpath = os.path.join(HERE, "cross_task_matrix_baseline.csv")
    dt, dm = read_matrix(dpath)
    bt, bm = read_matrix(bpath)
    common = [t for t in dt if t in bt]  # preserve distilled order; drop baseline-missing tasks
    dropped = [t for t in dt if t not in bt]
    dms = submatrix(dt, dm, common)
    bms = submatrix(bt, bm, common)

    out = ["========== R2.3 DISTILLED vs FROM-SCRATCH BASELINE cross-task transfer =========="]
    out.append(f"common tasks: {len(common)} (dropped {dropped or 'none'} -- absent in baseline)")
    binary = set(t for t in common if t not in MULTICLASS)
    multi = set(t for t in common if t in MULTICLASS)
    for label, group in [("BINARY", binary), ("MULTICLASS(3-label)", multi)]:
        dd, do = diag_off(dms, common, group)
        bd, bo = diag_off(bms, common, group)
        out.append(f"\n--- {label} ({len(group)} tasks) ---")
        out.append(f"  DISTILLED  diag={dd:+.4f}  off-diag={do:+.4f}  drop={dd - do:+.4f}")
        out.append(f"  BASELINE   diag={bd:+.4f}  off-diag={bo:+.4f}  drop={bd - bo:+.4f}")
        out.append(f"  DELTA (distilled - baseline):  diag={dd - bd:+.4f}  off-diag={do - bo:+.4f}")
        if not np.isnan(do) and not np.isnan(bo):
            verdict = ("distillation IMPROVES off-diagonal transfer"
                       if do > bo else "distillation does NOT improve off-diagonal transfer")
            out.append(f"  -> {verdict} (off-diag {do:+.4f} vs {bo:+.4f})")
    text = "\n".join(out)
    print(text)
    sp = os.path.join(HERE, "distilled_vs_baseline_summary.txt")
    with open(sp, "w") as f:
        f.write(text + "\n")
    print(f"\nSAVED {sp}")


if __name__ == "__main__":
    main()
