#!/usr/bin/env python3
"""Fast HP-grid progress counter. Counts unique completed (task, weight_kl, weight_mse, temperature)
RAW combos by parsing the run-dir PATH (which encodes the hyperparameters) instead of opening every
final_summary.json — the per-file open over the contended SSD is what makes gen_hp_specs time out.
A run dir looks like:
  .../deploy_120k/<TASK>/<ts>/<uuid>/distill_..._mse_normalizeFalse_temperature0.5_..._weight_kl0.5_weight_mse1.0_.../final_summary.json
so (task, kl, mse, T, variant) come straight from the string — no I/O beyond the directory walk.
Usage: python slurm/grid_progress.py [--base <dir>] [--total 1440]
"""
import argparse, glob, os, re

DEF_BASE = "/tmp/galaxy_srv_disk00/pengchx3/carbon_distillation"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=DEF_BASE)
    ap.add_argument("--total", type=int, default=1440)
    args = ap.parse_args()

    raw, l2 = set(), set()
    n = 0
    for f in glob.iglob(os.path.join(args.base, "**", "final_summary.json"), recursive=True):
        d = os.path.basename(os.path.dirname(f))            # the hyperparam-string dir
        m_task = re.search(r"deploy_120k/([^/]+)/", f)
        m_kl = re.search(r"weight_kl([0-9.]+)", d)
        m_mse = re.search(r"weight_mse([0-9.]+)", d)
        m_T = re.search(r"temperature([0-9.]+)", d)
        if not (m_task and m_kl and m_mse and m_T):
            continue
        n += 1
        key = (m_task.group(1), float(m_kl.group(1)), float(m_mse.group(1)), float(m_T.group(1)))
        (l2 if "mse_normalizeTrue" in d else raw).add(key)

    print(f"scanned {n} final_summary paths")
    print(f"RAW grid combos done: {len(raw)} / {args.total} ({100*len(raw)/args.total:.0f}%)")
    print(f"  (l2norm combos seen: {len(l2)})")


if __name__ == "__main__":
    main()
