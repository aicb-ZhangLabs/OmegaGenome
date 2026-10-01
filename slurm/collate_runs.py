#!/usr/bin/env python3
"""Collate EVERY distillation run into one flat table: one row per (task, hyperparams, seed) with its
val/test performance. Reads every final_summary.json under --base and emits CSV to stdout (stdlib only,
so it runs on any node — run it ON GALAXY where /srv/disk00 is local for speed). Columns:
  variant, task, weight_ce, weight_kl, weight_mse, temperature, lr, batch_size, random_state,
  best_val_mcc, best_test_mcc, final_test_mcc, best_test_f1, best_epoch, total_epochs, early_stopped, path
A summary (counts) is written to stderr so it doesn't pollute the CSV. Usage:
  python slurm/collate_runs.py --base <dir>  > results/carbon_grid_results.csv
"""

import os
import argparse
import csv
import glob
import json
import sys

COLS = [
    "variant",
    "task",
    "weight_ce",
    "weight_kl",
    "weight_mse",
    "temperature",
    "lr",
    "batch_size",
    "random_state",
    "best_val_mcc",
    "best_test_mcc",
    "final_test_mcc",
    "best_test_f1",
    "best_epoch",
    "total_epochs",
    "early_stopped",
    "path",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--base", default=os.environ.get("OG_SCRATCH", "output") + "/carbon_distillation"
    )
    args = ap.parse_args()

    w = csv.writer(sys.stdout)
    w.writerow(COLS)
    n = 0
    seeds, tasks = {}, set()
    for f in glob.iglob(os.path.join(args.base, "**", "final_summary.json"), recursive=True):
        try:
            s = json.load(open(f))
        except Exception:
            continue
        if s.get("task") is None or s.get("best_val_mcc") is None:
            continue
        hp = s.get("hyperparameters", {})
        variant = "l2norm" if "mse_normalizeTrue" in f else "raw"
        seed = s.get("random_state", "")
        w.writerow(
            [
                variant,
                s["task"],
                hp.get("weight_ce"),
                hp.get("weight_kl"),
                hp.get("weight_mse"),
                hp.get("temperature"),
                hp.get("lr"),
                hp.get("batch_size"),
                seed,
                s.get("best_val_mcc"),
                s.get("best_test_mcc"),
                s.get("final_test_mcc"),
                s.get("best_test_f1"),
                s.get("best_epoch"),
                s.get("total_epochs"),
                s.get("early_stopped"),
                f,
            ]
        )
        n += 1
        tasks.add(s["task"])
        seeds[seed] = seeds.get(seed, 0) + 1
    print(
        f"[collate] {n} runs | {len(tasks)} tasks | runs-per-seed {dict(sorted(seeds.items(), key=str))}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
