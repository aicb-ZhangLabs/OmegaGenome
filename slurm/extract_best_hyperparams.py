#!/usr/bin/env python3
"""Extract the best distillation hyperparameters per task from a completed/partial hyperparam search.

Walks every ``final_summary.json`` under the carbon distillation output tree, groups runs by
(variant, task) where variant = raw|l2norm (read from the run-dir path's ``mse_normalize{False,True}``),
and selects the run with the highest **validation** MCC per (variant, task) — selecting on val, never
test, so the chosen hyperparameters don't leak the test set. Writes a clean ``best_hyperparams.json``
the 3-seed runner consumes, and prints a summary.

Usage:
  python slurm/extract_best_hyperparams.py [--base <output_dir>] [--out best_hyperparams.json]
"""

import argparse
import glob
import json
import os

DEFAULT_BASE = os.environ.get("OG_SCRATCH", "output") + "/carbon_distillation"
HP_KEYS = ("weight_ce", "weight_kl", "weight_mse", "temperature")  # the swept knobs we re-apply


def variant_from_path(path: str) -> str:
    """raw vs l2norm is encoded in the run-dir name as mse_normalizeFalse / mse_normalizeTrue."""
    if "mse_normalizeTrue" in path:
        return "l2norm"
    if "mse_normalizeFalse" in path:
        return "raw"
    return "unknown"


def collect(base: str):
    best = {}  # (variant, task) -> summary dict (+ variant, path)
    n = 0
    for f in glob.glob(os.path.join(base, "**", "final_summary.json"), recursive=True):
        try:
            s = json.load(open(f))
        except Exception:
            continue
        task = s.get("task")
        val = s.get("best_val_mcc")
        if task is None or val is None:
            continue
        n += 1
        variant = variant_from_path(f)
        key = (variant, task)
        if key not in best or val > best[key]["best_val_mcc"]:
            s["_variant"] = variant
            s["_path"] = f
            best[key] = s
    # candidate counts per (variant, task)
    counts = {}
    for f in glob.glob(os.path.join(base, "**", "final_summary.json"), recursive=True):
        try:
            s = json.load(open(f))
        except Exception:
            continue
        if s.get("task") and s.get("best_val_mcc") is not None:
            counts[(variant_from_path(f), s["task"])] = (
                counts.get((variant_from_path(f), s["task"]), 0) + 1
            )
    return best, counts, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=DEFAULT_BASE)
    ap.add_argument("--out", default="best_hyperparams.json")
    args = ap.parse_args()

    best, counts, n = collect(args.base)
    # nested out: {variant: {task: {hyperparameters, best_val_mcc, best_test_mcc, n_candidates}}}
    out = {}
    for (variant, task), s in sorted(best.items()):
        hp = {
            k: s.get("hyperparameters", {}).get(k)
            for k in HP_KEYS
            if k in s.get("hyperparameters", {})
        }
        out.setdefault(variant, {})[task] = {
            "hyperparameters": hp,
            "best_val_mcc": round(s["best_val_mcc"], 4),
            "best_test_mcc": round(s.get("best_test_mcc", float("nan")), 4),
            "best_epoch": s.get("best_epoch"),
            "n_candidates": counts.get((variant, task), 1),
        }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)

    print(f"scanned {n} final_summary.json under {args.base}")
    for variant in sorted(out):
        rows = out[variant]
        print(f"\n=== {variant}: best-on-val hyperparameters ({len(rows)}/18 tasks) ===")
        print(f"{'task':<24}{'val':>7}{'test':>7}{'  n':>4}  hyperparams")
        for task in sorted(rows):
            r = rows[task]
            hp = ", ".join(
                f"{k.replace('weight_', 'w_')}={v}" for k, v in r["hyperparameters"].items()
            )
            print(
                f"{task:<24}{r['best_val_mcc']:>7}{r['best_test_mcc']:>7}{r['n_candidates']:>4}  {hp}"
            )
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
