#!/usr/bin/env python3
"""Aggregate the BEST-hyperparameter 3-seed runs into a per-task mean+/-std significance table.

Reads ``final_summary.json`` (authoritative — not log-scraping) and selects ONLY the multi-seed runs
by ``random_state`` in the requested seed set (default {0,1,2}). HP-search runs use the config default
seed 42, so they are excluded automatically — no contamination of the seed-0 cell (the bug that would
hit a log-scraping aggregator, where every HP-search run defaults to "seed 0"). Per task it reports
mean +/- population-std of ``best_test_mcc`` across the seeds, and (if best_hyperparams.json is given)
asserts each seed run actually used the best hyperparameters.

Usage:
  python slurm/aggregate_3seed.py [--base <output_dir>] [--seeds 0 1 2] [--best best_hyperparams.json]
"""
import argparse
import glob
import json
import os
import re
from collections import defaultdict
from statistics import mean, pstdev

DEFAULT_BASE = "/tmp/galaxy_srv_disk00/pengchx3/carbon_distillation"
HP_KEYS = ("weight_ce", "weight_kl", "weight_mse", "temperature")
ORDER = ["H3K27me3", "H3K36me3", "H4K20me1", "H2AFZ", "H3K27ac", "H3K4me1", "H3K4me2", "H3K4me3",
         "H3K9ac", "H3K9me3", "promoter_all", "promoter_tata", "promoter_no_tata", "enhancers",
         "enhancers_types", "splice_sites_all", "splice_sites_acceptors", "splice_sites_donors"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=DEFAULT_BASE)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--best", default=None, help="best_hyperparams.json to cross-check HP per seed run")
    ap.add_argument("--variant", default="raw", choices=["raw", "l2norm"])
    args = ap.parse_args()
    want = set(args.seeds)

    best = None
    if args.best and os.path.exists(args.best):
        best = json.load(open(args.best)).get(args.variant, {})

    # task -> seed -> best_test_mcc   (one entry per seed; last-write wins only on genuine duplicates)
    runs = defaultdict(dict)
    hp_seen = defaultdict(set)
    scanned = 0
    for f in glob.glob(os.path.join(args.base, "**", "final_summary.json"), recursive=True):
        try:
            s = json.load(open(f))
        except Exception:
            continue
        # missing random_state (pre-seed-field HP runs) -> treat as the config default 42 -> excluded
        seed = s.get("random_state", 42)
        if seed not in want:
            continue
        task, mcc = s.get("task"), s.get("best_test_mcc")
        if task is None or mcc is None:
            continue
        scanned += 1
        runs[task][seed] = float(mcc)
        hp = tuple(s.get("hyperparameters", {}).get(k) for k in HP_KEYS)
        hp_seen[task].add(hp)

    print(f"selected {scanned} seed runs (random_state in {sorted(want)}) under {args.base}\n")
    print(f"{'task':<24}{'mean+-std (best_test_mcc)':<26}{'seeds':<10}{'hp(ce,kl,mse,T)'}")
    print("-" * 90)
    means = []
    for task in [t for t in ORDER if t in runs] + [t for t in sorted(runs) if t not in ORDER]:
        d = runs[task]
        vals = [d[s] for s in sorted(d)]
        cell = f"{mean(vals):.4f}+-{pstdev(vals):.4f}" if len(vals) > 1 else f"{vals[0]:.4f}"
        means.append(mean(vals))
        hp = sorted(hp_seen[task])[0] if hp_seen[task] else ()
        hp_str = ",".join(str(x) for x in hp)
        warn = ""
        if len(hp_seen[task]) > 1:
            warn = "  <-- WARN: seeds used DIFFERENT hp"
        if best and task in best:
            want_hp = tuple(best[task]["hyperparameters"].get(k) for k in HP_KEYS)
            if hp_seen[task] and hp != want_hp:
                warn += f"  <-- WARN: hp != best {want_hp}"
        print(f"{task:<24}{cell:<26}{str(sorted(d)):<10}{hp_str}{warn}")
    print("-" * 90)
    full = [t for t in runs if len(runs[t]) == len(want)]
    print(f"{len(runs)}/18 tasks have any seed; {len(full)}/18 have all {len(want)} seeds")
    if means:
        print(f"mean best_test_mcc over {len(means)} tasks: {mean(means):.4f}")


if __name__ == "__main__":
    main()
