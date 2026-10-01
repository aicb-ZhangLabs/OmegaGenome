#!/usr/bin/env python3
"""Emit resume-aware carbon-raw-original HP specs (only unfinished base-grid combos) to stdout."""

import os
import glob
import json
import sys
from itertools import product

BASE = os.environ.get("OG_SCRATCH", "output") + "/carbon_distillation/original"
TASKS = [
    "H3K27me3",
    "H3K36me3",
    "H4K20me1",
    "H2AFZ",
    "H3K27ac",
    "H3K4me1",
    "H3K4me2",
    "H3K4me3",
    "H3K9ac",
    "H3K9me3",
    "promoter_all",
    "promoter_tata",
    "promoter_no_tata",
    "enhancers",
    "enhancers_types",
    "splice_sites_all",
    "splice_sites_acceptors",
    "splice_sites_donors",
]
KLS = [0.0, 0.25, 0.5, 1.0]
MSES = [0.0, 1.0, 2.0, 5.0]
TEMPS = [0.5, 1.0, 1.5, 2.0, 4.0]


def tf(kl):
    return [TEMPS[0]] if kl == 0.0 else TEMPS


done = set()
for fp in glob.glob(BASE + "/**/final_summary.json", recursive=True):
    if "mse_normalizeTrue" in fp:
        continue
    try:
        s = json.load(open(fp))
        h = s["hyperparameters"]
        if s["task"] in TASKS:
            done.add(
                (
                    s["task"],
                    0.5,
                    float(h["weight_kl"]),
                    float(h["weight_mse"]),
                    float(h["temperature"]),
                )
            )
    except Exception:
        pass

lines = []
for task, ce, kl, mse in product(TASKS, [0.5], KLS, MSES):
    for tp in tf(kl):
        if (task, ce, kl, mse, tp) in done:
            continue
        lines.append(
            f"carbon-raw-original --task-names {task} --distillation-config.weight-ce {ce} "
            f"--distillation-config.weight-kl {kl} --distillation-config.weight-mse {mse} "
            f"--distillation-config.temperature {tp} --trainer-config.early-stop-patience 100 --slurm-config.mode run"
        )
sys.stderr.write(f"carbon done: {len(done)}; gaps: {len(lines)}\n")
print("\n".join(lines))
