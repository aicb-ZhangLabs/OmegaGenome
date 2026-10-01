#!/usr/bin/env python3
"""Emit resume-aware Carbon 3-seed specs (best_hp_carbon config per task x seed {0,1,2}) to stdout.
Carbon's own 3-seed finals (5th teacher), gated on the completed HP grid -> best_hp_carbon.csv.
Output isolated under 3seed_finals/carbon/. No-patience (250) full-200ep protocol like the others."""

import os
import csv
import glob
import json
import sys

REPO = os.environ.get("OG_ROOT", os.getcwd())
OUT = os.environ.get("OG_SCRATCH", "output") + "/3seed_finals/carbon/vanilla_original"
SEEDS = (0, 1, 2)

done = set()  # (task, seed)
for fp in glob.glob(
    f"{os.environ.get('OG_SCRATCH', 'output')}/3seed_finals/carbon/**/final_summary.json",
    recursive=True,
):
    try:
        s = json.load(open(fp))
        done.add((s["task"], int(s.get("random_state", -1))))
    except Exception:
        pass


def num(x):
    x = str(x).strip()
    return x if x else None


lines = []
for r in csv.DictReader(open(f"{REPO}/best_hp_all_teachers/best_hp/best_hp_carbon.csv")):
    task = r["task"]
    ce, kl, mse, tp = (
        num(r["weight_ce"]),
        num(r["weight_kl"]),
        num(r["weight_mse"]),
        num(r["temperature"]),
    )
    if None in (ce, kl, mse, tp):
        continue
    for sd in SEEDS:
        if (task, sd) in done:
            continue
        lines.append(
            f"carbon-raw-original --task-names {task} --distillation-config.weight-ce {ce} "
            f"--distillation-config.weight-kl {kl} --distillation-config.weight-mse {mse} "
            f"--distillation-config.temperature {tp} --trainer-config.output-dir {OUT} "
            f"--trainer-config.early-stop-patience 250 --random-state {sd} --slurm-config.mode run"
        )
sys.stderr.write(f"carbon-3seed done cells: {len(done)}; gaps: {len(lines)}\n")
print("\n".join(lines))
