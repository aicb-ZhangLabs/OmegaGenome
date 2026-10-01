#!/usr/bin/env python3
"""Emit resume-aware 3-seed specs (only missing teacher x task x seed cells) to stdout.
Scans 3seed_finals/<teacher>/**/final_summary.json for completed (teacher, task, random_state)."""

import os
import csv
import glob
import json
import sys

F = os.environ.get("OG_SCRATCH", "output") + "/3seed_finals"
REPO = os.environ.get("OG_ROOT", os.getcwd())
CFG = {
    "nt": "nt_vanilla",
    "dnabert2": "dna_bert_v2",
    "caduceus": "caduceus_vanilla",
    "enformer": "enformer_vanilla",
}

done = set()
for t in CFG:
    for fp in glob.glob(f"{F}/{t}/**/final_summary.json", recursive=True):
        try:
            s = json.load(open(fp))
            done.add((t, s["task"], int(s.get("random_state", -1))))
        except Exception:
            pass


def num(x):
    x = str(x).strip()
    return x if x else None


lines = []
for t, cfg in CFG.items():
    for r in csv.DictReader(open(f"{REPO}/best_hp_all_teachers/best_hp/best_hp_{t}.csv")):
        ce, kl, mse, tp = (
            num(r["weight_ce"]),
            num(r["weight_kl"]),
            num(r["weight_mse"]),
            num(r["temperature"]),
        )
        if None in (ce, kl, mse, tp):
            continue
        for sd in (0, 1, 2):
            if (t, r["task"], sd) in done:
                continue
            lines.append(
                f"{cfg} --task-names {r['task']} --distillation-config.weight-ce {ce} "
                f"--distillation-config.weight-kl {kl} --distillation-config.weight-mse {mse} "
                f"--distillation-config.temperature {tp} --trainer-config.output-dir {F}/{t}/vanilla_original "
                f"--trainer-config.early-stop-patience 250 --random-state {sd} --slurm-config.mode run"
            )
sys.stderr.write(f"done cells: {len(done)}; gaps: {len(lines)}\n")
print("\n".join(lines))
