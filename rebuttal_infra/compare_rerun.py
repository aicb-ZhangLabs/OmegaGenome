#!/usr/bin/env python3
"""Aggregate the below-baseline rerun (final_summary.json files) into 3-seed means and
compare each (teacher, task) cell against the original 3-seed (@200 epochs) and the BPNet
baseline. Prints a table so we can pick, per cell, whether the rerun or the original is
better and which clears the baseline. Safe to run mid-sweep (reports whatever is done)."""
import os, json, glob, re
from collections import defaultdict
import pandas as pd

BASE = "/tmp/galaxy_srv_disk00/pengchx3/rerun_below_baseline"
REPO = "/home/pengchx3/text-dna/OmegaGenome_Revise_202606"
PRESET2FAM = {"dnabert2": "DNABERT-2", "caduceus": "Caduceus", "carbon": "Carbon-3B",
              "enformer": "Enformer", "nt": "NT-2.5B"}

# --- rerun results ---
rows = defaultdict(dict)  # (fam, task) -> {seed: (mcc, epochs, best_epoch)}
for f in glob.glob(f"{BASE}/**/final_summary.json", recursive=True):
    d = json.load(open(f))
    teacher_dir = f.replace(BASE + "/", "").split("/")[0]  # dnabert2/caduceus/...
    fam = PRESET2FAM.get(teacher_dir, teacher_dir)
    rows[(fam, d["task"])][d["random_state"]] = (
        d["best_test_mcc"], d["total_epochs"], d["best_epoch"])

# --- original 3-seed (DNABERT-2 was @100 epochs, all other teachers @200) ---
orig = pd.read_csv(f"{REPO}/code_carbon/rebuttal_infra/3seed_results_summary.csv")
orig_mean = {(r.teacher, r.task): r.mean_test_mcc for r in orig.itertuples()}
ORIG_EP = {"DNABERT-2": 100}  # default 200 for every other teacher

# --- BPNet baseline per task ---
form = pd.read_csv(f"{REPO}/plot_repo/data/model_comparison_5teacher_formal.csv")
base = form[form["Type"] == "Baseline"].set_index("Task")["Score"].to_dict()

print(f"{'Teacher':11s} {'Task':22s} {'ep':>4s} {'rerun':>7s} {'orig':>7s} {'oEp':>3s} "
      f"{'bpnet':>6s} {'rr-Δ':>7s} {'or-Δ':>7s}  verdict")
print("-" * 92)
import statistics as st
done = 0
for (fam, task), seeds in sorted(rows.items()):
    mccs = [v[0] for v in seeds.values()]
    if len(mccs) < 3:
        tag = f"(partial {len(mccs)}/3)"
    else:
        tag = ""
        done += 1
    rr = sum(mccs) / len(mccs)
    ep = list(seeds.values())[0][1]
    o = orig_mean.get((fam, task))
    bp = base.get(task)
    rrd = rr - bp if bp else float("nan")
    ord_ = (o - bp) if (o and bp) else float("nan")
    verdict = ""
    if bp:
        if rr >= bp and (o or 0) < bp:
            verdict = "RERUN CLEARS ✓"
        elif rr >= bp and (o or 0) >= bp:
            verdict = "both ok"
        elif rr < bp and (o or 0) >= bp:
            verdict = "rerun WORSE ✗"
        else:
            verdict = "both below"
    oep = ORIG_EP.get(fam, 200)
    print(f"{fam:11s} {task:22s} {ep:>4d} {rr:>7.4f} {(o or 0):>7.4f} {oep:>3d} "
          f"{(bp or 0):>6.4f} {rrd:>+7.4f} {ord_:>+7.4f}  {verdict} {tag}")
print("-" * 92)
print(f"cells with full 3 seeds: {done} / {len(rows)} started; "
      f"total final_summary files: {sum(len(s) for s in rows.values())}")
