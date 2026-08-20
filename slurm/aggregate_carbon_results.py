#!/usr/bin/env python3
"""Aggregate carbon distillation results from slurm logs into a raw-vs-l2norm comparison table,
with multi-seed mean +/- std (the significance view). Parses slurm-carbon-distill-*.out for: config
(carbon-raw/carbon-l2norm), task, seed (--random-state, default 0), teacher MCC, student best-test MCC.
Re-runnable as the campaign progresses. Usage: python slurm/aggregate_carbon_results.py
"""
import glob, re
from collections import defaultdict
from statistics import mean, pstdev

teacher = {}                                  # task -> teacher MCC
students = defaultdict(lambda: defaultdict(dict))  # cfg -> task -> {seed: mcc}
for f in glob.glob("slurm-carbon-distill-*.out"):
    try:
        t = open(f, errors="ignore").read()
    except Exception:
        continue
    m_cfg = re.search(r"carbon distillation: (carbon-raw|carbon-l2norm)", t)
    m_task = re.search(r"Starting Distillation: ([A-Za-z0-9_]+)", t)
    if not (m_cfg and m_task):
        continue
    cfg = "raw" if m_cfg.group(1) == "carbon-raw" else "l2norm"
    task = m_task.group(1)
    seed = int(m.group(1)) if (m := re.search(r"--random-state (\d+)", t)) else 0
    if (tm := re.findall(r"Teacher Test MCC: ([0-9.]+)", t)):
        teacher[task] = float(tm[0])
    if (sm := re.findall(r"Best test MCC: ([0-9.]+)", t)):
        students[cfg][task][seed] = float(sm[-1])

ORDER = ["H3K27me3","H3K36me3","H4K20me1","H2AFZ","H3K27ac","H3K4me1","H3K4me2","H3K4me3","H3K9ac",
         "H3K9me3","promoter_all","promoter_tata","promoter_no_tata","enhancers","enhancers_types",
         "splice_sites_all","splice_sites_acceptors","splice_sites_donors"]

def cell(cfg, task):
    d = students[cfg].get(task, {})
    if not d:
        return "-"
    vals = list(d.values())
    if len(vals) == 1:
        return f"{vals[0]:.4f} (s{list(d)[0]})"
    return f"{mean(vals):.4f}+-{pstdev(vals):.4f} (n{len(vals)})"

print(f"{'task':<24}{'teacher':>9}   {'raw':<24}{'l2norm':<24}")
print("-"*82)
for task in ORDER:
    te = f"{teacher[task]:.4f}" if task in teacher else "-"
    print(f"{task:<24}{te:>9}   {cell('raw',task):<24}{cell('l2norm',task):<24}")
print("-"*82)
for cfg in ("raw","l2norm"):
    nruns = sum(len(d) for d in students[cfg].values())
    ntask = len(students[cfg])
    seeds = sorted({s for d in students[cfg].values() for s in d})
    print(f"{cfg}: {ntask}/18 tasks, {nruns} runs, seeds={seeds}")
common = [t for t in ORDER if t in teacher and students['raw'].get(t)]
if common:
    tm = mean(teacher[t] for t in common)
    sm = mean(mean(students['raw'][t].values()) for t in common)
    print(f"\nMean over {len(common)} tasks: teacher={tm:.4f}  raw-student={sm:.4f}  ({100*sm/tm:.1f}% of teacher)")
