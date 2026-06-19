#!/usr/bin/env python3
"""Aggregate carbon distillation results from slurm logs into a raw-vs-l2norm comparison table.
Parses slurm-carbon-distill-*.out for: config (carbon-raw/carbon-l2norm), task, teacher MCC, student
best test MCC. Re-runnable as the campaign progresses. Usage: python slurm/aggregate_carbon_results.py
"""
import glob, re, sys
rows = {}  # (task) -> {"teacher":x, "raw":y, "l2norm":z}
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
    tmcc = re.findall(r"Teacher Test MCC: ([0-9.]+)", t)
    smcc = re.findall(r"Best test MCC: ([0-9.]+)", t)
    done = "done: carbon" in t
    r = rows.setdefault(task, {})
    if tmcc:
        r["teacher"] = float(tmcc[0])
    if smcc:
        r[cfg] = float(smcc[-1])
    r.setdefault(cfg + "_status", "running")
    if done:
        r[cfg + "_status"] = "done"

ORDER = ["H3K27me3","H3K36me3","H4K20me1","H2AFZ","H3K27ac","H3K4me1","H3K4me2","H3K4me3","H3K9ac",
         "H3K9me3","promoter_all","promoter_tata","promoter_no_tata","enhancers","enhancers_types",
         "splice_sites_all","splice_sites_acceptors","splice_sites_donors"]
print(f"{'task':<24}{'teacher':>9}{'raw':>9}{'l2norm':>9}{'  status'}")
print("-"*65)
nraw=nl2=0
for task in ORDER:
    r = rows.get(task, {})
    te = f"{r['teacher']:.4f}" if "teacher" in r else "-"
    rw = f"{r['raw']:.4f}" if "raw" in r else "-"
    l2 = f"{r['l2norm']:.4f}" if "l2norm" in r else "-"
    st = f"raw:{r.get('raw_status','-')} l2:{r.get('l2norm_status','-')}"
    nraw += "raw" in r; nl2 += "l2norm" in r
    print(f"{task:<24}{te:>9}{rw:>9}{l2:>9}  {st}")
print("-"*65)
print(f"raw with student MCC: {nraw}/18   l2norm: {nl2}/18")
