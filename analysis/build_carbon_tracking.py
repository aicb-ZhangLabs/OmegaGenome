#!/usr/bin/env python3
"""Build the Carbon-3B HP-tracking artifacts (no GPU; pure aggregation).

Produces, under analysis/tracking/ and results/:
  1. hp_grids/carbon_base_hp_grid.csv      -- the base HP grid (ce/kl/mse/T), NT-style grid doc
  2. hp_grids/carbon_stage1_2_3_settings.csv -- the mse-staged sweep (stage1/2/3 + manual slices)
  3. results/carbon_merged_results.csv     -- lab grid results + box (HF) results, one schema
  4. tracking/carbon_hp_gap.csv            -- per (task,mse,kl,T) combos still MISSING vs base grid
  5. tracking/distill_progress.csv         -- master teacher x task tracker (hp/3seed/hf/docx)

Inputs (all already on the lab node):
  * config/distillation/experiments/carbon.py  -- grid definition (mirrored here as constants)
  * results/carbon_grid_results_original*.csv   -- lab-side "original" BPNet student HP runs
  * results/box_synced/metrics/carbon_students_summary.csv -- box-side runs pulled from HF
"""

import os
import csv
from collections import defaultdict

REPO = os.environ.get("OG_ROOT", os.getcwd())
os.makedirs(f"{REPO}/analysis/hp_grids", exist_ok=True)
os.makedirs(f"{REPO}/analysis/tracking", exist_ok=True)

# ---- canonical grid (from experiments/carbon.py _GRID; kl=0 collapses temperature) ----
WEIGHT_CE = 0.5
WEIGHT_KLS = [0.0, 0.25, 0.5, 1.0]
TEMPS = [0.5, 1.0, 1.5, 2.0, 4.0]
BASE_MSES = [0.0, 1.0, 2.0, 5.0]  # base grid (per user: excludes 0.25 / 0.2)
STAGE_MSES = {1: [0.0, 1.0], 2: [0.25], 3: [2.0, 5.0]}
MANUAL_MSES = [0.2, 0.25]  # "for me checking" slices, outside the base grid

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
TEACHERS = ["carbon", "nt", "dnabert2", "caduceus", "enformer"]


def canonical_kl_temp():
    """Yield the canonical (kl, temperature) combos: kl=0 uses a single temp (T is a no-op)."""
    combos = [(0.0, 1.0)]  # kl=0 -> one representative temp
    for kl in [0.25, 0.5, 1.0]:
        for t in TEMPS:
            combos.append((kl, t))
    return combos  # 1 + 15 = 16


def base_grid_combos(mses=BASE_MSES):
    """Full per-task combo list = (kl,temp) x mse over the given mse set."""
    for kl, t in canonical_kl_temp():
        for m in mses:
            yield (WEIGHT_CE, kl, m, t)


# ---- 1. base HP grid CSV (documentation, NT-style) ----
with open(f"{REPO}/analysis/hp_grids/carbon_base_hp_grid.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["param", "values", "note"])
    w.writerow(["weight_ce", "0.5", "fixed"])
    w.writerow(["weight_kl", "0.0 | 0.25 | 0.5 | 1.0", "kl=0 collapses temperature (T no-op)"])
    w.writerow(["weight_mse", "0.0 | 1.0 | 2.0 | 5.0", "BASE grid (excludes 0.2, 0.25)"])
    w.writerow(["temperature", "0.5 | 1.0 | 1.5 | 2.0 | 4.0", "applied only when kl>0"])
    w.writerow(["zscore", "False", "fixed"])
    w.writerow(["distill_method", "vanilla", "fixed (kl_method=kl)"])
    w.writerow(
        [
            "student",
            "original_bpnet (~121k) + deploy_120k",
            "original = same student as NT/DNABERT2/Caduceus/Enformer",
        ]
    )
    w.writerow(["variant", "raw | l2norm", "MSE normalization of teacher features"])
    w.writerow(
        ["canonical_combos_per_task", str(len(list(base_grid_combos()))), "16 (kl,T) x 4 mse = 64"]
    )
    w.writerow(["tasks", str(len(TASKS)), "18 NT-revised downstream tasks"])

# ---- 2. stage1/2/3 settings CSV ----
with open(f"{REPO}/analysis/hp_grids/carbon_stage1_2_3_settings.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["stage", "weight_mses", "n_combos_per_task", "kind", "note"])
    for s, mses in STAGE_MSES.items():
        n = len(canonical_kl_temp()) * len(mses)
        kind = {1: "base", 2: "manual/extra", 3: "base"}[s]
        w.writerow(
            [
                f"stage{s}",
                " | ".join(str(m) for m in mses),
                n,
                kind,
                {1: "current lab default", 2: "mse=0.25 (for me checking)", 3: "mse=2,5 (box)"}[s],
            ]
        )
    w.writerow(
        [
            "manual",
            " | ".join(str(m) for m in MANUAL_MSES),
            "-",
            "manual/extra",
            "0.2 = single-config default; 0.25 = stage2 slice; NOT in base grid",
        ]
    )


# ---- helpers to normalize a float key ----
def fkey(x):
    try:
        return round(float(x), 4)
    except (TypeError, ValueError):
        return None


# ---- 3. merge lab + box results ----
merged = []  # unified rows
seen_combo = defaultdict(set)  # task -> set of (kl,mse,temp) present (any source)

# lab: carbon_grid_results_original_refresh.csv (latest) preferred, fall back to original
lab_csv = None
for cand in [
    "results/carbon_grid_results_original_refresh.csv",
    "results/carbon_grid_results_original.csv",
]:
    if os.path.exists(f"{REPO}/{cand}"):
        lab_csv = f"{REPO}/{cand}"
        break
if lab_csv:
    for r in csv.DictReader(open(lab_csv)):
        t = r.get("task")
        kl, m, temp = (
            fkey(r.get("weight_kl")),
            fkey(r.get("weight_mse")),
            fkey(r.get("temperature")),
        )
        merged.append(
            {
                "source": "lab",
                "task": t,
                "variant": r.get("variant", "raw"),
                "weight_ce": fkey(r.get("weight_ce")),
                "weight_kl": kl,
                "weight_mse": m,
                "temperature": temp,
                "seed": r.get("random_state"),
                "best_val_mcc": r.get("best_val_mcc"),
                "best_test_mcc": r.get("best_test_mcc"),
                "best_epoch": r.get("best_epoch"),
                "path": r.get("path", ""),
            }
        )
        if t and kl is not None:
            seen_combo[t].add((kl, m, temp if kl != 0.0 else 1.0))

# box: results/box_synced/metrics/carbon_students_summary.csv (pulled from HF)
box_csv = f"{REPO}/results/box_synced/metrics/carbon_students_summary.csv"
if os.path.exists(box_csv):
    for r in csv.DictReader(open(box_csv)):
        t = r.get("task")
        kl, m, temp = (
            fkey(r.get("hp_weight_kl")),
            fkey(r.get("hp_weight_mse")),
            fkey(r.get("hp_temperature")),
        )
        merged.append(
            {
                "source": "box",
                "task": t,
                "variant": "raw",
                "weight_ce": fkey(r.get("hp_weight_ce")),
                "weight_kl": kl,
                "weight_mse": m,
                "temperature": temp,
                "seed": "box",
                "best_val_mcc": r.get("best_val_mcc"),
                "best_test_mcc": r.get("teacher_test_mcc"),
                "best_epoch": r.get("best_epoch"),
                "path": r.get("rel_path", ""),
            }
        )
        if t and kl is not None:
            seen_combo[t].add((kl, m, temp if kl != 0.0 else 1.0))

cols = [
    "source",
    "task",
    "variant",
    "weight_ce",
    "weight_kl",
    "weight_mse",
    "temperature",
    "seed",
    "best_val_mcc",
    "best_test_mcc",
    "best_epoch",
    "path",
]
with open(f"{REPO}/results/carbon_merged_results.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=cols)
    w.writeheader()
    for row in merged:
        w.writerow(row)

# ---- 4. gap analysis vs base grid ----
gap_rows = []
for t in TASKS:
    have = seen_combo.get(t, set())
    for ce, kl, m, temp in base_grid_combos():
        key = (kl, m, temp if kl != 0.0 else 1.0)
        if key not in have:
            gap_rows.append(
                {"task": t, "weight_ce": ce, "weight_kl": kl, "weight_mse": m, "temperature": temp}
            )
with open(f"{REPO}/analysis/tracking/carbon_hp_gap.csv", "w", newline="") as f:
    w = csv.DictWriter(
        f, fieldnames=["task", "weight_ce", "weight_kl", "weight_mse", "temperature"]
    )
    w.writeheader()
    for row in gap_rows:
        w.writerow(row)

# per-task gap summary
gap_by_task = defaultdict(int)
for g in gap_rows:
    gap_by_task[g["task"]] += 1

# ---- 5. master progress tracker (teacher x task) ----
# HF student presence: carbon has carbon_bpnet_students/{task}.pt on omegagenome-distilled-students
carbon_hf_tasks = set(TASKS)  # all 18 present per earlier HF listing (carbon_bpnet_students/)
prog = []
for teacher in TEACHERS:
    for t in TASKS:
        if teacher == "carbon":
            n_grid_combos = 64
            n_have = n_grid_combos - gap_by_task.get(t, 0)
            hp_done = "yes" if gap_by_task.get(t, 0) == 0 else f"partial ({n_have}/{n_grid_combos})"
            hf = "yes(carbon_bpnet_students)" if t in carbon_hf_tasks else "no"
        else:
            hp_done = "yes(best_hp seed42)"  # NT/DNABERT2/Caduceus/Enformer best_hp_*.csv exist
            hf = "no"
        prog.append(
            {
                "teacher": teacher,
                "task": t,
                "hp_search": hp_done,
                "seed42": "-",
                "seed123": "-",
                "seed456": "-",
                "best_ckpt": "-",
                "hf_uploaded": hf,
                "docx_row": "-",
            }
        )
with open(f"{REPO}/analysis/tracking/distill_progress.csv", "w", newline="") as f:
    w = csv.DictWriter(
        f,
        fieldnames=[
            "teacher",
            "task",
            "hp_search",
            "seed42",
            "seed123",
            "seed456",
            "best_ckpt",
            "hf_uploaded",
            "docx_row",
        ],
    )
    w.writeheader()
    for row in prog:
        w.writerow(row)

# ---- report ----
print("=== ARTIFACTS WRITTEN ===")
for p in [
    "analysis/hp_grids/carbon_base_hp_grid.csv",
    "analysis/hp_grids/carbon_stage1_2_3_settings.csv",
    "results/carbon_merged_results.csv",
    "analysis/tracking/carbon_hp_gap.csv",
    "analysis/tracking/distill_progress.csv",
]:
    print(f"  {p}  ({os.path.getsize(REPO + '/' + p)} B)")
print(f"\nmerged rows: {len(merged)} (lab+box)")
print(f"total base-grid gap combos: {len(gap_rows)} across 18 tasks")
print("\nper-task Carbon HP coverage (have/64 base combos):")
for t in TASKS:
    have = 64 - gap_by_task.get(t, 0)
    flag = "  <-- COMPLETE" if gap_by_task.get(t, 0) == 0 else ""
    print(f"   {t:24s} {have:2d}/64{flag}")
