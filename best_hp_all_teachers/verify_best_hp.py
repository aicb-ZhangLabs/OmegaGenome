#!/usr/bin/env python3
"""Verify each best_hp_<teacher>.yaml against its source CSV.

Asserts per teacher: (1) exactly 18 task blocks; (2) each block's best_val_mcc equals the
max validation MCC over that task's rows in the CSV (rounded); (3) the recorded best_test_mcc
comes from that SAME best-val row (not the global max test). Reports any task with a missing
test MCC or other anomaly instead of silently dropping it.
"""

import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_best_hp import resolve_map, get_task, VAL_COL, TEST_COL

SRC = (
    os.environ.get("OG_STORE", "data")
    + "/OmegaGenome_different_version/OmegaGenome_11_1_fix_ckpt/OmegaGenome"
)
CSVS = {
    "nt": f"{SRC}/nt_hyperparam_flat_clean.csv",
    "dnabert2": f"{SRC}/dnabert2_hyperparam_flat_clean.csv",
    "caduceus": f"{SRC}/caduceus_hyperparam_1-15_flat_clean.csv",
    "enformer": f"{SRC}/enformer_hyperparam_flat_clean.csv",
}
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "best_hp")


def parse_yaml(path):
    """Minimal parser for our flat best_hp YAML: returns {task: {best_val_mcc, best_test_mcc}}."""
    out, cur = {}, None
    for line in open(path):
        if line.rstrip() and not line.startswith(" ") and line.rstrip().endswith(":"):
            cur = line.strip()[:-1]
            out[cur] = {}
        elif cur and line.startswith("  best_val_mcc:"):
            out[cur]["val"] = line.split(":", 1)[1].strip()
        elif cur and line.startswith("  best_test_mcc:"):
            out[cur]["test"] = line.split(":", 1)[1].strip()
    return out


def csv_truth(csv_path, teacher):
    """Return {task: (max_val, test_of_that_row)} computed directly from the CSV."""
    cmap = resolve_map(teacher)
    best = {}
    for row in csv.DictReader(open(csv_path, newline="")):
        v = row.get(VAL_COL, "")
        if not v or not str(v).strip():
            continue
        try:
            val = float(v)
        except ValueError:
            continue
        task = get_task(row, cmap)
        if task is None:
            continue
        if task not in best or val > best[task][0]:
            tm = row.get(TEST_COL, "")
            best[task] = (val, float(tm) if tm and str(tm).strip() else None)
    return best


def main():
    all_ok = True
    for teacher, csv_path in CSVS.items():
        yaml_path = os.path.join(OUT, f"best_hp_{teacher}.yaml")
        y = parse_yaml(yaml_path)
        truth = csv_truth(csv_path, teacher)
        n = len(y)
        problems, missing_test = [], []
        for task, rec in truth.items():
            yv = y.get(task)
            if yv is None:
                problems.append(f"{task}: missing in YAML")
                continue
            if abs(round(rec[0], 4) - float(yv["val"])) > 1e-6:
                problems.append(f"{task}: val {yv['val']} != csv-max {round(rec[0], 4)}")
            exp_test = "null" if rec[1] is None else str(round(rec[1], 4))
            if yv["test"] != exp_test:
                problems.append(f"{task}: test {yv['test']} != best-val-row test {exp_test}")
            if rec[1] is None:
                missing_test.append(task)
        ok = (n == 18) and not problems
        all_ok &= ok
        print(f"[{teacher}] tasks={n}/18  assertions={'PASS' if ok else 'FAIL'}")
        if missing_test:
            print(f"   missing best_test_mcc (reported, kept as null): {missing_test}")
        for p in problems:
            print("   PROBLEM:", p)
    print("\nOVERALL:", "ALL PASS" if all_ok else "SOME FAIL")


if __name__ == "__main__":
    main()
