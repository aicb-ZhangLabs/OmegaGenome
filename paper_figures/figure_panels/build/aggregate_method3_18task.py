"""
Aggregate the 3-method x 18-task x 3-seed distillation sweep into a tidy CSV.

Reads every final_summary.json under
  ${OG_SCRATCH}/method3_18task/<method>/<task>/<ts>/<uuid>/<config>/
for method in {dkd, dist, logit_standard}, computes 3-seed mean +/- std of
best_test_mcc (best-val model selection, same rule as the paper benchmark).

The 4th "method" is OmegaGenome (vanilla): its 18 per-task values are the NT
distilled-student column of data/model_comparison_5teacher_formal.csv
(Model == "Distilled Nucleotide Transformer", Type == "Student"). All four
methods share the NT-2.5B teacher + vanilla-best HP.

Writes data/method_comparison_18task.csv with columns:
  task, method in {OmegaGenome,DKD,DIST,LS}, mcc_mean, mcc_std, n_seeds
and prints the 18-task mean per method (mean over the 18 per-task means) with
its across-task s.d., plus sanity checks.
"""
import glob
import json
import os
import numpy as np
import pandas as pd

ROOT = os.environ.get("OG_SCRATCH", "output") + "/method3_18task"
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
CSV5T = os.path.join(REPO, "data", "model_comparison_5teacher_formal.csv")
OUT = os.path.join(REPO, "data", "method_comparison_18task.csv")

METHOD_LABEL = {"dkd": "DKD", "dist": "DIST", "logit_standard": "LS"}
TASKS = sorted(os.listdir(os.path.join(ROOT, "dkd")))

# ---- 1) new sweep runs -----------------------------------------------------
rows = []
for mkey, mlabel in METHOD_LABEL.items():
    for task in TASKS:
        files = glob.glob(os.path.join(ROOT, mkey, task, "*", "*", "*", "final_summary.json"))
        seed2mcc = {}
        for f in files:
            with open(f) as fh:
                d = json.load(fh)
            seed2mcc[d["random_state"]] = d["best_test_mcc"]
        vals = np.array(list(seed2mcc.values()), dtype=float)
        rows.append({
            "task": task,
            "method": mlabel,
            "mcc_mean": float(vals.mean()),
            "mcc_std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
            "n_seeds": int(len(vals)),
        })
        if len(vals) < 3:
            print(f"  [MISSING] {mlabel} / {task}: only {len(vals)} seeds "
                  f"(seeds present: {sorted(seed2mcc)})")

# ---- 2) OmegaGenome = NT distilled-student column --------------------------
c5 = pd.read_csv(CSV5T)
og = c5[(c5["Model"] == "Distilled Nucleotide Transformer") & (c5["Type"] == "Student")]
og = og.set_index("Task")
for task in TASKS:
    if task not in og.index:
        print(f"  [WARN] OmegaGenome NT column has no row for task {task}")
        continue
    r = og.loc[task]
    rows.append({
        "task": task,
        "method": "OmegaGenome",
        "mcc_mean": float(r["Score"]),
        "mcc_std": float(r["Std"]),
        "n_seeds": 3,  # NT benchmark = 3 seeds
    })

df = pd.DataFrame(rows)
# order: method category then task
mcat = pd.Categorical(df["method"], ["OmegaGenome", "DKD", "DIST", "LS"], ordered=True)
df = df.assign(_m=mcat).sort_values(["_m", "task"]).drop(columns="_m").reset_index(drop=True)
df.to_csv(OUT, index=False)
print(f"\nWrote {OUT}  ({len(df)} rows)")

# ---- 3) 18-task means + sanity checks --------------------------------------
print("\n=== 18-task mean per method (mean over 18 per-task means) ===")
summary = {}
for m in ["OmegaGenome", "DKD", "DIST", "LS"]:
    sub = df[df["method"] == m].sort_values("task")
    means = sub["mcc_mean"].values
    summary[m] = (means.mean(), means.std(ddof=1))
    print(f"  {m:12s}  {means.mean():.4f} +/- {means.std(ddof=1):.4f}  "
          f"(n_tasks={len(means)})")

print("\n=== splice_sites_all sanity check (paper Panel-A: "
      "OG .884 / DKD .880 / DIST .864 / LS .863) ===")
for m in ["OmegaGenome", "DKD", "DIST", "LS"]:
    v = df[(df["method"] == m) & (df["task"] == "splice_sites_all")]["mcc_mean"].values
    print(f"  {m:12s} splice_sites_all = {v[0]:.4f}" if len(v) else f"  {m}: MISSING")

print("\n=== MCC range sanity (all per-task means in [0,1]) ===")
bad = df[(df["mcc_mean"] < 0) | (df["mcc_mean"] > 1)]
print("  all in [0,1]" if bad.empty else f"  OUT OF RANGE:\n{bad}")

# wide pivot for readability
print("\n=== per-task table (mean) ===")
wide = df.pivot(index="task", columns="method", values="mcc_mean")[
    ["OmegaGenome", "DKD", "DIST", "LS"]]
with pd.option_context("display.float_format", lambda x: f"{x:.4f}"):
    print(wide)
