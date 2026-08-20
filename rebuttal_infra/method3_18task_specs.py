#!/usr/bin/env python3
"""Generate spec lines for the 3-KD-method x 18-task x 3-seed distillation sweep (NT-2.5B teacher).

PURPOSE
  Produce REAL per-method 18-task means for the three non-OmegaGenome KD methods
  (DKD, DIST, Logit-Standardization) to replace a fabricated rebuttal column.

  3 methods x 18 tasks x 3 seeds = 162 runs, one spec line per run, consumed verbatim by:
      python -m src.train.distill <line>
  (`--slurm-config.mode run` => train INLINE on the allocated GPU, NOT a nested submit --
  identical CLI to the reported 3-seed and to rerun_below_baseline_specs.py.)

METHOD CONFIG (exact, from config/distillation/experiments/nt_method_hyperparam.py +
src/model/distillation.py DistillationModelConfig; verified vs rebuttal_infra/dkd/runs/dkd_nt.json):
  * dkd            -> distill_method=dkd,  weight_mse=0 (DKD has no feature-matching term),
                     dkd_alpha=2.0, dkd_beta=4.0  (method-search best DKD params from the DKD
                     rebuttal, rebuttal_infra/dkd/runs/dkd_nt.json -- the only DKD-specific tuning
                     the paper's method search produced).
  * dist           -> distill_method=dist, weight_mse=0 (DIST uses inter/intra correlation, no MSE).
  * logit_standard -> distill_method=logit_standard, keeps the per-task vanilla weight_mse
                     (Logit-Standardization is a drop-in z-scored-KL replacement; the method-search
                     COMMON grid keeps the feature-MSE term).

HP SOURCE
  The paper's method search (nt_method_hyperparam.py) swept ONE task (splice_sites_donors) only,
  so there is NO per-task best-config for these methods across all 18 tasks. Per the standing
  instruction ("otherwise use the same base HP as OmegaGenome/vanilla for that task with the
  method's objective toggled on") we take each task's VANILLA best HP -- weight_ce, weight_kl,
  temperature (and weight_mse for logit_standard) -- from
      best_hp_all_teachers/best_hp/best_hp_nt.csv
  (authoritative NT best-HP table; same source rerun_below_baseline_specs.py uses) and toggle the
  method on. lr/batch_size/student are preset defaults (bs16, lr1e-4, original BPNet, max_len1000),
  identical to the reported vanilla NT students.

PROTOCOL (matches reported 3-seed EXCEPT longer + explicit, same as rerun_below_baseline):
  epochs=200, early-stop-patience=0 (NO early stop), seeds {0,1,2}, best-val-MCC ckpt is the
  trainer default (src/trainer/distill_trainer.py:443-458).

TEACHER CACHE CAVEAT (see report): the on-disk NT teacher logits/features cache
  (.../data/cache/2b5-.../{task}/) is keyed on train ORDER and only validates for random_state=42.
  Seeds 0/1/2 MISS the cache, so each run loads + forwards the NT-2.5B teacher unless the per-(task,
  seed) caches are pre-built first.

Usage:
  python rebuttal_infra/method3_18task_specs.py            # write specs + print table
  python rebuttal_infra/method3_18task_specs.py --dry-run  # print only, no file
"""
import argparse
import csv
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BEST_HP_CSV = os.path.join(REPO, "best_hp_all_teachers", "best_hp", "best_hp_nt.csv")
OUT_SPECS = os.path.join(REPO, "rebuttal_infra", "method3_18task_specs.txt")

SSD = "/tmp/galaxy_srv_disk00/pengchx3"
OUT_BASE = f"{SSD}/method3_18task"      # fresh, distinct output root (no collision)

PRESET = "nt_vanilla"                    # NT-2.5B teacher preset; method toggled below.
SEEDS = [0, 1, 2]
METHODS = ["dkd", "dist", "logit_standard"]
DKD_ALPHA = 2.0                          # method-search best (rebuttal_infra/dkd/runs/dkd_nt.json)
DKD_BETA = 4.0

ALL_18_TASKS = [
    "H2AFZ", "H3K27ac", "H3K27me3", "H3K36me3", "H3K4me1", "H3K4me2", "H3K4me3",
    "H3K9ac", "H3K9me3", "H4K20me1", "promoter_all", "promoter_tata", "promoter_no_tata",
    "enhancers", "enhancers_types", "splice_sites_all", "splice_sites_acceptors",
    "splice_sites_donors",
]


def load_best_hp():
    """Return {task: (weight_ce, weight_kl, weight_mse, temperature)} from best_hp_nt.csv (str)."""
    out = {}
    with open(BEST_HP_CSV) as f:
        for row in csv.DictReader(f):
            out[row["task"]] = (
                row["weight_ce"], row["weight_kl"], row["weight_mse"], row["temperature"],
            )
    return out


def build_spec(method, task, seed, hp):
    """One `python -m src.train.distill` spec line for (method, task, seed) with vanilla-best HP."""
    ce, kl, mse, temp = hp
    outdir = f"{OUT_BASE}/{method}"
    mse_use = "0" if method in ("dkd", "dist") else mse   # dkd/dist logit-only; ls keeps vanilla mse
    line = (
        f"{PRESET} --task-names {task} "
        f"--distillation-config.distill-method {method} "
        f"--distillation-config.weight-ce {ce} "
        f"--distillation-config.weight-kl {kl} "
        f"--distillation-config.weight-mse {mse_use} "
        f"--distillation-config.temperature {temp} "
    )
    if method == "dkd":
        line += (
            f"--distillation-config.dkd-alpha {DKD_ALPHA} "
            f"--distillation-config.dkd-beta {DKD_BETA} "
        )
    line += (
        f"--trainer-config.epochs 200 "
        f"--trainer-config.early-stop-patience 0 "
        f"--trainer-config.output-dir {outdir} "
        f"--random-state {seed} "
        f"--slurm-config.mode run"
    )
    return line


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="print only, do not write specs file")
    args = ap.parse_args()

    hp = load_best_hp()
    for t in ALL_18_TASKS:
        if t not in hp:
            raise SystemExit(f"no best-HP row for {t} in {BEST_HP_CSV}")

    lines = []   # method-major: all dkd (54), then dist (54), then logit_standard (54)
    for method in METHODS:
        for task in ALL_18_TASKS:
            for seed in SEEDS:
                lines.append(build_spec(method, task, seed, hp[task]))

    print(f"# {len(METHODS)} methods x {len(ALL_18_TASKS)} tasks x {len(SEEDS)} seeds = {len(lines)} runs")
    print(f"# HP source: {BEST_HP_CSV} (per-task vanilla best; method toggled on)")
    print(f"# output root: {OUT_BASE}/<method>/  (dkd|dist|logit_standard)")
    print(f"# {'method':16}{'task':24}{'ce,kl,mse,T (mse=0 for dkd/dist)':34}")
    for method in METHODS:
        for task in ALL_18_TASKS:
            ce, kl, mse, temp = hp[task]
            mse_use = "0" if method in ("dkd", "dist") else mse
            extra = f"  alpha={DKD_ALPHA},beta={DKD_BETA}" if method == "dkd" else ""
            print(f"# {method:16}{task:24}{','.join([ce,kl,mse_use,temp]):34}{extra}")

    if not args.dry_run:
        with open(OUT_SPECS, "w") as f:
            f.write("\n".join(lines) + "\n")
        print(f"\nwrote {len(lines)} spec lines -> {OUT_SPECS}")
    else:
        print("\n--- spec lines (dry-run, not written) ---")
        for ln in lines:
            print(ln)


if __name__ == "__main__":
    main()
