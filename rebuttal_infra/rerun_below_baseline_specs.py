#!/usr/bin/env python3
"""Generate the rerun spec lines for the below-baseline / DNABERT-2 completion sweep.

RERUN SET = 27 (teacher, task) cells x 3 seeds = 81 runs, one spec line per run:

  1. ALL 18 DNABERT-2 tasks. The `dna_bert_v2` preset uniquely used the generic
     trainer_config (epochs=100, config/distillation/trainer.py:13) while every other
     teacher trained 200 epochs -> the DNABERT-2 students were undertrained AND compared
     against a 200-epoch from-scratch baseline. Rerunning all 18 at 200 epochs removes that
     inconsistency (not just the 5 below-baseline DNABERT-2 cells).
  2. 9 non-DNABERT-2 below-baseline pairs (see NON_DNABERT below).

Protocol for every cell (matches the reported 3-seed protocol EXCEPT longer + explicit):
  * best hyperparameters read from best_hp_all_teachers/best_hp/best_hp_<teacher>.csv
    (authoritative best-HP table; verified to match the reported runs' final_summary.json).
  * epochs=200, early-stop-patience=0 (NO early stopping).
  * seeds 0,1,2 (the seed set the 3-seed aggregator selects).
  * the trainer already REPORTS the best-val-MCC checkpoint's test MCC
    (src/trainer/distill_trainer.py:443-458 + aggregate_3seed.py reads `best_test_mcc`),
    so "select best-val checkpoint" needs no code change.

Each spec line is consumed verbatim by:  python -m src.train.distill <line>
(the same CLI the original 3-seed used; `--slurm-config.mode run` = train INLINE on the
allocated GPU, NOT a nested submit -- see slurm/carbon_distill.sbatch:57).

Usage:
  python rebuttal_infra/rerun_below_baseline_specs.py            # write specs + print table
  python rebuttal_infra/rerun_below_baseline_specs.py --dry-run  # print only, no file
"""
import argparse
import csv
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BEST_HP_DIR = os.path.join(REPO, "best_hp_all_teachers", "best_hp")
OUT_SPECS = os.path.join(REPO, "rebuttal_infra", "rerun_below_baseline_specs.txt")
# Fresh output root so the reruns do NOT overwrite the original 3-seed dirs and the
# aggregator can score them per-teacher (aggregate keys by task only, so keep teachers apart).
SSD = "/tmp/galaxy_srv_disk00/pengchx3"
OUT_BASE = f"{SSD}/rerun_below_baseline"
SEEDS = [0, 1, 2]

# teacher_key -> tyro preset name in the `python -m src.train.distill` config registry.
PRESET = {
    "dnabert2": "dna_bert_v2",
    "caduceus": "caduceus_vanilla",
    "enformer": "enformer_vanilla",
    "nt": "nt_vanilla",
    "carbon": "carbon-raw-original",
}

ALL_18_TASKS = [
    "H2AFZ", "H3K27ac", "H3K27me3", "H3K36me3", "H3K4me1", "H3K4me2", "H3K4me3",
    "H3K9ac", "H3K9me3", "H4K20me1", "promoter_all", "promoter_tata", "promoter_no_tata",
    "enhancers", "enhancers_types", "splice_sites_all", "splice_sites_acceptors",
    "splice_sites_donors",
]

# The 9 non-DNABERT-2 below-baseline pairs (teacher_key, task).
NON_DNABERT = [
    ("caduceus", "splice_sites_donors"),
    ("carbon", "H3K4me3"),
    ("caduceus", "H3K9ac"),
    ("caduceus", "splice_sites_acceptors"),
    ("caduceus", "H3K4me3"),
    ("enformer", "H3K9ac"),
    ("carbon", "H3K9ac"),
    ("caduceus", "promoter_tata"),
    ("nt", "H3K4me3"),
]

# The 27 (teacher, task) cells = all 18 DNABERT-2 + the 9 above.
CELLS = [("dnabert2", t) for t in ALL_18_TASKS] + NON_DNABERT


def load_best_hp(teacher_key):
    """Return {task: (ce, kl, mse, temperature)} from best_hp_<teacher>.csv (str values)."""
    path = os.path.join(BEST_HP_DIR, f"best_hp_{teacher_key}.csv")
    out = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            out[row["task"]] = (
                row["weight_ce"], row["weight_kl"], row["weight_mse"], row["temperature"],
            )
    return out


def build_spec(teacher_key, task, seed, hp):
    ce, kl, mse, temp = hp
    preset = PRESET[teacher_key]
    outdir = f"{OUT_BASE}/{teacher_key}"
    return (
        f"{preset} --task-names {task} "
        f"--distillation-config.weight-ce {ce} "
        f"--distillation-config.weight-kl {kl} "
        f"--distillation-config.weight-mse {mse} "
        f"--distillation-config.temperature {temp} "
        f"--trainer-config.epochs 200 "
        f"--trainer-config.early-stop-patience 0 "
        f"--trainer-config.output-dir {outdir} "
        f"--random-state {seed} "
        f"--slurm-config.mode run"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="print only, do not write specs file")
    args = ap.parse_args()

    hp_cache = {tk: load_best_hp(tk) for tk in PRESET}
    lines = []
    table = []
    for teacher_key, task in CELLS:
        hp = hp_cache[teacher_key].get(task)
        if hp is None:
            raise SystemExit(f"no best-HP row for {teacher_key}/{task} in best_hp_{teacher_key}.csv")
        table.append((teacher_key, task, hp))
        for seed in SEEDS:
            lines.append(build_spec(teacher_key, task, seed, hp))

    print(f"# {len(CELLS)} cells x {len(SEEDS)} seeds = {len(lines)} runs")
    print(f"# {'teacher':10}{'task':24}{'ce,kl,mse,T':20}preset")
    for tk, task, hp in table:
        print(f"# {tk:10}{task:24}{','.join(hp):20}{PRESET[tk]}")

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
