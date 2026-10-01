#!/usr/bin/env python3
"""Generate spec lines for the TARGETED HP-fix rerun of the 9 DIST/LS cells that COLLAPSED
in the method3 sweep (they reused OmegaGenome's vanilla HP; diagnosis: weight_kl=1.0 traps a
constant-prediction basin, weight_kl<=0.5 escapes it).

GOAL
  Fix exactly those 9 cells with ONE known-good config (no kl sweep) so the 3-method comparison
  is fair. The config -- weight_kl=0.5, temperature=4, weight_ce=1.0 (CE >= KD) -- is the one that
  let splice_sites_donors reach 0.95. Everything else mirrors method3_18task_specs.py exactly.

  9 cells x 1 config x 3 seeds = 27 runs, one spec line per run, consumed verbatim by:
      python -m src.train.distill <line>
  (`--slurm-config.mode run` => train INLINE on the allocated GPU, identical CLI to method3.)

THE 9 COLLAPSED CELLS
  * dist            (weight_mse=0, inter/intra correlation loss):
      H3K27ac, H3K27me3, H3K4me1, H3K9me3, enhancers, splice_sites_acceptors
  * logit_standard  (z-scored KL; keeps the task's VANILLA weight_mse -- which is 0 for all 3):
      H3K9me3, splice_sites_acceptors, splice_sites_donors

FIXED CONFIG (all 9 cells, all seeds)
  weight_ce=1.0, weight_kl=0.5, temperature=4, weight_mse=<vanilla for LS / 0 for DIST>.
  lr/batch_size/student are preset defaults (bs16, lr1e-4, original BPNet, max_len1000), identical
  to method3 / the reported vanilla NT students.

PROTOCOL (identical to method3): epochs=200, early-stop-patience=0 (NO early stop), seeds {0,1,2},
best-val-MCC ckpt is the trainer default.

TEACHER CACHE: NT-2.5B teacher logits/features cache is reused (same as method3); verified present
for all 9 tasks under the project_path cache. weight_mse=0 => logit-only, features not required.

Usage:
  python analysis/method_hpfix_specs.py            # write specs + print table
  python analysis/method_hpfix_specs.py --dry-run  # print only, no file
"""

import argparse
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_SPECS = os.path.join(REPO, "analysis", "method_hpfix_specs.txt")

SSD = os.environ.get("OG_SCRATCH", "output")
OUT_BASE = f"{SSD}/method_hpfix"  # fresh, distinct output root (no overlap with method3/size)

PRESET = "nt_vanilla"  # NT-2.5B teacher preset; method toggled below.
SEEDS = [0, 1, 2]

# Fixed known-good config (the fix). Diagnosis: kl<=0.5 escapes the constant-prediction basin.
WEIGHT_CE = "1.0"
WEIGHT_KL = "0.5"
TEMPERATURE = "4"

# The 9 collapsed cells, method-major then task order. weight_mse per cell:
#   dist -> 0 (logit-only correlation loss); logit_standard -> that task's VANILLA weight_mse
#   (best_hp_nt.csv: H3K9me3=0, splice_sites_acceptors=0, splice_sites_donors=0 -> all 0).
CELLS = [
    ("dist", "H3K27ac", "0"),
    ("dist", "H3K27me3", "0"),
    ("dist", "H3K4me1", "0"),
    ("dist", "H3K9me3", "0"),
    ("dist", "enhancers", "0"),
    ("dist", "splice_sites_acceptors", "0"),
    ("logit_standard", "H3K9me3", "0"),
    ("logit_standard", "splice_sites_acceptors", "0"),
    ("logit_standard", "splice_sites_donors", "0"),
]


def build_spec(method, task, mse, seed):
    """One `python -m src.train.distill` spec line for (method, task, seed) with the fix HP.

    Output dir: {OUT_BASE}/{method}/{task}/seed{seed} (per-seed clean separation; the trainer
    additionally nests {task}/{uuid}/{hp} beneath, so runs never collide).
    """
    outdir = f"{OUT_BASE}/{method}/{task}/seed{seed}"
    return (
        f"{PRESET} --task-names {task} "
        f"--distillation-config.distill-method {method} "
        f"--distillation-config.weight-ce {WEIGHT_CE} "
        f"--distillation-config.weight-kl {WEIGHT_KL} "
        f"--distillation-config.weight-mse {mse} "
        f"--distillation-config.temperature {TEMPERATURE} "
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

    lines = []  # cell-major (method,task), seeds interleaved
    for method, task, mse in CELLS:
        for seed in SEEDS:
            lines.append(build_spec(method, task, mse, seed))

    print(f"# {len(CELLS)} cells x 1 config x {len(SEEDS)} seeds = {len(lines)} runs")
    print(
        f"# fix HP: weight_ce={WEIGHT_CE} weight_kl={WEIGHT_KL} temperature={TEMPERATURE} "
        f"(mse=0 dist / vanilla LS)"
    )
    print(f"# output root: {OUT_BASE}/<method>/<task>/seed<N>/")
    print(f"# {'idx':4}{'method':16}{'task':26}{'ce,kl,mse,T':16}")
    i = 0
    for method, task, mse in CELLS:
        for seed in SEEDS:
            print(
                f"# {i:<4}{method:16}{task:26}{WEIGHT_CE},{WEIGHT_KL},{mse},{TEMPERATURE}  seed{seed}"
            )
            i += 1

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
