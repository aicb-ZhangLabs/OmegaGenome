#!/usr/bin/env python3
"""Generate spec lines for the student-SIZE scaling sweep across the 17 non-splice_sites_all tasks.

PURPOSE
  Turn Fig 6B (student-size scaling) from a splice_sites_all-only curve into an 18-task-mean
  curve. splice_sites_all was already swept across sizes (the existing SIZE_CSV / XL_CSV the
  plot reads); this sweep adds the OTHER 17 tasks at the SAME 6 non-`original` sizes, so the
  panel can average all 18 tasks per size. NT-2.5B teacher, per-task best NT HP, 3 seeds.

  6 sizes x 17 tasks x 3 seeds = 306 runs, one spec line per run, consumed verbatim by:
      python -m src.train.distill <line>
  (`--slurm-config.mode run` => train INLINE on the allocated GPU, NOT a nested submit --
  identical CLI to method3_18task_specs.py / rerun_below_baseline_specs.py.)

SIZES (student model_size config strings — the ACTUAL StudentConfig-accepted values, from
src/model/bpnet_classifier.py:45 Literal + the _create_bpnet_backbone branches). The plot
(plot_repo/.../fig6_method_size_hp.py Panel B) remaps ultra_tiny->ultra_small and
extra_tiny->extra_small for display, so the config strings are the RAW ones:
  pico            (7 ch,  ~2K params)   -> plot label "2K"
  ultra_tiny      (14 ch, ~7K)          -> plot label "ultra_small" / "7K"
  extra_tiny      (30 ch, ~28K)         -> plot label "extra_small" / "28K"
  extra_large_fix (170 ch, ~0.8M)       -> plot label "extra_large_fix" / "0.8M"
  large           (256 ch, ~1.8M)       -> plot label "large" / "1.8M"
  xxlarge         (363 ch, ~3.6M)       -> plot label "xxlarge" / "3.6M"
  * `original` (0.1M, 64 ch) is SKIPPED — it is the default benchmark NT student already run on
    all 18 tasks (the reported vanilla NT 3-seed); the curve's 0.1M point reuses those.

EXTRA_LARGE_FIX — the architecture fix (NOT a flag we pass, it is a distinct model_size string):
  The plain `extra_large` branch caps dilation at 2**min(i,6) (max dilation 64), starving the
  receptive field -> collapse / spurious 0.86 dip (bpnet_classifier.py:482-498, and
  config/best_hyperparams.py:528 marks it "<-- BROKEN (dilation cap 6)"). The FIXED branch
  `extra_large_fix` (bpnet_classifier.py:500-530) is the SAME 170-channel width but with
  UNCAPPED dilation 2**i (2,4,...,512), matching original BPNet. We therefore use
  model_size=extra_large_fix and NEVER submit the buggy `extra_large`.

HP SOURCE
  Per-task best NT HP (weight_ce/weight_kl/weight_mse/temperature) from
      best_hp_all_teachers/best_hp/best_hp_nt.csv
  (authoritative NT best-HP table; same source rerun_below_baseline / method3 use), applied
  ACROSS ALL SIZES for that task (the CSV's own model_size column is ignored — we sweep it).
  Vanilla KD (no method toggle). lr/batch_size are preset defaults (bs16, lr1e-4, max_len1000).

PROTOCOL (matches reported 3-seed EXCEPT longer + explicit):
  epochs=200, early-stop-patience=0 (NO early stop), seeds {0,1,2}, best-val-MCC ckpt is the
  trainer default (src/trainer/distill_trainer.py:443-458).

ORDERING (seed-outer -> task-middle -> size-inner): seed 0 across ALL (task x size), THEN seed 1,
  THEN seed 2. If the array is cut short, seed 0 already yields a full 17-task x 6-size picture.

TEACHER CACHE: the data split is built from config.dataset_config.random_state (fixed 42;
  distill.py:1595 overrides only task_name, NOT the seed) and does NOT depend on model_size
  (student-only) — so the on-disk NT teacher cache (data/cache/2b5-.../{task}/) is valid+reused
  for EVERY (task,size,seed). Logit-only tasks (weight_mse=0) skip the 3B load entirely; tasks
  with weight_mse>0 still forward the teacher for features (features_computed=false on disk),
  identical to method3/vanilla — size/seed-independent either way.

Usage:
  python rebuttal_infra/size18task_specs.py            # write specs + print table
  python rebuttal_infra/size18task_specs.py --dry-run  # print only, no file
"""
import argparse
import csv
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BEST_HP_CSV = os.path.join(REPO, "best_hp_all_teachers", "best_hp", "best_hp_nt.csv")
OUT_SPECS = os.path.join(REPO, "rebuttal_infra", "size18task_specs.txt")

SSD = "/tmp/galaxy_srv_disk00/pengchx3"
OUT_BASE = f"{SSD}/size18task"          # fresh, distinct output root (no collision)

PRESET = "nt_vanilla"                    # NT-2.5B teacher preset; vanilla KD.
SEEDS = [0, 1, 2]

# model_size config strings, small -> large (size-inner order). extra_large_fix = FIXED
# uncapped-dilation 0.8M variant (NOT the buggy `extra_large`).
SIZES = ["pico", "ultra_tiny", "extra_tiny", "extra_large_fix", "large", "xxlarge"]

# 17 tasks = all 18 EXCEPT splice_sites_all (already swept across sizes).
TASKS_17 = [
    "H2AFZ", "H3K27ac", "H3K27me3", "H3K36me3", "H3K4me1", "H3K4me2", "H3K4me3",
    "H3K9ac", "H3K9me3", "H4K20me1", "promoter_all", "promoter_no_tata", "promoter_tata",
    "enhancers", "enhancers_types", "splice_sites_acceptors", "splice_sites_donors",
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


def build_spec(size, task, seed, hp):
    """One `python -m src.train.distill` spec line for (size, task, seed) with per-task best NT HP."""
    ce, kl, mse, temp = hp
    outdir = f"{OUT_BASE}/{task}/{size}/seed{seed}"
    return (
        f"{PRESET} --task-names {task} "
        f"--student-config.model-size {size} "
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

    hp = load_best_hp()
    for t in TASKS_17:
        if t not in hp:
            raise SystemExit(f"no best-HP row for {t} in {BEST_HP_CSV}")

    # seed-outer -> task-middle -> size-inner
    lines = []
    for seed in SEEDS:
        for task in TASKS_17:
            for size in SIZES:
                lines.append(build_spec(size, task, seed, hp[task]))

    assert len(lines) == len(SEEDS) * len(TASKS_17) * len(SIZES) == 306, len(lines)

    print(f"# {len(SIZES)} sizes x {len(TASKS_17)} tasks x {len(SEEDS)} seeds = {len(lines)} runs")
    print(f"# ordering: seed-outer -> task-middle -> size-inner (seed 0 block = lines 1..102)")
    print(f"# sizes (small->large): {SIZES}")
    print(f"# HP source: {BEST_HP_CSV} (per-task best NT HP; applied across all sizes)")
    print(f"# output root: {OUT_BASE}/<task>/<size>/seed<N>/")
    print(f"# {'task':24}{'ce,kl,mse,T':20}")
    for task in TASKS_17:
        ce, kl, mse, temp = hp[task]
        print(f"# {task:24}{','.join([ce, kl, mse, temp]):20}")

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
