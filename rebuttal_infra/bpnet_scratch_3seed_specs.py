#!/usr/bin/env python3
"""Generate the 54 spec lines for the 3-seed FROM-SCRATCH BPNet baseline sweep.

WHY: the reported BPNet from-scratch baseline (plot_repo/data/formal_5teacher_benchmark.csv,
`Family=BPNet, Type=Baseline`, Std=0.0) is SINGLE-SEED — it has no error bars, while every
distilled student is reported as 3-seed mean+-std. This driver re-runs the from-scratch
baseline with seeds {0,1,2} x 18 tasks = 54 runs so the baseline gets matching error bars.

WHAT "FROM-SCRATCH BPNet" MEANS HERE (verified against the codebase):
  * Student = the `original` BPNet (full receptive field), the SAME student every distilled
    teacher (NT / Enformer / Caduceus / DNABERT-2 / Carbon) distills into. Confirmed by the
    reported baseline's splice_sites_donors = 0.85273, which is the full-RF `original` value
    (~0.85), NOT the capped `deploy_120k` value (~0.63). The `carbon-raw-original` preset
    (config/distillation/experiments/carbon.py:69, :196) is the ONE preset that pairs the
    `original` student (bpnet.py:9, model_size="original") with the standard student recipe.
  * PURE SUPERVISED / NO TEACHER = weight_kl=0 AND weight_mse=0 (only the CE task loss). With
    both zero, `train_distill_task` never forwards the teacher (distill.py `_config_needs_teacher`
    / the needs_logits/needs_features gate), so the run trains the student on labels ALONE.
    weight_ce is kept at 0.5 to match the distilled students' CE weight EXACTLY (the from-scratch
    counterpart = "students' loss minus the teacher terms"). Note the optimizer is AdamW
    (distill_trainer.py:349) with a FLAT lr, so scaling CE by 0.5 vs 1.0 is ~invariant anyway.
  * Recipe (carbon_original_trainer_config -> carbon_trainer_config, trainer.py): epochs=200,
    batch_size=16, lr=1e-4 (AdamW, no scheduler), max_len=1000 — identical to the 3-seed students.

The teacher is still NEEDED as a *checkpoint reference* only: the `distill` entrypoint calls
`find_teacher_checkpoint` (returns None -> skip if absent) and `evaluate_and_log_teacher` (logs
the teacher's own test MCC). Both are satisfied by the warm Carbon LoRA teacher caches
(~/carbon_teachers/carbon_3b_lora/{task}_finetuned/teacher_evaluation.json exist for all 18
tasks), so no 3B teacher forward is needed for the STUDENT's pure-CE training and there is no
"missing teacher checkpoint" error. The student loss is teacher-independent by construction.

Each spec line is consumed verbatim by:  python -m src.train.distill <line>
(`--slurm-config.mode run` == train INLINE on the allocated GPU, NOT a nested submit).

Fresh output-dir (bpnet_scratch_3seed/bpnet) so these runs never collide with the distilled
`original`/`deploy_120k` dirs, and so aggregate_3seed.py --base <that dir> scores ONLY these
54 from-scratch runs (the aggregator keys by task, filters random_state in {0,1,2}).

Usage:
  python rebuttal_infra/bpnet_scratch_3seed_specs.py            # write specs + print summary
  python rebuttal_infra/bpnet_scratch_3seed_specs.py --dry-run  # print only, no file
"""
import argparse
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_SPECS = os.path.join(REPO, "rebuttal_infra", "bpnet_scratch_3seed_specs.txt")

# tyro preset: `original` BPNet student + carbon_original_trainer_config (200ep/bs16/lr1e-4/ml1000).
PRESET = "carbon-raw-original"

# Fresh output root (galaxy SSD) so from-scratch runs are isolated from the distilled dirs and
# aggregate_3seed.py --base points here to score ONLY these 54 runs.
OUT_DIR = "/tmp/galaxy_srv_disk00/pengchx3/bpnet_scratch_3seed/bpnet"

SEEDS = [0, 1, 2]

# Pure-supervised (NO teacher) loss weights. weight_ce=0.5 matches the distilled students' CE
# weight; weight_kl=weight_mse=0 removes ALL teacher terms -> student trains on labels only.
WEIGHT_CE = "0.5"
WEIGHT_KL = "0"
WEIGHT_MSE = "0"

EPOCHS = "200"  # match the 3-seed students' epoch budget (carbon_original_trainer_config default)

# The 18 NT-revised classification tasks (aggregate_3seed.py ORDER).
ALL_18_TASKS = [
    "H3K27me3", "H3K36me3", "H4K20me1", "H2AFZ", "H3K27ac", "H3K4me1", "H3K4me2", "H3K4me3",
    "H3K9ac", "H3K9me3", "promoter_all", "promoter_tata", "promoter_no_tata", "enhancers",
    "enhancers_types", "splice_sites_all", "splice_sites_acceptors", "splice_sites_donors",
]


def build_spec(task, seed):
    """Return the one-line `python -m src.train.distill` spec for a (task, seed) from-scratch run."""
    return (
        f"{PRESET} --task-names {task} "
        f"--distillation-config.weight-ce {WEIGHT_CE} "
        f"--distillation-config.weight-kl {WEIGHT_KL} "
        f"--distillation-config.weight-mse {WEIGHT_MSE} "
        f"--trainer-config.epochs {EPOCHS} "
        f"--trainer-config.early-stop-patience 0 "
        f"--trainer-config.output-dir {OUT_DIR} "
        f"--random-state {seed} "
        f"--slurm-config.mode run"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="print only, do not write specs file")
    args = ap.parse_args()

    # Task-major, seed-minor: lines 1-3 = task0 seeds{0,1,2}, lines 4-6 = task1, ...
    lines = [build_spec(task, seed) for task in ALL_18_TASKS for seed in SEEDS]

    print(f"# {len(ALL_18_TASKS)} tasks x {len(SEEDS)} seeds = {len(lines)} from-scratch runs")
    print(f"# preset={PRESET} (original BPNet student, 200ep/bs16/lr1e-4/ml1000)")
    print(f"# pure-supervised: weight_ce={WEIGHT_CE} weight_kl={WEIGHT_KL} weight_mse={WEIGHT_MSE}")
    print(f"# output-dir={OUT_DIR}")

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
