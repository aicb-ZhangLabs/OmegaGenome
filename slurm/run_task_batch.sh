#!/bin/bash
# Per-task BATCH distillation driver: load the Carbon-3B teacher ONCE for a task and sweep
# all of that task's HP configs in a SINGLE process (amortizing the ~3-5 min/config
# 3B-load + LoRA-merge + teacher-eval over the task's ~32 configs).
#
# This is the H100 (/root/run_all.sh) replacement: one process per task instead of one
# process per config. It greps the task's spec lines out of an HP spec file and hands them
# to src.train.distill_task, which calls prepare_task ONCE then train_student per config.
#
# It does NOT change the SLURM per-config path: src.train.distill is untouched.
#
# Usage:
#   slurm/run_task_batch.sh <task> [config_list_file] [base_experiment]
#     <task>             e.g. H3K27ac
#     config_list_file   default: hp_original_stage1.txt
#     base_experiment    default: carbon-raw-original
#
# Drive ALL 7 laggard tasks (sequential, one teacher load each), e.g. on the H100 box:
#   for t in splice_sites_all splice_sites_acceptors splice_sites_donors enhancers \
#            enhancers_types H3K27ac H3K4me1; do
#     slurm/run_task_batch.sh "$t" hp_original_stage1.txt; done
# (or run each in its own GPU process / background to parallelize across GPUs.)

set -euo pipefail
# Derive REPO from this script's own location (slurm/<this>) so the launcher is portable:
# works from the lab repo (/home/.../code_carbon) AND the non-SLURM vast.ai box
# (/root/omega_carbon) without editing the path.
_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$_SCRIPT_DIR/.." && pwd)"
cd "$REPO"

TASK="${1:?usage: run_task_batch.sh <task> [config_list_file] [base_experiment]}"
CONFIG_LIST="${2:-hp_original_stage1.txt}"
BASE_EXPERIMENT="${3:-carbon-raw-original}"

# Reuse the shared SLURM env (HF cache, OUTPUT_PATH, wandb, portable venv) if present.
# On the vast.ai box this file is absent -> skipped, which is fine.
if [ -f slurm/env_setup.sh ]; then
  source slurm/env_setup.sh
fi
# Node-aware SSD HF override (mirrors carbon_distill.sbatch) ONLY when a galaxy SSD mount
# exists. On the vast.ai box neither path exists, so we leave HF cache at its default
# (public Carbon-3B needs no special cache location).
for _b in /tmp/galaxy_srv_disk00/pengchx3 /srv/disk00/sshfs/pengchx3; do
  [ -d "$_b" ] && { _SSD="$_b"; break; }
done
if [ -n "${_SSD:-}" ]; then
  export HF_OVERRIDE="$_SSD/hf_cache_shared"
  export HF_HOME="$HF_OVERRIDE" HF_HUB_CACHE="$HF_OVERRIDE/hub" \
         HUGGINGFACE_HUB_CACHE="$HF_OVERRIDE/hub" HF_DATASETS_CACHE="$HF_OVERRIDE/datasets" \
         TRANSFORMERS_CACHE="$HF_OVERRIDE/hub" HF_ASSETS_CACHE="$HF_OVERRIDE/assets"
  mkdir -p "$HF_OVERRIDE/hub" "$HF_OVERRIDE/datasets" "$HF_OVERRIDE/assets"
fi
# Lab HF token (gated artifacts) ONLY if present. The public Carbon-3B teacher needs no
# token, so on the box (no token file) we simply run unauthenticated.
for _tok in /home/pengchx3/text-dna/huggingface-token-0616.txt "$REPO/.hf_token"; do
  [ -f "$_tok" ] && { export HF_TOKEN="$(cat "$_tok")"; break; }
done
export WANDB_INIT_TIMEOUT=300
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Prefer the repo's portable venv if present; else fall back to whatever python is on PATH
# (the vast.ai box may use a system/conda python instead of .venv_carbon_portable).
if [ -x "$REPO/.venv_carbon_portable/bin/python" ]; then
  PY="$REPO/.venv_carbon_portable/bin/python"
else
  PY="$(command -v python3 || command -v python)"
fi

N=$(grep -c -- "--task-names $TASK " "$CONFIG_LIST" || true)
echo "=== batch distill task=$TASK : $N configs from $CONFIG_LIST (single teacher load) ==="
"$PY" -m src.train.distill_task \
  --task "$TASK" \
  --config-list "$CONFIG_LIST" \
  --base-experiment "$BASE_EXPERIMENT"
echo "=== done: task=$TASK ==="
