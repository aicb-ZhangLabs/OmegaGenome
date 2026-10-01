#!/bin/bash
# SEED-0 PRIORITY split of the student-SIZE scaling sweep (was loom_size18task / 253021, all-3-seeds).
# This array runs ONLY the not-yet-done seed0 cells so the seed-0 curve completes first; seeds 1-2
# live in the sibling size_seed12.sbatch (submitted then whole-job scontrol-held). Env, resources,
# and spec-reading logic mirror size18task.sbatch EXACTLY. Reads size_seed0_specs.txt (the /seed0
# lines of size18task_specs.txt with completed (task,size,seed0) cells removed). Same /seed0/ output
# dirs => already-done + new runs merge cleanly for aggregation.
#
#   sbatch analysis/size_seed0.sbatch    # remaining seed0, 7 concurrent, runs now
#
#SBATCH --job-name=loom_size-seed0
#SBATCH --array=0-74%7
#SBATCH --nodes=1
#SBATCH --partition=zhanglab.p
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=64G
#SBATCH --requeue
#SBATCH --time=12:00:00
#SBATCH --output=slurm-%x-%A_%a.out

set -euo pipefail
REPO=${OG_ROOT:-$PWD}
cd "$REPO"
source slurm/env_setup.sh

# ---- HF cache on the node-aware SSD (identical to size18task.sbatch / method3_18task.sbatch) ----
for _b in ${OG_SCRATCH:-$PWD/output} ${OG_SCRATCH:-$PWD/output}; do [ -d "$_b" ] && { _SSD="$_b"; break; }; done
[ -n "${_SSD:-}" ] && export HF_OVERRIDE="$_SSD/hf_cache_shared"
if [ -n "${HF_OVERRIDE:-}" ]; then
  export HF_HOME="$HF_OVERRIDE" HF_HUB_CACHE="$HF_OVERRIDE/hub" HUGGINGFACE_HUB_CACHE="$HF_OVERRIDE/hub" \
         HF_DATASETS_CACHE="$HF_OVERRIDE/datasets" TRANSFORMERS_CACHE="$HF_OVERRIDE/hub" HF_ASSETS_CACHE="$HF_OVERRIDE/assets"
  mkdir -p "$HF_OVERRIDE/hub" "$HF_OVERRIDE/datasets" "$HF_OVERRIDE/assets"
fi
export HF_TOKEN="$(cat ${HF_TOKEN_FILE})"
export WANDB_INIT_TIMEOUT=300 WANDB_MODE=offline
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRITON_CACHE_DIR="/tmp/${USER}_triton_cache"; mkdir -p "$TRITON_CACHE_DIR"
PY="$REPO/.venv_carbon_portable/bin/python"

SPECS="$REPO/analysis/size_seed0_specs.txt"
[ -f "$SPECS" ] || { echo "missing $SPECS -- run analysis/build (seed0/seed12 split) first"; exit 1; }

# SLURM_ARRAY_TASK_ID is 0-based; specs file is 1-based.
LINE_NO=$(( ${SLURM_ARRAY_TASK_ID:-0} + 1 ))
SPEC=$(sed -n "${LINE_NO}p" "$SPECS")
[ -n "$SPEC" ] || { echo "no spec on line $LINE_NO of $SPECS"; exit 1; }

echo "=== [array ${SLURM_ARRAY_TASK_ID}] line ${LINE_NO}: $SPEC ==="
# shellcheck disable=SC2086
"$PY" -m src.train.distill $SPEC
echo "=== done line ${LINE_NO} ==="
