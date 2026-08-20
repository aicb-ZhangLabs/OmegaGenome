#!/bin/bash
# TINY ESCALATION of the 2 H3K9me3 cells (DIST + logit_standard) that did NOT recover at weight_kl=0.5
# in the method_hpfix rerun. Retry at weight_kl=0.25 (further out of the constant-prediction basin).
# Everything else replicates method_hpfix EXACTLY: temperature=4, weight_ce=1.0, weight_mse=0,
# epochs 200, early-stop-patience 0, best-val ckpt. 2 cells x 3 seeds = 6 runs.
#
# ONE array task == ONE run. Reads its spec line from method_hpfix_kl025_specs.txt and trains it
# INLINE on the allocated GPU via `python -m src.train.distill <spec>` (mode=run => in-process,
# NOT a nested submit). Env mirrors method_hpfix.sbatch / size18task.sbatch EXACTLY (HF cache on
# SSD, portable venv, triton cache).
#
# MULTI-NODE (FIXED pattern): --nodes=1 + --partition=zhanglab.p and NO --nodelist -> the array
# floats across all 3 GPU nodes. NO dependency. --array=0-5%2 caps it at 2 concurrent GPUs so it
# runs courteously alongside the size sweep and leaves GPUs for others. NO --nice (normal priority).
#
# TEACHER CACHE: NT-2.5B teacher logits cache reused (verified present for H3K9me3; the kl=0.5
# method_hpfix H3K9me3 runs already completed against it); weight_mse=0 => logit-only.
#
# Pipeline already proven (method_hpfix ran 27 of these). Submit with:
#   sbatch rebuttal_infra/method_hpfix_kl025.sbatch                 # all 6, 2 concurrent, normal prio
#
#SBATCH --job-name=loom-hpfix-kl025
#SBATCH --array=0-5%2
#SBATCH --nodes=1
#SBATCH --partition=zhanglab.p
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=64G
#SBATCH --requeue
#SBATCH --time=12:00:00
#SBATCH --output=slurm-%x-%A_%a.out

set -euo pipefail
REPO=/home/pengchx3/text-dna/OmegaGenome_Revise_202606/code_carbon
cd "$REPO"
source slurm/env_setup.sh

# ---- HF cache on the node-aware SSD (identical to method_hpfix.sbatch / size18task.sbatch) ----
for _b in /tmp/galaxy_srv_disk00/pengchx3 /srv/disk00/sshfs/pengchx3; do [ -d "$_b" ] && { _SSD="$_b"; break; }; done
[ -n "${_SSD:-}" ] && export HF_OVERRIDE="$_SSD/hf_cache_shared"
if [ -n "${HF_OVERRIDE:-}" ]; then
  export HF_HOME="$HF_OVERRIDE" HF_HUB_CACHE="$HF_OVERRIDE/hub" HUGGINGFACE_HUB_CACHE="$HF_OVERRIDE/hub" \
         HF_DATASETS_CACHE="$HF_OVERRIDE/datasets" TRANSFORMERS_CACHE="$HF_OVERRIDE/hub" HF_ASSETS_CACHE="$HF_OVERRIDE/assets"
  mkdir -p "$HF_OVERRIDE/hub" "$HF_OVERRIDE/datasets" "$HF_OVERRIDE/assets"
fi
export HF_TOKEN="$(cat /home/pengchx3/text-dna/huggingface-token-0616.txt)"
export WANDB_INIT_TIMEOUT=300 WANDB_MODE=offline
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRITON_CACHE_DIR="/tmp/${USER}_triton_cache"; mkdir -p "$TRITON_CACHE_DIR"
PY="$REPO/.venv_carbon_portable/bin/python"

SPECS="$REPO/rebuttal_infra/method_hpfix_kl025_specs.txt"
[ -f "$SPECS" ] || { echo "missing $SPECS"; exit 1; }

# SLURM_ARRAY_TASK_ID is 0-based; specs file is 1-based.
LINE_NO=$(( ${SLURM_ARRAY_TASK_ID:-0} + 1 ))
SPEC=$(sed -n "${LINE_NO}p" "$SPECS")
[ -n "$SPEC" ] || { echo "no spec on line $LINE_NO of $SPECS"; exit 1; }

echo "=== [array ${SLURM_ARRAY_TASK_ID}] line ${LINE_NO}: $SPEC ==="
# shellcheck disable=SC2086
"$PY" -m src.train.distill $SPEC
echo "=== done line ${LINE_NO} ==="
