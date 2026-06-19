#!/bin/bash
# Carbon-3B -> deploy_120k BPNet distillation runner. Sources the shared env (HF cache, venv, wandb)
# and runs a distillation config INLINE on the allocated GPU (mode="run" configs: carbon-smoke).
# For the parallel multi-task runs (carbon-raw / carbon-l2norm, mode="slurm") submit from the login
# node directly so slurm_fn fans out one job per task.
#   sbatch slurm/carbon_distill.sbatch carbon-smoke
#SBATCH --job-name=carbon-distill
#SBATCH --nodelist=laniakea
#SBATCH --partition=zhanglab.p
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --nice=0
#SBATCH --requeue
#SBATCH --time=12:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail
REPO=/home/pengchx3/text-dna/OmegaGenome_Revise_202606/code_carbon
cd "$REPO"
source slurm/env_setup.sh
# Optional HF-cache override (e.g. big base-model downloads -> SSD, not the tight /home quota).
# Submit with: sbatch --export=ALL,HF_OVERRIDE=/tmp/galaxy_srv_disk00/pengchx3/hf_cache_shared ...
if [ -n "${HF_OVERRIDE:-}" ]; then
  export HF_HOME="$HF_OVERRIDE" HF_HUB_CACHE="$HF_OVERRIDE/hub" HUGGINGFACE_HUB_CACHE="$HF_OVERRIDE/hub" \
         HF_DATASETS_CACHE="$HF_OVERRIDE/datasets" TRANSFORMERS_CACHE="$HF_OVERRIDE/hub" HF_ASSETS_CACHE="$HF_OVERRIDE/assets"
  mkdir -p "$HF_OVERRIDE/hub" "$HF_OVERRIDE/datasets" "$HF_OVERRIDE/assets"
  echo "HF cache overridden -> $HF_OVERRIDE"
fi
export HF_TOKEN="$(cat /home/pengchx3/text-dna/huggingface-token-0616.txt)"
export WANDB_INIT_TIMEOUT=300
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True  # reduce fragmentation OOM
PY="$REPO/.venv_carbon_portable/bin/python"

CONFIG="${1:-carbon-smoke}"
echo "=== carbon distillation: $CONFIG (inline) ==="
"$PY" -m src.train.distill "$CONFIG"
echo "=== done: $CONFIG ==="
