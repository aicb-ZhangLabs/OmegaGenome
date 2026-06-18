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
export HF_TOKEN="$(cat /home/pengchx3/text-dna/huggingface-token-0616.txt)"
PY="$REPO/.venv_carbon_portable/bin/python"

CONFIG="${1:-carbon-smoke}"
echo "=== carbon distillation: $CONFIG (inline) ==="
"$PY" -m src.train.distill "$CONFIG"
echo "=== done: $CONFIG ==="
