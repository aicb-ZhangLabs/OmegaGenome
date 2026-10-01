#!/bin/bash
#SBATCH --job-name=dkd-dnabert2
#SBATCH --partition=zhanglab.p
#SBATCH --nodelist=laniakea
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=slurm-%x-%j.out
set -uo pipefail
cd ${OG_ROOT:-$PWD}
PV="$PWD/.venv_carbon_portable/bin/python"
export HF_HOME=${HF_HOME} TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True WANDB_MODE=offline
echo "=== DKD dnabert2 $(hostname) $(date +%T) ==="
"$PV" -u -m src.train.run_single_experiment --config-path analysis/dkd/runs/dkd_dnabert2.json
echo "=== EXIT=$? $(date +%T) ==="
