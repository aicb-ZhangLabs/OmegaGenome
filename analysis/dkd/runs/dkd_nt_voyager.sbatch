#!/bin/bash
#SBATCH --job-name=dkd-nt-v
#SBATCH --partition=zhanglab.p
#SBATCH --nodelist=voyager
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=slurm-%x-%j.out
set -uo pipefail
cd ${OG_ROOT:-$PWD}
PV="$PWD/.venv_carbon_portable/bin/python"
# Override ALL HF cache vars (bashrc pins them to galaxy-local /srv, invisible on voyager)
HC=${HF_HOME}
export HF_HOME=$HC HF_HUB_CACHE=$HC/hub HUGGINGFACE_HUB_CACHE=$HC/hub TRANSFORMERS_CACHE=$HC HF_DATASETS_CACHE=$HC/datasets
export TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True WANDB_MODE=offline
echo "=== DKD nt (voyager) $(hostname) $(date +%T) ==="
"$PV" -u -m src.train.run_single_experiment --config-path analysis/dkd/runs/dkd_nt.json
echo "=== EXIT=$? $(date +%T) ==="
