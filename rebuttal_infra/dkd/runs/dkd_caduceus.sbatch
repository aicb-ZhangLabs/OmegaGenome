#!/bin/bash
#SBATCH --job-name=dkd-caduceus
#SBATCH --partition=zhanglab.p
#SBATCH --nodelist=laniakea
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=slurm-%x-%j.out
set -uo pipefail
cd /home/pengchx3/text-dna/OmegaGenome_Revise_202606/code_carbon
PV="$PWD/.venv_carbon_portable/bin/python"
export HF_HOME=/home/pengchx3/.cache/huggingface TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True WANDB_MODE=offline
echo "=== DKD caduceus $(hostname) $(date +%T) ==="
"$PV" -u -m src.train.run_single_experiment --config-path rebuttal_infra/dkd/runs/dkd_caduceus.json
echo "=== EXIT=$? $(date +%T) ==="
