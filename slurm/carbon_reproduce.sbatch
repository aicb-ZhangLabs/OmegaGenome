#!/bin/bash
# Validate the Carbon install + weights (sequence-recovery proxy) on one GPU.
#   sbatch slurm/carbon_reproduce.sbatch
# Confirm with Jing before submitting (lab policy).
# GPU: laniakea=A6000 48G (default), voyager=H100 80G (ask Jing; cap 3), galaxy=3090 24G.
#SBATCH --job-name=carbon_reproduce
#SBATCH --nodes=1
#SBATCH --nodelist=laniakea
#SBATCH --partition=zhanglab.p
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail
REPO=/home/pengchx3/text-dna/OmegaGenome_Revise_202606/code_carbon
cd "$REPO"
source slurm/env_setup.sh

"$PY" -m src.train.carbon_reproduce --model_id HuggingFaceBio/Carbon-3B --num_seqs 64 --task promoter_all
