#!/bin/bash
# Fine-tune Carbon as a teacher, inline on the allocated GPU.
#   sbatch slurm/carbon_finetune.sbatch                      # carbon_3b_debug (1 task, 1 epoch)
#   sbatch slurm/carbon_finetune.sbatch carbon_3b            # full 18-task run, one job
#   sbatch slurm/carbon_finetune.sbatch carbon_3b H3K4me3    # single task (used by the fan-out)
# Confirm with Jing before submitting (lab policy).
# GPU: laniakea=A6000 48G (default), voyager=H100 80G (ask Jing; cap 3), galaxy=3090 24G.
#SBATCH --job-name=carbon_finetune
#SBATCH --nodes=1
#SBATCH --nodelist=laniakea
#SBATCH --partition=zhanglab.p
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=30-00:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail
EXP="${1:-carbon_3b_debug}"   # experiment name from config/distillation/teacher_finetune.py
REPO=/home/pengchx3/text-dna/OmegaGenome_Revise_202606/code_carbon
cd "$REPO"
source slurm/env_setup.sh
export OMEGA_FINETUNE_LOCAL=1  # run inline on this GPU instead of re-submitting per task

# Everything after the config name is passed through as tyro overrides (the CLI uses
# overridable_config_cli, so overrides are --config.<field>), e.g.:
#   sbatch slurm/carbon_finetune.sbatch carbon_3b --config.task-names H3K27ac --config.epochs 40
echo "Running teacher fine-tune: exp=$EXP  overrides=${*:2}"
"$PY" -m src.train.finetune_teacher "$EXP" "${@:2}"
