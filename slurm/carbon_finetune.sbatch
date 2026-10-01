#!/bin/bash
# Fine-tune Carbon as a teacher, inline on the allocated GPU.
#   sbatch slurm/carbon_finetune.sbatch                      # carbon_3b_debug (1 task, 1 epoch)
#   sbatch slurm/carbon_finetune.sbatch carbon_3b            # full 18-task run, one job
#   sbatch slurm/carbon_finetune.sbatch carbon_3b H3K4me3    # single task (used by the fan-out)
# Confirm with the lab before submitting (GPU-sharing policy).
# GPU: laniakea=A6000 48G (default), voyager=H100 80G (limited availability), galaxy=3090 24G.
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
REPO=${OG_ROOT:-$PWD}
cd "$REPO"
source slurm/env_setup.sh
export OMEGA_FINETUNE_LOCAL=1  # run inline on this GPU instead of re-submitting per task

# Everything after the config name is passed through as tyro overrides. The CLI uses
# overridable_config_cli, so the config's fields are TOP-LEVEL flags (NOT --config.<field>),
# e.g. --task-names / --epochs / --lora-r:
#   sbatch slurm/carbon_finetune.sbatch carbon_3b --task-names H3K27ac --epochs 40 --lora-r 32 --lora-alpha 64
echo "Running teacher fine-tune: exp=$EXP  overrides=${*:2}"
"$PY" -m src.train.finetune_teacher "$EXP" "${@:2}"
