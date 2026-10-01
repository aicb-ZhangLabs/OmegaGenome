#!/bin/bash
# Evaluate the 3 already-trained FULL-fine-tune (no-LoRA) NT-2.5B teachers
# (H2AFZ, H3K27ac, promoter_tata) for TEST MCC, to compare against the LoRA teacher ceiling.
# Inference only: NT-2.5B ~11GB at fp16, fits galaxy 24G. Runs in minutes.
#   sbatch slurm/eval_fullft_nt.sbatch
#SBATCH --job-name=ft-eval-fullft
#SBATCH --nodelist=galaxy
#SBATCH --partition=zhanglab.p
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --nice=2000
#SBATCH --requeue
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail
REPO=${OG_ROOT:-$PWD}
cd "$REPO"
source slurm/env_setup.sh

# Full-FT (the headline numbers) + LoRA r32 test MCC (apples-to-apples) for all 3 tasks.
# Both are inference-only; total runtime is minutes. Results -> analysis/results/.
"$PY" analysis/eval_fullft_nt.py \
    --tasks H2AFZ H3K27ac promoter_tata \
    --kinds full_ft lora \
    --batch-size 32
