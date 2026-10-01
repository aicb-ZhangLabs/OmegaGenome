#!/bin/bash
# GPU smoke for the faithful NTv3 FT pipeline: validates the real model's headless-core extraction,
# the full train/val/test loop, loss/metric, and best-model save/load — tiny seq + few steps so it
# fits any GPU and finishes in minutes. Node-flexible. NOT a results run.
#   sbatch slurm/ntv3_finetune_smoke.sbatch
#SBATCH --job-name=ntv3-ft-smoke
#SBATCH --nodes=1
#SBATCH --partition=zhanglab.p
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --nice=0
#SBATCH --requeue
#SBATCH --time=2:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail
REPO=${OG_NTV3_ROOT:-$PWD}
cd "$REPO"
HOME_HF=${OG_ROOT:-$PWD}/.hf_cache
export HF_HOME="$HOME_HF" HF_HUB_CACHE="$HOME_HF/hub" TRANSFORMERS_CACHE="$HOME_HF/hub"
export HF_TOKEN="$(cat ${HF_TOKEN_FILE})"
export TOKENIZERS_PARALLELISM=false
PV=${OG_ROOT:-$PWD}/.venv_carbon_portable/bin/python

SSD=${OG_SCRATCH:-$PWD/output}
mountpoint -q "$SSD" 2>/dev/null || SSD=${OG_SCRATCH:-$PWD/output}
DATA="$SSD/ntv3_benchmark_data"
STAGE="${SLURM_TMPDIR:-/tmp/ntv3_stage_$USER}"

# --smoke shrinks steps/val; small seq + mini_batch 1 keeps it on a 24GB GPU.
"$PV" -u -m src.train.finetune_ntv3 --data_dir "$DATA" --model InstaDeepAI/NTv3_650M_post \
      --out "$SSD/ntv3_targets/ntv3_ft_smoke" --stage_dir "$STAGE" \
      --smoke --sequence_length 4096 --mini_batch_size 1 --num_workers 4
echo "=== smoke done: $SSD/ntv3_targets/ntv3_ft_smoke/ntv3_finetune_result.json ==="
