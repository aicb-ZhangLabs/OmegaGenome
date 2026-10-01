#!/bin/bash
# DIAGNOSTIC SMOKE: validate Enformer + DNABERT-2 run under fp16 in benchmark_teacher_inference.
# Confirms the precision-unification fix (autocast fp16) produces correct-shape logits and no
# half/float dtype mismatch. 1 task, just the two formerly-fp32 teachers, 2 timed batches.
#   sbatch slurm/benchmark_fp16_fix_smoke.sbatch
#SBATCH --job-name=bench-fp16fix-smoke
#SBATCH --nodes=1
#SBATCH --partition=zhanglab.p
#SBATCH --nodelist=laniakea
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --requeue
#SBATCH --time=2:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail
REPO=${OG_ROOT:-$PWD}
cd "$REPO"

HOME_HF="$REPO/.hf_cache"
export HF_HOME="$HOME_HF" HF_HUB_CACHE="$HOME_HF/hub" HUGGINGFACE_HUB_CACHE="$HOME_HF/hub"
export HF_DATASETS_CACHE="$HOME_HF/datasets" TRANSFORMERS_CACHE="$HOME_HF/hub" HF_ASSETS_CACHE="$HOME_HF/assets"
export HF_TOKEN="$(cat ${HF_TOKEN_FILE})"
export TOKENIZERS_PARALLELISM=false WANDB_MODE=offline
export HF_HUB_DISABLE_XET=1
export HF_XET_CACHE="$REPO/.hf_cache/xet"
export OUTPUT_PATH=${OG_SCRATCH:-$PWD/output}/latency_18task/fp16fix_smoke

PY="$REPO/.venv_carbon_portable/bin/python"

echo "=== node + GPU ==="; hostname; ldd --version | head -1
"$PY" -c "import torch; print('device:', torch.cuda.get_device_name(0))"

"$PY" -u -m src.train.benchmark_inference full_benchmark \
    --task-names splice_sites_all \
    --teachers enformer dnabert2 \
    --student-sizes \
    --precision fp16 \
    --batch-size 16 \
    --max-length 1000 \
    --warmup-batches 2 \
    --num-batches 2 \
    --skip-on-error

echo "=== FP16-FIX SMOKE DONE; CSVs under $OUTPUT_PATH/benchmark/ ==="
find "$OUTPUT_PATH/benchmark" -name "*.csv" 2>/dev/null | head
