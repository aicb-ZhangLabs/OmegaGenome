#!/bin/bash
# Complete Enformer's 18-task GPU latency/memory at fp32 (the honest Enformer deployment
# precision; its architecture emits an internal fp32 one-hot that breaks fp16 batchnorm).
# The earlier fp16 sweep failed all 18; only 1 fp32 task (splice_sites_all) was measured, so
# the all-18 total was extrapolated from that one. This runs ALL 18 tasks fp32 on the SAME
# galaxy RTX 3090 (hardware-consistent with the other legs) so Enformer becomes 18/18 measured.
#   sbatch slurm/benchmark_enformer_fp32_18task.sbatch
#SBATCH --job-name=bench-enformer-fp32-18task
#SBATCH --nodes=1
#SBATCH --partition=zhanglab.p
#SBATCH --nodelist=galaxy
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --requeue
#SBATCH --time=4:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail
REPO=${OG_ROOT:-$PWD}
cd "$REPO"

HOME_HF="$REPO/.hf_cache"
export HF_HOME="$HOME_HF" HF_HUB_CACHE="$HOME_HF/hub" HUGGINGFACE_HUB_CACHE="$HOME_HF/hub"
export HF_DATASETS_CACHE="$HOME_HF/datasets" TRANSFORMERS_CACHE="$HOME_HF/hub" HF_ASSETS_CACHE="$HOME_HF/assets"
export HF_TOKEN="$(cat ${HF_TOKEN_FILE})"
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=offline
export HF_HUB_DISABLE_XET=1
export HF_XET_CACHE="$REPO/.hf_cache/xet"
export OUTPUT_PATH=${OG_WORKSPACE:-$PWD/..}/analysis/benchmark_enformer_fp32_18task

PY="$REPO/.venv_carbon_portable/bin/python"
echo "=== node + GPU ==="; hostname; "$PY" -c "import torch; print('device:', torch.cuda.get_device_name(0))"

"$PY" -u -m src.train.benchmark_inference full_benchmark \
    --task-names H2AFZ H3K27ac H3K27me3 H3K36me3 H3K4me1 H3K4me2 H3K4me3 H3K9ac H3K9me3 H4K20me1 promoter_all promoter_no_tata promoter_tata enhancers enhancers_types splice_sites_all splice_sites_acceptors splice_sites_donors \
    --teachers enformer \
    --student-sizes \
    --precision fp32 \
    --batch-size 16 \
    --max-length 1000 \
    --warmup-batches 10 \
    --num-batches None \
    --skip-on-error

echo "=== DONE; per-task CSVs under $OUTPUT_PATH/benchmark/<task>/ ==="
find "$OUTPUT_PATH" -name "all_benchmark.csv" 2>/dev/null | wc -l
