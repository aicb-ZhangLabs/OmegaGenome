#!/bin/bash
# Complete NT-2.5B's 2 missing splice tasks (acceptors, donors). The 18-task fp16 sweep
# stopped after splice_sites_all, so NT had 16/18. NT latency loads the BASE model
# (task-independent, architecture-constant), so these are directly measurable the same way
# the other 16 NT tasks were: fp16, --num-batches 8 (extrapolated whole-task). Galaxy RTX 3090.
#SBATCH --job-name=bench-nt-splice-fp16
#SBATCH --nodes=1
#SBATCH --partition=zhanglab.p
#SBATCH --nodelist=galaxy
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --requeue
#SBATCH --time=2:00:00
#SBATCH --output=slurm-%x-%j.out
set -euo pipefail
REPO=/home/pengchx3/text-dna/OmegaGenome_Revise_202606/code_carbon
cd "$REPO"
HOME_HF="$REPO/.hf_cache"
export HF_HOME="$HOME_HF" HF_HUB_CACHE="$HOME_HF/hub" HUGGINGFACE_HUB_CACHE="$HOME_HF/hub"
export HF_DATASETS_CACHE="$HOME_HF/datasets" TRANSFORMERS_CACHE="$HOME_HF/hub" HF_ASSETS_CACHE="$HOME_HF/assets"
export HF_TOKEN="$(cat /home/pengchx3/text-dna/huggingface-token-0616.txt)"
export TOKENIZERS_PARALLELISM=false WANDB_MODE=offline HF_HUB_DISABLE_XET=1
export HF_XET_CACHE="$REPO/.hf_cache/xet"
export OUTPUT_PATH=/home/pengchx3/text-dna/OmegaGenome_Revise_202606/rebuttal_infra/benchmark_nt_splice_fp16
PY="$REPO/.venv_carbon_portable/bin/python"
echo "=== node + GPU ==="; hostname; "$PY" -c "import torch; print('device:', torch.cuda.get_device_name(0))"
"$PY" -u -m src.train.benchmark_inference full_benchmark \
    --task-names splice_sites_acceptors splice_sites_donors \
    --teachers nt \
    --student-sizes \
    --precision fp16 \
    --batch-size 16 \
    --max-length 1000 \
    --warmup-batches 3 \
    --num-batches 8 \
    --skip-on-error
echo "=== DONE; CSVs under $OUTPUT_PATH/benchmark/ ==="
find "$OUTPUT_PATH" -name "*benchmark*.csv" 2>/dev/null
