#!/bin/bash
# Follow-up GALAXY (RTX 3090) run for Enformer + DNABERT-2 at fp32.
# Both load fine on the 3090 but hit fp16 input/bias dtype mismatches (Enformer takes
# float seq input; DNABERT-2's PyTorch-attention fallback builds a float bias) -> run them
# at fp32. They are small (DNABERT-2 117M; Enformer) so fp32 is the honest deployment number
# and avoids the fp16 cast landmines. SAME galaxy GPU as the fp16 NT/student run -> merged
# into one table by gpu_name. DNABERT-2 Triton flash-attn is auto-disabled in-process
# (PyTorch attention fallback) so it no longer needs a separate env.
#   sbatch slurm/benchmark_galaxy_fp32.sbatch
#SBATCH --job-name=bench-galaxy-fp32
#SBATCH --nodes=1
#SBATCH --partition=zhanglab.p
#SBATCH --nodelist=galaxy
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --nice=0
#SBATCH --requeue
#SBATCH --time=12:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail
REPO=/home/pengchx3/text-dna/OmegaGenome_Revise_202606/code_carbon
cd "$REPO"

HOME_HF="$REPO/.hf_cache"
export HF_HOME="$HOME_HF" HF_HUB_CACHE="$HOME_HF/hub" HUGGINGFACE_HUB_CACHE="$HOME_HF/hub"
export HF_DATASETS_CACHE="$HOME_HF/datasets" TRANSFORMERS_CACHE="$HOME_HF/hub" HF_ASSETS_CACHE="$HOME_HF/assets"
export HF_TOKEN="$(cat /home/pengchx3/text-dna/huggingface-token-0616.txt)"
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=offline
export HF_HUB_DISABLE_XET=1
export HF_XET_CACHE="$REPO/.hf_cache/xet"
export OUTPUT_PATH=/home/pengchx3/text-dna/OmegaGenome_Revise_202606/rebuttal_infra/benchmark_runs_galaxy

PY="$REPO/.venv_carbon_portable/bin/python"

echo "=== GPU on this node ==="
"$PY" -c "import torch; print('device:', torch.cuda.get_device_name(0))"

"$PY" -u -m src.train.benchmark_inference full_benchmark \
    --task-names splice_sites_all \
    --teachers enformer dnabert2 \
    --student-sizes \
    --precision fp32 \
    --batch-size 16 \
    --max-length 1000 \
    --warmup-batches 10 \
    --num-batches None \
    --skip-on-error

echo "=== GALAXY fp32 (enformer+dnabert2) DONE; CSVs under $OUTPUT_PATH/benchmark/splice_sites_all/ ==="
ls -lt "$OUTPUT_PATH/benchmark/splice_sites_all/" 2>/dev/null | head
