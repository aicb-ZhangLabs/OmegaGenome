#!/bin/bash
# SMOKE test for the R1.11/R2.2 inference latency+memory benchmark.
# Tiny: caduceus teacher (self-contained ckpt, no big download) + BPNet student on
# splice_sites_all, num_batches=2, fp16. Confirms the pipeline runs end-to-end and
# writes a CSV. NOT a results run. Fits a 24GB GPU (galaxy).
#   sbatch slurm/benchmark_smoke.sbatch
#SBATCH --job-name=bench-smoke
#SBATCH --nodes=1
#SBATCH --partition=zhanglab.p
#SBATCH --nodelist=galaxy
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --nice=0
#SBATCH --requeue
#SBATCH --time=1:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail
REPO=/home/pengchx3/text-dna/OmegaGenome_Revise_202606/code_carbon
cd "$REPO"

# Node-agnostic HF cache (the .hf_cache that already holds the NT dataset + NT tokenizer).
HOME_HF="$REPO/.hf_cache"
export HF_HOME="$HOME_HF" HF_HUB_CACHE="$HOME_HF/hub" HUGGINGFACE_HUB_CACHE="$HOME_HF/hub"
export HF_DATASETS_CACHE="$HOME_HF/datasets" TRANSFORMERS_CACHE="$HOME_HF/hub" HF_ASSETS_CACHE="$HOME_HF/assets"
export HF_TOKEN="$(cat /home/pengchx3/text-dna/huggingface-token-0616.txt)"
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=offline
# Pin benchmark output under /home (persisted, not /tmp).
export OUTPUT_PATH=/home/pengchx3/text-dna/OmegaGenome_Revise_202606/rebuttal_infra/benchmark_runs

PY="$REPO/.venv_carbon_portable/bin/python"

"$PY" -u -m src.train.benchmark_inference full_benchmark \
    --task-names splice_sites_all \
    --teachers caduceus \
    --student-sizes original \
    --precision fp16 \
    --batch-size 16 \
    --max-length 1000 \
    --warmup-batches 2 \
    --num-batches 2

echo "=== SMOKE DONE; CSVs under $OUTPUT_PATH/benchmark/splice_sites_all/ ==="
ls -lt "$OUTPUT_PATH/benchmark/splice_sites_all/" 2>/dev/null | head
