#!/bin/bash
# CPU inference-latency for the FOUR teacher models (Enformer, DNABERT-2, Caduceus, Carbon-3B)
# for the OmegaGenome rebuttal CPU-latency table (T18 / R1.11). Runs on galaxy (AMD EPYC 7662
# 64-core) so numbers match the existing NT-2.5B / BPNet CPU rows (same box). fp32, batch 16,
# max_length 1000 -- apples-to-apples with the GPU T15 measurement path (benchmark_teacher_inference).
#
# Carbon-3B (~3.45B, ~14 GB fp32 weights) is the reason for SLURM: needs lots of RAM and is
# minutes/batch on CPU (the deployment-impracticality point). Caduceus is re-timed here on the
# reference 64-core box too (its mamba_ssm CPU reference-scan is very slow -- see script patches).
#
#   module load slurm && sbatch slurm/cpu_teacher_latency_galaxy.sbatch
#SBATCH --job-name=cpu-lat-nt
#SBATCH --nodes=1
#SBATCH --partition=zhanglab.p
#SBATCH --nodelist=galaxy
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --requeue
#SBATCH --time=01:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail
REPO=/home/pengchx3/text-dna/OmegaGenome_Revise_202606/code_carbon
cd "$REPO"

export PYTHONPATH="$REPO:${PYTHONPATH:-}"
HOME_HF="$REPO/.hf_cache"
export HF_HOME="$HOME_HF" HF_HUB_CACHE="$HOME_HF/hub" HUGGINGFACE_HUB_CACHE="$HOME_HF/hub"
export HF_DATASETS_CACHE="$HOME_HF/datasets" TRANSFORMERS_CACHE="$HOME_HF/hub" HF_ASSETS_CACHE="$HOME_HF/assets"
export HF_TOKEN="$(cat /home/pengchx3/text-dna/huggingface-token-0616.txt)"
export TOKENIZERS_PARALLELISM=false WANDB_MODE=offline
export HF_HUB_DISABLE_XET=1 HF_XET_CACHE="$HOME_HF/xet"
export CARBON_TEACHER_DIR=/home/pengchx3/carbon_teachers/carbon_3b_lora
export CUDA_VISIBLE_DEVICES=""

PY="$REPO/.venv_carbon_portable/bin/python"
OUT="$REPO/rebuttal_infra/cpu_teacher_latency/results_galaxy.jsonl"

echo "=== CPU node ==="; hostname; grep -m1 "model name" /proc/cpuinfo; nproc
"$PY" -c "import torch; print('cuda avail (should be False):', torch.cuda.is_available())"

# Fast teachers: full warmup + 5 timed batches.
for T in nt; do
  echo "=== $T (warmup 2, 5 batches) ==="
  "$PY" -u scripts/cpu_teacher_latency.py --teacher "$T" --threads 16 --num-batches 3 --warmup 1 --out-jsonl "$OUT"
done

# Caduceus: mamba_ssm CPU reference-scan is very slow -> warmup 1, 3 timed batches.
echo "=== caduceus (warmup 1, 3 batches) ==="

# Carbon-3B (~3.45B): minutes/batch -> warmup 1, 2 timed batches (task says 2-3 is fine).
echo "=== carbon3b (warmup 1, 2 batches) ==="

echo "=== DONE. results -> $OUT ==="
cat "$OUT"
