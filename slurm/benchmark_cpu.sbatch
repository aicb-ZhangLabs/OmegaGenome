#!/bin/bash
# CPU inference latency + memory (RSS) benchmark for the deployment claim ("inference may
# take minutes"). On CPU the ~0.12M BPNet student runs in ms/sample while
# NT-2.5B is orders of magnitude slower -> exactly the deployment argument.
#
# Two phases share the SAME CPU (same-hardware), recorded via cpu_name + device=cpu:
#   Phase A (teachers, SLOW on CPU): NT-2.5B, Enformer, DNABERT-2 with FEW timed batches
#     (warmup 1, num_batches 3) so it finishes; per-sample latency + RSS still meaningful.
#   Phase B (student, FAST): full/large num_batches.
# fp32 on CPU (fp16 CPU kernels are not the deployment path); batch_size=16, seqlen 1000.
# Caduceus excluded (mamba/triton GLIBC issue, and it's GPU-oriented).
#   sbatch slurm/benchmark_cpu.sbatch
#SBATCH --job-name=bench-cpu
#SBATCH --nodes=1
#SBATCH --partition=zhanglab.p
#SBATCH --nodelist=galaxy
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --nice=0
#SBATCH --requeue
#SBATCH --time=12:00:00
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
export OUTPUT_PATH=${OG_WORKSPACE:-$PWD/..}/analysis/benchmark_runs_cpu
# Hide GPUs so torch picks CPU even though galaxy has a 3090.
export CUDA_VISIBLE_DEVICES=""

PY="$REPO/.venv_carbon_portable/bin/python"

echo "=== CPU on this node ==="
grep -m1 "model name" /proc/cpuinfo || true
"$PY" -c "import torch; print('cuda avail (should be False):', torch.cuda.is_available())"

# Phase A: teachers on CPU, FEW timed batches (warmup 1, 3 timed) so it terminates.
echo "=== PHASE A: teachers on CPU (few batches) ==="
"$PY" -u -m src.train.benchmark_inference full_benchmark \
    --task-names splice_sites_all \
    --teachers nt enformer dnabert2 \
    --student-sizes \
    --device cpu \
    --precision fp32 \
    --batch-size 16 \
    --max-length 1000 \
    --warmup-batches 1 \
    --num-batches 3 \
    --skip-on-error

# Phase B: student on CPU, large batch count (it is fast).
echo "=== PHASE B: BPNet student on CPU (full) ==="
"$PY" -u -m src.train.benchmark_inference full_benchmark \
    --task-names splice_sites_all \
    --teachers \
    --student-sizes original \
    --device cpu \
    --precision fp32 \
    --batch-size 16 \
    --max-length 1000 \
    --warmup-batches 5 \
    --num-batches None \
    --skip-on-error

echo "=== CPU BENCHMARK DONE; CSVs under $OUTPUT_PATH/benchmark/splice_sites_all/ ==="
ls -lt "$OUTPUT_PATH/benchmark/splice_sites_all/" 2>/dev/null | head
