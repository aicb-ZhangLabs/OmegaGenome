#!/bin/bash
# UNIFIED CPU inference-latency for ALL SIX models in ONE job on ONE node under identical
# conditions, for the OmegaGenome rebuttal CPU-latency table (T18 / R1.11):
#   NT-2.5B, Carbon-3B, DNABERT-2, Enformer, Caduceus, and the BPNet student (re-measured here,
#   NOT reused from the old run) -> every row shares the exact same hardware + thread count + no
#   contention. fp32, batch 16, max_length 1000, task splice_sites_all -- identical across all 6.
#
# FIXED allocation: 64 physical cores (galaxy = 2x64-core EPYC 7662; N=64 = one socket's worth,
# a clean round number matching the existing table's 64-thread convention). N=64 for BOTH
# torch.set_num_threads AND OMP_NUM_THREADS/MKL_NUM_THREADS -> NO SMT oversubscription (unlike the
# discarded login-node run that used 64 threads on a contended 16-core box). Requests a fixed
# --cpus-per-task=64 dedicated to this task. NOTE: this cluster has ConstrainCores=no, so true
# node-exclusivity (--exclusive) would be ideal but is blocked behind other long-running jobs;
# galaxy is currently ~84% idle so cross-job contention is minimal. The job PRINTS the live node
# load (sinfo CPUs A/I/O/T) at start so the recorded numbers can be qualified. All 6 models run
# back-to-back in THIS one job on THIS one node with identical threads/precision -> apples-to-apples.
#
#   module load slurm && sbatch slurm/cpu_latency_all6_galaxy.sbatch
#SBATCH --job-name=cpu-lat-all6
#SBATCH --nodes=1
#SBATCH --partition=zhanglab.p
#SBATCH --nodelist=galaxy
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --requeue
#SBATCH --time=30-00:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail
REPO=/home/pengchx3/text-dna/OmegaGenome_Revise_202606/code_carbon
cd "$REPO"

N=64
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export OMP_NUM_THREADS=$N MKL_NUM_THREADS=$N
HOME_HF="$REPO/.hf_cache"
export HF_HOME="$HOME_HF" HF_HUB_CACHE="$HOME_HF/hub" HUGGINGFACE_HUB_CACHE="$HOME_HF/hub"
export HF_DATASETS_CACHE="$HOME_HF/datasets" TRANSFORMERS_CACHE="$HOME_HF/hub" HF_ASSETS_CACHE="$HOME_HF/assets"
export HF_TOKEN="$(cat /home/pengchx3/text-dna/huggingface-token-0616.txt)"
export TOKENIZERS_PARALLELISM=false WANDB_MODE=offline
export HF_HUB_DISABLE_XET=1 HF_XET_CACHE="$HOME_HF/xet"
export CARBON_TEACHER_DIR=/home/pengchx3/carbon_teachers/carbon_3b_lora
export CUDA_VISIBLE_DEVICES=""

PY="$REPO/.venv_carbon_portable/bin/python"
OUT="$REPO/rebuttal_infra/cpu_teacher_latency/results_all6.jsonl"
: > "$OUT"   # fresh file for the unified run

echo "=== CPU node ==="; hostname; grep -m1 "model name" /proc/cpuinfo
echo "physical cores:"; lscpu 2>/dev/null | grep -E "^CPU\(s\)|Core\(s\) per socket|Socket|Thread\(s\) per core"
echo "=== live node load at job start (CPUs A/I/O/T; lower A = less contention) ==="
( module load slurm 2>/dev/null; sinfo -N -n galaxy -o "%.10N %.6t %.14C %.10m" 2>/dev/null ) || true
echo "SLURM_CPUS_ON_NODE=${SLURM_CPUS_ON_NODE:-?} SLURM_JOB_CPUS_PER_NODE=${SLURM_JOB_CPUS_PER_NODE:-?}"
echo "N (torch+omp threads) = $N ; OMP_NUM_THREADS=$OMP_NUM_THREADS"
"$PY" -c "import torch; print('cuda avail (should be False):', torch.cuda.is_available(), '| torch threads:', torch.get_num_threads())"

# Fast models: full warmup (2) + 5 timed batches. Slow models: warmup 1 + fewer timed batches
# (still runs to completion -- Caduceus/Carbon-3B are minutes/batch, that's the point).
run () { echo "=== $1 (warmup $2, $3 batches) ==="; "$PY" -u scripts/cpu_teacher_latency.py \
         --teacher "$1" --num-batches "$3" --warmup "$2" --threads $N --out-jsonl "$OUT"; }

run bpnet     2 20    # BPNet student, re-measured in-job (fast, more batches for a stable mean)
run enformer  2 5
run dnabert2  2 5
run nt        1 3     # NT-2.5B ~minutes/batch
run carbon3b  1 3     # Carbon-3B ~minutes/batch
run caduceus  1 3     # Caduceus CPU reference-scan is very slow but runs to completion

echo "=== DONE. unified results -> $OUT ==="
cat "$OUT"
