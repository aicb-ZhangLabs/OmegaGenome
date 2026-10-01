#!/bin/bash
# FULL 18-task CPU latency+memory (RSS) sweep on galaxy (AMD EPYC). The deployment claim
# ("inference may take minutes"): NT-2.5B / Carbon-3B are ~minutes/batch on CPU.
# Two phases share the same CPU (cpu_name recorded, device=cpu):
#   Phase A (big teachers, SLOW): NT-2.5B, Carbon-3B, Enformer with FEW timed batches
#     (warmup 0/1, num_batches 1-2) -> per-batch latency + EXTRAPOLATED whole-dataset time.
#   Phase B (student, FAST): full test set per task.
# fp32 on CPU. Caduceus excluded (galaxy GLIBC 2.31 < 2.34). DNABERT-2 excluded on CPU
# (its disabled-flash path still pokes a Triton/driver call that errors with no GPU).
#   sbatch slurm/benchmark_18task_cpu.sbatch
#SBATCH --job-name=bench18-cpu
#SBATCH --nodes=1
#SBATCH --partition=zhanglab.p
#SBATCH --nodelist=galaxy
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --requeue
#SBATCH --time=30-00:00:00
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
export CARBON_TEACHER_DIR=${CARBON_TEACHER_DIR:-$PWD/data/finetuned_models}/carbon_3b_lora
export CUDA_VISIBLE_DEVICES=""
# On galaxy the SSD is local at ${OG_SCRATCH:-$PWD/output}.
export OUTPUT_PATH=${OG_SCRATCH:-$PWD/output}/latency_18task/cpu

PY="$REPO/.venv_carbon_portable/bin/python"
# Space-separated for tyro List[str] (NOT a bracket string).
TASKS="H2AFZ H3K27ac H3K27me3 H3K36me3 H3K4me1 H3K4me2 H3K4me3 H3K9ac H3K9me3 H4K20me1 promoter_all promoter_no_tata promoter_tata enhancers enhancers_types splice_sites_all splice_sites_acceptors splice_sites_donors"

echo "=== CPU node ==="; hostname; grep -m1 "model name" /proc/cpuinfo
"$PY" -c "import torch; print('cuda avail (should be False):', torch.cuda.is_available())"

echo "=== PHASE A: big teachers on CPU (few batches, extrapolated) ==="
"$PY" -u -m src.train.benchmark_inference full_benchmark \
    --task-names $TASKS \
    --teachers nt carbon3b enformer \
    --student-sizes \
    --device cpu --precision fp32 \
    --batch-size 16 --max-length 1000 \
    --warmup-batches 1 --num-batches 2 \
    --skip-on-error

echo "=== PHASE B: BPNet student on CPU (full) ==="
"$PY" -u -m src.train.benchmark_inference full_benchmark \
    --task-names $TASKS \
    --teachers \
    --student-sizes original \
    --device cpu --precision fp32 \
    --batch-size 16 --max-length 1000 \
    --warmup-batches 5 --num-batches None \
    --skip-on-error

echo "=== 18-TASK CPU SWEEP DONE; CSVs under $OUTPUT_PATH/benchmark/ ==="
find "$OUTPUT_PATH/benchmark" -name "all_benchmark.csv" 2>/dev/null | head
