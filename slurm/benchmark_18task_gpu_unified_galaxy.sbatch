#!/bin/bash
# UNIFIED 18-task latency+memory sweep on GPU, SINGLE PRECISION (fp16) for ALL models:
# NT-2.5B, Caduceus, Enformer, DNABERT-2, Carbon-3B teachers + BPNet `original` student.
# Replaces the old apples-to-oranges split (NT/Caduceus/Carbon3b fp16 in benchmark_18task_gpu
# + Enformer/DNABERT-2 fp32 in benchmark_18task_gpu_fp32). The fp16 fix: model.half() for
# genuine fp16 weights/memory, PLUS a CUDA fp16 autocast around the forward that promotes the
# stray fp32 bias/LayerNorm tensors in Enformer/DNABERT-2 (which previously forced the fp32 leg).
# The BPNet student runs the same fp16 autocast (it can't be half()-cast -> forward one-hots
# input_ids to float32). Every CSV row's `precision` column == fp16 (single-precision table, R1.11).
#
# DNABERT-2 Triton flash-attn auto-disabled in-process (PyTorch attention fallback).
# Big teachers (NT-2.5B, Carbon-3B): time a fixed 8 batches/task -> extrapolated whole-set time
# (flagged); Caduceus/Enformer/DNABERT-2 small, also timed on the same 8 batches for a uniform
# per-batch latency basis; student runs the same batch count. GALAXY (RTX 3090 24G, GLIBC 2.31)
# -- teachers are loaded/benchmarked/freed one-at-a-time (del model + empty_cache between models),
# so the largest single model (NT-2.5B ~11.5GB fp16) fits on the 3090. GALAXY is where the
# carbon venv is built/works (the voyager GLIBC-2.39 mismatch killed job 243789 at exit 13) and
# is hardware-consistent with the earlier fp32/3090 latency legs.
#   sbatch slurm/benchmark_18task_gpu_unified_galaxy.sbatch
#SBATCH --job-name=bench18-gpu-fp16-galaxy
#SBATCH --nodes=1
#SBATCH --partition=zhanglab.p
#SBATCH --nodelist=galaxy
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=80G
#SBATCH --requeue
#SBATCH --time=30-00:00:00
#SBATCH --output=slurm-%x-%j.out

# NOTE: deliberately NOT using `pipefail`. Job 243789 (voyager) and 243940 (galaxy) both died
# with ExitCode=141 (128+13 = SIGPIPE) right after the header: `ldd --version | head -1` makes
# `ldd` write into a pipe that `head -1` closes after one line -> SIGPIPE -> pipefail+`-e` aborts
# the whole job in <1s before any benchmarking. `set -eu` (no pipefail) keeps error-checking on
# the real python step while tolerating the harmless SIGPIPE on the diagnostic `| head` lines.
set -eu
REPO=/home/pengchx3/text-dna/OmegaGenome_Revise_202606/code_carbon
cd "$REPO"

HOME_HF="$REPO/.hf_cache"
export HF_HOME="$HOME_HF" HF_HUB_CACHE="$HOME_HF/hub" HUGGINGFACE_HUB_CACHE="$HOME_HF/hub"
export HF_DATASETS_CACHE="$HOME_HF/datasets" TRANSFORMERS_CACHE="$HOME_HF/hub" HF_ASSETS_CACHE="$HOME_HF/assets"
export HF_TOKEN="$(cat /home/pengchx3/text-dna/huggingface-token-0616.txt)"
export TOKENIZERS_PARALLELISM=false WANDB_MODE=offline
export HF_HUB_DISABLE_XET=1
export HF_XET_CACHE="$REPO/.hf_cache/xet"
export CARBON_TEACHER_DIR=/home/pengchx3/carbon_teachers/carbon_3b_lora
# Galaxy SSD output. On galaxy, /tmp/galaxy_srv_disk00/pengchx3 and /srv/disk00/sshfs/pengchx3
# are the SAME directory (same inode), so this writes exactly where the tracker expects.
# NEVER /home (quota) or /extra-write.
export OUTPUT_PATH=/tmp/galaxy_srv_disk00/pengchx3/latency_18task/gpu_fp16_unified

PY="$REPO/.venv_carbon_portable/bin/python"

echo "=== node + GPU ==="; hostname; ldd --version | head -1
"$PY" -c "import torch; print('device:', torch.cuda.get_device_name(0))"

"$PY" -u -m src.train.benchmark_inference full_benchmark \
    --task-names H2AFZ H3K27ac H3K27me3 H3K36me3 H3K4me1 H3K4me2 H3K4me3 H3K9ac H3K9me3 H4K20me1 promoter_all promoter_no_tata promoter_tata enhancers enhancers_types splice_sites_all splice_sites_acceptors splice_sites_donors \
    --teachers nt caduceus enformer dnabert2 carbon3b \
    --student-sizes original \
    --precision fp16 \
    --batch-size 16 \
    --max-length 1000 \
    --warmup-batches 3 \
    --num-batches 8 \
    --skip-on-error

echo "=== UNIFIED 18-TASK GPU fp16 SWEEP DONE; CSVs under $OUTPUT_PATH/benchmark/ ==="
find "$OUTPUT_PATH/benchmark" -name "all_benchmark.csv" 2>/dev/null | head
