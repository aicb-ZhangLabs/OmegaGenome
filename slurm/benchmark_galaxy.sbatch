#!/bin/bash
# PRIMARY rebuttal benchmark on GALAXY (consumer RTX 3090, 24G) = the paper's deployment
# target ("real-time on consumer GPUs"). SAME-HARDWARE comparison: NT-2.5B, Enformer,
# DNABERT-2 teachers + the paper's BPNet student (`original`, ~0.12M params), ALL in one
# process on one 3090 GPU, FULL splice_sites_all test set, fp16, batch_size=16,
# max_length=1000 bp. CSV records gpu_name, total_time_s, num_samples, seq_length,
# batch_size, peak_memory_mb, params per model.
#
# Caduceus is EXCLUDED on galaxy: galaxy GLIBC 2.31 < 2.34 required by mamba/triton kernels
# (established empirically). Caduceus is separately measured on laniakea (job 243599) and
# reported as a different-hardware footnote, NOT mixed into this same-HW table.
#
# NT-2.5B fp16 weights ~5GB -> expected to fit 24G at bs16/seqlen1000. If it OOMs, lower
# --batch-size (8 then 4); the chosen batch_size is recorded in every CSV row.
#   sbatch slurm/benchmark_galaxy.sbatch
#SBATCH --job-name=bench-galaxy
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
# Disable hf_xet/CAS downloader (its multi-GB scratch blew the home quota); plain HTTP
# writes straight into blobs/. NT-2.5B shards should already be fully cached by now.
export HF_HUB_DISABLE_XET=1
export HF_XET_CACHE="$REPO/.hf_cache/xet"
export OUTPUT_PATH=/home/pengchx3/text-dna/OmegaGenome_Revise_202606/rebuttal_infra/benchmark_runs_galaxy

PY="$REPO/.venv_carbon_portable/bin/python"

echo "=== GPU on this node ==="
"$PY" -c "import torch; print('device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'); print('total mem GB:', round(torch.cuda.get_device_properties(0).total_memory/1e9,1) if torch.cuda.is_available() else 0)"

# num-batches None => ENTIRE test set. Caduceus excluded (glibc on galaxy).
"$PY" -u -m src.train.benchmark_inference full_benchmark \
    --task-names splice_sites_all \
    --teachers nt enformer dnabert2 \
    --student-sizes original \
    --precision fp16 \
    --batch-size 16 \
    --max-length 1000 \
    --warmup-batches 10 \
    --num-batches None \
    --skip-on-error

echo "=== GALAXY BENCHMARK DONE; CSVs under $OUTPUT_PATH/benchmark/splice_sites_all/ ==="
ls -lt "$OUTPUT_PATH/benchmark/splice_sites_all/" 2>/dev/null | head
