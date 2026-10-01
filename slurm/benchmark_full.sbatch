#!/bin/bash
# FULL inference latency + peak-memory benchmark for the efficiency claims.
# ALL teachers (NT-2.5B, Caduceus, Enformer, DNABERT-2) + the paper's BPNet student
# (`original`, ~0.12M params), on the FULL splice_sites_all test set, fp16,
# batch_size=16, seq_len(max_length)=1000 bp.
# CSV reports per model: total_time_s (whole-dataset wall-clock), num_samples (N),
# seq_length (bp), batch_size, mean/median/p95 latency, throughput, peak_memory_mb, params.
#
# NT-2.5B needs a big-mem GPU at fp16 (~5GB weights) -> laniakea (6000Ada 49G). DNABERT-2
# may fail on Triton flash-attn; skip_on_error=True keeps the rest going. mamba_ssm +
# causal_conv1d prebuilt wheels (cu11/torch2.6/cp311 cxx11abiFALSE) are already installed
# in the carbon venv, so Caduceus runs in-process.
#   sbatch slurm/benchmark_full.sbatch
#SBATCH --job-name=bench-full
#SBATCH --nodes=1
#SBATCH --partition=zhanglab.p
#SBATCH --nodelist=laniakea
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
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
# Disable the hf_xet/CAS downloader: it stages a multi-GB scratch copy that blew the home
# quota mid-download. Plain-HTTP download writes straight into blobs/ (no double space).
export HF_HUB_DISABLE_XET=1
export HF_XET_CACHE="$REPO/.hf_cache/xet"   # if xet ever runs, keep its scratch under the repo cache
export OUTPUT_PATH=${OG_WORKSPACE:-$PWD/..}/analysis/benchmark_runs

PY="$REPO/.venv_carbon_portable/bin/python"

# Caduceus needs mamba_ssm + causal_conv1d (Mamba kernels). Already installed in the venv;
# verify the import resolves on the GPU node before the run.
echo "=== mamba_ssm import check ==="
"$PY" -c "import causal_conv1d, mamba_ssm; print('mamba_ssm', mamba_ssm.__version__, 'causal_conv1d', causal_conv1d.__version__)" \
    || echo "WARN: mamba_ssm import failed; caduceus may be skipped via --skip-on-error."

# num-batches None => use the ENTIRE test set (whole-dataset total_time_s reported per model).
"$PY" -u -m src.train.benchmark_inference full_benchmark \
    --task-names splice_sites_all \
    --teachers nt caduceus enformer dnabert2 \
    --student-sizes original \
    --precision fp16 \
    --batch-size 16 \
    --max-length 1000 \
    --warmup-batches 10 \
    --num-batches None \
    --skip-on-error

echo "=== FULL BENCHMARK DONE; CSVs under $OUTPUT_PATH/benchmark/splice_sites_all/ ==="
ls -lt "$OUTPUT_PATH/benchmark/splice_sites_all/" 2>/dev/null | head
