#!/bin/bash
# Signal-track figure inference: joint-34 650M teacher | distilled 8M
# student | from-scratch 8M baseline, over a few held-out TEST windows. GALAXY-ONLY (NTv3 policy).
#   sbatch --gres=gpu:1 slurm/signal_track.sbatch
#SBATCH --job-name=sigtrack
#SBATCH --nodes=1
#SBATCH --nodelist=galaxy
#SBATCH --partition=zhanglab.p
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --nice=0
#SBATCH --requeue
#SBATCH --time=12:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail
REPO=${OG_NTV3_ROOT:-$PWD}
cd "$REPO"
HOME_HF=${OG_ROOT:-$PWD}/.hf_cache
export HF_HOME="$HOME_HF" HF_HUB_CACHE="$HOME_HF/hub" TRANSFORMERS_CACHE="$HOME_HF/hub"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export HF_TOKEN="$(cat ${HF_TOKEN_FILE} 2>/dev/null || true)"
export TOKENIZERS_PARALLELISM=false
export TRITON_CACHE_DIR="/tmp/triton_${USER}_$(hostname -s)"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PV=${OG_ROOT:-$PWD}/.venv_carbon_portable/bin/python

SSD=${OG_SCRATCH:-$PWD/output}
mountpoint -q "$SSD" 2>/dev/null || SSD=${OG_SCRATCH:-$PWD/output}
export SSD
export NTV3_DATA_DIR="$SSD/ntv3_benchmark_data"
OUT="$SSD/ntv3_latency/signal_track_preds.npz"

echo "=== signal-track inference -> $OUT ==="
"$PV" -u scripts/signal_track_inference.py --out "$OUT" --n-windows "${NWIN:-8}"
echo "=== done: $OUT ==="
