#!/bin/bash
# NTv3 multi-track inference LATENCY + PEAK MEMORY benchmark.
# Times the FULL NTv3 test set (10531 windows @ 32768 bp) for the 650M teacher + 8M/100M/0.12M
# students, single-track (t0..t33) + joint-34. GALAXY-ONLY (data-job policy), SSD node-local.
#
#   SPECS=<specs.json> DEV=cuda  sbatch --gres=gpu:1 slurm/ntv3_latency.sbatch  # GPU sweep (needs --gres!)
#   SPECS=<specs.json> DEV=cpu   sbatch slurm/ntv3_latency.sbatch               # CPU sweep (no GPU)
#   SPECS=<smoke.json> DEV=cuda  sbatch --gres=gpu:1 slurm/ntv3_latency.sbatch  # smoke
# NOTE: no #SBATCH --gres line here so the CPU sweep doesn't reserve a GPU; pass --gres=gpu:1 at
# submit time for the cuda sweeps.
# Extra args pass through to the module (e.g. --max-test-windows 200).
#SBATCH --job-name=ntv3lat
#SBATCH --nodes=1
#SBATCH --nodelist=galaxy
#SBATCH --partition=zhanglab.p
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --nice=0
#SBATCH --requeue
#SBATCH --time=30-00:00:00
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
DATA="$SSD/ntv3_benchmark_data"
STAGE="${SLURM_TMPDIR:-/tmp/ntv3_stage_$USER}"

DEV="${DEV:-cuda}"
SPECS="${SPECS:-$SSD/ntv3_latency/latency_specs.json}"
TAG="${TAG:-$(basename "$SPECS" .json)_${DEV}}"
OUTP="$SSD/ntv3_latency/results_${TAG}"

# CPU sweep: the 650M teacher is timed over --cpu-batches and extrapolated (handled in the module).
AMP_FLAG=""; [ "$DEV" = "cuda" ] && AMP_FLAG="--amp"

echo "=== ntv3 latency: dev=$DEV specs=$SPECS out=$OUTP ==="
"$PV" -u -m src.train.benchmark_latency_ntv3 \
      --data_dir "$DATA" --stage_dir "$STAGE" \
      --device "$DEV" $AMP_FLAG \
      --batch_size "${BS:-4}" --num_workers 8 \
      --specs "$SPECS" --out-prefix "$OUTP" "$@"
echo "=== done: ${OUTP}.csv ==="
