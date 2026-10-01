#!/bin/bash
# NTv3 multi-track distillation v1 (K562, 9 tracks): generate NTv3-650M per-bp targets, then
# distill into a BPNetRegressor student and report per-track Pearson. One voyager (H100) job.
#   sbatch slurm/ntv3_distill_v1.sbatch
# Storage: model cache -> home (faster load, no sshfs memmap risk); targets+results -> SSD
# (big, accessible from any node). Requires the galaxy SSD mounted on voyager; falls back to
# home if not. Confirm with the lab before submitting.
#SBATCH --job-name=ntv3-distill-v1
#SBATCH --nodes=1
#SBATCH --nodelist=voyager
#SBATCH --partition=zhanglab.p
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --nice=2000
#SBATCH --requeue
#SBATCH --time=1-00:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail
REPO=${OG_NTV3_ROOT:-$PWD}
cd "$REPO"

# model cache on home (fast, fits); HF token for gated 650M
HOME_HF=${OG_ROOT:-$PWD}/.hf_cache
export HF_HOME="$HOME_HF" HF_HUB_CACHE="$HOME_HF/hub" HUGGINGFACE_HUB_CACHE="$HOME_HF/hub" \
       TRANSFORMERS_CACHE="$HOME_HF/hub" TOKENIZERS_PARALLELISM=false
export HF_TOKEN="$(cat ${HF_TOKEN_FILE})"

# big outputs on SSD if mounted on this node, else home
SSD=${OG_SCRATCH:-$PWD/output}
if mountpoint -q "$SSD" 2>/dev/null; then OUT="$SSD/ntv3_targets/v1"; else OUT="$REPO/.ntv3_run/v1"; fi
mkdir -p "$OUT"
echo "outputs -> $OUT"

PY="$REPO/.venv_carbon_portable/bin/python"
# this worktree shares code_carbon's portable venv
[ -x "$PY" ] || PY=${OG_ROOT:-$PWD}/.venv_carbon_portable/bin/python

echo "=== 1. generate NTv3-650M per-bp targets ==="
"$PY" -m src.train.ntv3_gen_targets --model InstaDeepAI/NTv3_650M_post --out "$OUT" \
      --window 16384 --n_train 2000 --n_test 400 --batch_size 2

echo "=== 2. distill into BPNetRegressor student + per-track Pearson ==="
"$PY" -m src.train.distill_tracks --data "$OUT" --out "$OUT/student" \
      --model_size medium --epochs 40 --lr 1e-3 --batch_size 8
echo "=== done. result: $OUT/student/track_distill_result.json ==="
