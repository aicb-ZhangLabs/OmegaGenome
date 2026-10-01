#!/bin/bash
# NTv3-100M multi-track distillation (K562, 9 tracks) — node-flexible (NOT pinned to voyager) so it
# runs on the first free GPU on any node; 100M fits a 24GB 3090. Smaller-teacher preliminary while
# the 650M run (voyager-only) waits for an 80GB slot.
#   sbatch slurm/ntv3_distill_100m.sbatch
#SBATCH --job-name=ntv3-distill-100m
#SBATCH --nodes=1
#SBATCH --partition=zhanglab.p
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --nice=2000
#SBATCH --requeue
#SBATCH --time=1-00:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail
REPO=${OG_NTV3_ROOT:-$PWD}
cd "$REPO"
HOME_HF=${OG_ROOT:-$PWD}/.hf_cache
export HF_HOME="$HOME_HF" HF_HUB_CACHE="$HOME_HF/hub" TRANSFORMERS_CACHE="$HOME_HF/hub" TOKENIZERS_PARALLELISM=false

SSD=${OG_SCRATCH:-$PWD/output}
mountpoint -q "$SSD" 2>/dev/null || SSD=${OG_SCRATCH:-$PWD/output}   # galaxy-native path
OUT="$SSD/ntv3_targets/v1_100m"
mkdir -p "$OUT"; echo "outputs -> $OUT"
PY=${OG_ROOT:-$PWD}/.venv_carbon_portable/bin/python
M100="$SSD/ntv3_local/100m_post"   # 100M snapshot on the SSD (not /extra)

echo "=== 1. generate NTv3-100M per-bp targets ==="
"$PY" -m src.train.ntv3_gen_targets --model "$M100" --out "$OUT" \
      --window 16384 --n_train 2000 --n_test 400 --batch_size 8
echo "=== 2. distill into BPNetRegressor + per-track Pearson ==="
"$PY" -m src.train.distill_tracks --data "$OUT" --out "$OUT/student" \
      --model_size medium --epochs 40 --lr 1e-3 --batch_size 8
echo "=== done: $OUT/student/track_distill_result.json ==="
