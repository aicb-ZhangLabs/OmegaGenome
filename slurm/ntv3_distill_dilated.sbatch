#!/bin/bash
# Train the DilatedTrackNet student on the ALREADY-CACHED 8k-window 100M targets (no re-gen) ->
# direct apples-to-apples comparison vs the BPNet baseline at the same data. Node-flexible.
#   sbatch slurm/ntv3_distill_dilated.sbatch
#SBATCH --job-name=ntv3-dilated
#SBATCH --nodes=1
#SBATCH --partition=zhanglab.p
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --nice=2000
#SBATCH --requeue
#SBATCH --time=1-00:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail
REPO=${OG_NTV3_ROOT:-$PWD}
cd "$REPO"
HOME_HF=${OG_ROOT:-$PWD}/.hf_cache
export HF_HOME="$HOME_HF" HF_HUB_CACHE="$HOME_HF/hub" TRANSFORMERS_CACHE="$HOME_HF/hub"
PY=${OG_ROOT:-$PWD}/.venv_carbon_portable/bin/python
DATA=${OG_SCRATCH:-$PWD/output}/ntv3_targets/v1_100m_8k   # 8k-window cached 100M targets

# dilated student (RF ~32kb covers the 16kb window); mse (ablation: mse~poisson, both > correlation)
"$PY" -m src.train.distill_tracks --data "$DATA" --out "$DATA/student_dilated" \
      --student dilated --hidden 128 --n_blocks 14 --loss mse \
      --epochs 60 --lr 1e-3 --batch_size 4
echo "=== done: $DATA/student_dilated/track_distill_result.json ==="
