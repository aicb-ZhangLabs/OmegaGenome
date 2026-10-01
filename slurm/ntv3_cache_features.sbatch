#!/bin/bash
# Build the generalist teacher FEATURE cache (one epoch over all train windows). Args pass to
# src.train.cache_teacher_features. Writes to the separate `feat/` namespace (no conflict with logit caches).
#SBATCH --job-name=ntv3featcache
#SBATCH --nodes=1
#SBATCH --partition=zhanglab.p
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=slurm-%x-%j.out
set -euo pipefail
REPO=${OG_NTV3_ROOT:-$PWD}
cd "$REPO"
HOME_HF=${OG_ROOT:-$PWD}/.hf_cache
export HF_HOME="$HOME_HF" HF_HUB_CACHE="$HOME_HF/hub" TRANSFORMERS_CACHE="$HOME_HF/hub"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export HF_TOKEN="$(cat ${HF_TOKEN_FILE})"
export TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PV=${OG_ROOT:-$PWD}/.venv_carbon_portable/bin/python
SSD=${OG_SCRATCH:-$PWD/output}
mountpoint -q "$SSD" 2>/dev/null || SSD=${OG_SCRATCH:-$PWD/output}
"$PV" -u -m src.train.cache_teacher_features \
      --data_dir "$SSD/ntv3_benchmark_data" --tokenizer "$SSD/ntv3_models/8m_pre" \
      --teacher "$SSD/ntv3_targets/ntv3_ft_faithful_s0/best_model.pth" \
      --out_dir "$SSD/ntv3_cache/teacher_feat" "$@"
echo "=== feature cache build done ==="
