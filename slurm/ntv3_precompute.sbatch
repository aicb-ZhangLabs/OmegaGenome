#!/bin/bash
# Precompute a teacher's per-bp track logits over a fixed window set (so cached KD needs no teacher fwd).
# Args pass through to src.train.precompute_teacher_logits (e.g. --teacher ... --track_subset 12 ...).
#SBATCH --job-name=ntv3precomp
#SBATCH --nodes=1
#SBATCH --partition=zhanglab.p
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=30-00:00:00
#SBATCH --output=slurm-%x-%j.out
set -euo pipefail
REPO=${OG_NTV3_ROOT:-$PWD}
cd "$REPO"
HOME_HF=${OG_ROOT:-$PWD}/.hf_cache
export HF_HOME="$HOME_HF" HF_HUB_CACHE="$HOME_HF/hub" TRANSFORMERS_CACHE="$HOME_HF/hub"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1            # never write /home (quota)
export HF_TOKEN="$(cat ${HF_TOKEN_FILE})"
export TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PV=${OG_ROOT:-$PWD}/.venv_carbon_portable/bin/python
SSD=${OG_SCRATCH:-$PWD/output}
mountpoint -q "$SSD" 2>/dev/null || SSD=${OG_SCRATCH:-$PWD/output}
"$PV" -u -m src.train.precompute_teacher_logits --data_dir "$SSD/ntv3_benchmark_data" \
      --tokenizer "$SSD/ntv3_models/8m_pre" "$@"
echo "=== precompute done ==="
