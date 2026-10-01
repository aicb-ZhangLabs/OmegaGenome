#!/bin/bash
# Measure 650M teacher feature compressibility (PCA spectrum, positional autocorr, logit-residual) so we know
# empirically how much a compressed feature cache would lose. Short job. Args pass to measure_feature_spectrum.
#SBATCH --job-name=ntv3featspec
#SBATCH --nodes=1
#SBATCH --partition=zhanglab.p
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
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
"$PV" -u -m src.train.measure_feature_spectrum \
      --data_dir "$SSD/ntv3_benchmark_data" --tokenizer "$SSD/ntv3_models/8m_pre" \
      --teacher "$SSD/ntv3_targets/ntv3_ft_faithful_s0/best_model.pth" \
      --out "$SSD/ntv3_cache/feature_spectrum.json" "$@"
echo "=== feature-spectrum measurement done ==="
