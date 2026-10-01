#!/bin/bash
# Per-teacher panels: extract second_to_last FEATURES + class LOGITS for an HF teacher
# (NT / DNABERT-2 / Caduceus) + its distilled BPNet student + from-scratch BPNet baseline on the FULL
# splice_sites_all test set. Driven by env vars TEACHER, OMEGA, DKD, BASE.
#   TEACHER=caduceus OMEGA=<vanilla.pt> DKD=<dkd_or_same.pt> BASE=<model_final.pt> \
#     sbatch --nodelist=voyager slurm/extract_feats_logits_multi.sbatch
#SBATCH --job-name=featlogit-multi
#SBATCH --nodes=1
#SBATCH --partition=zhanglab.p
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --requeue
#SBATCH --time=30-00:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail
REPO=${OG_ROOT:-$PWD}
cd "$REPO"

HOME_HF="$REPO/.hf_cache"
export HF_HOME="$HOME_HF" HF_HUB_CACHE="$HOME_HF/hub" HUGGINGFACE_HUB_CACHE="$HOME_HF/hub"
export HF_DATASETS_CACHE="$HOME_HF/datasets" TRANSFORMERS_CACHE="$HOME_HF/hub" HF_ASSETS_CACHE="$HOME_HF/assets"
export HF_XET_CACHE="$HOME_HF/xet" HF_HUB_DISABLE_XET=1
export HF_TOKEN="$(cat ${HF_TOKEN_FILE})"
export TOKENIZERS_PARALLELISM=false WANDB_MODE=offline

PY="$REPO/.venv_carbon_portable/bin/python"
N="${N:-100000}"

echo "=== node / GPU ==="; hostname
"$PY" -c "import torch; print('cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"

"$PY" -u analysis/extract_feats_logits_multi.py \
    --teacher "$TEACHER" --omega-ckpt "$OMEGA" --dkd-ckpt "$DKD" --baseline-ckpt "$BASE" --n "$N"

echo "=== DONE; feature_logit_data_${TEACHER}.npz in analysis/figs/ ==="
ls -la "$REPO/analysis/figs/feature_logit_data_${TEACHER}.npz"
