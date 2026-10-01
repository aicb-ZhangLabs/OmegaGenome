#!/bin/bash
# Feature-space comparison: DKD vs OmegaGenome student vs Enformer teacher on
# splice_sites_all. Small GPU job (one Enformer forward over a few hundred seqs + two tiny BPNets).
#   sbatch slurm/feature_space_compare.sbatch         # full (n=300)
#   N=8 sbatch slurm/feature_space_compare.sbatch     # smoke
#SBATCH --job-name=featcos
#SBATCH --nodes=1
#SBATCH --partition=zhanglab.p
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --requeue
#SBATCH --time=30-00:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail
REPO=${OG_ROOT:-$PWD}
cd "$REPO"

# HF cache pinned to repo .hf_cache (the ~/.bashrc /srv trap doesn't apply: every HF_* set here).
HOME_HF="$REPO/.hf_cache"
export HF_HOME="$HOME_HF" HF_HUB_CACHE="$HOME_HF/hub" HUGGINGFACE_HUB_CACHE="$HOME_HF/hub"
export HF_DATASETS_CACHE="$HOME_HF/datasets" TRANSFORMERS_CACHE="$HOME_HF/hub" HF_ASSETS_CACHE="$HOME_HF/assets"
export HF_XET_CACHE="$HOME_HF/xet" HF_HUB_DISABLE_XET=1
export HF_TOKEN="$(cat ${HF_TOKEN_FILE})"
export TOKENIZERS_PARALLELISM=false WANDB_MODE=offline

PY="$REPO/.venv_carbon_portable/bin/python"
N="${N:-300}"

echo "=== node / GPU ==="; hostname; (ldd --version 2>/dev/null || true) | sed -n '1p' || true
"$PY" -c "import torch; print('cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"

"$PY" -u analysis/feature_space_compare.py --n "$N"

echo "=== DONE; outputs in analysis/figs/ ==="
ls -la "$REPO/analysis/figs/"
