#!/bin/bash
# 1-epoch dry-run for the HP-fix pipeline: dist / splice_sites_acceptors / seed0.
# Mirrors method_hpfix.sbatch env EXACTLY. Confirms teacher cache HIT + student is LEARNING
# (not stuck at 0.0 collapse) before the 27-run array is submitted.
set -euo pipefail
REPO=/home/pengchx3/text-dna/OmegaGenome_Revise_202606/code_carbon
cd "$REPO"
source slurm/env_setup.sh
for _b in /tmp/galaxy_srv_disk00/pengchx3 /srv/disk00/sshfs/pengchx3; do [ -d "$_b" ] && { _SSD="$_b"; break; }; done
[ -n "${_SSD:-}" ] && export HF_OVERRIDE="$_SSD/hf_cache_shared"
if [ -n "${HF_OVERRIDE:-}" ]; then
  export HF_HOME="$HF_OVERRIDE" HF_HUB_CACHE="$HF_OVERRIDE/hub" HUGGINGFACE_HUB_CACHE="$HF_OVERRIDE/hub" \
         HF_DATASETS_CACHE="$HF_OVERRIDE/datasets" TRANSFORMERS_CACHE="$HF_OVERRIDE/hub" HF_ASSETS_CACHE="$HF_OVERRIDE/assets"
  mkdir -p "$HF_OVERRIDE/hub" "$HF_OVERRIDE/datasets" "$HF_OVERRIDE/assets"
fi
export HF_TOKEN="$(cat /home/pengchx3/text-dna/huggingface-token-0616.txt)"
export WANDB_INIT_TIMEOUT=300 WANDB_MODE=offline
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRITON_CACHE_DIR="/tmp/${USER}_triton_cache"; mkdir -p "$TRITON_CACHE_DIR"
PY="$REPO/.venv_carbon_portable/bin/python"
echo "=== DRY-RUN host=$(hostname) start=$(date) ==="
"$PY" -m src.train.distill nt_vanilla --task-names splice_sites_acceptors \
  --distillation-config.distill-method dist --distillation-config.weight-ce 1.0 \
  --distillation-config.weight-kl 0.5 --distillation-config.weight-mse 0 \
  --distillation-config.temperature 4 --trainer-config.epochs 1 \
  --trainer-config.eval-every-n-epochs 1 --trainer-config.early-stop-patience 0 \
  --trainer-config.output-dir /tmp/galaxy_srv_disk00/pengchx3/method_hpfix_DRYRUN/dist/splice_sites_acceptors/seed0 \
  --random-state 0 --slurm-config.mode run
echo "=== DRY-RUN done=$(date) ==="
