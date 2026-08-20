#!/bin/bash
# 1-config 1-epoch DRY-RUN for the method3 pipeline: dkd / H3K9ac / seed 0.
# Validates method dispatch + teacher-load(cache MISS @ seed0) + precompute + student train +
# best-val ckpt + test eval + final_summary.json. NOT dependent on BPNet; voyager (80G, no OOM).
#SBATCH --job-name=loom_method3-dryrun
#SBATCH --nodelist=voyager
#SBATCH --partition=zhanglab.p
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=64G
#SBATCH --time=1:30:00
#SBATCH --output=slurm-%x-%j.out

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

echo "=== DRYRUN START $(hostname) $(date +%T) ==="
SECONDS=0
"$PY" -m src.train.distill nt_vanilla --task-names H3K9ac \
  --distillation-config.distill-method dkd --distillation-config.weight-ce 0.5 \
  --distillation-config.weight-kl 1 --distillation-config.weight-mse 0 \
  --distillation-config.temperature 2 --distillation-config.dkd-alpha 2.0 \
  --distillation-config.dkd-beta 4.0 --trainer-config.epochs 1 \
  --trainer-config.eval-every-n-epochs 1 --trainer-config.early-stop-patience 0 \
  --trainer-config.output-dir /tmp/galaxy_srv_disk00/pengchx3/method3_18task_DRYRUN/dkd \
  --random-state 0 --slurm-config.mode run
echo "=== DRYRUN EXIT=$? elapsed=${SECONDS}s $(date +%T) ==="
