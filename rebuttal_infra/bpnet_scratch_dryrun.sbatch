#!/bin/bash
# 1-epoch dry-run of the BPNet from-scratch (pure-CE, no teacher) pipeline. Validates before the 54-run sweep.
#SBATCH --job-name=loom_bpnet-scratch-dryrun
#SBATCH --nodelist=laniakea
#SBATCH --partition=zhanglab.p
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=24G
#SBATCH --time=0:30:00
#SBATCH --output=slurm-%x-%j.out
set -euo pipefail
cd /home/pengchx3/text-dna/OmegaGenome_Revise_202606/code_carbon
source slurm/env_setup.sh
for _b in /tmp/galaxy_srv_disk00/pengchx3 /srv/disk00/sshfs/pengchx3; do [ -d "$_b" ] && { _SSD="$_b"; break; }; done
export HF_OVERRIDE="$_SSD/hf_cache_shared" HF_HOME="$_SSD/hf_cache_shared" \
  HF_HUB_CACHE="$_SSD/hf_cache_shared/hub" HUGGINGFACE_HUB_CACHE="$_SSD/hf_cache_shared/hub" \
  HF_DATASETS_CACHE="$_SSD/hf_cache_shared/datasets" TRANSFORMERS_CACHE="$_SSD/hf_cache_shared/hub"
export HF_TOKEN="$(cat /home/pengchx3/text-dna/huggingface-token-0616.txt)" WANDB_MODE=offline
export TRITON_CACHE_DIR="/tmp/${USER}_triton_cache"; mkdir -p "$TRITON_CACHE_DIR"
.venv_carbon_portable/bin/python -m src.train.distill carbon-raw-original --task-names H3K4me3 \
  --distillation-config.weight-ce 0.5 --distillation-config.weight-kl 0 --distillation-config.weight-mse 0 \
  --trainer-config.epochs 1 --trainer-config.eval-every-n-epochs 1 --trainer-config.early-stop-patience 0 \
  --trainer-config.output-dir /tmp/galaxy_srv_disk00/pengchx3/bpnet_scratch_3seed_DRYRUN/bpnet \
  --random-state 0 --slurm-config.mode run
echo "=== DRY-RUN summary ==="
find /tmp/galaxy_srv_disk00/pengchx3/bpnet_scratch_3seed_DRYRUN/bpnet -name final_summary.json -exec cat {} \;
echo "=== BPNET_SCRATCH_DRYRUN_OK ==="
