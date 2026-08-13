#!/bin/bash
# 1-epoch dry-run of the below-baseline rerun pipeline (preflight before the 81-run sweep).
# Exercises: teacher load -> teacher-logit precompute -> student train (1 epoch) -> best-val ckpt
# -> test eval -> final_summary.json. Uses DNABERT-2 (small teacher) so it is cheap. Runs on a
# free voyager H100 slot (laniakea is at the 8-GPU cap). Same env as rerun_below_baseline.sbatch.
#SBATCH --job-name=loom_rerun-dryrun-distill
#SBATCH --nodelist=voyager
#SBATCH --partition=zhanglab.p
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=24G
#SBATCH --time=0:30:00
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

echo "=== DRY-RUN: dnabert2 / H3K9ac / 1 epoch / seed 0 ==="
"$PY" -m src.train.distill dna_bert_v2 --task-names H3K9ac \
  --distillation-config.weight-ce 0.5 --distillation-config.weight-kl 1 --distillation-config.weight-mse 2 \
  --distillation-config.temperature 4 --trainer-config.epochs 1 --trainer-config.eval-every-n-epochs 1 \
  --trainer-config.early-stop-patience 0 \
  --trainer-config.output-dir /tmp/galaxy_srv_disk00/pengchx3/rerun_below_baseline_DRYRUN/dnabert2 \
  --random-state 0 --slurm-config.mode run
echo "=== DRY-RUN done; checking final_summary.json ==="
find /tmp/galaxy_srv_disk00/pengchx3/rerun_below_baseline_DRYRUN/dnabert2 -name final_summary.json -exec cat {} \;
echo "=== DRY-RUN OK ==="
