#!/bin/bash
# Carbon-3B -> deploy_120k BPNet distillation runner. Sources the shared env (HF cache, venv, wandb)
# and runs a distillation config INLINE on the allocated GPU (mode="run" configs: carbon-smoke).
# For the parallel multi-task runs (carbon-raw / carbon-l2norm, mode="slurm") submit from the login
# node directly so slurm_fn fans out one job per task.
#   sbatch slurm/carbon_distill.sbatch carbon-smoke
#SBATCH --job-name=carbon-distill
#SBATCH --nodelist=laniakea
#SBATCH --partition=zhanglab.p
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --nice=0
#SBATCH --requeue
#SBATCH --time=12:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail
REPO=/home/pengchx3/text-dna/OmegaGenome_Revise_202606/code_carbon
cd "$REPO"
source slurm/env_setup.sh
# Node-aware SSD root: the galaxy SSD is an sshfs mount (/tmp/galaxy_srv_disk00) on laniakea/voyager but
# a LOCAL disk (/srv/disk00/sshfs) on galaxy itself — same physical disk, different path. Pin HF_OVERRIDE
# to whichever exists so galaxy jobs also find the shared HF cache (no-op on laniakea/voyager where the
# sshfs path resolves first). CARBON_OUTPUT_BASE (cache+ckpts) is already node-aware via _default_output_base.
for _b in /tmp/galaxy_srv_disk00/pengchx3 /srv/disk00/sshfs/pengchx3; do [ -d "$_b" ] && { _SSD="$_b"; break; }; done
[ -n "${_SSD:-}" ] && export HF_OVERRIDE="$_SSD/hf_cache_shared"
# Optional HF-cache override (big base-model downloads -> SSD, not the tight /home quota).
if [ -n "${HF_OVERRIDE:-}" ]; then
  export HF_HOME="$HF_OVERRIDE" HF_HUB_CACHE="$HF_OVERRIDE/hub" HUGGINGFACE_HUB_CACHE="$HF_OVERRIDE/hub" \
         HF_DATASETS_CACHE="$HF_OVERRIDE/datasets" TRANSFORMERS_CACHE="$HF_OVERRIDE/hub" HF_ASSETS_CACHE="$HF_OVERRIDE/assets"
  mkdir -p "$HF_OVERRIDE/hub" "$HF_OVERRIDE/datasets" "$HF_OVERRIDE/assets"
  echo "HF cache overridden -> $HF_OVERRIDE"
fi
export HF_TOKEN="$(cat /home/pengchx3/text-dna/huggingface-token-0616.txt)"
export WANDB_INIT_TIMEOUT=300
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True  # reduce fragmentation OOM
PY="$REPO/.venv_carbon_portable/bin/python"

# Pass ALL args through to the distill CLI so per-task runs work, e.g.:
#   sbatch slurm/carbon_distill.sbatch carbon-raw --task-names H3K27me3 --slurm-config.mode run
ARGS=("$@"); [ ${#ARGS[@]} -eq 0 ] && ARGS=(carbon-smoke)
echo "=== carbon distillation: ${ARGS[*]} (inline) ==="
"$PY" -m src.train.distill "${ARGS[@]}"
echo "=== done: ${ARGS[*]} ==="
