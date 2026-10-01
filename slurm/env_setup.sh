# Shared environment for Carbon SLURM jobs. Sourced by the sbatch scripts.
#
# IMPORTANT: ~/.bashrc pins HF caches to /srv/disk00/sshfs/... (the galaxy SSD), which is
# only writable on galaxy. To run on laniakea/voyager too we override the whole HF cache
# family to a node-agnostic home path. Also forces wandb online.
REPO=${OG_ROOT:-$PWD}

export HF_HOME="$REPO/.hf_cache"
export HF_HUB_CACHE="$HF_HOME/hub"
export HUGGINGFACE_HUB_CACHE="$HF_HUB_CACHE"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TRANSFORMERS_CACHE="$HF_HOME/hub"
export HF_ASSETS_CACHE="$HF_HOME/assets"
export TOKENIZERS_PARALLELISM=false

export WANDB_MODE=online
export WANDB_CACHE_DIR="$REPO/.wandb_cache"
export WANDB_DIR="$REPO"

# Pin the output dir. Otherwise nntool timestamps it per process, scattering the 18
# per-task jobs into separate folders. Pinning puts every {task}_finetuned/ under one
# parent so results consolidate and stage-2 distillation can find all teacher checkpoints.
export OUTPUT_PATH=${OG_STORE:-$PWD/data}/OmegaGenome_carbon_runs

mkdir -p "$HF_HUB_CACHE" "$HF_DATASETS_CACHE" "$HF_ASSETS_CACHE" "$WANDB_CACHE_DIR"

# Portable venv (uv-managed python under ~/.local, shared across nodes) so jobs run on
# galaxy too. The plain .venv_carbon points its interpreter at laniakea's /usr/bin/python3.11
# which is absent on galaxy.
PY="$REPO/.venv_carbon_portable/bin/python"
