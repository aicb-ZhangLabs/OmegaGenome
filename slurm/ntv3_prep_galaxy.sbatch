#!/bin/bash
# Build the benchmark prep cache on GALAXY (native /srv/disk00 SSD = ~6 GB/s reads, ~50x sshfs) with
# threaded bigWig reads — minutes, not hours. CPU-only (prep needs no GPU). The cache lands on the
# shared SSD, so the voyager training job then loads it instantly.
#   sbatch slurm/ntv3_prep_galaxy.sbatch
#SBATCH --job-name=ntv3-prep
#SBATCH --nodelist=galaxy
#SBATCH --partition=zhanglab.p
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --nice=2000
#SBATCH --requeue
#SBATCH --time=6:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail
REPO=${OG_NTV3_ROOT:-$PWD}
cd "$REPO"
HOME_HF=${OG_ROOT:-$PWD}/.hf_cache
export HF_HOME="$HOME_HF" HF_HUB_CACHE="$HOME_HF/hub" TRANSFORMERS_CACHE="$HOME_HF/hub" CUDA_VISIBLE_DEVICES=""
export HF_TOKEN="$(cat ${HF_TOKEN_FILE})"
PV=${OG_ROOT:-$PWD}/.venv_carbon_portable/bin/python

SSD=${OG_SCRATCH:-$PWD/output}
mountpoint -q "$SSD" 2>/dev/null || SSD=${OG_SCRATCH:-$PWD/output}   # on galaxy: native /srv/disk00
DATA="$SSD/ntv3_benchmark_data"

# prep_only: reads bigWigs natively on galaxy (fast) + caches; no GPU, no model training.
"$PV" -m src.train.finetune_ntv3 --data_dir "$DATA" --model InstaDeepAI/NTv3_650M_post \
      --out "$SSD/ntv3_targets/ntv3_finetune" --prep_only \
      --window 16384 --n_train 2000 --n_test 400
echo "=== prep cache built on galaxy native SSD; voyager training can now load it ==="