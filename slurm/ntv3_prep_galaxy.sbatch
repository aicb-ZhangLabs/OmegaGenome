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
REPO=/home/pengchx3/text-dna/OmegaGenome_Revise_202606/code_ntv3
cd "$REPO"
HOME_HF=/home/pengchx3/text-dna/OmegaGenome_Revise_202606/code_carbon/.hf_cache
export HF_HOME="$HOME_HF" HF_HUB_CACHE="$HOME_HF/hub" TRANSFORMERS_CACHE="$HOME_HF/hub" CUDA_VISIBLE_DEVICES=""
export HF_TOKEN="$(cat /home/pengchx3/text-dna/huggingface-token-0616.txt)"
PV=/home/pengchx3/text-dna/OmegaGenome_Revise_202606/code_carbon/.venv_carbon_portable/bin/python

SSD=/tmp/galaxy_srv_disk00/pengchx3
mountpoint -q "$SSD" 2>/dev/null || SSD=/srv/disk00/sshfs/pengchx3   # on galaxy: native /srv/disk00
DATA="$SSD/ntv3_benchmark_data"

# prep_only: reads bigWigs natively on galaxy (fast) + caches; no GPU, no model training.
"$PV" -m src.train.finetune_ntv3 --data_dir "$DATA" --model InstaDeepAI/NTv3_650M_post \
      --out "$SSD/ntv3_targets/ntv3_finetune" --prep_only \
      --window 16384 --n_train 2000 --n_test 400
echo "=== prep cache built on galaxy native SSD; voyager training can now load it ==="