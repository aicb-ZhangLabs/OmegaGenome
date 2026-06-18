#!/bin/bash
# Fine-tune NTv3 on the NTv3 Benchmark functional tracks (their data) -> the benchmark teacher.
# Full-FT of 650M needs an 80GB H100 (voyager). For head-only (cheap), add --freeze_backbone and it
# runs node-flexibly. Compare the result to the published CSV.
#   sbatch slurm/ntv3_finetune.sbatch                      # full-FT 650M (voyager)
#   sbatch slurm/ntv3_finetune.sbatch --freeze_backbone    # head-only baseline
#SBATCH --job-name=ntv3-finetune
#SBATCH --nodes=1
#SBATCH --nodelist=voyager
#SBATCH --partition=zhanglab.p
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=110G
#SBATCH --nice=2000
#SBATCH --requeue
#SBATCH --time=2-00:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail
REPO=/home/pengchx3/text-dna/OmegaGenome_Revise_202606/code_ntv3
cd "$REPO"
HOME_HF=/home/pengchx3/text-dna/OmegaGenome_Revise_202606/code_carbon/.hf_cache
export HF_HOME="$HOME_HF" HF_HUB_CACHE="$HOME_HF/hub" TRANSFORMERS_CACHE="$HOME_HF/hub"
export HF_TOKEN="$(cat /home/pengchx3/text-dna/huggingface-token-0616.txt)"
PV=/home/pengchx3/text-dna/OmegaGenome_Revise_202606/code_carbon/.venv_carbon_portable/bin/python

SSD=/tmp/galaxy_srv_disk00/pengchx3
mountpoint -q "$SSD" 2>/dev/null || SSD=/srv/disk00/sshfs/pengchx3
DATA="$SSD/ntv3_benchmark_data"

"$PV" -m src.train.finetune_ntv3 --data_dir "$DATA" --model InstaDeepAI/NTv3_650M_post \
      --out "$SSD/ntv3_targets/ntv3_finetune" --window 16384 --n_train 2000 --n_test 400 \
      --epochs 20 --lr 1e-4 --batch_size 2 --loss poisson "$@"
echo "=== done: $SSD/ntv3_targets/ntv3_finetune/ntv3_finetune_result.json ==="
