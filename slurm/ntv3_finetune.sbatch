#!/bin/bash
# Faithful NTv3 benchmark fine-tune — direct port of the official notebook
# 03_fine_tuning_posttrained_model_biwig.ipynb ("reproduce paper results"). Full-FT of 650M at
# 32,768 bp, eff. batch 32, ~20.9B tokens, ~28h on an H100 (voyager). Reference human mean Pearson ~0.605.
#   sbatch slurm/ntv3_finetune.sbatch                 # full faithful run (voyager H100)
#   sbatch slurm/ntv3_finetune.sbatch --use_lora      # cheap LoRA variant
# Extra args pass through to src.train.finetune_ntv3 (e.g. --smoke, --sequence_length 4096).
#SBATCH --job-name=ntv3-finetune
#SBATCH --nodes=1
#SBATCH --nodelist=voyager
#SBATCH --partition=zhanglab.p
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --nice=0
#SBATCH --requeue
#SBATCH --time=30-00:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail
REPO=/home/pengchx3/text-dna/OmegaGenome_Revise_202606/code_ntv3
cd "$REPO"
HOME_HF=/home/pengchx3/text-dna/OmegaGenome_Revise_202606/code_carbon/.hf_cache
export HF_HOME="$HOME_HF" HF_HUB_CACHE="$HOME_HF/hub" TRANSFORMERS_CACHE="$HOME_HF/hub"
export HF_TOKEN="$(cat /home/pengchx3/text-dna/huggingface-token-0616.txt)"
export TOKENIZERS_PARALLELISM=false
PV=/home/pengchx3/text-dna/OmegaGenome_Revise_202606/code_carbon/.venv_carbon_portable/bin/python

SSD=/tmp/galaxy_srv_disk00/pengchx3
mountpoint -q "$SSD" 2>/dev/null || SSD=/srv/disk00/sshfs/pengchx3
DATA="$SSD/ntv3_benchmark_data"
STAGE="${SLURM_TMPDIR:-/tmp/ntv3_stage_$USER}"

"$PV" -u -m src.train.finetune_ntv3 --data_dir "$DATA" --model InstaDeepAI/NTv3_650M_post \
      --out "$SSD/ntv3_targets/ntv3_finetune_faithful" --stage_dir "$STAGE" --wandb "$@"
echo "=== done: $SSD/ntv3_targets/ntv3_finetune_faithful/ntv3_finetune_result.json ==="
