#!/bin/bash
#SBATCH --job-name=rabl-ptata-r64
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=pengchx3@ics.uci.edu
#SBATCH --nodes=1
#SBATCH --nodelist=laniakea
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --partition=zhanglab.p
#SBATCH --time=20-00:00:00
#SBATCH --output=/extra/zhanglab0/INDV/pengchx3/NT/2b5-INCREASE-R-ablation-0629/slurm-%x-%j.out

# --- shared HF cache so the 2.5B model is NOT redownloaded per job ---
export HF_HOME=/extra/zhanglab0/INDV/pengchx3/NT/hf_cache_rabl
export HF_HUB_CACHE=/extra/zhanglab0/INDV/pengchx3/NT/hf_cache_rabl/hub
export HUGGINGFACE_HUB_CACHE=/extra/zhanglab0/INDV/pengchx3/NT/hf_cache_rabl/hub
export TRANSFORMERS_CACHE=/extra/zhanglab0/INDV/pengchx3/NT/hf_cache_rabl/hub
export HF_DATASETS_CACHE=/extra/zhanglab0/INDV/pengchx3/NT/hf_cache_rabl/datasets
export HF_ASSETS_CACHE=/extra/zhanglab0/INDV/pengchx3/NT/hf_cache_rabl/assets
unset HF_TOKEN_PATH HF_TOKEN
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

export LORA_R=64
export LORA_ALPHA=128   # alpha = 2r (task-requested scaling)
export TASK=promoter_tata

PYTHONPATH=/extra/zhanglab0/INDV/pengchx3/NT/peft_overlay_019 \
  /home/pengchx3/.conda/envs/NT/bin/python \
  /extra/zhanglab0/INDV/pengchx3/NT/nucleotide-transformer/examples/nt-finetune-increase-r-ablation-tf5fix.py
