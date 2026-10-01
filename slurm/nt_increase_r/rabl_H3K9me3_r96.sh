#!/bin/bash
#SBATCH --job-name=rabl-H3K9me3-r96
#SBATCH --nodes=1
#SBATCH --nodelist=laniakea
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --partition=zhanglab.p
#SBATCH --time=20-240:00
#SBATCH --output=logs/slurm-%x-%j.out

# --- shared HF cache so the 2.5B model is NOT redownloaded per job ---
export HF_HOME=${OG_STORE:-$PWD/data}/NT/hf_cache_rabl
export HF_HUB_CACHE=${OG_STORE:-$PWD/data}/NT/hf_cache_rabl/hub
export HUGGINGFACE_HUB_CACHE=${OG_STORE:-$PWD/data}/NT/hf_cache_rabl/hub
export TRANSFORMERS_CACHE=${OG_STORE:-$PWD/data}/NT/hf_cache_rabl/hub
export HF_DATASETS_CACHE=${OG_STORE:-$PWD/data}/NT/hf_cache_rabl/datasets
export HF_ASSETS_CACHE=${OG_STORE:-$PWD/data}/NT/hf_cache_rabl/assets
unset HF_TOKEN_PATH HF_TOKEN
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

export LORA_R=96
export TASK=H3K9me3

PYTHONPATH=${OG_STORE:-$PWD/data}/NT/peft_overlay_019 python ${OG_STORE:-$PWD/data}/NT/nucleotide-transformer/examples/nt-finetune-increase-r-ablation.py
