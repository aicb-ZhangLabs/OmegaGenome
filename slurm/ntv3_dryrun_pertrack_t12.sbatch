#!/bin/bash
# DRY-RUN of the PER-TRACK t12 config on ntv3-4m: one batch train->val->test then exit. Verifies the
# 1-track head reads gen/t12.pt (track_subset==[12], ENCSR325NFE), reads bigWig SSD-direct, and steps.
# Identical flags/paths to slurm/ntv3_size_sweep_pertrack_t12.sbatch, plus --dry_run. ZERO node-local /tmp.
#SBATCH --job-name=loom_ntv3-pertrack-t12-dryrun
#SBATCH --nodes=1
#SBATCH --nodelist=laniakea
#SBATCH --partition=zhanglab.p
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --nice=200
#SBATCH --time=02:00:00
#SBATCH --output=slurm-loom_ntv3-pertrack-t12-dryrun-%j.out
set -euo pipefail
REPO=/home/pengchx3/text-dna/OmegaGenome_Revise_202606/code_ntv3
cd "$REPO"
SSD=/tmp/galaxy_srv_disk00/pengchx3
mountpoint -q "$SSD" 2>/dev/null || SSD=/srv/disk00/sshfs/pengchx3
[ -d "$SSD/ntv3_targets/teacher_cache_joint34" ] || SSD=/srv/disk00/sshfs/pengchx3
[ -d "$SSD/ntv3_targets/teacher_cache_joint34" ] || { echo "FATAL: SSD mount missing at $SSD"; exit 1; }
HOME_HF="$SSD/hf_cache_archive_from_code_carbon"
export HF_HOME="$HOME_HF" HF_HUB_CACHE="$HOME_HF" TRANSFORMERS_CACHE="$HOME_HF"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export HF_TOKEN="$(cat /home/pengchx3/text-dna/huggingface-token-0616.txt)"
export TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRITON_CACHE_DIR="$SSD/triton_cache_${USER}_$(hostname -s)"
mkdir -p "$TRITON_CACHE_DIR"
export WANDB_MODE=offline WANDB_DIR="$SSD/wandb_logs" WANDB_CACHE_DIR="$SSD/wandb_cache"
mkdir -p "$WANDB_DIR" "$WANDB_CACHE_DIR"
PV=/home/pengchx3/text-dna/OmegaGenome_Revise_202606/code_carbon/.venv_carbon_portable/bin/python

SIZE=ntv3-4m
OUT="$SSD/ntv3_targets/size_sweep_pertrack/_dryrun/${SIZE}_t12_dryrun"
T12_CACHE="$SSD/ntv3_cache/teacher_logits/gen/t12.pt"
[ -f "$T12_CACHE" ] || { echo "FATAL: per-track cache missing at $T12_CACHE"; exit 1; }
mkdir -p "$(dirname "$OUT")"
echo "=== DRY-RUN $SIZE track=12/ENCSR325NFE on $(hostname -s); SSD=$SSD; cache=$T12_CACHE ==="

"$PV" -u -m src.train.finetune_ntv3 \
      --data_dir "$SSD/ntv3_benchmark_data" \
      --model "$SIZE" \
      --track_subset 12 \
      --cached_teacher_logits "$T12_CACHE" \
      --kd_w_ce 0.5 --kd_w_kl 0.5 --kd_w_mse 0.0 \
      --kd_gt_loss poisson_multinomial --kd_distill_loss standardized_mse \
      --kd_multinomial_weight 5.0 \
      --sequence_length 32768 --mini_batch_size 4 --num_accumulation_gradient 8 \
      --seed 0 --amp --dry_run \
      --out "$OUT"
echo "=== DRY-RUN done $SIZE ==="
