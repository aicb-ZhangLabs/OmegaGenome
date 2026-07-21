#!/bin/bash
# FROM-SCRATCH distilled NTv3 size sweep — RAM-CACHE variant (ZERO node-local /tmp).
# Same recipe as ntv3_size_sweep_scratch_kd.sbatch (kd_w_ce 0.5 / kd_w_kl 0.5 / standardized_mse / w_mse 0,
# 19932 steps, seq 32768, eff-batch 32, amp, best-val ckpt, seed 0) and the SAME out dirs, but the
# sshfs-memmap-hang mitigation is now RAM-resident + SSD-direct instead of node-local /tmp staging:
#   - teacher cache: read from the SSD mount and loaded WHOLE into RAM once (--cache_in_ram); no /tmp copy.
#   - benchmark data: read DIRECTLY from the SSD mount (pyBigWig/pyfaidx indexed file-handle reads, NOT
#     numpy memmap -> the memmap-hang failure mode does not apply); NO --stage_dir, no /tmp copy.
#   - TRITON cache -> SSD, never node-local /tmp.
# The ONLY /tmp path used is the SSD mount /tmp/galaxy_srv_disk00/pengchx3. Restarts idx 3/4 (100m/300m).
#   sbatch slurm/ntv3_size_sweep_ram_kd.sbatch
#SBATCH --job-name=loom_ntv3-size-ram
#SBATCH --array=3-4%2
#SBATCH --nodes=1
#SBATCH --nodelist=laniakea
#SBATCH --partition=zhanglab.p
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=180G
#SBATCH --requeue
#SBATCH --time=30-00:00:00
#SBATCH --output=slurm-%x-%A_%a.out
set -euo pipefail
REPO=/home/pengchx3/text-dna/OmegaGenome_Revise_202606/code_ntv3
cd "$REPO"
SSD=/tmp/galaxy_srv_disk00/pengchx3
mountpoint -q "$SSD" 2>/dev/null || SSD=/srv/disk00/sshfs/pengchx3
# GUARD: SSD mount must be present (all data/cache/caches live there); never fall back to node-local /tmp.
[ -d "$SSD/ntv3_targets/teacher_cache_joint34" ] || { echo "FATAL: SSD mount missing at $SSD"; exit 1; }
# HF cache -> SSD archive (tokenizer template resolve only; students are random-init, teacher-free).
HOME_HF="$SSD/hf_cache_archive_from_code_carbon"
export HF_HOME="$HOME_HF" HF_HUB_CACHE="$HOME_HF" TRANSFORMERS_CACHE="$HOME_HF"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export HF_TOKEN="$(cat /home/pengchx3/text-dna/huggingface-token-0616.txt)"
export TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# TRITON cache -> SSD mount (NEVER node-local /tmp).
export TRITON_CACHE_DIR="$SSD/triton_cache_${USER}_$(hostname -s)"
mkdir -p "$TRITON_CACHE_DIR"
export WANDB_MODE=offline WANDB_DIR="$SSD/wandb_logs" WANDB_CACHE_DIR="$SSD/wandb_cache"
mkdir -p "$WANDB_DIR" "$WANDB_CACHE_DIR"
PV=/home/pengchx3/text-dna/OmegaGenome_Revise_202606/code_carbon/.venv_carbon_portable/bin/python

SIZES=(ntv3-4m ntv3-8m ntv3-30m ntv3-100m ntv3-300m)
SIZE="${SIZES[${SLURM_ARRAY_TASK_ID:-0}]}"
OUT="$SSD/ntv3_targets/size_sweep/${SIZE}_scratch_kd_s0"
echo "=== [array ${SLURM_ARRAY_TASK_ID}] size=$SIZE -> $OUT (host $(hostname -s)) ==="

# Joint-34 teacher cache read from the SSD mount + loaded WHOLE into RAM (--cache_in_ram): one sequential
# read, then in-memory fancy-index per batch => no per-batch random sshfs memmap reads (finding
# 'sshfs memmap hang'), and NO node-local /tmp staging. ~53GB fp16 fits laniakea's ~1TB RAM.
SRC_CACHE="$SSD/ntv3_targets/teacher_cache_joint34"
[ -f "$SRC_CACHE/logits.npy" ] && [ -f "$SRC_CACHE/meta.pt" ] || { echo "FATAL: joint-34 cache missing at $SRC_CACHE"; exit 1; }

"$PV" -u -m src.train.finetune_ntv3 \
      --data_dir "$SSD/ntv3_benchmark_data" \
      --model "$SIZE" \
      --cached_teacher_logits "$SRC_CACHE" --cache_in_ram \
      --kd_w_ce 0.5 --kd_w_kl 0.5 --kd_w_mse 0.0 \
      --kd_gt_loss poisson_multinomial --kd_distill_loss standardized_mse \
      --kd_multinomial_weight 5.0 \
      --sequence_length 32768 --mini_batch_size 4 --num_accumulation_gradient 8 \
      --num_steps_training 19932 --num_steps_warmup 598 \
      --seed 0 --amp \
      --out "$OUT"
echo "=== done $SIZE: $OUT/ntv3_finetune_result.json ==="
