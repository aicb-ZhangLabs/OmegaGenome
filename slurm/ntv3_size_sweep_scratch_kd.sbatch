#!/bin/bash
# FROM-SCRATCH distilled NTv3 size sweep (series 1 of the two-line size-ladder design).
# 5 random-init tiers (ntv3-4m/8m/30m/100m/300m), each KD'd from the 650M teacher via the precomputed
# JOINT-34 memmap cache -> TEACHER-FREE. SAME KD recipe as the existing distilled runs
# (kd_w_ce 0.5 / kd_w_kl 0.5 / standardized_mse / w_mse 0, 19932 steps, seq 32768, eff-batch 32, amp),
# best-val ckpt, seed 0. Distinct out dirs; does NOT touch the existing pretrained 8M/100M points.
#   sbatch --dependency=afterok:<precompute_jobid> slurm/ntv3_size_sweep_scratch_kd.sbatch
#SBATCH --job-name=loom_ntv3-size
#SBATCH --array=0-4%2
#SBATCH --nodes=1
#SBATCH --nodelist=laniakea
#SBATCH --partition=zhanglab.p
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --requeue
#SBATCH --time=30-00:00:00
#SBATCH --output=slurm-%x-%A_%a.out
set -euo pipefail
REPO=${OG_NTV3_ROOT:-$PWD}
cd "$REPO"
SSD=${OG_SCRATCH:-$PWD/output}
mountpoint -q "$SSD" 2>/dev/null || SSD=${OG_SCRATCH:-$PWD/output}
# HF cache -> SSD archive (only needed for the tokenizer template resolve; students are random-init, teacher-free).
HOME_HF="$SSD/hf_cache_archive_from_code_carbon"
export HF_HOME="$HOME_HF" HF_HUB_CACHE="$HOME_HF" TRANSFORMERS_CACHE="$HOME_HF"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export HF_TOKEN="$(cat ${HF_TOKEN_FILE})"
export TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRITON_CACHE_DIR="/tmp/triton_${USER}_$(hostname -s)"
export WANDB_MODE=offline WANDB_DIR="$SSD/wandb_logs" WANDB_CACHE_DIR="$SSD/wandb_cache"
mkdir -p "$WANDB_DIR" "$WANDB_CACHE_DIR"
PV=${OG_ROOT:-$PWD}/.venv_carbon_portable/bin/python

SIZES=(ntv3-4m ntv3-8m ntv3-30m ntv3-100m ntv3-300m)
SIZE="${SIZES[${SLURM_ARRAY_TASK_ID:-0}]}"
OUT="$SSD/ntv3_targets/size_sweep/${SIZE}_scratch_kd_s0"
echo "=== [array ${SLURM_ARRAY_TASK_ID}] size=$SIZE -> $OUT (host $(hostname -s)) ==="

# --- stage the joint-34 memmap cache node-local (avoid random sshfs memmap reads over hours: finding
#     'sshfs memmap hang'). Sequential copy = low hang risk; flock so only the first array task copies. ---
SRC_CACHE="$SSD/ntv3_targets/teacher_cache_joint34"
STAGE_CACHE="/tmp/ntv3_cache_joint34_${USER}"
[ -f "$SRC_CACHE/logits.npy" ] && [ -f "$SRC_CACHE/meta.pt" ] || { echo "FATAL: joint-34 cache missing at $SRC_CACHE"; exit 1; }
mkdir -p "$STAGE_CACHE"
(
  flock -x 200
  for f in meta.pt logits.npy; do
    s="$SRC_CACHE/$f"; d="$STAGE_CACHE/$f"
    if [ ! -f "$d" ] || [ "$(stat -c%s "$s")" != "$(stat -c%s "$d")" ]; then
      echo "staging cache $f node-local -> $d ..."; cp -f "$s" "$d.tmp.$$" && mv -f "$d.tmp.$$" "$d"
    fi
  done
) 200>"$STAGE_CACHE/.stage.lock"
echo "cache staged node-local: $(du -sh "$STAGE_CACHE" | cut -f1)"

STAGE_DATA="/tmp/ntv3_stage_${USER}"   # shared; stage_to_local is idempotent + atomic (safe under concurrency)

"$PV" -u -m src.train.finetune_ntv3 \
      --data_dir "$SSD/ntv3_benchmark_data" --stage_dir "$STAGE_DATA" \
      --model "$SIZE" \
      --cached_teacher_logits "$STAGE_CACHE" \
      --kd_w_ce 0.5 --kd_w_kl 0.5 --kd_w_mse 0.0 \
      --kd_gt_loss poisson_multinomial --kd_distill_loss standardized_mse \
      --kd_multinomial_weight 5.0 \
      --sequence_length 32768 --mini_batch_size 4 --num_accumulation_gradient 8 \
      --num_steps_training 19932 --num_steps_warmup 598 \
      --seed 0 --amp \
      --out "$OUT"
echo "=== done $SIZE: $OUT/ntv3_finetune_result.json ==="
