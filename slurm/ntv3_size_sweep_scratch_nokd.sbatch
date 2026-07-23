#!/bin/bash
# FROM-SCRATCH, NO-KD (labels-only) NTv3 size-ladder baseline — the missing "scratch (no-KD)" column to
# compare 1:1 against the from-scratch+KD sweep (ntv3_size_sweep_ram_kd.sbatch). 5 random-init tiers
# (ntv3-4m/8m/30m/100m/300m, NTV3_SCALED_SIZES from_config -> fresh weights), each trained on the
# ground-truth bigWig LABELS ONLY with ZERO teacher/distillation signal.
#
# NO-KD CONFIG (verified against src/train/finetune_ntv3.py):
#   * We pass NEITHER --teacher NOR --cached_teacher_logits. Then `cached_logits` stays None (l.238) and
#     `kd_cfg` stays None (l.347-348: `if args.teacher or cached_logits is not None`). With kd_cfg None the
#     training loop takes the ELSE branch (l.489-491): `loss = poisson_multinomial_loss(logits, targets)` —
#     pure GT label loss, NO teacher forward, NO KL, NO distill/feature MSE. The --kd_w_* flags are IGNORED
#     when there is no teacher, so the correct "teacher off" path is simply omitting the teacher inputs
#     (not --kd_w_kl 0 etc.). result.json records "kd": null (l.539).
#   * We ALSO pass --kd_w_ce 1.0 --kd_w_kl 0 --kd_w_mse 0 EXPLICITLY, for self-documentation only. These are
#     INERT on this path and change NOTHING numerically: `kd_w_ce` occurs only at argparse (l.119) and inside
#     the kd_cfg block (l.351/356/540), and `track_kd_loss` is called ONLY at l.486 inside `if kd_cfg is not
#     None`. With no teacher the guard at l.348 is False, so TrackKDConfig is never constructed and the KL /
#     feature-MSE terms are never COMPUTED AT ALL (not against zeros or garbage — no teacher tensor exists).
#     The GT loss at l.491 is a bare, UNWEIGHTED `poisson_multinomial_loss(logits, targets)` — i.e. the label
#     loss is at FULL weight here regardless of --kd_w_ce (the 0.5 default never scales it). NOTE: because
#     kd_cfg stays None, result.json's "kd" block stays null; the explicit weights surface in result["config"].
#   * The GT loss is a clean 1:1 with the KD sweep's GT term: no-KD calls
#     poisson_multinomial_loss(logits, targets) with default shape_loss_coefficient=5.0 (track_losses.py
#     l.23-24); KD-mode's gt_loss=poisson_multinomial calls the SAME function with
#     shape_loss_coefficient=--kd_multinomial_weight (=5.0) (track_distill.py l.204-205, l.247). Identical.
#   * Everything else matches the KD sweep exactly: seq 32768, 19932 steps, warmup 598, eff-batch 32
#     (mini 4 x accum 8), seed 0, amp, best-val ckpt, default LR schedule. ONLY the teacher/KD is removed.
#
# /tmp POLICY (HARD): the ONLY "/tmp" path used is the galaxy SSD mount /tmp/galaxy_srv_disk00/pengchx3.
# NO node-local /tmp data staging (no --stage_dir, no cache copy): bigWigs are read DIRECTLY from the SSD
# via pyBigWig/pyfaidx indexed file-handle reads (NOT numpy memmap -> the sshfs memmap-hang failure mode
# does not apply), with the hardened per-window sshfs-retry loader + num_workers 8. TRITON/HF/WANDB -> SSD.
# These runs load NO teacher cache (no ~53GB RAM load) => LOW-RAM backfill: --mem 48G, nice=200, %2 throttle,
# courteous to the running scratch+KD sweep and the per-track t12 ladder on laniakea.
#   sbatch slurm/ntv3_size_sweep_scratch_nokd.sbatch          # full ladder 0-4 (4m/8m/30m/100m/300m)
#SBATCH --job-name=loom_ntv3-size-nokd
#SBATCH --array=0-4%2
#SBATCH --nice=200
#SBATCH --nodes=1
#SBATCH --nodelist=laniakea
#SBATCH --partition=zhanglab.p
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --requeue
#SBATCH --time=30-00:00:00
#SBATCH --output=slurm-%x-%A_%a.out
set -euo pipefail
REPO=/home/pengchx3/text-dna/OmegaGenome_Revise_202606/code_ntv3
cd "$REPO"
SSD=/tmp/galaxy_srv_disk00/pengchx3
mountpoint -q "$SSD" 2>/dev/null || SSD=/srv/disk00/sshfs/pengchx3
# GUARD: SSD mount must be present (all data lives there); never fall back to node-local /tmp.
[ -d "$SSD/ntv3_benchmark_data" ] || { echo "FATAL: SSD mount / benchmark data missing at $SSD"; exit 1; }
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
# Distinct out dir (parallel to, and NEVER overlapping, the KD sweep's ${SIZE}_scratch_kd_s0).
OUT="$SSD/ntv3_targets/size_sweep/${SIZE}_scratch_nokd_s0"
# Per-size squeue visibility (array shares one job-name; log file stays slurm-%x-%A_%a.out).
command -v scontrol >/dev/null 2>&1 && scontrol update JobId="${SLURM_JOB_ID}" Name="loom_ntv3-size-nokd-${SIZE}" 2>/dev/null || true
echo "=== [array ${SLURM_ARRAY_TASK_ID}] NO-KD size=$SIZE -> $OUT (host $(hostname -s)) ==="

# NO --teacher, NO --cached_teacher_logits, NO --stage_dir: pure GT-labels-only training, teacher-free,
# SSD-direct bigWig reads with the hardened sshfs-retry loader. See header for the verified no-KD path.
"$PV" -u -m src.train.finetune_ntv3 \
      --data_dir "$SSD/ntv3_benchmark_data" \
      --model "$SIZE" \
      --kd_w_ce 1.0 --kd_w_kl 0 --kd_w_mse 0 \
      --kd_gt_loss poisson_multinomial --kd_multinomial_weight 5.0 \
      --sequence_length 32768 --mini_batch_size 4 --num_accumulation_gradient 8 \
      --num_steps_training 19932 --num_steps_warmup 598 \
      --num_workers 8 \
      --seed 0 --amp \
      --out "$OUT"
echo "=== done $SIZE: $OUT/ntv3_finetune_result.json ==="
