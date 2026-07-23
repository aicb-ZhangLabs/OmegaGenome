#!/bin/bash
# PER-TRACK FROM-SCRATCH distilled NTv3 size ladder — SINGLE track t12 = ENCSR325NFE (ATAC-seq).
# Same recipe / hardening as the joint sweep (slurm/ntv3_size_sweep_ram_kd.sbatch): kd_w_ce 0.5 /
# kd_w_kl 0.5 / kd_w_mse 0, poisson_multinomial + standardized_mse, 19932 steps, seq 32768, eff-batch
# 32, amp, best-val ckpt, seed 0, from-scratch ntv3-<size> registry keys, num_workers=8 + the
# sshfs-retry bigWig loader, SSD-direct data, TRITON/HF/WANDB on the SSD mount. ZERO node-local /tmp.
#
# DIFFERENCES vs the joint sweep:
#   - --track_subset 12  => a 1-track head (ENCSR325NFE only).
#   - --cached_teacher_logits <SSD>/ntv3_cache/teacher_logits/gen/t12.pt : the TINY (~1.5GB) single-track
#     .pt cache (channel 12 sliced from teacher_cache_joint34, provably identical to the canonical
#     precompute). torch.load pulls it fully into RAM — NO --cache_in_ram (that flag is only for the DIR
#     memmap layout). So each run uses LITTLE host RAM and can coexist with the 53GB joint sweep without
#     competing for RAM (the pending joint 8m/30m need 180G; these need ~32G).
#   - CLEAN non-overlapping out dir size_sweep_pertrack/t12_ENCSR325NFE/ (never size_sweep/ or old
#     ntv3_8m_t12_* dirs).
#   - Low --mem (32G) + --nice (deprioritized below the joint pending jobs) + %2 throttle: courteous to
#     the running/pending joint sweep. Submit e.g.:
#       ssh voyager 'bash -lc "module load slurm; cd <REPO>; sbatch --array=0-4%2 slurm/ntv3_size_sweep_pertrack_t12.sbatch"'
#     Array idx -> size:  0=4m 1=8m 2=30m 3=100m 4=300m.
#
# NODE PLACEMENT (2026-07-23): NOT pinned to laniakea any more (the NTv3 sweep was monopolising it).
# `--exclude=galaxy` leaves laniakea AND voyager eligible so a tier starts on whichever frees first;
# galaxy's 24G 3090s cannot hold these seq-32768 tiers. nice=200 keeps it a backfill citizen.
#SBATCH --job-name=loom_ntv3-pertrack-t12
#SBATCH --array=0-4%2
#SBATCH --nodes=1
#SBATCH --exclude=galaxy
#SBATCH --partition=zhanglab.p
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --nice=200
#SBATCH --requeue
#SBATCH --time=30-00:00:00
#SBATCH --output=slurm-loom_ntv3-pertrack-t12-%a-%A.out
set -euo pipefail
REPO=/home/pengchx3/text-dna/OmegaGenome_Revise_202606/code_ntv3
cd "$REPO"
# MOUNT GUARD (see slurm/_mount_guard.sh): sourced FIRST because resolve_ssd_mount picks the SSD root.
source "$REPO/slurm/_mount_guard.sh"
# Node-agnostic SSD resolution + HARD /tmp policy: refuses any root on the node's own /tmp filesystem
# (voyager's /tmp is only 49G and has overflowed before) and dies rather than falling back to it.
SSD="$(resolve_ssd_mount ntv3_benchmark_data/human/genome.fasta \
                         ntv3_cache/teacher_logits/gen/t12.pt)" || exit 1
# Everything scratch-like goes on the SSD, incl. TMPDIR: nothing this job writes may touch node-local /tmp.
export TMPDIR="$SSD/tmp_scratch_${USER}_$(hostname -s)"; mkdir -p "$TMPDIR"
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
TAG="${SIZE#ntv3-}"                       # 4m / 8m / 30m / 100m / 300m
# Per-task job name + a size-named log symlink (squeue/logs show the exact tier).
scontrol update JobId="${SLURM_JOB_ID}" JobName="loom_ntv3-pertrack-t12-${TAG}" 2>/dev/null || true
ln -sf "slurm-loom_ntv3-pertrack-t12-${SLURM_ARRAY_TASK_ID}-${SLURM_ARRAY_JOB_ID}.out" \
       "slurm-loom_ntv3-pertrack-t12-${TAG}-${SLURM_ARRAY_JOB_ID}.out" 2>/dev/null || true

OUT="$SSD/ntv3_targets/size_sweep_pertrack/t12_ENCSR325NFE/${SIZE}_scratch_kd_s0"
T12_CACHE="$SSD/ntv3_cache/teacher_logits/gen/t12.pt"
[ -f "$T12_CACHE" ] || { echo "FATAL: per-track cache missing at $T12_CACHE"; exit 1; }
echo "=== [array ${SLURM_ARRAY_TASK_ID}] size=$SIZE track=12/ENCSR325NFE -> $OUT (host $(hostname -s)) ==="
echo "    per-track cache: $T12_CACHE"

# MOUNT GUARD (sourced above): the 2026-07-23 sshfs outage EPERM'd every SSD path — it killed the t12
# 30m/100m runs mid-training and the 300m run in 15s on the HF tokenizer module under $HOME_HF (hence
# that path is preflighted too). On a mount-caused crash, requeue instead of dying; the trainer then
# RESUMES from latest_state.pth.
GUARD_PATHS=("$SSD/ntv3_benchmark_data/human/genome.fasta"
             "$SSD/ntv3_benchmark_data/human/genome.fasta.fai"
             "$SSD/ntv3_benchmark_data/human/functional_tracks"
             "$T12_CACHE" "$HOME_HF" "$HOME_HF/modules/transformers_modules")
mount_preflight "${GUARD_PATHS[@]}" || exit 1

# --track_subset 12 => 1-track head; the .pt cache (track_subset==[12]) is loaded whole into RAM by
# torch.load (no --cache_in_ram; that is the DIR-memmap flag). bigWig data read DIRECTLY off the SSD
# (indexed file-handle reads, NOT numpy memmap) with the per-window sshfs-retry loader; NO /tmp staging.
set +e
"$PV" -u -m src.train.finetune_ntv3 \
      --data_dir "$SSD/ntv3_benchmark_data" \
      --model "$SIZE" \
      --track_subset 12 \
      --cached_teacher_logits "$T12_CACHE" \
      --kd_w_ce 0.5 --kd_w_kl 0.5 --kd_w_mse 0.0 \
      --kd_gt_loss poisson_multinomial --kd_distill_loss standardized_mse \
      --kd_multinomial_weight 5.0 \
      --sequence_length 32768 --mini_batch_size 4 --num_accumulation_gradient 8 \
      --num_steps_training 19932 --num_steps_warmup 598 \
      --num_workers 8 \
      --seed 0 --amp \
      --out "$OUT"
rc=$?
set -e
requeue_if_mount_failure "$rc" "${GUARD_PATHS[@]}" || exit "$rc"
echo "=== done $SIZE: $OUT/ntv3_finetune_result.json ==="
