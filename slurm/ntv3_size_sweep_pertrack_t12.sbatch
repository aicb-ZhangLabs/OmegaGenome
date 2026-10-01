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
REPO=${OG_NTV3_ROOT:-$PWD}
cd "$REPO"
# MOUNT GUARD (see slurm/_mount_guard.sh): sourced FIRST because resolve_ssd_mount picks the SSD root.
source "$REPO/slurm/_mount_guard.sh"
# HANG WATCHDOG (see slurm/_hang_watchdog.sh): the mount guard classifies non-zero EXITS; a HUNG trainer
# never exits. Sourced here, ARMED just before the trainer, DISARMED by the EXIT trap below.
source "$REPO/slurm/_hang_watchdog.sh"
# Node-agnostic SSD resolution + HARD /tmp policy: refuses any root on the node's own /tmp filesystem
# (voyager's /tmp is only 49G and has overflowed before) and dies rather than falling back to it.
SSD="$(resolve_ssd_mount ntv3_benchmark_data/human/genome.fasta \
                         ntv3_cache/teacher_logits/gen/t12.pt)" || exit 1
# TMPDIR MUST STAY NODE-LOCAL (do NOT move it onto the SSD mount).
# It holds ONLY multiprocessing's AF_UNIX rendezvous sockets (a few bytes each) and small python
# tempfiles -- never DATA, so it does not violate the "no node-local /tmp data staging" policy.
# WHY IT MUST NOT BE ON THE SSD: sshfs's mknod() returns EPERM for every non-regular file, so
# bind()ing an AF_UNIX socket under an sshfs path ALWAYS fails with
#   PermissionError: [Errno 1] Operation not permitted   (multiprocessing/connection.py, _socket.bind)
# torch's default `file_descriptor` sharing strategy routes every DataLoader worker->main batch
# through multiprocessing.resource_sharer.DupFd, which needs exactly such a listener socket. The
# exception is raised in the queue FEEDER THREAD, so the trainer does not exit -- it HANGS at step 0
# holding a GPU. Commit 7f72c3d pointed TMPDIR at "$SSD/tmp_scratch_..." and thereby hung every job
# submitted after it (2026-07-23: 257235_4 and 257236_4 burned ~1 GPU-hour each producing 0 steps;
# 257441_1/_2, 257439_2, 257444_4 reproduced it identically). Job-scoped so concurrent tiers on one
# node never collide, and removed on exit so nothing accumulates on the node's /tmp.
export TMPDIR="/tmp/ntv3_sock_${USER}_${SLURM_JOB_ID:-$$}"; mkdir -p "$TMPDIR"
# hang_watchdog_stop FIRST on every exit path (success, failure, requeue): a finished job must never
# leave a watchdog process behind that could requeue it after the fact.
trap 'hang_watchdog_stop; rm -rf "$TMPDIR"' EXIT
# HF cache -> SSD archive (tokenizer template resolve only; students are random-init, teacher-free).
HOME_HF="$SSD/hf_cache_archive_from_code_carbon"
export HF_HOME="$HOME_HF" HF_HUB_CACHE="$HOME_HF" TRANSFORMERS_CACHE="$HOME_HF"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export HF_TOKEN="$(cat ${HF_TOKEN_FILE})"
export TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# TRITON cache -> SSD mount (NEVER node-local /tmp).
export TRITON_CACHE_DIR="$SSD/triton_cache_${USER}_$(hostname -s)"
mkdir -p "$TRITON_CACHE_DIR"
export WANDB_MODE=offline WANDB_DIR="$SSD/wandb_logs" WANDB_CACHE_DIR="$SSD/wandb_cache"
mkdir -p "$WANDB_DIR" "$WANDB_CACHE_DIR"
PV=${OG_ROOT:-$PWD}/.venv_carbon_portable/bin/python

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
# HANG WATCHDOG armed here, on THIS job's own stdout: if the log stops growing for 4h before the first
# `step N/` line (startup grace: covers the teacher-cache load) or 4h after it (stall timeout), the job
# is requeued and RESUMES from latest_state.pth. Thresholds measured from this ladder's own logs; see
# slurm/_hang_watchdog.sh. It shares the mount guard's requeue budget and its one-requeue-per-dispatch
# stamp, so the two mechanisms can never both requeue the same dispatch.
hang_watchdog_start
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
