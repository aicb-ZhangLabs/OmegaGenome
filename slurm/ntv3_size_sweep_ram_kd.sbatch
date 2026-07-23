#!/bin/bash
# FROM-SCRATCH distilled NTv3 size sweep — RAM-CACHE variant (ZERO node-local /tmp).
# Same recipe as ntv3_size_sweep_scratch_kd.sbatch (kd_w_ce 0.5 / kd_w_kl 0.5 / standardized_mse / w_mse 0,
# 19932 steps, seq 32768, eff-batch 32, amp, best-val ckpt, seed 0) and the SAME out dirs, but the
# sshfs-memmap-hang mitigation is now RAM-resident + SSD-direct instead of node-local /tmp staging:
#   - teacher cache: read from the SSD mount and loaded WHOLE into RAM once (--cache_in_ram); no /tmp copy.
#   - benchmark data: read DIRECTLY from the SSD mount (pyBigWig/pyfaidx indexed file-handle reads, NOT
#     numpy memmap -> the memmap-hang failure mode does not apply); NO --stage_dir, no /tmp copy.
#   - TRITON cache -> SSD, never node-local /tmp.
# The ONLY /tmp path used is the SSD mount /tmp/galaxy_srv_disk00/pengchx3. Full sweep idx 0-4
# (4m/8m/30m/100m/300m). After the 2026-07-20 sshfs saturation (two jobs' cache-load->read transitions
# storming sshfs at once crashed 4m), THREE mitigations: (a) per-window bigWig reads retry transient sshfs
# errors with a widened ~63s window (src/data/ntv3_ft_data.py _retry_sshfs, 8 attempts); (b) --num_workers 8
# (halved from 16) to cut the per-job open-storm at the transition; (c) %1 throttle so no two jobs transition
# at once. Submit staggered:  sbatch --array=0,3,4%1 slurm/ntv3_size_sweep_ram_kd.sbatch  (8m/30m already run).
#
# NODE PLACEMENT (2026-07-23): NOT pinned to laniakea any more — this sweep was monopolising it.
# `--exclude=galaxy` leaves laniakea (RTX 6000 Ada 49G) AND voyager (H100 NVL 96G) eligible, so a tier
# starts on whichever frees first; galaxy stays excluded because its 3090s (24G) cannot hold these
# seq-32768 tiers. Both eligible nodes have >=1TB RAM for the 53GB in-RAM teacher cache (--mem 180G).
# `--nice=200` keeps it behind the other projects' jobs, so it backfills rather than crowds.
#SBATCH --job-name=loom_ntv3-size-ram
#SBATCH --array=0-4%1
#SBATCH --nodes=1
#SBATCH --exclude=galaxy
#SBATCH --partition=zhanglab.p
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=180G
#SBATCH --nice=200
#SBATCH --requeue
#SBATCH --time=30-00:00:00
#SBATCH --output=slurm-%x-%A_%a.out
set -euo pipefail
REPO=/home/pengchx3/text-dna/OmegaGenome_Revise_202606/code_ntv3
cd "$REPO"
# MOUNT GUARD (see slurm/_mount_guard.sh): sourced FIRST because resolve_ssd_mount picks the SSD root.
source "$REPO/slurm/_mount_guard.sh"
# Node-agnostic SSD resolution + HARD /tmp policy: refuses any root on the node's own /tmp filesystem
# (voyager's /tmp is only 49G and has overflowed before) and dies rather than falling back to it.
SSD="$(resolve_ssd_mount ntv3_benchmark_data/human/genome.fasta \
                         ntv3_targets/teacher_cache_joint34/logits.npy)" || exit 1
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
trap 'rm -rf "$TMPDIR"' EXIT
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

# MOUNT GUARD (sourced above): the 2026-07-23 sshfs outage EPERM'd every SSD path and killed this sweep
# at step 19500/19932. Preflight the exact files the run reads, and on a mount-caused crash requeue
# instead of dying — the trainer then RESUMES from latest_state.pth.
GUARD_PATHS=("$SSD/ntv3_benchmark_data/human/genome.fasta"
             "$SSD/ntv3_benchmark_data/human/genome.fasta.fai"
             "$SSD/ntv3_benchmark_data/human/functional_tracks"
             "$SRC_CACHE/logits.npy" "$SRC_CACHE/meta.pt" "$HOME_HF")
mount_preflight "${GUARD_PATHS[@]}" || exit 1

set +e
"$PV" -u -m src.train.finetune_ntv3 \
      --data_dir "$SSD/ntv3_benchmark_data" \
      --model "$SIZE" \
      --cached_teacher_logits "$SRC_CACHE" --cache_in_ram \
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
