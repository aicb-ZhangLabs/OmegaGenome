#!/bin/bash
# PER-TRACK FROM-SCRATCH, NO-KD (labels-only) NTv3 size-ladder baseline — the missing per-track "scratch
# (no-KD)" column, the 1-track analog of the joint no-KD arm (slurm/ntv3_size_sweep_scratch_nokd.sbatch).
# SINGLE track t12 = ENCSR325NFE (ATAC-seq), trained on the ground-truth bigWig LABELS ONLY, ZERO teacher.
# Pairing this with the per-track+KD arm (slurm/ntv3_size_sweep_pertrack_t12.sbatch, *_scratch_kd_s0)
# yields the PER-TRACK KD DELTA:  (per-track+KD) - (per-track no-KD).
#
# THIS SCRIPT = the INTERSECTION of two already-working sbatches (no new mechanism invented):
#   (A) PER-TRACK  slurm/ntv3_size_sweep_pertrack_t12.sbatch : --track_subset 12  => a 1-track head
#       (ENCSR325NFE only).  We take ONLY the --track_subset 12; we drop its teacher entirely (no
#       gen/t12.pt, no --cached_teacher_logits) so this arm is labels-only.
#   (B) NO-KD MATCHED  slurm/ntv3_size_sweep_scratch_nokd.sbatch : labels-only + DATA-MATCH. We take its
#       exact no-teacher path AND its window-set matching (--train_overlap 0 --max_train_samples <cache
#       limit>, with the limit read from teacher_cache_joint34/meta.pt), so the ONLY thing that differs
#       from the per-track+KD arm is teacher on/off — same 63,707-window stream, same recipe.
#
# NO-KD CONFIG (verified against src/train/finetune_ntv3.py — identical reasoning to sbatch (B)):
#   * NEITHER --teacher NOR --cached_teacher_logits => `cached_logits` stays None, `kd_cfg` stays None
#     (l.347-348), so the training loop takes the ELSE branch (l.489-491):
#     `loss = poisson_multinomial_loss(logits, targets)` — pure GT label loss, NO teacher forward, NO KL,
#     NO distill/feature MSE. result.json records "kd": null (l.539); teacher==None, cached==None in config.
#   * --kd_w_ce 1.0 --kd_w_kl 0 --kd_w_mse 0 are passed for SELF-DOCUMENTATION only; they are INERT on the
#     no-teacher path (the kd_cfg guard at l.348 is False, so the KL/MSE terms are never computed) and the
#     GT loss at l.491 is a bare UNWEIGHTED poisson_multinomial_loss — the label loss is at FULL weight.
#   * The GT loss is a clean 1:1 with the KD sweep's GT term: shape_loss_coefficient=5.0 both sides.
#   * Everything else matches the per-track+KD arm: --track_subset 12, seq 32768, 19932 steps, warmup 598,
#     eff-batch 32 (mini 4 x accum 8), seed 0, amp, best-val ckpt, random-init ntv3-<size>. ONLY KD removed.
#
# ── DATA-STREAM MATCHING (same fix as sbatch (B)) ────────────────────────────────────────────────────
# The per-track+KD arm passes --cached_teacher_logits gen/t12.pt, which is channel 12 sliced from
# teacher_cache_joint34 and so carries that cache's OVERRIDE (finetune_ntv3.py l.253): overlap 0.0 +
# limit 64000 -> GenomeBigWigDataset stride 32768 (non-overlapping) -> 63,707 DISTINCT train windows.
# A no-teacher run gets NO such override -> it would default to overlap 0.999 (stride 32, ~65M windows),
# confounding "teacher off" with "much more diverse data". THE FIX: reproduce the SAME window set WITHOUT
# any logits, via the two flags the trainer exposes for exactly this (l.65-67):
#       --train_overlap 0  --max_train_samples <cache meta limit_num_samples>
# GenomeBigWigDataset derives its windows deterministically from (split regions, seq_len, overlap, limit)
# via a SEEDED sample_regions_for_total_length, so same regions + seq_len + overlap 0 + limit 64000 =>
# the SAME 63,707 windows in the SAME order as the cache the per-track KD arm consumed. We read the LIMIT
# from teacher_cache_joint34/meta.pt (never hardcoded) and NEVER touch logits.npy — no teacher signal, no
# 53GB load. (t12.pt shares that same joint-34 window set, so matching joint34's meta matches t12.pt too.)
#
# DRY-RUN (do this before every ladder submit): NTV3_DRY_RUN=1 runs THIS EXACT script/flags for 2 steps
# + 1 val + 1 test batch into a _dryrun/ out dir, so the thing verified is the thing queued. It prints
# `windows: train N` — N MUST be the per-track KD arm's 63707.
#   sbatch --export=ALL,NTV3_DRY_RUN=1 --array=0 --time=02:00:00 slurm/ntv3_size_sweep_pertrack_t12_nokd.sbatch
#
# /tmp POLICY (HARD): the ONLY "/tmp" path used is the galaxy SSD mount /tmp/galaxy_srv_disk00/pengchx3.
# NO node-local /tmp data staging: bigWigs are read DIRECTLY off the SSD via pyBigWig/pyfaidx indexed
# file-handle reads (NOT numpy memmap -> the sshfs memmap-hang mode does not apply) with the hardened
# per-window sshfs-retry loader + num_workers 8. Only meta.pt (~1MB) is opened from the cache (never
# logits) => LOW-RAM backfill citizen: --mem 32G, nice=200, %2 throttle, --exclude=galaxy (node-flexible;
# galaxy's 24G 3090s cannot hold these seq-32768 tiers). Courteous to the running scratch+KD sweeps.
#   sbatch slurm/ntv3_size_sweep_pertrack_t12_nokd.sbatch     # full ladder 0-4 (4m/8m/30m/100m/300m)
#   Array idx -> size:  0=4m 1=8m 2=30m 3=100m 4=300m.
#SBATCH --job-name=loom_ntv3-pertrack-t12-nokd
#SBATCH --array=0-4%2
#SBATCH --nice=200
#SBATCH --nodes=1
#SBATCH --exclude=galaxy
#SBATCH --partition=zhanglab.p
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --requeue
#SBATCH --time=30-00:00:00
#SBATCH --output=slurm-%x-%A_%a.out
set -euo pipefail
REPO=/home/pengchx3/text-dna/OmegaGenome_Revise_202606/code_ntv3
cd "$REPO"
# MOUNT GUARD (see slurm/_mount_guard.sh): sourced FIRST because resolve_ssd_mount picks the SSD root.
source "$REPO/slurm/_mount_guard.sh"
# HANG WATCHDOG (see slurm/_hang_watchdog.sh): the mount guard classifies non-zero EXITS; a HUNG trainer
# never exits. Sourced here, ARMED just before the trainer, DISARMED by the EXIT trap below.
source "$REPO/slurm/_hang_watchdog.sh"
# Node-agnostic SSD resolution + HARD /tmp policy: refuses any root on the node's own /tmp filesystem
# (voyager's /tmp is only 49G and has overflowed before) and dies rather than falling back to it.
# The teacher cache's meta.pt is REQUIRED here (not its logits): it defines the window set this arm must
# match, so a root that lacks it is the wrong root. (No t12.pt is needed — this arm is teacher-free.)
SSD="$(resolve_ssd_mount ntv3_benchmark_data/human/genome.fasta \
                         ntv3_targets/teacher_cache_joint34/meta.pt)" || exit 1
# TMPDIR MUST STAY NODE-LOCAL (do NOT move it onto the SSD mount) — commit bb281a7.
# It holds ONLY multiprocessing's AF_UNIX rendezvous sockets (a few bytes each) and small python
# tempfiles -- never DATA, so it does not violate the "no node-local /tmp data staging" policy.
# WHY IT MUST NOT BE ON THE SSD: sshfs's mknod() returns EPERM for every non-regular file, so
# bind()ing an AF_UNIX socket under an sshfs path ALWAYS fails with
#   PermissionError: [Errno 1] Operation not permitted   (multiprocessing/connection.py, _socket.bind)
# torch's default `file_descriptor` sharing strategy routes every DataLoader worker->main batch
# through multiprocessing.resource_sharer.DupFd, which needs exactly such a listener socket. The
# exception is raised in the queue FEEDER THREAD, so the trainer does not exit -- it HANGS at step 0
# holding a GPU. Commit 7f72c3d pointed TMPDIR at "$SSD/tmp_scratch_..." and thereby hung every job
# submitted after it. Job-scoped so concurrent tiers on one node never collide, and removed on exit so
# nothing accumulates on the node's /tmp.
export TMPDIR="/tmp/ntv3_sock_${USER}_${SLURM_JOB_ID:-$$}"; mkdir -p "$TMPDIR"
# hang_watchdog_stop FIRST on every exit path (success, failure, requeue): a finished job must never
# leave a watchdog process behind that could requeue it after the fact.
trap 'hang_watchdog_stop; rm -rf "$TMPDIR"' EXIT
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
# CLEAN out dir: parallel to the per-track+KD arm's ${SIZE}_scratch_kd_s0, NEVER overlapping it.
OUT="$SSD/ntv3_targets/size_sweep_pertrack/t12_ENCSR325NFE/${SIZE}_scratch_nokd_matched_s0"
DRY=()
if [ "${NTV3_DRY_RUN:-0}" = "1" ]; then   # same script, same flags, 2 steps -> proves the queued config
    DRY=(--dry_run)
    OUT="$SSD/ntv3_targets/size_sweep_pertrack/t12_ENCSR325NFE/_dryrun/${SIZE}_scratch_nokd_matched_dryrun"
fi
mkdir -p "$OUT"
# Per-size squeue visibility + a size-named log symlink (array shares one job-name).
command -v scontrol >/dev/null 2>&1 && scontrol update JobId="${SLURM_JOB_ID}" Name="loom_ntv3-pertrack-t12-nokd-${TAG}" 2>/dev/null || true
ln -sf "slurm-loom_ntv3-pertrack-t12-nokd-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out" \
       "slurm-loom_ntv3-pertrack-t12-nokd-${TAG}-${SLURM_ARRAY_JOB_ID}.out" 2>/dev/null || true
echo "=== [array ${SLURM_ARRAY_TASK_ID}] PER-TRACK NO-KD size=$SIZE track=12/ENCSR325NFE -> $OUT (host $(hostname -s)) ==="

# ── window-set spec READ FROM THE KD ARM'S OWN CACHE (never hardcoded) ───────────────────────────────
# Only meta.pt (~1MB) is opened; logits.npy is NEVER touched, so no teacher signal can leak in and there
# is no 53GB RAM load. The asserts fail loudly if the cache is ever rebuilt with a different tiling.
SRC_CACHE="$SSD/ntv3_targets/teacher_cache_joint34"
[ -f "$SRC_CACHE/meta.pt" ] || { echo "FATAL: joint-34 cache meta missing at $SRC_CACHE"; exit 1; }
CACHE_SPEC="$("$PV" -c "
import torch
m = torch.load('$SRC_CACHE/meta.pt', map_location='cpu')
assert float(m.get('overlap', 0.0)) == 0.0, ('cache overlap', m.get('overlap'))
assert int(m['sequence_length']) == 32768, ('cache seq_len', m['sequence_length'])
assert m.get('track_subset') is None, ('cache is a track SUBSET', m.get('track_subset'))
print(int(m['limit_num_samples']), int(m['num_windows']))
")" || { echo "FATAL: could not read the window spec from $SRC_CACHE/meta.pt"; exit 1; }
CACHE_LIMIT="${CACHE_SPEC%% *}"; CACHE_NWIN="${CACHE_SPEC##* }"
echo "DATA-MATCH: reproducing the per-track KD arm's window set from $SRC_CACHE/meta.pt -> "\
     "--train_overlap 0 --max_train_samples $CACHE_LIMIT (expect 'windows: train $CACHE_NWIN')"

# MOUNT GUARD (sourced above): on a mount-caused crash, requeue instead of dying; the trainer then
# RESUMES from latest_state.pth (or restarts clean if none was written yet).
GUARD_PATHS=("$SSD/ntv3_benchmark_data/human/genome.fasta"
             "$SSD/ntv3_benchmark_data/human/genome.fasta.fai"
             "$SSD/ntv3_benchmark_data/human/functional_tracks"
             "$SRC_CACHE/meta.pt" "$HOME_HF")
mount_preflight "${GUARD_PATHS[@]}" || exit 1

# --track_subset 12 => 1-track head (ENCSR325NFE). NO --teacher, NO --cached_teacher_logits, NO
# --stage_dir: pure GT-labels-only training, teacher-free, SSD-direct bigWig reads with the hardened
# sshfs-retry loader. --train_overlap 0 --max_train_samples $CACHE_LIMIT: DATA-MATCH to the per-track KD
# arm (see header) -> the SAME 63,707-window stream, so ONLY the teacher differs.
# HANG WATCHDOG armed here, on THIS job's own stdout: if the log stops growing for 4h before the first
# `step N/` line (startup grace) or 4h after it (stall timeout), the job is requeued and RESUMES from
# latest_state.pth. It shares the mount guard's requeue budget + one-requeue-per-dispatch stamp.
hang_watchdog_start
set +e
"$PV" -u -m src.train.finetune_ntv3 \
      --data_dir "$SSD/ntv3_benchmark_data" \
      --model "$SIZE" \
      --track_subset 12 \
      --train_overlap 0 --max_train_samples "$CACHE_LIMIT" \
      --kd_w_ce 1.0 --kd_w_kl 0 --kd_w_mse 0 \
      --kd_gt_loss poisson_multinomial --kd_multinomial_weight 5.0 \
      --sequence_length 32768 --mini_batch_size 4 --num_accumulation_gradient 8 \
      --num_steps_training 19932 --num_steps_warmup 598 \
      --num_workers 8 \
      --seed 0 --amp \
      "${DRY[@]}" \
      --out "$OUT"
rc=$?
set -e
requeue_if_mount_failure "$rc" "${GUARD_PATHS[@]}" || exit "$rc"
echo "=== done $SIZE: $OUT/ntv3_finetune_result.json ==="
