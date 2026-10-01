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
# ── DATA-STREAM MATCHING (added 2026-07-24 — fixes a CONFOUND in the first no-KD ladder) ─────────────
# The first no-KD ladder (out dirs *_scratch_nokd_s0, of which only ntv3-4m finished) was NOT data-matched
# to the KD arm, so its "KD delta" mixed teacher-on/off with a data-diversity difference:
#   KD arm    : --cached_teacher_logits <dir> makes finetune_ntv3.py OVERRIDE the data config at l.253
#               `train_limit, args.train_overlap = _cm["limit_num_samples"], _cm.get("overlap", 0.0)`
#               -> overlap 0.0 + limit 64000 -> GenomeBigWigDataset stride = 32768 (non-overlapping)
#               -> 63,707 DISTINCT train windows, revisited ~10x over 19932 steps (637,824 window draws).
#   no-KD arm : no cache -> NOTHING overrides the defaults -> --train_overlap 0.999 (l.50) and
#               --max_train_samples None (l.238 `train_limit = args.max_train_samples`) -> stride 32
#               -> ~65.05M candidate windows, i.e. an effectively all-distinct stream.
# The arms therefore differed in BOTH teacher on/off AND data diversity/repetition. (The no-KD arm was if
# anything ADVANTAGED by the more diverse stream, so the measured +0.0213 KD gain at 4m is a conservative
# lower bound — but it is not a clean KD-only delta.)
# THE FIX (this script): reproduce the cache's window set WITHOUT loading its logits, using the two flags
# the trainer already exposes for exactly this ("cap train windows for a cached-baseline matched to a
# cache's fixed window set; pair with --train_overlap 0", l.65-67):
#       --train_overlap 0  --max_train_samples <cache meta limit_num_samples>
# That is sufficient AND exact: GenomeBigWigDataset derives its windows deterministically from
# (split regions, sequence_length, overlap, limit) —
#   l.180  self.stride = max(1, int((1 - overlap) * sequence_length))
#   l.182  regions = sample_regions_for_total_length(regions, limit_num_samples * sequence_length)
# and sample_regions_for_total_length is SEEDED (seed=0, not --seed) and depends only on the region list
# and the byte budget. Same regions + same seq_len + overlap 0 + limit 64000 => the SAME 63,707 windows in
# the SAME order as the cache (this is precisely the identity the KD run asserts at l.313-322 against the
# cache's stored coords). We read the LIMIT from the cache's own meta.pt rather than hardcoding it, so the
# two arms cannot silently drift apart, and we do NOT touch logits.npy (no 53GB load, no KD signal).
# OUT DIRS: *_scratch_nokd_MATCHED_s0 — deliberately distinct from the unmatched *_scratch_nokd_s0, which
# stays on disk (and on HF) as a valid "labels-only, more-diverse-data" datapoint. Distinct dirs also stop
# the trainer resuming a matched run from an UNMATCHED latest_state.pth.
#
# DRY-RUN (do this before every ladder submit): NTV3_DRY_RUN=1 runs THIS EXACT script/flags for 2 steps
# + 1 val + 1 test batch into a _dryrun/ out dir, so the thing verified is the thing queued (no drift
# between a separate dry-run script and the real one). It prints `windows: train N / val ...` — N MUST be
# the KD arm's 63707.
#   sbatch --export=ALL,NTV3_DRY_RUN=1 --array=0 --time=02:00:00 slurm/ntv3_size_sweep_scratch_nokd.sbatch
#
# /tmp POLICY (HARD): the ONLY "/tmp" path used is the galaxy SSD mount ${OG_SCRATCH:-$PWD/output}.
# NO node-local /tmp data staging (no --stage_dir, no cache copy): bigWigs are read DIRECTLY from the SSD
# via pyBigWig/pyfaidx indexed file-handle reads (NOT numpy memmap -> the sshfs memmap-hang failure mode
# does not apply), with the hardened per-window sshfs-retry loader + num_workers 8. TRITON/HF/WANDB -> SSD.
# These runs read ONLY the teacher cache's meta.pt (~1MB, for the window spec) and never logits.npy, so
# there is no ~53GB RAM load => LOW-RAM backfill: --mem 48G, nice=200, %1 throttle (2026-07-24: dropped
# from %2 so that this ladder + the running scratch+KD sweep together stay within a ~1-2 GPU NTv3
# footprint), courteous to the scratch+KD sweep and the per-track t12 ladder on the shared nodes.
#   sbatch slurm/ntv3_size_sweep_scratch_nokd.sbatch          # full ladder 0-4 (4m/8m/30m/100m/300m)
#
# NODE PLACEMENT (2026-07-23): NOT pinned to laniakea any more (the NTv3 sweep was monopolising it).
# `--exclude=galaxy` leaves laniakea AND voyager eligible so a tier starts on whichever frees first;
# galaxy's 24G 3090s cannot hold these seq-32768 tiers. nice=200 keeps it a backfill citizen.
#SBATCH --job-name=loom_ntv3-size-nokd
#SBATCH --array=0-4%1
#SBATCH --nice=200
#SBATCH --nodes=1
#SBATCH --exclude=galaxy
#SBATCH --partition=zhanglab.p
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --requeue
#SBATCH --time=30-00:00:00
#SBATCH --output=slurm-%x-%A_%a.out
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
# The teacher cache's meta.pt is REQUIRED here (not its logits): it defines the window set this arm must
# match, so a root that lacks it is the wrong root.
SSD="$(resolve_ssd_mount ntv3_benchmark_data/human/genome.fasta \
                         ntv3_targets/teacher_cache_joint34/meta.pt)" || exit 1
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
# Distinct out dir (parallel to, and NEVER overlapping, the KD sweep's ${SIZE}_scratch_kd_s0 NOR the
# superseded UNMATCHED ${SIZE}_scratch_nokd_s0 — see DATA-STREAM MATCHING above).
OUT="$SSD/ntv3_targets/size_sweep/${SIZE}_scratch_nokd_matched_s0"
DRY=()
if [ "${NTV3_DRY_RUN:-0}" = "1" ]; then   # same script, same flags, 2 steps -> proves the queued config
    DRY=(--dry_run)
    OUT="$SSD/ntv3_targets/size_sweep/_dryrun/${SIZE}_scratch_nokd_matched_dryrun"
fi
mkdir -p "$OUT"
# Per-size squeue visibility (array shares one job-name; log file stays slurm-%x-%A_%a.out).
command -v scontrol >/dev/null 2>&1 && scontrol update JobId="${SLURM_JOB_ID}" Name="loom_ntv3-size-nokd-${SIZE}" 2>/dev/null || true
echo "=== [array ${SLURM_ARRAY_TASK_ID}] NO-KD size=$SIZE -> $OUT (host $(hostname -s)) ==="

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
echo "DATA-MATCH: reproducing the KD arm's window set from $SRC_CACHE/meta.pt -> "\
     "--train_overlap 0 --max_train_samples $CACHE_LIMIT (expect 'windows: train $CACHE_NWIN')"

# MOUNT GUARD (sourced above): the 2026-07-23 sshfs outage EPERM'd every SSD path and killed the 8m/30m
# runs at step ~500 (before their first checkpoint). On a mount-caused crash, requeue instead of dying;
# the trainer then RESUMES from latest_state.pth (or restarts clean if none was written yet).
# NOTE: this ladder renames itself (scontrol update Name=...), which used to defeat the outage-signature
# check — see the MOUNT_GUARD_LOG note in _mount_guard.sh; that is why 257151_3/257224_1 were not requeued.
GUARD_PATHS=("$SSD/ntv3_benchmark_data/human/genome.fasta"
             "$SSD/ntv3_benchmark_data/human/genome.fasta.fai"
             "$SSD/ntv3_benchmark_data/human/functional_tracks"
             "$SRC_CACHE/meta.pt" "$HOME_HF")
mount_preflight "${GUARD_PATHS[@]}" || exit 1

# NO --teacher, NO --cached_teacher_logits, NO --stage_dir: pure GT-labels-only training, teacher-free,
# SSD-direct bigWig reads with the hardened sshfs-retry loader. See header for the verified no-KD path.
# --train_overlap 0 --max_train_samples $CACHE_LIMIT: DATA-MATCH to the KD arm (see header). These are the
# ONLY difference from the pre-2026-07-24 unmatched no-KD ladder; the loss path is unchanged.
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
