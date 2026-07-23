#!/bin/bash
# Shared SSD-mount (sshfs) resilience helpers for the NTv3 size-ladder SLURM jobs.
#
# WHY THIS EXISTS
#   On 2026-07-23 a transient sshfs outage on laniakea made every path under
#   /tmp/galaxy_srv_disk00/pengchx3 return `PermissionError: [Errno 1] Operation not permitted`.
#   It surfaced as `pyfaidx.FastaNotFoundError` inside DataLoader workers (and as an EPERM on the
#   HF tokenizer module for a job still in startup) and killed 7 runs at once, exit 1:0 — one of
#   them at step 19500/19932 (98%).
#   `#SBATCH --requeue` alone does NOT cover this: SLURM auto-requeues on node failure / preemption,
#   never on a non-zero *script* exit. These two helpers close that gap. Combined with the trainer's
#   `latest_state.pth` resume (src/train/finetune_ntv3.py), a mount blip becomes self-healing:
#   requeue -> re-dispatch -> RESUME at the last validation step instead of losing hours.
#
# USAGE (sourced by the sbatch after $SSD is resolved, before launching the trainer):
#     source "$REPO/slurm/_mount_guard.sh"
#     mount_preflight <path>...            # blocks until readable; requeues instead of burning a GPU
#     set +e
#     "$PV" -u -m src.train.finetune_ntv3 ... ; rc=$?
#     set -e
#     requeue_if_mount_failure "$rc" <path>...   # only requeues for MOUNT failures; real bugs stay FAILED
#     exit "$rc"
#
# Tunables (env): MOUNT_GUARD_TRIES (default 10), MOUNT_GUARD_SLEEP (30s), MOUNT_GUARD_MAX_REQUEUE (5).

MOUNT_GUARD_TRIES="${MOUNT_GUARD_TRIES:-10}"
MOUNT_GUARD_SLEEP="${MOUNT_GUARD_SLEEP:-30}"
MOUNT_GUARD_MAX_REQUEUE="${MOUNT_GUARD_MAX_REQUEUE:-5}"

# _mount_readable <path>... : 0 iff EVERY path exists AND is actually readable right now.
# Existence alone is not enough — during the outage `stat` succeeded while open() returned EPERM,
# so files are probed with a 1-byte read and directories with a listing.
_mount_readable() {
    local p
    for p in "$@"; do
        if [ -d "$p" ]; then
            ls -A "$p" >/dev/null 2>&1 || return 1
        else
            [ -f "$p" ] || return 1
            head -c 1 -- "$p" >/dev/null 2>&1 || return 1
        fi
    done
    return 0
}

# _job_stdout : path of THIS job's SLURM stdout file (works for any --output pattern).
_job_stdout() {
    [ -n "${SLURM_JOB_ID:-}" ] || return 0
    scontrol show job "$SLURM_JOB_ID" 2>/dev/null | tr ' ' '\n' | sed -n 's/^StdOut=//p' | head -1
}

# _requeue_or_die <reason> : put THIS job back in the queue (it resumes from latest_state.pth on the
# next dispatch), unless MOUNT_GUARD_MAX_REQUEUE restarts have already been spent — then fail loudly
# so a genuinely dead mount is visible instead of thrashing the scheduler forever.
_requeue_or_die() {
    local reason="$1" n="${SLURM_RESTART_COUNT:-0}"
    if [ -z "${SLURM_JOB_ID:-}" ] || ! command -v scontrol >/dev/null 2>&1; then
        echo "[mount_guard] FATAL: $reason (no SLURM context to requeue into)" >&2
        return 1
    fi
    if [ "$n" -ge "$MOUNT_GUARD_MAX_REQUEUE" ]; then
        echo "[mount_guard] FATAL: $reason — already requeued $n time(s) (max $MOUNT_GUARD_MAX_REQUEUE)." \
             "The mount looks persistently broken; giving up so a human sees it." >&2
        return 1
    fi
    echo "[mount_guard] $reason — requeueing job $SLURM_JOB_ID (restart #$((n + 1)) of $MOUNT_GUARD_MAX_REQUEUE);" \
         "it will RESUME from latest_state.pth." >&2
    scontrol requeue "$SLURM_JOB_ID" || { echo "[mount_guard] FATAL: scontrol requeue failed" >&2; return 1; }
    sleep 120   # SLURM kills the job during this sleep; never race on into the trainer
    return 1
}

# mount_preflight <path>... : assert the SSD mount answers reads before a GPU slot is committed.
# Retries MOUNT_GUARD_TRIES x MOUNT_GUARD_SLEEP (default 5 min) to ride out a blip, then requeues.
mount_preflight() {
    local i
    for i in $(seq 1 "$MOUNT_GUARD_TRIES"); do
        if _mount_readable "$@"; then
            echo "[mount_guard] preflight OK on attempt $i/$MOUNT_GUARD_TRIES ($# paths readable on $(hostname -s))"
            return 0
        fi
        echo "[mount_guard] preflight attempt $i/$MOUNT_GUARD_TRIES FAILED — SSD mount not readable;" \
             "sleeping ${MOUNT_GUARD_SLEEP}s" >&2
        sleep "$MOUNT_GUARD_SLEEP"
    done
    _requeue_or_die "SSD mount unreadable at startup after $MOUNT_GUARD_TRIES attempts"
}

# requeue_if_mount_failure <trainer_exit_code> <path>... : classify a non-zero trainer exit.
# Requeue ONLY when it was the mount (mount unreadable now, or the outage signature in this job's
# log). Any other failure — a real bug, OOM, bad flag — is left FAILED so it stays visible.
requeue_if_mount_failure() {
    local rc="$1"; shift
    [ "$rc" -eq 0 ] && return 0
    local log mount_failed=0
    if ! _mount_readable "$@"; then
        mount_failed=1
        echo "[mount_guard] trainer exit $rc AND the SSD mount is unreadable right now." >&2
    else
        log="$(_job_stdout)"
        if [ -n "$log" ] && [ -f "$log" ] && tail -n 400 -- "$log" | grep -qE \
             'Operation not permitted|FastaNotFoundError|Transport endpoint is not connected|Input/output error|Stale file handle'; then
            mount_failed=1
            echo "[mount_guard] trainer exit $rc with an SSD-outage signature in $log (mount has since recovered)." >&2
        fi
    fi
    if [ "$mount_failed" -ne 1 ]; then
        echo "[mount_guard] trainer exit $rc is NOT a mount failure — leaving the job FAILED for inspection." >&2
        return 1
    fi
    _requeue_or_die "trainer exit $rc caused by an SSD-mount failure"
}
