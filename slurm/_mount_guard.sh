#!/bin/bash
# Shared SSD-mount (sshfs) resilience helpers for the NTv3 size-ladder SLURM jobs.
#
# WHY THIS EXISTS
#   On 2026-07-23 a transient sshfs outage on laniakea made every path under
#   ${OG_SCRATCH:-$PWD/output} return `PermissionError: [Errno 1] Operation not permitted`.
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
# Tunables (env): MOUNT_GUARD_TRIES (default 10), MOUNT_GUARD_SLEEP (30s), MOUNT_GUARD_MAX_REQUEUE (5),
# GUARD_REQUEUE_STAMP_DIR ($HOME/.ntv3_guard_requeue — the one-requeue-per-dispatch interlock shared
# with slurm/_hang_watchdog.sh, which covers the HUNG-but-never-exits failure mode this file cannot see).

MOUNT_GUARD_TRIES="${MOUNT_GUARD_TRIES:-10}"
MOUNT_GUARD_SLEEP="${MOUNT_GUARD_SLEEP:-30}"
MOUNT_GUARD_MAX_REQUEUE="${MOUNT_GUARD_MAX_REQUEUE:-5}"

# resolve_ssd_mount <required-relative-path>... : echo the galaxy-SSD root for THIS node, or fail.
#   SSD="$(resolve_ssd_mount ntv3_benchmark_data/human/genome.fasta)" || exit 1
#
# HARD /tmp POLICY (node-agnostic; these jobs may land on laniakea OR voyager OR galaxy).
# The only acceptable "/tmp" path is the galaxy SSD, reachable as:
#   laniakea/voyager : /tmp/galaxy_srv_disk00/$USER  = a real sshfs MOUNTPOINT
#   galaxy           : /tmp/galaxy_srv_disk00        = a SYMLINK to the local /srv/disk00/sshfs
# A same-named plain directory on the node's own scratch must NEVER be used: voyager's /tmp is a 49G
# LV that has overflowed catastrophically, and silently "succeeding" onto it would both fill the node
# and train on absent/stale data. TEST USED: a candidate is refused if it sits on the SAME FILESYSTEM
# as the node's /tmp (st_dev comparison). That is node-agnostic — it holds on laniakea/voyager (where
# /tmp is a real LV and the SSD is an sshfs mount) and on galaxy (where /tmp is a symlink to
# /lv_scratch/tmp while the SSD is /dev/sda1), unlike a path-prefix or `mountpoint` test, both of
# which this repo has now seen misfire. A candidate must also already hold the required data.
# MOUNT_GUARD_SSD_CANDIDATES: search order, overridable ONLY so the self-test can exercise this exact
# function against synthetic roots (never override it in a real job).
: "${MOUNT_GUARD_SSD_CANDIDATES:=/tmp/galaxy_srv_disk00/$USER /srv/disk00/sshfs/$USER /srv/disk00/$USER}"
resolve_ssd_mount() {
    local cand rel ok tmpdev
    tmpdev="$(stat -c %d -- /tmp/. 2>/dev/null || echo NONE)"
    for cand in $MOUNT_GUARD_SSD_CANDIDATES; do
        [ -d "$cand" ] || continue
        if [ "$(stat -c %d -- "$cand" 2>/dev/null || echo X)" = "$tmpdev" ]; then
            echo "[mount_guard] REFUSING $cand on $(hostname -s): it is on the node-local /tmp filesystem, not the SSD mount" >&2
            continue
        fi
        ok=1
        for rel in "$@"; do
            [ -e "$cand/$rel" ] || { echo "[mount_guard] $cand is missing required data: $rel" >&2; ok=0; break; }
        done
        [ "$ok" = 1 ] || continue
        echo "$cand"; return 0
    done
    echo "[mount_guard] FATAL: no usable galaxy-SSD mount on $(hostname -s) (never falling back to node-local /tmp)" >&2
    return 1
}

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

# MOUNT_GUARD_LOG : the file THIS batch script's stdout is actually attached to, resolved ONCE at
# source time (fd 1 of the sourcing shell, i.e. the SLURM --output file).
# WHY: `scontrol show job` re-expands the --output pattern against the job's CURRENT name, so any
# sbatch that renames itself (`scontrol update JobId=... Name=...`, as the no-KD ladder does) gets
# back a path that DOES NOT EXIST. `[ -f "$log" ]` then failed silently and the outage-signature
# branch never ran — on 2026-07-23 that misfiled two genuine sshfs-blip deaths (257151_3, 257224_1)
# as "NOT a mount failure", so they were never requeued. fd 1 is immune to renaming.
MOUNT_GUARD_LOG="${MOUNT_GUARD_LOG:-$(readlink -f "/proc/$$/fd/1" 2>/dev/null || true)}"

# _job_stdout : path of THIS job's SLURM stdout file (works for any --output pattern, and for jobs
# that rename themselves). Prefers the fd-1 target; falls back to scontrol only if that is not a file.
_job_stdout() {
    if [ -n "${MOUNT_GUARD_LOG:-}" ] && [ -f "$MOUNT_GUARD_LOG" ]; then
        echo "$MOUNT_GUARD_LOG"; return 0
    fi
    [ -n "${SLURM_JOB_ID:-}" ] || return 0
    scontrol show job "$SLURM_JOB_ID" 2>/dev/null | tr ' ' '\n' | sed -n 's/^StdOut=//p' | head -1
}

# _requeue_budget_left : 0 iff THIS job may still be requeued — i.e. there IS a SLURM context and the
# MOUNT_GUARD_MAX_REQUEUE budget is not yet spent. Exposed separately (and used by _requeue_or_die, so
# the predicate has exactly one definition) because slurm/_hang_watchdog.sh must know BEFORE it acts
# whether a requeue is still possible: when it is not, the hang watchdog kills the trainer so the job
# FAILS visibly rather than hanging on forever. The budget is SHARED — the mount guard and the hang
# watchdog together may requeue a job at most MOUNT_GUARD_MAX_REQUEUE times.
_requeue_budget_left() {
    [ -n "${SLURM_JOB_ID:-}" ] || return 1
    command -v scontrol >/dev/null 2>&1 || return 1
    [ "${SLURM_RESTART_COUNT:-0}" -lt "$MOUNT_GUARD_MAX_REQUEUE" ]
}

# GUARD_REQUEUE_STAMP_DIR : where the one-requeue-per-dispatch stamp lives. The stamp is keyed by
# job id AND restart count, so it is scoped to the CURRENT dispatch and a later dispatch is never
# blocked by a stale file. WHY: the mount guard and the hang watchdog can both decide to requeue the
# same dispatch (the watchdog requeues a hang; SLURM then SIGTERMs the trainer; the batch script sees a
# non-zero rc and a 'Operation not permitted' signature in the log and would requeue AGAIN). Two
# requeues for one dispatch burn two restarts of the shared budget and can race the scheduler.
# Overridable only so the self-tests can point it at a scratch dir.
: "${GUARD_REQUEUE_STAMP_DIR:=$HOME/.ntv3_guard_requeue}"

# _requeue_or_die <reason> : put THIS job back in the queue (it resumes from latest_state.pth on the
# next dispatch), unless the shared requeue budget is spent or this dispatch has already been requeued
# — then fail loudly so a genuinely dead mount / persistent hang is visible instead of thrashing the
# scheduler forever. ALWAYS returns non-zero: on the success path it sleeps waiting for SLURM to kill
# the job, so callers can uniformly write `_requeue_or_die ... || exit 1`.
# GUARD_TAG names the mechanism in the log lines (mount_guard | hang_watchdog).
_requeue_or_die() {
    local reason="$1" n="${SLURM_RESTART_COUNT:-0}" tag="${GUARD_TAG:-mount_guard}" stamp
    if ! _requeue_budget_left; then
        if [ -z "${SLURM_JOB_ID:-}" ] || ! command -v scontrol >/dev/null 2>&1; then
            echo "[$tag] FATAL: $reason (no SLURM context to requeue into)" >&2
        else
            echo "[$tag] FATAL: $reason — already requeued $n time(s) (max $MOUNT_GUARD_MAX_REQUEUE)." \
                 "It looks persistently broken; giving up so a human sees it." >&2
        fi
        return 1
    fi
    stamp="$GUARD_REQUEUE_STAMP_DIR/${SLURM_JOB_ID}.${n}"
    if [ -e "$stamp" ]; then
        echo "[$tag] $reason — but THIS dispatch was already requeued (stamp $stamp);" \
             "not requeueing twice. Letting the job end so SLURM's pending requeue takes over." >&2
        return 1
    fi
    mkdir -p "$GUARD_REQUEUE_STAMP_DIR" 2>/dev/null && : > "$stamp" 2>/dev/null
    echo "[$tag] $reason — requeueing job $SLURM_JOB_ID (restart #$((n + 1)) of $MOUNT_GUARD_MAX_REQUEUE);" \
         "it will RESUME from latest_state.pth." >&2
    scontrol requeue "$SLURM_JOB_ID" || { echo "[$tag] FATAL: scontrol requeue failed" >&2; return 1; }
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
