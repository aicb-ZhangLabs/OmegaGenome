#!/bin/bash
# STEP-PROGRESS HANG WATCHDOG for the NTv3 size-ladder SLURM jobs.
#
# WHY THIS EXISTS (the last uncovered failure mode)
#   slurm/_mount_guard.sh classifies non-zero EXITS. A HUNG trainer never exits. On 2026-07-23 commit
#   7f72c3d moved TMPDIR onto the sshfs mount; sshfs mknod() returns EPERM for AF_UNIX sockets, so
#   torch's DataLoader `file_descriptor` sharing raised PermissionError inside the queue FEEDER THREAD.
#   The main process did NOT exit — it sat at step 0 holding a GPU. 257235_4 burned 1h12m and 257236_4
#   49m producing ZERO steps, and only a human looking at the logs noticed. Unattended, --time=30-00:00:00
#   means such a job holds a GPU for a MONTH.
#   This watchdog is the missing half: it watches this job's own stdout for FORWARD PROGRESS and, if the
#   log goes silent past a (deliberately huge) threshold, requeues the job so it resumes from
#   latest_state.pth — exactly what the mount guard does on a mount-caused exit.
#
# THE OVERRIDING RISK IS A FALSE POSITIVE. Killing a healthy 98%-done run costs days; a late kill costs
# a few extra GPU-hours. Every design choice below is the conservative one, and both thresholds are
# derived from MEASURED gaps in this repo's own slurm-loom_ntv3-*.out logs (see THRESHOLDS below).
#
# LIVENESS SIGNAL = "the log file GREW since the last poll" (mtime/size advancing), which by construction
#   subsumes every `step N/19932`, `[val] step N`, `new best val ... -> saved`, `[save-retry ...]`,
#   `RESUMED from ...` and `TEST mean Pearson` line, plus any warning the trainer emits. It is the most
#   FALSE-POSITIVE-SAFE choice: a phase that legitimately prints nothing but a save-retry still counts as
#   alive. Requiring specifically `step` lines would have fired during the end-of-run test evaluation
#   (10531 windows, ZERO output between `step 19900/19932` and `TEST mean Pearson`).
#   It still catches the TMPDIR hang: MEASURED, that hang emits exactly 16 tracebacks
#   (`grep -c "PermissionError: \[Errno 1\]"` == 16 in 257235_4 / 257236_4 / 257441_1 — one per
#   DataLoader worker x2) and then the file goes PERMANENTLY silent, so growth stops long before the
#   threshold. Step/val lines are still parsed, but only to decide the PHASE and to name what was last
#   seen in the failure message.
#
# PHASES
#   STARTUP  (until the first `step N/...` line ever appears)  -> HANG_WATCHDOG_STARTUP_GRACE
#            Covers the 53 GB teacher-cache RAM load of the KD arms, the 65M-window enumeration of the
#            no-KD arm, CUDA/Triton warmup and the `latest_state.pth` resume load.
#   RUNNING  (after the first step line)                       -> HANG_WATCHDOG_STALL_TIMEOUT
#
# ON STALL: log a LOUD, unambiguous reason, then `scontrol requeue` (the job resumes from
#   latest_state.pth) — subject to the SAME MOUNT_GUARD_MAX_REQUEUE budget and the SAME
#   one-requeue-per-dispatch stamp as the mount guard, so the two mechanisms can never double-requeue or
#   thrash the scheduler. When the budget is spent (or there is no SLURM context) it does NOT requeue:
#   it SIGTERM/SIGKILLs the trainer so the job FAILS visibly instead of hanging forever.
#
# USAGE (in the sbatch, after mount_preflight, before launching the trainer):
#     source "$REPO/slurm/_mount_guard.sh"
#     source "$REPO/slurm/_hang_watchdog.sh"
#     trap 'hang_watchdog_stop; rm -rf "$TMPDIR"' EXIT     # never leave an orphan watchdog behind
#     ...
#     hang_watchdog_start
#     "$PV" -u -m src.train.finetune_ntv3 ...
#
# ---------------------------------------------------------------------------------------------------
# THRESHOLDS — DERIVED FROM MEASURED DATA, NOT GUESSED (2026-07-24; see the analysis in the commit msg)
#
#   MEASURED startup (job start -> first `step N/` line), per arm:
#     no-KD arm (no teacher cache)   : <= 12 min  (257224_1 8m: step 500 reached within its 11m53s life;
#                                                  257446_1 8m: step 1000 within 23m01s)
#     joint/KD arm (53 GB RAM load)  : ~8-27 min  (UPPER BOUNDS, since these logs carry no timestamps:
#                                                  253825_3/_4 loaded the 53.2 GB cache AND entered the
#                                                  DataLoader inside 27m24s / 27m17s;  257439_2 was
#                                                  still mid-load at 4m52s;  and the running joint 300m
#                                                  257449_4 gives ~8 min by arithmetic --- 24848 s from
#                                                  job start to the step-18500 checkpoint mtime, minus
#                                                  2500 steps x 9.3 s and 5 val+save cycles)
#     per-track t12 arm (1.5 GB .pt) : < 5 min    (257451_4 reached step 1000 well inside its first hour)
#     PLUS the project record for this 53 GB RAM load under slow sshfs: ~55 min (not re-measurable from
#     these logs, so it is carried as the worst case rather than discarded).
#     => STARTUP_GRACE = 4 h = 4.4 x that 55-min worst case (>= the required 3x), and 8.8 x the largest
#        startup this repo's logs can actually bound (27 min).
#
#   MEASURED gap BETWEEN consecutive liveness lines during healthy training (log cadence = 50 steps):
#   These were sampled LIVE (2 s polling of the log's size) on the running joint 300m 257449_4, the
#   slowest tier in the ladder, over steps 18750-19000 on laniakea while 3 other GPU jobs shared it:
#     300m, per 50-step line :  429 s, 467 s, 469 s, 485 s   -> MAX MEASURED GAP 485 s (8.1 min)
#     300m, validation (997 windows, every 500 steps) : 95 s  (`step 19000` -> `[val] step 19000`)
#     4m/8m/30m/100m tiers   :  strictly faster (their whole runs finish in 6-28 h vs the 300m's 2 d)
#     300m, val + 3.6 GB latest_state.pth checkpoint write : 105 s end to end (the `[val] step 19000`
#                              line at t=1784883657, latest_state.pth mtime t=1784883667) -- i.e. the
#                              validate+save cycle is SHORTER than one ordinary 50-step gap, not longer.
#     LARGEST no-output phase in the whole run = the FINAL TEST EVAL: between `step 19900/19932` and
#                              `TEST mean Pearson` the trainer prints NOTHING while it runs 32 steps
#                              plus 10531 test windows. From the MEASURED val rate (997 windows in
#                              95 s = 0.095 s/window) that is ~1000 s = 17 min, ~35 min if the node is
#                              twice as contended as it was during the measurement.
#     => STALL_TIMEOUT = 4 h = 30 x the max MEASURED gap (485 s), 14 x the derived 17-min test eval and
#        ~7 x its contention-doubled 35 min -- comfortably past the required 5x on every reading.
#        Deliberately 4 h and not 3 h: a systematically too-short threshold would kill runs during the
#        final test eval, requeue them to redo it, and eventually burn the budget and mark a FINISHED
#        run FAILED. NOTHING in the observed data comes within 4x of this threshold.
#
#   Cost asymmetry check: a FALSE kill costs a requeue + re-run from the last 500-step checkpoint
#   (<= ~1.5 GPU-h) but, worse, risks doing that repeatedly; a LATE kill costs at most 3-4 extra
#   GPU-hours versus the ~2 h the 2026-07-23 hang already burned unnoticed. Both thresholds are set on
#   the "late" side of that trade on purpose.
# ---------------------------------------------------------------------------------------------------
#
# Tunables (env, measured defaults baked in):
#   HANG_WATCHDOG_STARTUP_GRACE (14400s = 4h)  no first step line -> stall
#   HANG_WATCHDOG_STALL_TIMEOUT (14400s = 4h)  no new output after the first step line -> stall
#   HANG_WATCHDOG_POLL          (60s)          how often the log is sampled
#   HANG_WATCHDOG_RECHECK       (60s)          final re-look before firing (completion/race safety)
#   HANG_WATCHDOG_KILL_GRACE    (30s)          SIGTERM -> SIGKILL delay when the requeue budget is spent
#   MOUNT_GUARD_MAX_REQUEUE     (5)            SHARED requeue budget with the mount guard

HANG_WATCHDOG_STARTUP_GRACE="${HANG_WATCHDOG_STARTUP_GRACE:-14400}"
HANG_WATCHDOG_STALL_TIMEOUT="${HANG_WATCHDOG_STALL_TIMEOUT:-14400}"
HANG_WATCHDOG_POLL="${HANG_WATCHDOG_POLL:-60}"
HANG_WATCHDOG_RECHECK="${HANG_WATCHDOG_RECHECK:-60}"
HANG_WATCHDOG_KILL_GRACE="${HANG_WATCHDOG_KILL_GRACE:-30}"

# Patterns are kept as data (not inlined) so the self-test exercises the REAL ones.
#   _STEP_RE : the marker that ends the STARTUP phase (first training step logged).
#   _PROGRESS_RE : every "forward progress" line, used ONLY to report what was last seen.
#   _DONE_RE : the run finished — the watchdog must never requeue after this appears.
HANG_WATCHDOG_STEP_RE='step [0-9]+/[0-9]+'
HANG_WATCHDOG_PROGRESS_RE='step [0-9]+/[0-9]+|\[val\] step [0-9]+|new best val|\[save-retry|RESUMED from|preflight OK|TEST mean Pearson'
HANG_WATCHDOG_DONE_RE='TEST mean Pearson|^=== done '

# The mount guard supplies _job_stdout (the fd-1-first stdout resolution, commit 7f72c3d) and
# _requeue_or_die / _requeue_budget_left (the shared requeue budget + one-requeue-per-dispatch stamp).
# Source it if the sbatch has not already done so, so this file is usable stand-alone.
if ! declare -F _requeue_or_die >/dev/null 2>&1; then
    # shellcheck disable=SC1090,SC1091
    source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_mount_guard.sh"
fi

# _hw_say <msg>... : one log line, tagged, on stderr (which SLURM merges into this job's stdout).
# The watchdog writes to the very file it monitors, so it must never write PERIODICALLY — every call
# site below is followed by a last_size resync so the watchdog can never mistake its own output for
# trainer progress.
_hw_say() { echo "[hang_watchdog] $*" >&2; }

# _hw_size <log> : current size in bytes of the monitored log, or the EMPTY STRING if it cannot be
# stat'ed right now. Size (not mtime) is the liveness counter: monotone, cheap, and immune to clock
# skew between the compute node and the shared filesystem. Empty deliberately means "no information"
# rather than "zero bytes" — callers must SKIP such a poll, never treat an unreadable log as a hang.
_hw_size() { stat -c %s -- "$1" 2>/dev/null; }

# _hw_has <regex> <log> : 0 iff the log currently contains a line matching regex (binary-safe).
_hw_has() { grep -aqE "$1" -- "$2" 2>/dev/null; }

# _hw_completed <log> : 0 iff the run has already produced its final result line. THE completion
# predicate — a completed (or completing) run must NEVER be requeued, even if the log has been silent
# for longer than the threshold. Isolated in its own function so the self-test can mutate exactly this
# decision and prove the race-safety cases constrain it.
_hw_completed() { _hw_has "$HANG_WATCHDOG_DONE_RE" "$1"; }

# _hw_last_progress_line <log> : the most recent forward-progress marker, for the failure message.
# Only the log tail is scanned so this stays O(1) on a multi-hundred-KB log.
_hw_last_progress_line() {
    tail -c 65536 -- "$1" 2>/dev/null | grep -aoE "$HANG_WATCHDOG_PROGRESS_RE" | tail -1
}

# _hw_stalled <now_epoch> <last_progress_epoch> <threshold_s> : THE detection predicate.
# 0 iff the log has produced nothing for at least <threshold_s>. Isolated in its own function so the
# self-test can mutate exactly this decision (never-fires / always-fires) and prove every case
# constrains it.
_hw_stalled() { [ "$(( $1 - $2 ))" -ge "$3" ]; }

# _hw_alive <pid> : 0 iff that process still exists (the batch shell that owns the trainer).
_hw_alive() { kill -0 "$1" 2>/dev/null; }

# _hw_children <pid> : pids of the direct children of <pid>, excluding the watchdog itself.
_hw_children() {
    local p out
    out="$(pgrep -P "$1" 2>/dev/null || ps -o pid= --ppid "$1" 2>/dev/null)"
    for p in $out; do [ "$p" = "${BASHPID:-$$}" ] || echo "$p"; done
}

# _hw_kill_trainer <batch_shell_pid> : escalate SIGTERM -> SIGKILL to the trainer.
# Used ONLY when the job may not be requeued (budget spent / no SLURM context): the job must then FAIL
# loudly and give the GPU back, never keep hanging.
_hw_kill_trainer() {
    local ppid="$1" p kids
    kids="$(_hw_children "$ppid")"
    [ -n "$kids" ] || { _hw_say "no trainer child of $ppid left to kill"; return 0; }
    for p in $kids; do _hw_say "SIGTERM -> trainer child pid $p"; kill -TERM "$p" 2>/dev/null; done
    sleep "$HANG_WATCHDOG_KILL_GRACE"
    for p in $kids; do
        if kill -0 "$p" 2>/dev/null; then _hw_say "pid $p survived SIGTERM -> SIGKILL"; kill -KILL "$p" 2>/dev/null; fi
    done
}

# _hw_fire <log> <phase> <threshold_s> <gap_s> <batch_shell_pid> : the loud report + the action.
# Returns 0 if the job was requeued, 1 if it was left to fail (budget spent / no SLURM context).
_hw_fire() {
    local log="$1" phase="$2" thresh="$3" gap="$4" ppid="$5" last
    last="$(_hw_last_progress_line "$log")"
    _hw_say "================== HANG DETECTED — STEP-PROGRESS WATCHDOG FIRING =================="
    _hw_say "job=${SLURM_JOB_ID:-<none>} restart=${SLURM_RESTART_COUNT:-0} host=$(hostname -s) log=$log"
    _hw_say "phase=$phase  threshold=${thresh}s  observed silence=${gap}s (log has not grown by one byte)"
    if [ "$phase" = startup ]; then
        _hw_say "NOT ONE '${HANG_WATCHDOG_STEP_RE}' LINE was ever produced — this is the step-0 hang signature"
        _hw_say "(TMPDIR-on-sshfs AF_UNIX bind EPERM in the DataLoader feeder thread, 2026-07-23 / 7f72c3d)"
    fi
    _hw_say "last forward-progress line seen: ${last:-<NONE — the run never logged any progress>}"
    _hw_say "this is the HANG watchdog, not the mount guard: the trainer never EXITED, it stopped MOVING"
    if _requeue_budget_left; then
        GUARD_TAG=hang_watchdog _requeue_or_die \
            "trainer produced no output for ${gap}s (>= ${thresh}s $phase threshold) — HUNG, not crashed"
        return 0   # _requeue_or_die always returns 1 (it sleeps waiting to be killed); the requeue is the outcome
    fi
    _hw_say "FATAL: no requeue budget left (SLURM_RESTART_COUNT=${SLURM_RESTART_COUNT:-0}," \
            "max $MOUNT_GUARD_MAX_REQUEUE) or no SLURM context — this job hangs REPEATEDLY."
    _hw_say "FATAL: killing the trainer so the job FAILS visibly and returns the GPU. A human must look."
    _hw_kill_trainer "$ppid"
    return 1
}

# _hw_still_stalled <log> <batch_shell_pid> <last_size> : THE race-safety gate, run AFTER a threshold
# has been crossed and BEFORE anything is killed. Pauses HANG_WATCHDOG_RECHECK seconds and looks once
# more. Returns 0 ("fire") only if the run is STILL silent, STILL running and STILL not finished.
# Sets HW_RECHECK_VERDICT = fire | continue | stop  and HW_RECHECK_SIZE = the log size it last saw.
# WHY: the biggest silent phase in a healthy run is the end-of-run test evaluation (10531 windows,
# zero output) — it can cross the threshold and then print everything at once. Isolated in its own
# function so the self-test can mutate exactly this gate and prove the race cases constrain it.
_hw_still_stalled() {
    local log="$1" ppid="$2" last_size="$3" size
    HW_RECHECK_VERDICT=fire
    HW_RECHECK_SIZE="$last_size"
    sleep "$HANG_WATCHDOG_RECHECK"
    if ! _hw_alive "$ppid"; then
        _hw_say "threshold crossed but the batch shell exited during the re-check — NOT requeueing"
        HW_RECHECK_VERDICT=stop; return 1
    fi
    if _hw_completed "$log"; then
        _hw_say "threshold crossed but the run COMPLETED during the re-check — NOT requeueing"
        HW_RECHECK_VERDICT=stop; return 1
    fi
    size="$(_hw_size "$log")"
    if [ -z "$size" ]; then                      # log unreadable at the decisive moment -> stand down
        _hw_say "threshold crossed but $log cannot be read right now — NOT requeueing on missing information"
        HW_RECHECK_VERDICT=continue; return 1
    fi
    HW_RECHECK_SIZE="$size"
    if [ "$size" != "$last_size" ]; then
        _hw_say "threshold crossed but the log grew during the re-check ($last_size -> $size bytes)" \
                "— alive after all, NOT requeueing"
        HW_RECHECK_VERDICT=continue; return 1
    fi
    return 0
}

# hang_watchdog_monitor <log> <batch_shell_pid> : the polling loop. Runs as a BACKGROUND process.
# Exits 0 (cleanly, no requeue) when the batch shell is gone or the run has finished; only ever
# requeues/kills after a full threshold of ZERO log growth plus a final re-check.
hang_watchdog_monitor() {
    local log="$1" ppid="$2"
    set +e                       # a background monitor must never die on a non-zero grep/stat
    local now size last_size last_progress phase thresh gap warned_unreadable=0
    last_progress="$(date +%s)"
    last_size="$(_hw_size "$log")"
    phase=startup
    _hw_has "$HANG_WATCHDOG_STEP_RE" "$log" && phase=running   # a requeued job may already have steps
    while :; do
        sleep "$HANG_WATCHDOG_POLL"
        if ! _hw_alive "$ppid"; then
            _hw_say "batch shell $ppid has exited — trainer finished, watchdog stopping cleanly"
            return 0
        fi
        if _hw_completed "$log"; then
            _hw_say "run COMPLETED (found the final-result line) — watchdog stopping cleanly, no requeue"
            return 0
        fi
        now="$(date +%s)"
        size="$(_hw_size "$log")"
        if [ -z "$size" ]; then
            # The log cannot be stat'ed (NFS blip on $HOME, file rotated, ...). That is MISSING
            # INFORMATION, not evidence of a hang: skip the poll without advancing the stall clock and
            # without firing. Warned only ONCE — the watchdog writes into the file it monitors, so it
            # must never log periodically.
            [ "$warned_unreadable" = 0 ] && { _hw_say "WARNING: cannot stat $log — polls are being" \
                "skipped until it is readable again; the watchdog will NOT fire on missing information"
                warned_unreadable=1; }
            continue
        fi
        warned_unreadable=0
        if [ "$size" != "$last_size" ]; then last_size="$size"; last_progress="$now"; fi
        if [ "$phase" = startup ] && _hw_has "$HANG_WATCHDOG_STEP_RE" "$log"; then
            phase=running
            _hw_say "first step line seen — leaving STARTUP grace (${HANG_WATCHDOG_STARTUP_GRACE}s)," \
                    "now enforcing the stall timeout (${HANG_WATCHDOG_STALL_TIMEOUT}s)"
            last_size="$(_hw_size "$log")"     # do NOT let the line just written count as trainer progress
        fi
        if [ "$phase" = startup ]; then thresh="$HANG_WATCHDOG_STARTUP_GRACE"; else thresh="$HANG_WATCHDOG_STALL_TIMEOUT"; fi
        _hw_stalled "$now" "$last_progress" "$thresh" || continue
        _hw_still_stalled "$log" "$ppid" "$last_size" || {
            case "$HW_RECHECK_VERDICT" in
                stop) return 0 ;;                                    # finished / batch shell gone
                *)    last_size="$HW_RECHECK_SIZE"; last_progress="$(date +%s)"; continue ;;
            esac
        }
        gap=$(( $(date +%s) - last_progress ))
        _hw_fire "$log" "$phase" "$thresh" "$gap" "$ppid"
        return $?
    done
}

# hang_watchdog_start [<log>] : arm the watchdog in the background on THIS job's stdout.
# Sets HANG_WATCHDOG_PID. Resolves the log via the mount guard's _job_stdout, which prefers the fd-1
# target: `scontrol show job` re-expands --output against the job's CURRENT name, so a self-renaming
# sbatch (the no-KD and per-track ladders both `scontrol update ... Name=`) gets back a path that does
# not exist — the bug fixed in 7f72c3d. If no log can be resolved the watchdog stays DISARMED (a
# watchdog that cannot see the log must never guess that the job is hung).
hang_watchdog_start() {
    local log="${1:-$(_job_stdout)}"
    if [ -z "$log" ] || [ ! -f "$log" ]; then
        _hw_say "DISARMED: cannot resolve this job's stdout file (got '${log:-}') — monitoring nothing"
        HANG_WATCHDOG_PID=""
        return 0
    fi
    hang_watchdog_monitor "$log" "$$" &
    HANG_WATCHDOG_PID=$!
    _hw_say "armed (pid $HANG_WATCHDOG_PID) on $log — startup grace ${HANG_WATCHDOG_STARTUP_GRACE}s," \
            "stall timeout ${HANG_WATCHDOG_STALL_TIMEOUT}s, poll ${HANG_WATCHDOG_POLL}s," \
            "requeue budget shared with the mount guard (max $MOUNT_GUARD_MAX_REQUEUE)"
}

# hang_watchdog_stop : disarm. Idempotent, safe to call from an EXIT trap on every path (success,
# failure, requeue) so a finished job never leaves a watchdog process behind to requeue it later.
#
# EVERY command here is `|| true`-guarded and the function always returns 0. WHY (caught by the
# integration dry-run of the wired sbatch): the sbatches run `set -euo pipefail`, which is still in
# force inside the EXIT trap. `wait` on a process we just killed returns 143, so an unguarded `wait`
# ABORTED THE TRAP — the shell exited 143 instead of the trainer's real status (a SUCCESSFUL run would
# have been recorded FAILED by SLURM) and `rm -rf "$TMPDIR"`, the rest of the trap, never ran.
hang_watchdog_stop() {
    [ -n "${HANG_WATCHDOG_PID:-}" ] || return 0
    kill "$HANG_WATCHDOG_PID" 2>/dev/null || true
    wait "$HANG_WATCHDOG_PID" 2>/dev/null || true
    HANG_WATCHDOG_PID=""
    return 0
}
