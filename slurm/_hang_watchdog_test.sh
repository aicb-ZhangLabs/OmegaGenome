#!/bin/bash
# CPU self-test for slurm/_hang_watchdog.sh. Pure shell: NO SLURM, NO GPU, NO SSD, NO network.
# `scontrol` is stubbed to record its arguments, the "trainer" is a `sleep` process and the SLURM log
# is a plain file the test appends to.
#
#   bash slurm/_hang_watchdog_test.sh            # run the suite against the real watchdog
#   bash slurm/_hang_watchdog_test.sh --prove    # ALSO prove the suite constrains: run the whole suite
#                                                # against 9 deliberately-broken variants and require
#                                                # every primary case to FAIL against at least one.
#
# WHY --prove EXISTS (finding "tests that pass but do not constrain"): a watchdog test suite is easy to
# write so that a no-op watchdog passes it. --prove asserts NO PRIMARY CASE SURVIVES EVERY MUTANT
# (a case that passes against all of them constrains nothing and is reported as a proof failure), and
# prints the full case x mutant kill matrix. The mutants each break exactly one real decision:
#   MUT-NEVER      _hw_stalled never fires            MUT-ALWAYS     _hw_stalled always fires
#   MUT-NODONE     blind to a finished run            MUT-NORECHECK  no race-safety re-look
#   MUT-ZEROSIZE   unreadable log reads as 0 bytes    MUT-STRICTSTOP unguarded kill/wait in the EXIT trap
#   MUT-NOCAP      ignores the requeue cap            MUT-SCONTROLLOG trusts scontrol StdOut (7f72c3d)
#   MUT-TRIGGERHAPPY always stalled AND no re-look -> requeues on the first poll no matter what
# See _mutant_body/_mutant_why below for the exact one-function override each one applies.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WD="${HANG_WATCHDOG_SRC:-$HERE/_hang_watchdog.sh}"

# ------------------------------------------------------------------------------- MUTATION PROOF MODE
# Builds broken variants of the REAL file (bash takes the last definition of a function, so appending
# an override is a genuine mutation of that one decision), runs this whole suite against each, prints
# the case x mutant kill matrix, and asserts the invariant: NO PRIMARY CASE SURVIVES EVERY MUTANT. One that
# passes against all of them constrains nothing and is a proof FAILURE, not a silent pass.
# HANG_WATCHDOG_PRIMARY_CASES : the case ids that MUST each be killed by at least one mutant. These are
# the numbered tests (7 is split into its two independent halves, 7a/7b); the lettered sub-assertions
# such as 1b/5c/12d are supporting detail and are reported in the matrix but not required individually.
# The list is a declared contract, checked BOTH ways: every id here must exist in the suite output, and
# every id here must fail against some mutant.
HANG_WATCHDOG_PRIMARY_CASES="1 2 3 4 5 6 7a 7b 8 9 10 11 12 13 14"
# HANG_WATCHDOG_MUTANTS : the broken variants, one per real decision the watchdog makes.
HANG_WATCHDOG_MUTANTS="MUT-NEVER MUT-ALWAYS MUT-TRIGGERHAPPY MUT-NODONE MUT-NORECHECK MUT-ZEROSIZE MUT-STRICTSTOP MUT-NOCAP MUT-SCONTROLLOG"

# _mutant_body <name> : the single-function override that breaks exactly one real decision. Appended to
# a copy of the REAL file, where bash's last-definition-wins makes it a genuine mutation.
_mutant_body() {
    case "$1" in
      MUT-NEVER)     echo '_hw_stalled() { return 1; }' ;;
      MUT-ALWAYS)    echo '_hw_stalled() { return 0; }' ;;
      MUT-TRIGGERHAPPY) echo '_hw_stalled() { return 0; }
_hw_still_stalled() { HW_RECHECK_VERDICT=fire; HW_RECHECK_SIZE=$3; return 0; }' ;;
      MUT-NODONE)    echo '_hw_completed() { return 1; }' ;;
      MUT-NORECHECK) echo '_hw_still_stalled() { HW_RECHECK_VERDICT=fire; HW_RECHECK_SIZE=$3; return 0; }' ;;
      MUT-ZEROSIZE)  echo '_hw_size() { stat -c %s -- "$1" 2>/dev/null || echo 0; }' ;;
      MUT-STRICTSTOP) echo 'hang_watchdog_stop() { [ -n "${HANG_WATCHDOG_PID:-}" ] || return 0; kill "$HANG_WATCHDOG_PID" 2>/dev/null; wait "$HANG_WATCHDOG_PID" 2>/dev/null; HANG_WATCHDOG_PID=; return 0; }' ;;
      MUT-NOCAP)     echo '_requeue_budget_left() { return 0; }' ;;
      MUT-SCONTROLLOG) echo '_job_stdout() { scontrol show job "${SLURM_JOB_ID:-1}" 2>/dev/null | tr " " "\n" | sed -n "s/^StdOut=//p" | head -1; }' ;;
    esac
}
# _mutant_why <name> : what that mutation breaks, for the printed matrix.
_mutant_why() {
    case "$1" in
      MUT-NEVER)      echo '_hw_stalled never says stalled -> a watchdog that never fires' ;;
      MUT-ALWAYS)     echo '_hw_stalled always says stalled -> the FALSE POSITIVE we fear most' ;;
      MUT-TRIGGERHAPPY) echo 'always stalled AND no race-safety re-look -> requeues on the very first poll' ;;
      MUT-NODONE)     echo '_hw_completed never says done -> blind to a finished run' ;;
      MUT-NORECHECK)  echo '_hw_still_stalled always fires -> no race-safety re-look before killing' ;;
      MUT-ZEROSIZE)   echo '_hw_size reports 0 for an unreadable log -> an NFS blip looks like a hang' ;;
      MUT-STRICTSTOP) echo 'unguarded kill/wait in hang_watchdog_stop -> aborts the sbatch set -e EXIT trap' ;;
      MUT-NOCAP)      echo '_requeue_budget_left always true -> the requeue cap is ignored' ;;
      MUT-SCONTROLLOG) echo '_job_stdout trusts scontrol StdOut -> the 7f72c3d self-renaming-job bug' ;;
    esac
}

_prove() {
    local real_out="$1" dir bad=0 name got all union missing c hit
    dir="$(mktemp -d "${TMPDIR:-$HOME}/hangwd_mutants.XXXXXX")"
    all="$(sed -n 's/^  \(PASS\|FAIL\)  \([0-9a-z]*\)\..*/\2/p' "$real_out" | tr '\n' ' ')"
    echo "cases in the suite: $all"
    # contract check #1: every declared primary case really exists (a renamed or deleted case cannot
    # quietly drop out of the contract).
    for c in $HANG_WATCHDOG_PRIMARY_CASES; do
        case " $all " in *" $c "*) ;; *) echo "PROOF FAILURE: declared primary case $c is not in the suite"; bad=1;; esac
    done
    echo
    # The mutant suites are independent (each makes its own scratch dir, scontrol stub and stamp dir),
    # so they run CONCURRENTLY -- 9 sequential suites would take ~40 min and the cases are mostly
    # sleeping, not computing.
    for name in $HANG_WATCHDOG_MUTANTS; do
        { cat "$HERE/_hang_watchdog.sh"; _mutant_body "$name"; } > "$dir/$name.sh"
        HANG_WATCHDOG_SRC="$dir/$name.sh" bash "$0" > "$dir/$name.out" 2>&1 &
    done
    wait
    union=""
    for name in $HANG_WATCHDOG_MUTANTS; do
        echo "### $name — $(_mutant_why "$name")"
        got="$(sed -n 's/^  FAIL  \([0-9a-z]*\)\..*/\1/p' "$dir/$name.out" | tr '\n' ' ')"
        echo "    cases that FAILED against it: ${got:-<none>}"
        hit=""
        for c in $HANG_WATCHDOG_PRIMARY_CASES; do case " $got " in *" $c "*) hit="$hit $c";; esac; done
        echo "    primary cases it kills:${hit:- <none>}"
        [ -n "$got" ] || { echo "    PROOF FAILURE: this broken watchdog passed the ENTIRE suite"; bad=1; }
        union="$union $got"
    done
    rm -rf "$dir"
    # contract check #2 (THE invariant): no primary case may survive every mutant. A case that passes
    # against all of them constrains nothing, and that is a FAILURE rather than a silent pass.
    missing=""
    for c in $HANG_WATCHDOG_PRIMARY_CASES; do
        case " $union " in *" $c "*) ;; *) missing="$missing $c";; esac
    done
    echo
    if [ -n "$missing" ]; then
        echo "PROOF FAILURE: primary case(s)$missing passed against EVERY broken watchdog — they constrain nothing"
        bad=1
    fi
    if [ "$bad" -eq 0 ]; then echo "MUTATION PROOF PASSED (every primary case is killed by at least one broken watchdog)"
    else echo "MUTATION PROOF FAILED"; fi
    return "$bad"
}
if [ "${1:-}" = "--prove" ]; then
    echo "== proving the suite constrains: the real watchdog first, then every broken variant"
    _real="$(mktemp "${TMPDIR:-$HOME}/hangwd_real.XXXXXX")"
    bash "$0" 2>&1 | tee "$_real"
    grep -q "ALL HANG-WATCHDOG TESTS PASSED" "$_real" || {
        echo "REAL WATCHDOG FAILED ITS OWN SUITE — fix that first"; rm -f "$_real"; exit 1; }
    echo
    _prove "$_real"; _rc=$?
    rm -f "$_real"
    exit "$_rc"
fi
TD="$(mktemp -d "${TMPDIR:-$HOME}/hangwd_test.XXXXXX")"
# Cleanup: kill any stragglers, then remove the scratch dir. Retried quietly because on NFS a killed
# process can leave a .nfsXXXX silly-rename file behind for a moment ("Device or resource busy").
trap 'jobs -p | xargs -r kill 2>/dev/null; sleep 1; rm -rf "$TD" 2>/dev/null; sleep 1; rm -rf "$TD" 2>/dev/null' EXIT

fails=0
ok()  { echo "  PASS  $1"; }
bad() { echo "  FAIL  $1"; fails=$((fails + 1)); }

# ---------------------------------------------------------------------------- fake SLURM environment
mkdir -p "$TD/bin"
cat > "$TD/bin/scontrol" <<EOF
#!/bin/bash
echo "\$@" >> "$TD/scontrol.calls"
[ "\$1" = "show" ] && echo "JobId=1 StdOut=$TD/scontrol-says.log Foo=bar"
exit 0
EOF
chmod +x "$TD/bin/scontrol"
export PATH="$TD/bin:$PATH"
export GUARD_REQUEUE_STAMP_DIR="$TD/stamps"      # never touch the real $HOME stamp dir
export SLURM_JOB_ID=1 SLURM_RESTART_COUNT=0 MOUNT_GUARD_MAX_REQUEUE=5

# Test-speed thresholds. Every case sets the two it cares about explicitly.
export HANG_WATCHDOG_POLL=1 HANG_WATCHDOG_RECHECK=1 HANG_WATCHDOG_KILL_GRACE=1

# shellcheck disable=SC1090,SC1091
source "$HERE/_mount_guard.sh"     # sourced first so a mutant copy of the watchdog need not find it
# shellcheck disable=SC1090,SC1091
source "$WD"

# ---------------------------------------------------------------------------------------- utilities
# reset_case <name> : fresh log, fresh scontrol call log, fresh requeue stamp (= a fresh dispatch).
reset_case() {
    CASE="$1"; LOG="$TD/job.log"
    : > "$LOG"; : > "$TD/scontrol.calls"; : > "$TD/mon.out"; rm -rf "$TD/stamps"
}
# n_requeues : how many times `scontrol requeue` was invoked in this case.
# (grep -c prints 0 AND exits 1 on no match, so the count must NOT be piped through `|| echo 0`.)
n_requeues() { local n; n="$(grep -c '^requeue' "$TD/scontrol.calls" 2>/dev/null)"; echo "${n:-0}"; }
# say_to_log <text>... : append a line to the monitored log (a "trainer" print).
say_to_log() { echo "$*" >> "$LOG"; }
# start_parent [<seconds>] : stand-in for the sbatch shell, holding ONE child (the "trainer") so the
# budget-exhausted path has something to SIGTERM. Sets PARENT and CHILD.
start_parent() {
    rm -f "$TD/child.pid"
    bash -c "sleep ${1:-120} & echo \$! > '$TD/child.pid'; wait" &
    PARENT=$!
    local i
    for i in $(seq 1 25); do [ -s "$TD/child.pid" ] && break; sleep 0.2; done
    CHILD="$(cat "$TD/child.pid" 2>/dev/null)"
}
# start_monitor : run the watchdog loop in the background against $LOG/$PARENT. Sets MON.
# Its stderr goes to mon.out (NOT the monitored log) so the assertions are about the trainer's output.
start_monitor() { hang_watchdog_monitor "$LOG" "$PARENT" >>"$TD/mon.out" 2>&1 & MON=$!; }
# stop_all : tear the case down (the fire path sleeps 120s waiting for SLURM, so always kill it).
stop_all() { kill "${MON:-0}" "${PARENT:-0}" "${CHILD:-0}" 2>/dev/null; wait "${MON:-0}" 2>/dev/null; MON=""; }
# gone_within <pid> <secs> : 0 iff that pid disappears inside the window.
gone_within() {
    local i; for i in $(seq 1 $(( $2 * 5 ))); do kill -0 "$1" 2>/dev/null || return 0; sleep 0.2; done
    return 1
}
# alive <pid>
alive() { kill -0 "$1" 2>/dev/null; }

echo "== watchdog under test: $WD"

# 1 --------------------------------------------------------------- genuine stall -> requeue ONCE
# A run that has been training normally goes completely silent for longer than STALL_TIMEOUT.
reset_case stall
say_to_log "step 4500/19932  lr=2.6e-05  loss=2.31  train_pearson=0.65"
HANG_WATCHDOG_STARTUP_GRACE=600 HANG_WATCHDOG_STALL_TIMEOUT=3
export HANG_WATCHDOG_STARTUP_GRACE HANG_WATCHDOG_STALL_TIMEOUT
start_parent 120; start_monitor
sleep 9
[ "$(n_requeues)" = "1" ] && ok "1. genuine stall requeues EXACTLY once (got $(n_requeues))" \
                          || bad "1. genuine stall requeues EXACTLY once (got $(n_requeues))"
grep -q "HANG DETECTED" "$TD/mon.out" && ok "1b. fires with a loud, unambiguous reason line" \
                                      || bad "1b. fires with a loud, unambiguous reason line"
grep -q "step 4500/19932" "$TD/mon.out" && ok "1c. reason names the last progress actually seen" \
                                        || bad "1c. reason names the last progress actually seen"
stop_all

# 2 ------------------------------------------------------- healthy progress -> NEVER fires (FP guard)
# The single most important case: a long, slow but LIVE run must not be touched.
reset_case progressing
say_to_log "step 100/19932"
export HANG_WATCHDOG_STARTUP_GRACE=600 HANG_WATCHDOG_STALL_TIMEOUT=3
start_parent 120; start_monitor
for i in $(seq 1 12); do sleep 1; say_to_log "step $((100 + i * 50))/19932"; done
sleep 2
[ "$(n_requeues)" = "0" ] && ok "2. steady progress NEVER requeues (0 requeues over 4x the timeout)" \
                          || bad "2. steady progress NEVER requeues (got $(n_requeues))"
alive "$MON" && ok "2b. watchdog is still watching (did not exit early)" \
             || bad "2b. watchdog is still watching (did not exit early)"
stop_all

# 3 ----------------------------------------------- inside STARTUP_GRACE, no step line yet -> no fire
# The KD arms print nothing while loading the 53 GB teacher cache. STALL_TIMEOUT is set ABSURDLY short
# here, so this case fails unless the STARTUP threshold is the one being applied.
reset_case startup_grace
say_to_log "[mount_guard] preflight OK on attempt 1/10 (6 paths readable on laniakea)"
say_to_log "open_memmap_logit_cache: loading logits.npy fully into RAM (one sequential read) ..."
export HANG_WATCHDOG_STARTUP_GRACE=60 HANG_WATCHDOG_STALL_TIMEOUT=1
start_parent 120; start_monitor
sleep 8
[ "$(n_requeues)" = "0" ] && ok "3. silent startup inside the grace window does NOT fire" \
                          || bad "3. silent startup inside the grace window does NOT fire (got $(n_requeues))"
stop_all

# 4 ------------------------------------------- step-0 hang (the real TMPDIR bug) -> fires after grace
# Reproduces 257235_4 / 257236_4: the job resumes, the DataLoader feeder thread raises
# PermissionError 16 times (once per worker x2), and then the log is silent forever at step 0.
reset_case step0_hang
say_to_log "RESUMED from /ssd/.../latest_state.pth at step 16000 (best_val=0.5555)"
for i in $(seq 1 16); do
    say_to_log "Traceback (most recent call last):"
    say_to_log "PermissionError: [Errno 1] Operation not permitted"
done
export HANG_WATCHDOG_STARTUP_GRACE=4 HANG_WATCHDOG_STALL_TIMEOUT=99999
start_parent 120; start_monitor
sleep 10
[ "$(n_requeues)" = "1" ] && ok "4. step-0 hang fires once after the startup grace elapses" \
                          || bad "4. step-0 hang fires once after the startup grace elapses (got $(n_requeues))"
grep -q "step-0 hang signature" "$TD/mon.out" && ok "4b. names it as the step-0 signature" \
                                              || bad "4b. names it as the step-0 signature"
stop_all

# 5 ----------------------------------------- requeue budget exhausted -> NO requeue, FAIL loudly
# A job that hangs every single time must stop thrashing the scheduler and become visibly FAILED.
reset_case budget
say_to_log "step 50/19932"
export HANG_WATCHDOG_STARTUP_GRACE=600 HANG_WATCHDOG_STALL_TIMEOUT=3
export SLURM_RESTART_COUNT=5 MOUNT_GUARD_MAX_REQUEUE=5   # this dispatch is the 5th of a max of 5
start_parent 120; start_monitor
sleep 9
[ "$(n_requeues)" = "0" ] && ok "5. at the requeue cap it does NOT requeue" \
                          || bad "5. at the requeue cap it does NOT requeue (got $(n_requeues))"
grep -q "FATAL" "$TD/mon.out" && ok "5b. at the cap it says FATAL loudly" \
                              || bad "5b. at the cap it says FATAL loudly"
gone_within "$CHILD" 6 && ok "5c. at the cap it kills the trainer so the job FAILS and frees the GPU" \
                       || bad "5c. at the cap it kills the trainer so the job FAILS and frees the GPU"
stop_all
export SLURM_RESTART_COUNT=0

# 6 ------------------------------------------------- trainer finishes -> clean exit, no orphan, no requeue
# The batch shell goes away; the watchdog must notice and stop, never requeue a job that is done.
reset_case trainer_ends
say_to_log "step 19900/19932"
export HANG_WATCHDOG_STARTUP_GRACE=600 HANG_WATCHDOG_STALL_TIMEOUT=3
start_parent 4; start_monitor
for i in 1 2 3; do sleep 1; say_to_log "step $((19900 + i))/19932"; done   # alive until the shell exits
gone_within "$MON" 12 && ok "6. watchdog exits when the batch shell exits (no orphan)" \
                      || bad "6. watchdog exits when the batch shell exits (no orphan)"
[ "$(n_requeues)" = "0" ] && ok "6b. a finished job is never requeued" \
                          || bad "6b. a finished job is never requeued (got $(n_requeues))"
grep -q "watchdog stopping cleanly" "$TD/mon.out" && ok "6c. exits cleanly, with a reason" \
                                                  || bad "6c. exits cleanly, with a reason"
stop_all

# 7 ---------------------------------------- COMPLETED at the threshold boundary -> never requeued
# 7a: the run finished (final result line present) and the log is legitimately silent afterwards.
reset_case completed
say_to_log "step 19900/19932"
say_to_log ""
say_to_log "best-val 0.5555 -> TEST mean Pearson 0.5412 | tracks 34"
export HANG_WATCHDOG_STARTUP_GRACE=600 HANG_WATCHDOG_STALL_TIMEOUT=2
start_parent 120; start_monitor
sleep 8
[ "$(n_requeues)" = "0" ] && ok "7a. a COMPLETED run past the stall timeout is not requeued" \
                          || bad "7a. a COMPLETED run past the stall timeout is not requeued (got $(n_requeues))"
stop_all
# 7b: RACE — the threshold is crossed first and the completion line lands during the re-check window.
# This is the end-of-run test evaluation (10531 windows, zero output) finishing a moment too late.
reset_case completed_race
say_to_log "step 19900/19932"
export HANG_WATCHDOG_STARTUP_GRACE=600 HANG_WATCHDOG_STALL_TIMEOUT=3 HANG_WATCHDOG_RECHECK=6
start_parent 120; start_monitor
sleep 5      # the stall threshold has now elapsed; the watchdog is inside its re-check pause
say_to_log "best-val 0.5555 -> TEST mean Pearson 0.5412 | tracks 34"
sleep 6
[ "$(n_requeues)" = "0" ] && ok "7b. finishing DURING the re-check window is not requeued (race safe)" \
                          || bad "7b. finishing DURING the re-check window is not requeued (got $(n_requeues))"
grep -q "COMPLETED during the re-check" "$TD/mon.out" && ok "7c. says why it stood down" \
                                                      || bad "7c. says why it stood down"
stop_all
export HANG_WATCHDOG_RECHECK=1

# 8 ------------------------------------------------- late output during the re-check -> stands down
# Same race, but the run is merely SLOW rather than finished: one new byte must call off the kill.
reset_case late_output
say_to_log "step 900/19932"
export HANG_WATCHDOG_STARTUP_GRACE=600 HANG_WATCHDOG_STALL_TIMEOUT=3 HANG_WATCHDOG_RECHECK=6
start_parent 120; start_monitor
sleep 5
say_to_log "  [val] step 1000  mean_pearson=0.4569"
sleep 6
[ "$(n_requeues)" = "0" ] && ok "8. output arriving during the re-check calls off the requeue" \
                          || bad "8. output arriving during the re-check calls off the requeue (got $(n_requeues))"
stop_all
export HANG_WATCHDOG_RECHECK=1

# 9 ------------------------------------ phase transition: startup grace -> stall timeout takes over
# Proves the two thresholds are wired to the right phases (not one threshold used for both).
reset_case phase_switch
export HANG_WATCHDOG_STARTUP_GRACE=600 HANG_WATCHDOG_STALL_TIMEOUT=3
say_to_log "windows: train 63707 / val 997 / test 10531"
start_parent 120; start_monitor
sleep 5
[ "$(n_requeues)" = "0" ] || bad "9. startup phase must use the LONG grace (fired too early)"
say_to_log "step 50/19932"          # first step line: the short stall timeout now governs
sleep 8
[ "$(n_requeues)" = "1" ] && ok "9. after the first step line the (shorter) stall timeout governs" \
                          || bad "9. after the first step line the (shorter) stall timeout governs (got $(n_requeues))"
grep -q "leaving STARTUP grace" "$TD/mon.out" && ok "9b. logs the phase transition" \
                                              || bad "9b. logs the phase transition"
stop_all

# 10 ------------------------------------- interlock: the two guards never requeue the same dispatch
# After the watchdog has requeued, the mount guard's exit classifier must stand down (the trainer will
# exit non-zero with an 'Operation not permitted' signature in the log, which otherwise looks exactly
# like a mount outage to requeue_if_mount_failure).
reset_case interlock
say_to_log "step 50/19932"
export HANG_WATCHDOG_STARTUP_GRACE=600 HANG_WATCHDOG_STALL_TIMEOUT=3
start_parent 120; start_monitor
sleep 9
before="$(n_requeues)"
: > "$TD/scontrol.calls"
echo "PermissionError: [Errno 1] Operation not permitted" >> "$LOG"
MOUNT_GUARD_LOG="$LOG" requeue_if_mount_failure 1 "$LOG" >>"$TD/mon.out" 2>&1
[ "$before" = "1" ] && [ "$(n_requeues)" = "0" ] \
  && ok "10. mount guard does NOT double-requeue a dispatch the watchdog already requeued" \
  || bad "10. mount guard does NOT double-requeue (watchdog=$before, mount guard=$(n_requeues))"
stop_all

# 11 -------------------------------- the watchdog's own log lines must not count as trainer progress
# In a real job the watchdog's stderr is merged into the very file it monitors. If it ever treated its
# own output as liveness it could keep itself alive forever.
reset_case self_write
say_to_log "step 50/19932"
export HANG_WATCHDOG_STARTUP_GRACE=600 HANG_WATCHDOG_STALL_TIMEOUT=3
start_parent 120
hang_watchdog_monitor "$LOG" "$PARENT" >>"$LOG" 2>&1 & MON=$!    # stderr INTO the monitored log
sleep 9
[ "$(n_requeues)" = "1" ] && ok "11. still fires when its own output shares the monitored log" \
                          || bad "11. still fires when its own output shares the monitored log (got $(n_requeues))"
stop_all

# 12 ------------------------------- log resolution survives a self-renaming job (regression 7f72c3d)
# `scontrol show job` re-expands --output against the job's CURRENT name, so the no-KD and per-track
# ladders (which `scontrol update ... Name=`) get back a path that does not exist. hang_watchdog_start
# must use the fd-1 target instead, and must stay DISARMED rather than guess when nothing resolves.
reset_case log_resolve
: > "$TD/real-fd1.log"
MOUNT_GUARD_LOG="$TD/real-fd1.log" hang_watchdog_start >>"$TD/mon.out" 2>&1
[ -n "${HANG_WATCHDOG_PID:-}" ] && ok "12. arms on the fd-1 log even when scontrol reports a bogus StdOut" \
                                || bad "12. arms on the fd-1 log even when scontrol reports a bogus StdOut"
grep -q "armed .* on $TD/real-fd1.log" "$TD/mon.out" && ok "12b. armed on the fd-1 path, not scontrol's" \
                                                     || bad "12b. armed on the fd-1 path, not scontrol's"
hang_watchdog_stop
[ -z "${HANG_WATCHDOG_PID:-}" ] && ok "12c. hang_watchdog_stop disarms (EXIT-trap safe, idempotent)" \
                               || bad "12c. hang_watchdog_stop disarms (EXIT-trap safe, idempotent)"
hang_watchdog_stop && ok "12d. hang_watchdog_stop is idempotent" || bad "12d. hang_watchdog_stop is idempotent"
: > "$TD/mon.out"
MOUNT_GUARD_LOG="$TD/does-not-exist.log" hang_watchdog_start >>"$TD/mon.out" 2>&1
grep -q "DISARMED" "$TD/mon.out" && ok "12e. no resolvable log -> DISARMED (never guesses a hang)" \
                                 || bad "12e. no resolvable log -> DISARMED (never guesses a hang)"

# 13 ------------------- REGRESSION: hang_watchdog_stop must be safe inside a `set -e` EXIT trap
# The sbatches run `set -euo pipefail`, which is STILL IN FORCE inside the EXIT trap. `wait` on the
# watchdog we just killed returns 143, so an unguarded `wait` aborts the trap: the job exits 143
# instead of the trainer's real status (SLURM would record a SUCCESSFUL run as FAILED) and the rest of
# the trap -- `rm -rf "$TMPDIR"` -- never runs. Exercised through a miniature of the REAL sbatch
# structure so the thing tested is the thing shipped.
reset_case exit_trap
cat > "$TD/fake_sbatch.sh" <<EOF
#!/bin/bash
set -euo pipefail
source "$HERE/_mount_guard.sh"
source "$WD"
export TMPDIR="\$1/tmpdir"; mkdir -p "\$TMPDIR"
trap 'hang_watchdog_stop; rm -rf "\$TMPDIR"' EXIT      # exactly the sbatches' trap
hang_watchdog_start
echo "\${HANG_WATCHDOG_PID:-NONE}" > "\$1/wd.pid"
set +e; bash -c "echo 'step 50/19932'; exit \$2"; rc=\$?; set -e
exit "\$rc"
EOF
chmod +x "$TD/fake_sbatch.sh"
mkdir -p "$TD/sb"; rm -rf "${TD:?}/sb"/*
HANG_WATCHDOG_POLL=5 bash "$TD/fake_sbatch.sh" "$TD/sb" 0 > "$TD/sb/job.log" 2>&1
rc=$?
[ "$rc" = "0" ] && ok "13. a successful job still exits 0 (the EXIT trap does not leak 143)" \
                || bad "13. a successful job still exits 0 (the EXIT trap does not leak 143) - got $rc"
[ ! -d "$TD/sb/tmpdir" ] && ok "13b. the rest of the EXIT trap still runs (TMPDIR cleaned up)" \
                         || bad "13b. the rest of the EXIT trap still runs (TMPDIR cleaned up)"
WDPID="$(cat "$TD/sb/wd.pid" 2>/dev/null)"
{ [ -n "$WDPID" ] && ! alive "$WDPID"; } && ok "13c. no orphan watchdog after the job ends" \
                                         || bad "13c. no orphan watchdog after the job ends"
rm -rf "${TD:?}/sb"/*
HANG_WATCHDOG_POLL=5 bash "$TD/fake_sbatch.sh" "$TD/sb" 7 > "$TD/sb/job.log" 2>&1
rc=$?
[ "$rc" = "7" ] && ok "13d. a FAILING trainer keeps its own exit code through the trap" \
                || bad "13d. a FAILING trainer keeps its own exit code through the trap - got $rc"

# 14 ------------------- an UNREADABLE log is missing information, not evidence of a hang
# $HOME (where the SLURM --output file lives) is NFS and does blip. If `stat` on the log fails, the
# watchdog must skip the poll rather than see "0 bytes, unchanged" and conclude the job is hung.
reset_case unreadable
say_to_log "step 300/19932"
export HANG_WATCHDOG_STARTUP_GRACE=600 HANG_WATCHDOG_STALL_TIMEOUT=2
start_parent 120; start_monitor
sleep 2
rm -f "$LOG"                       # the log becomes unstat-able
sleep 8
[ "$(n_requeues)" = "0" ] && ok "14. an unreadable log NEVER triggers a requeue (no info != hang)" \
                          || bad "14. an unreadable log NEVER triggers a requeue (got $(n_requeues))"
grep -q "cannot stat" "$TD/mon.out" && ok "14b. says once that it is skipping polls" \
                                    || bad "14b. says once that it is skipping polls"
[ "$(grep -c 'cannot stat' "$TD/mon.out")" -le 2 ] \
  && ok "14c. does not log every poll (it writes into the file it monitors)" \
  || bad "14c. does not log every poll (logged $(grep -c 'cannot stat' "$TD/mon.out") times)"
stop_all

echo
if [ "$fails" -eq 0 ]; then echo "ALL HANG-WATCHDOG TESTS PASSED"; else echo "$fails HANG-WATCHDOG TEST(S) FAILED"; fi
[ "$fails" -eq 0 ] || exit 1
exit 0
