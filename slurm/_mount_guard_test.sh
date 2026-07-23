#!/bin/bash
# CPU self-test for slurm/_mount_guard.sh. Pure shell, no SLURM, no GPU, no SSD needed.
# Each case is a MUTATION test: it checks the guard says NO when it should, not just YES when healthy.
#   bash slurm/_mount_guard_test.sh
set -uo pipefail
GUARD="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_mount_guard.sh"
TD="$(mktemp -d "${TMPDIR:-$HOME}/mountguard_test.XXXXXX")"
trap 'chmod -R u+rwX "$TD" 2>/dev/null; rm -rf "$TD"' EXIT
fails=0
ok()   { echo "  PASS  $1"; }
bad()  { echo "  FAIL  $1"; fails=$((fails + 1)); }
check(){ [ "$2" = "$3" ] && ok "$1 (rc=$2)" || bad "$1 (rc=$2, expected $3)"; }

echo "hello" > "$TD/good.txt"; mkdir -p "$TD/gooddir"; touch "$TD/gooddir/x"
echo "hello" > "$TD/noperm.txt"; chmod 000 "$TD/noperm.txt"

# Fake `scontrol`: records its args so we can prove requeue was/wasn't attempted.
mkdir -p "$TD/bin"
cat > "$TD/bin/scontrol" <<EOF
#!/bin/bash
echo "\$@" >> "$TD/scontrol.calls"
[ "\$1" = "show" ] && echo "JobId=1 StdOut=$TD/fake.log Foo=bar"
exit 0
EOF
chmod +x "$TD/bin/scontrol"
export PATH="$TD/bin:$PATH"

# shellcheck disable=SC1090
source "$GUARD"
export MOUNT_GUARD_TRIES=2 MOUNT_GUARD_SLEEP=0 MOUNT_GUARD_MAX_REQUEUE=1

echo "== _mount_readable"
_mount_readable "$TD/good.txt" "$TD/gooddir"; check "healthy file+dir accepted" $? 0
_mount_readable "$TD/good.txt" "$TD/missing.txt"; check "missing path rejected" $? 1
if [ "$(id -u)" -ne 0 ]; then
  _mount_readable "$TD/noperm.txt"; check "EPERM-on-open file rejected (the outage signature)" $? 1
else
  echo "  SKIP  EPERM case (running as root)"
fi

echo "== mount_preflight"
( unset SLURM_JOB_ID; mount_preflight "$TD/good.txt" >/dev/null ); check "healthy preflight passes" $? 0
( unset SLURM_JOB_ID; mount_preflight "$TD/missing.txt" >/dev/null 2>&1 ); check "dead-mount preflight fails (no SLURM ctx)" $? 1

echo "== requeue_if_mount_failure"
( unset SLURM_JOB_ID; requeue_if_mount_failure 0 "$TD/missing.txt" >/dev/null 2>&1 ); check "rc=0 short-circuits, never requeues" $? 0

: > "$TD/scontrol.calls"; : > "$TD/fake.log"; echo "RuntimeError: CUDA out of memory" > "$TD/fake.log"
SLURM_JOB_ID=1 SLURM_RESTART_COUNT=0 requeue_if_mount_failure 1 "$TD/good.txt" >/dev/null 2>&1
check "non-mount failure stays FAILED" $? 1
grep -q requeue "$TD/scontrol.calls" && bad "non-mount failure must NOT requeue" || ok "non-mount failure did not requeue"

: > "$TD/scontrol.calls"; echo "PermissionError: [Errno 1] Operation not permitted: /x/genome.fasta" > "$TD/fake.log"
SLURM_JOB_ID=1 SLURM_RESTART_COUNT=0 MOUNT_GUARD_MAX_REQUEUE=1 \
  timeout 5 bash -c 'source "$0"; requeue_if_mount_failure 1 "$1"' "$GUARD" "$TD/good.txt" >/dev/null 2>&1
grep -q "requeue 1" "$TD/scontrol.calls" && ok "recovered-mount outage signature in log DOES requeue" \
  || bad "recovered-mount outage signature in log DOES requeue"

: > "$TD/scontrol.calls"; : > "$TD/fake.log"
SLURM_JOB_ID=1 SLURM_RESTART_COUNT=0 \
  timeout 5 bash -c 'source "$0"; requeue_if_mount_failure 1 "$1"' "$GUARD" "$TD/missing.txt" >/dev/null 2>&1
grep -q "requeue 1" "$TD/scontrol.calls" && ok "currently-dead mount DOES requeue" \
  || bad "currently-dead mount DOES requeue"

# REGRESSION (2026-07-23): a job that renames itself makes `scontrol show job` report a StdOut path
# that does not exist (--output %x is re-expanded against the NEW name). The outage signature must
# still be found, via the fd-1 log, or a real mount blip gets misfiled as "NOT a mount failure".
cat > "$TD/bin/scontrol" <<EOF
#!/bin/bash
echo "\$@" >> "$TD/scontrol.calls"
[ "\$1" = "show" ] && echo "JobId=1 StdOut=$TD/renamed-does-not-exist.log Foo=bar"
exit 0
EOF
chmod +x "$TD/bin/scontrol"
: > "$TD/scontrol.calls"
echo "pyfaidx.FastaNotFoundError: Cannot read FASTA from file /x/genome.fasta" > "$TD/fake.log"
SLURM_JOB_ID=1 SLURM_RESTART_COUNT=0 MOUNT_GUARD_MAX_REQUEUE=1 MOUNT_GUARD_LOG="$TD/fake.log" \
  timeout 5 bash -c 'source "$0"; requeue_if_mount_failure 1 "$1"' "$GUARD" "$TD/good.txt" >/dev/null 2>&1
grep -q "requeue 1" "$TD/scontrol.calls" && ok "renamed job (bogus scontrol StdOut) still requeues via fd-1 log" \
  || bad "renamed job (bogus scontrol StdOut) still requeues via fd-1 log"
# ...and the fd-1 path must not turn every failure into a requeue: a real bug still stays FAILED.
: > "$TD/scontrol.calls"; echo "RuntimeError: CUDA out of memory" > "$TD/fake.log"
SLURM_JOB_ID=1 SLURM_RESTART_COUNT=0 MOUNT_GUARD_LOG="$TD/fake.log" \
  requeue_if_mount_failure 1 "$TD/good.txt" >/dev/null 2>&1
check "non-mount failure still stays FAILED with fd-1 log" $? 1
grep -q requeue "$TD/scontrol.calls" && bad "non-mount failure must NOT requeue (fd-1 path)" \
  || ok "non-mount failure did not requeue (fd-1 path)"

: > "$TD/scontrol.calls"
SLURM_JOB_ID=1 SLURM_RESTART_COUNT=9 MOUNT_GUARD_MAX_REQUEUE=5 \
  requeue_if_mount_failure 1 "$TD/missing.txt" >/dev/null 2>&1
check "requeue budget exhausted -> hard fail" $? 1
grep -q "requeue 1" "$TD/scontrol.calls" && bad "must not requeue past the cap" || ok "did not requeue past the cap"

echo "== resolve_ssd_mount (node-local /tmp must never be accepted)"
# Exercises the REAL function; only its candidate list is redirected at synthetic roots.
mkdir -p "$TD/fakessd/data"; touch "$TD/fakessd/data/genome.fasta"
# A REAL node-local /tmp dir that looks perfectly valid — the voyager-overflow trap. Must be refused.
NL="/tmp/mountguard_nodelocal_test_$$"; mkdir -p "$NL/data"; touch "$NL/data/genome.fasta"
trap 'chmod -R u+rwX "$TD" 2>/dev/null; rm -rf "$TD" "$NL"' EXIT

out="$(MOUNT_GUARD_SSD_CANDIDATES="$TD/fakessd" resolve_ssd_mount data/genome.fasta 2>/dev/null)"
check "non-/tmp root holding the data is accepted" $? 0
[ "$out" = "$TD/fakessd" ] && ok "returns the accepted root verbatim" || bad "returned '$out'"

( MOUNT_GUARD_SSD_CANDIDATES="$TD/fakessd" resolve_ssd_mount data/missing.fasta >/dev/null 2>&1 )
check "root missing the required data is refused" $? 1

out="$(MOUNT_GUARD_SSD_CANDIDATES="$NL" resolve_ssd_mount data/genome.fasta 2>/dev/null)"
check "node-local /tmp dir REFUSED even though the data is there" $? 1
[ -z "$out" ] && ok "refused candidate is not echoed" || bad "leaked node-local root '$out'"

# ...and it still falls THROUGH a bad /tmp candidate to a good one rather than dying.
out="$(MOUNT_GUARD_SSD_CANDIDATES="$NL $TD/fakessd" resolve_ssd_mount data/genome.fasta 2>/dev/null)"
[ "$out" = "$TD/fakessd" ] && ok "falls through the /tmp trap to the real root" || bad "got '$out'"

# Live host: must resolve to a real SSD root, or fail loudly — never silently pick node-local /tmp.
if out="$(resolve_ssd_mount ntv3_benchmark_data/human/genome.fasta 2>/dev/null)"; then
  case "$out" in /tmp/galaxy_srv_disk00/*|/srv/disk00/*) ok "live resolve_ssd_mount -> $out" ;;
                 *) bad "live resolve_ssd_mount returned unexpected root: $out" ;; esac
else ok "live resolve_ssd_mount failed loudly on this host (no SSD data visible here)"; fi

echo
[ "$fails" -eq 0 ] && { echo "ALL GUARD TESTS PASSED"; exit 0; } || { echo "$fails GUARD TEST(S) FAILED"; exit 1; }
