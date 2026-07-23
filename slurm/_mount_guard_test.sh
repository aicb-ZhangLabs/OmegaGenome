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

: > "$TD/scontrol.calls"
SLURM_JOB_ID=1 SLURM_RESTART_COUNT=9 MOUNT_GUARD_MAX_REQUEUE=5 \
  requeue_if_mount_failure 1 "$TD/missing.txt" >/dev/null 2>&1
check "requeue budget exhausted -> hard fail" $? 1
grep -q "requeue 1" "$TD/scontrol.calls" && bad "must not requeue past the cap" || ok "did not requeue past the cap"

echo
[ "$fails" -eq 0 ] && { echo "ALL GUARD TESTS PASSED"; exit 0; } || { echo "$fails GUARD TEST(S) FAILED"; exit 1; }
