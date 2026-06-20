#!/bin/bash
# Self-healing HP-grid reconciler. The primary submitter (auto_submit_specs.sh hp_specs.txt) is ONE-PASS:
# any combo whose job FAILED (e.g. a transient sshfs blip on the shared galaxy SSD) is consumed from the
# list and never retried, leaving a permanent hole in the grid. This watcher closes those holes:
#   0) wait for the in-flight one-pass submitter (PID $1) to exit AND the queue to fully drain
#      (so gen_hp_specs — which keys off final_summary.json — never re-submits an in-flight combo),
#   1) loop: regenerate ONLY the missing combos (resume-aware) and submit them, drain, repeat,
#      until 0 combos are missing (or MAX_CYCLES safety cap is hit).
# Re-runs are now robust because save_checkpoint retries transient FS errors (_retry_io). Bounded by
# MAX_CYCLES so a genuinely-broken combo can't spin forever. Usage: bash slurm/reconcile_hp.sh [PID]
set -uo pipefail
cd /home/pengchx3/text-dna/OmegaGenome_Revise_202606/code_carbon
SB=/pkg/slurm/22.05.3/bin
PY=.venv_carbon_portable/bin/python
EXISTING="${1:-}"
MAX_CYCLES=6
SPECS=hp_specs_reconcile.txt

qcount(){ "$SB/squeue" -u pengchx3 -h -o "%j" 2>/dev/null | grep -c carbon-distill; }
drain(){ while [ "$(qcount)" -gt 0 ]; do sleep 300; done; }

echo "[reconcile $(date +%F_%H:%M:%S)] start; waiting for primary submitter PID=${EXISTING:-none} + queue drain"
[ -n "$EXISTING" ] && while kill -0 "$EXISTING" 2>/dev/null; do sleep 300; done
drain
echo "[reconcile $(date +%F_%H:%M:%S)] primary pass drained; entering self-heal loop"

cycle=0
while [ "$cycle" -lt "$MAX_CYCLES" ]; do
  cycle=$((cycle+1))
  PYTHONPATH=. "$PY" slurm/gen_hp_specs.py --out "$SPECS"
  N=$(grep -vcE '^\s*(#|$)' "$SPECS" 2>/dev/null || echo 0)
  echo "[reconcile $(date +%F_%H:%M:%S)] cycle $cycle/$MAX_CYCLES: $N combos still missing"
  [ "$N" -eq 0 ] && { echo "[reconcile] HP GRID COMPLETE (0 holes)"; touch HP_GRID_COMPLETE; exit 0; }
  bash slurm/auto_submit_specs.sh "$SPECS"
  drain
done
echo "[reconcile $(date +%F_%H:%M:%S)] STOPPED after $MAX_CYCLES cycles with $N combos still missing — inspect these (persistent failures, not transient)."
PYTHONPATH=. "$PY" slurm/gen_hp_specs.py --out "$SPECS" >/dev/null 2>&1
echo "remaining specs in $SPECS:"; cat "$SPECS"
