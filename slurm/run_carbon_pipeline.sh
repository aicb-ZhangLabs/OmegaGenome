#!/bin/bash
# Master carbon pipeline (starts the HP search immediately; runs unattended):
#   phase B: HP search (BASE grid, raw MSE, 80 combos x 18 = 1440 runs, resume-aware) via spec submitter
#   phase C: auto-extract best-on-val hyperparameters per task -> best_hyperparams.json
#   phase D: 3-seed on the best hyperparameters per task
# The spec submitter is cap-aware (laniakea<=6, voyager<=0 here; voyager reserved for ntv3) and KEEPS
# submitting as GPU slots free until the whole grid is launched. Drain-waits gate C and D on completion.
# Usage: nohup bash slurm/run_carbon_pipeline.sh &>/tmp/carbon_pipeline.log &
set -uo pipefail
cd /home/pengchx3/text-dna/OmegaGenome_Revise_202606/code_carbon
SB=/pkg/slurm/22.05.3/bin
PV=.venv_carbon_portable/bin/python
log(){ echo "[pipeline $(date '+%m-%d %H:%M:%S')] $*"; }
carbon_inflight(){ "$SB/squeue" -u pengchx3 -h -t R,PD -o "%j" 2>/dev/null | grep -c carbon-distill; }
wait_drain(){ while [ "$(carbon_inflight)" -gt 0 ]; do sleep 180; done; }

log "PHASE B: HP search (BASE grid, raw MSE, resume-aware) — starting NOW, fills slots continuously"
"$PV" slurm/gen_hp_specs.py --out hp_specs.txt
# Resume-safe grid: skip any config already completed, so a re-run / stale specs file can't waste
# compute. Grid only (NOT the phase-D 3-seed below) — though the skip is seed-aware so 3-seed
# (--seeds 0 1 2) would be safe regardless. Propagates to jobs via auto_submit_specs.sh's --export=ALL.
CARBON_SKIP_IF_DONE=1 bash slurm/auto_submit_specs.sh hp_specs.txt
log "PHASE B: all HP jobs submitted; draining"
wait_drain
log "PHASE B done"

log "PHASE C: extract best-on-val hyperparameters"
"$PV" slurm/extract_best_hyperparams.py --out best_hyperparams.json
log "PHASE C done -> best_hyperparams.json"

log "PHASE D: 3-seed on best hyperparameters"
"$PV" slurm/gen_3seed_best_specs.py --best best_hyperparams.json --seeds 0 1 2 --out best_3seed_specs.txt
bash slurm/auto_submit_specs.sh best_3seed_specs.txt
log "PHASE D: all submitted; draining"
wait_drain
log "PHASE D done. Pipeline complete. Final table: python slurm/aggregate_carbon_results.py"
