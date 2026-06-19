#!/bin/bash
# Master carbon pipeline (runs unattended after the current 3-seed campaign):
#   phase A: wait for the in-flight multi-seed (raw seeds 1,2) to finish
#   phase B: HP search  (focused grid, 324 raw runs, resume-aware) via spec submitter
#   phase C: auto-extract best-on-val hyperparameters per task -> best_hyperparams.json
#   phase D: 3-seed on the best hyperparameters per task
# Each phase's submitter is cap-aware (laniakea<=6, voyager<=3). Drain-waits between phases keep one
# campaign at a time. Fully logged. Usage: nohup bash slurm/run_carbon_pipeline.sh &>/tmp/carbon_pipeline.log &
set -uo pipefail
cd /home/pengchx3/text-dna/OmegaGenome_Revise_202606/code_carbon
SB=/pkg/slurm/22.05.3/bin
PV=.venv_carbon_portable/bin/python
log(){ echo "[pipeline $(date '+%m-%d %H:%M:%S')] $*"; }
carbon_inflight(){ "$SB/squeue" -u pengchx3 -h -t R,PD -o "%j" 2>/dev/null | grep -c carbon-distill; }
wait_drain(){ while [ "$(carbon_inflight)" -gt 0 ]; do sleep 180; done; }

log "PHASE A: waiting for in-flight multi-seed (raw seeds 1,2) submitter + jobs to finish"
while pgrep -f "run_multiseed_raw.sh|auto_submit_carbon.sh" >/dev/null 2>&1; do sleep 120; done
wait_drain
log "PHASE A done (no carbon jobs in queue)"

log "PHASE B: HP search (focused grid, resume-aware)"
"$PV" slurm/gen_hp_specs.py --out hp_specs.txt
bash slurm/auto_submit_specs.sh hp_specs.txt
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
