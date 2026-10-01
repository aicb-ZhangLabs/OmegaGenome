#!/bin/bash
# MATCHED-PAIR full launcher: submit one task's 5-arm best-HP comparison as SEPARATE SLURM jobs
# (so they parallelize across free slots / queue), into ONE per-task results CSV.
#
# Arms & dependency chaining (so the per-bp embedding cache is computed ONCE, never doubly):
#   onehot                      -> independent (no embedding cache)
#   replaceK_nt                 -> CACHE PRIMER for NT embeddings (computes cache_embedding/<task> mid)
#   latefuse_onehot_nt          -> afterok:replaceK_nt   (HITs the NT cache)
#   replaceK_dnabert2           -> CACHE PRIMER for DNABERT-2 embeddings
#   latefuse_onehot_dnabert2    -> afterok:replaceK_dnabert2 (HITs the DNABERT-2 cache)
# Each arm runs ARMS=<single arm> through run_matched_capacity.sh at the task's best-HP + strong teacher.
#
# Usage:  ./launch_matched_capacity.sh <task_name> <node>     e.g.  ./launch_matched_capacity.sh splice_sites_all voyager
# Honors fair-share caps (laniakea<=8, voyager<=4, galaxy<=5): waits for a free slot per submit.
set -uo pipefail
module load slurm 2>/dev/null || true
SB=/pkg/slurm/22.05.3/bin
REPO=${OG_ROOT:-$PWD}
SCRIPT="$REPO/slurm/run_matched_capacity.sh"

TASK="${1:?usage: launch_matched_capacity.sh <task> <node> [model_size_override]}"
NODE="${2:?usage: launch_matched_capacity.sh <task> <node> [model_size_override]}"
# Optional 3rd arg / MODEL_SIZE env: override the best-HP student size (e.g. 'original' to force the
# 0.12M deployable student for splice_sites_all, which best_hp lists as medium). When overridden, the
# CSV is suffixed so the original-size run never overwrites the medium best-HP run.
SIZE_OVERRIDE="${3:-${MODEL_SIZE:-}}"
SSD=${OG_SCRATCH:-$PWD/output}; [ -d ${OG_SCRATCH:-$PWD/output} ] && SSD=${OG_SCRATCH:-$PWD/output}

# Effective model_size: override if given, else the task's best-HP value.
BESTHP_SIZE=$(/usr/bin/python3 -c "import yaml;print(yaml.safe_load(open('${OG_WORKSPACE:-$PWD/..}/analysis/best_hp/best_hp_nt.yaml'))['$TASK']['metadata']['model_size'])" 2>/dev/null || echo original)
if [ -n "$SIZE_OVERRIDE" ]; then
  MODEL_SIZE="$SIZE_OVERRIDE"
  RESULTS_CSV="$SSD/nt_runs/run/results/matched_capacity_${TASK}_${MODEL_SIZE}.csv"
else
  MODEL_SIZE="$BESTHP_SIZE"
  RESULTS_CSV="$SSD/nt_runs/run/results/matched_capacity_${TASK}.csv"
fi
echo "[$(date)] TASK=$TASK best-HP_size=$BESTHP_SIZE  effective_model_size=$MODEL_SIZE  CSV=$RESULTS_CSV"

cap_of() { case "$1" in voyager) echo 4;; galaxy) echo 5;; *) echo 8;; esac; }
run_on() { $SB/squeue -u "$(whoami)" -h -t RUNNING,PENDING -w "$1" --format='%i' 2>/dev/null | grep -c .; }
wait_slot() { local cap; cap=$(cap_of "$NODE")
  while [ "$(run_on "$NODE")" -ge "$cap" ]; do echo "[$(date +%H:%M)] $NODE at cap $cap; waiting..."; sleep 90; done; }

submit() { # $1=arm name(for job suffix)  $2..=extra sbatch args (e.g. dependency)
  local arm="$1"; shift
  wait_slot
  local jid
  jid=$($SB/sbatch --parsable --nodelist="$NODE" --job-name="mcap-${TASK:0:6}-${arm}" \
        --export=ALL,TASK="$TASK",RESULTS_CSV="$RESULTS_CSV",ARMS="$arm",MODEL_SIZE="$MODEL_SIZE" "$@" "$SCRIPT")
  echo "$jid"
}

echo "[$(date)] launching matched arms for TASK=$TASK on $NODE -> $RESULTS_CSV"
J_OH=$(submit onehot);                                   echo "  onehot                   -> $J_OH"
J_RKN=$(submit replaceK_nt);                             echo "  replaceK_nt (NT primer)  -> $J_RKN"
J_RKD=$(submit replaceK_dnabert2);                       echo "  replaceK_dnabert2 (primer)-> $J_RKD"
if [ "$MODEL_SIZE" = "original" ]; then
  submit latefuse_onehot_nt --dependency=afterok:$J_RKN  | sed 's/^/  latefuse_onehot_nt       -> /'
  submit latefuse_onehot_dnabert2 --dependency=afterok:$J_RKD | sed 's/^/  latefuse_onehot_dnabert2  -> /'
else
  echo "  [skip latefuse_onehot arms: model_size=$MODEL_SIZE (latefuse needs 'original')]"
fi
echo "[$(date)] all matched arms submitted for $TASK"
