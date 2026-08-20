#!/bin/bash
# Cap-respecting auto-submitter (one task per job, mode=run inline). Keeps my total (running+pending)
# per node <= cap by SELF-TRACKING submitted job IDs ("$SB/squeue" -w <node> misses pending jobs, the bug
# that over-submitted before). Usage: bash slurm/auto_submit_carbon.sh <config> <task1> <task2> ...
set -uo pipefail
cd /home/pengchx3/text-dna/OmegaGenome_Revise_202606/code_carbon
SB=/pkg/slurm/22.05.3/bin
CONFIG="${1:-carbon-raw}"; shift || true
TASKS=("$@")
HFSSD=/tmp/galaxy_srv_disk00/pengchx3/hf_cache_shared
CAPS_FILE="slurm/submit_caps.env"   # live per-node caps (single source of truth; re-read each loop)
read_caps(){
  LAN_CAP=$(grep -oE '^LAN_CAP=[0-9]+' "$CAPS_FILE" 2>/dev/null | grep -oE '[0-9]+$' | tail -1); LAN_CAP=${LAN_CAP:-7}
  VOY_CAP=$(grep -oE '^VOY_CAP=[0-9]+' "$CAPS_FILE" 2>/dev/null | grep -oE '[0-9]+$' | tail -1); VOY_CAP=${VOY_CAP:-3}
  GAL_CAP=$(grep -oE '^GAL_CAP=[0-9]+' "$CAPS_FILE" 2>/dev/null | grep -oE '[0-9]+$' | tail -1); GAL_CAP=${GAL_CAP:-0}
  [ "$LAN_CAP" -gt 8 ] 2>/dev/null && LAN_CAP=8
  [ "$VOY_CAP" -gt 4 ] 2>/dev/null && VOY_CAP=4
  [ "$GAL_CAP" -gt 6 ] 2>/dev/null && GAL_CAP=6   # galaxy has 6 GPUs
}
pick_node(){
  [ "$(count_node laniakea)" -lt "$LAN_CAP" ] && { echo laniakea; return; }
  [ "$(count_node voyager)"  -lt "$VOY_CAP" ] && { echo voyager;  return; }
  [ "$(count_node galaxy)"   -lt "$GAL_CAP" ] && { echo galaxy;   return; }
}
# Optional multi-seed: SEED=N env -> appends --random-state N (training seed; cache self-validates).
SEED_ARGS=(); [ -n "${SEED:-}" ] && SEED_ARGS=(--random-state "$SEED")
declare -A JOBS   # node -> "id id ..."
count_node(){     # AUTHORITATIVE: all my jobs (R+PD) targeting $1, regardless of which submitter made
                  # them. RUNNING via squeue -w (accurate); PENDING via scontrol ReqNodeList (squeue -w
                  # misses pending). So concurrent/seed submitters all see the true count -> no over-submit.
  local node="$1" r pd=0 j rn
  r=$("$SB/squeue" -u pengchx3 -w "$node" -h -t R 2>/dev/null | wc -l)
  for j in $("$SB/squeue" -u pengchx3 -h -t PD -o "%i" 2>/dev/null); do
    rn=$("$SB/scontrol" show job "$j" 2>/dev/null | grep -oE 'ReqNodeList=[^ ]+' | head -1)
    [[ "$rn" == *"$node"* ]] && pd=$((pd+1))
  done
  echo $((r + pd))
}
i=0
while [ $i -lt ${#TASKS[@]} ]; do
  read_caps                         # live caps from submit_caps.env each iteration
  t=${TASKS[$i]}
  node=$(pick_node)
  if [ -n "$node" ]; then
    jid=$("$SB/sbatch" --parsable --nodelist=$node --export=ALL,HF_OVERRIDE=$HFSSD \
          slurm/carbon_distill.sbatch "$CONFIG" --task-names "$t" "${SEED_ARGS[@]}" --slurm-config.mode run 2>/dev/null)
    JOBS[$node]="${JOBS[$node]:-} $jid"
    echo "[$(date +%H:%M:%S)] $CONFIG/$t seed=${SEED:-0} -> $node (job $jid)  lan=$(count_node laniakea)/$LAN_CAP voy=$(count_node voyager)/$VOY_CAP"
    i=$((i+1)); sleep 5
  else
    sleep 120  # both at cap; wait for a slot to free
  fi
done
echo "[$(date +%H:%M:%S)] all ${#TASKS[@]} tasks submitted"
