#!/bin/bash
# Cap-respecting auto-submitter (one task per job, mode=run inline). Keeps my total (running+pending)
# per node <= cap by SELF-TRACKING submitted job IDs (squeue -w <node> misses pending jobs, the bug
# that over-submitted before). Usage: bash slurm/auto_submit_carbon.sh <config> <task1> <task2> ...
set -uo pipefail
cd /home/pengchx3/text-dna/OmegaGenome_Revise_202606/code_carbon
module load slurm 2>/dev/null||true
CONFIG="${1:-carbon-raw}"; shift || true
TASKS=("$@")
HFSSD=/tmp/galaxy_srv_disk00/pengchx3/hf_cache_shared
LAN_CAP=6; VOY_CAP=3
declare -A JOBS   # node -> "id id ..."
count_node(){     # total that WILL run on $1 = (all my RUNNING on node) + (my tracked PENDING for node)
  local node="$1" r p=0 j st
  r=$(squeue -u pengchx3 -w "$node" -h -t R 2>/dev/null | wc -l)
  for j in ${JOBS[$node]:-}; do
    st=$(squeue -j "$j" -h -o "%t" 2>/dev/null)
    [ "$st" = "PD" ] && p=$((p+1))
  done
  echo $((r + p))
}
i=0
while [ $i -lt ${#TASKS[@]} ]; do
  t=${TASKS[$i]}
  node=""
  [ "$(count_node laniakea)" -lt "$LAN_CAP" ] && node=laniakea || { [ "$(count_node voyager)" -lt "$VOY_CAP" ] && node=voyager; }
  if [ -n "$node" ]; then
    jid=$(sbatch --parsable --nodelist=$node --export=ALL,HF_OVERRIDE=$HFSSD \
          slurm/carbon_distill.sbatch "$CONFIG" --task-names "$t" --slurm-config.mode run 2>/dev/null)
    JOBS[$node]="${JOBS[$node]:-} $jid"
    echo "[$(date +%H:%M:%S)] $CONFIG/$t -> $node (job $jid)  lan=$(count_node laniakea)/$LAN_CAP voy=$(count_node voyager)/$VOY_CAP"
    i=$((i+1)); sleep 5
  else
    sleep 120  # both at cap; wait for a slot to free
  fi
done
echo "[$(date +%H:%M:%S)] all ${#TASKS[@]} tasks submitted"
