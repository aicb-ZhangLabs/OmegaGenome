#!/bin/bash
# General cap-respecting submitter. Reads a SPECS file (one job per line; each line = the CLI args to
# pass to carbon_distill.sbatch, e.g. "carbon-raw --task-names H3K9me3 --distillation-config.temperature
# 4.0 --slurm-config.mode run"). Submits each within per-node caps (laniakea<=6, voyager<=3), one at a
# time as slots free. count_node is AUTHORITATIVE (running-on-node + all my pending requesting node via
# scontrol) so concurrent submitters can't over-submit. Usage:
#   bash slurm/auto_submit_specs.sh <specs_file>
set -uo pipefail
cd /home/pengchx3/text-dna/OmegaGenome_Revise_202606/code_carbon
SB=/pkg/slurm/22.05.3/bin
SPECS="${1:?need specs file}"
HFSSD=/tmp/galaxy_srv_disk00/pengchx3/hf_cache_shared
LAN_CAP=6; VOY_CAP=3
count_node(){
  local node="$1" r pd=0 j rn
  r=$("$SB/squeue" -u pengchx3 -w "$node" -h -t R 2>/dev/null | wc -l)
  for j in $("$SB/squeue" -u pengchx3 -h -t PD -o "%i" 2>/dev/null); do
    rn=$("$SB/scontrol" show job "$j" 2>/dev/null | grep -oE 'ReqNodeList=[^ ]+' | head -1)
    [[ "$rn" == *"$node"* ]] && pd=$((pd+1))
  done
  echo $((r + pd))
}
mapfile -t LINES < <(grep -vE '^\s*(#|$)' "$SPECS")
echo "[specs $(date +%H:%M:%S)] $SPECS: ${#LINES[@]} jobs to submit"
i=0
while [ $i -lt ${#LINES[@]} ]; do
  spec="${LINES[$i]}"
  node=""
  [ "$(count_node laniakea)" -lt "$LAN_CAP" ] && node=laniakea || { [ "$(count_node voyager)" -lt "$VOY_CAP" ] && node=voyager; }
  if [ -n "$node" ]; then
    jid=$("$SB/sbatch" --parsable --nodelist=$node --export=ALL,HF_OVERRIDE=$HFSSD \
          slurm/carbon_distill.sbatch $spec 2>/dev/null)
    echo "[$(date +%H:%M:%S)] ($((i+1))/${#LINES[@]}) -> $node job=$jid :: $spec"
    i=$((i+1)); sleep 5
  else
    sleep 120
  fi
done
echo "[specs $(date +%H:%M:%S)] all ${#LINES[@]} submitted"
