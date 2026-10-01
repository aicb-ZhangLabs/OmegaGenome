#!/bin/bash
# General cap-respecting submitter. Reads a SPECS file (one job per line; each line = the CLI args to
# pass to carbon_distill.sbatch, e.g. "carbon-raw --task-names H3K9me3 --distillation-config.temperature
# 4.0 --slurm-config.mode run"). Submits each within per-node caps (laniakea<=6, voyager<=3), one at a
# time as slots free. count_node is AUTHORITATIVE (running-on-node + all my pending requesting node via
# scontrol) so concurrent submitters can't over-submit. Usage:
#   bash slurm/auto_submit_specs.sh <specs_file>
set -uo pipefail
cd ${OG_ROOT:-$PWD}
SB=/pkg/slurm/22.05.3/bin
SPECS="${1:?need specs file}"
HFSSD=${OG_SCRATCH:-$PWD/output}/hf_cache_shared
CAPS_FILE="slurm/submit_caps.env"   # live per-node caps — edit it and it's picked up each loop (no restart)
read_caps(){                        # robust: integer-only parse, defaults if missing, clamp to GPU counts
  LAN_CAP=$(grep -oE '^LAN_CAP=[0-9]+' "$CAPS_FILE" 2>/dev/null | grep -oE '[0-9]+$' | tail -1); LAN_CAP=${LAN_CAP:-7}
  VOY_CAP=$(grep -oE '^VOY_CAP=[0-9]+' "$CAPS_FILE" 2>/dev/null | grep -oE '[0-9]+$' | tail -1); VOY_CAP=${VOY_CAP:-3}
  GAL_CAP=$(grep -oE '^GAL_CAP=[0-9]+' "$CAPS_FILE" 2>/dev/null | grep -oE '[0-9]+$' | tail -1); GAL_CAP=${GAL_CAP:-0}
  [ "$LAN_CAP" -gt 8 ] 2>/dev/null && LAN_CAP=8   # laniakea has 8 GPUs
  [ "$VOY_CAP" -gt 4 ] 2>/dev/null && VOY_CAP=4   # voyager has 4 GPUs
  [ "$GAL_CAP" -gt 6 ] 2>/dev/null && GAL_CAP=6   # galaxy has 6 GPUs
}
pick_node(){                        # first node (priority order) with room under its cap; "" if all full
  [ "$(count_node laniakea)" -lt "$LAN_CAP" ] && { echo laniakea; return; }
  [ "$(count_node voyager)"  -lt "$VOY_CAP" ] && { echo voyager;  return; }
  [ "$(count_node galaxy)"   -lt "$GAL_CAP" ] && { echo galaxy;   return; }
}
count_node(){
  local node="$1" r pd=0 j rn
  r=$("$SB/squeue" -u $USER -w "$node" -h -t R 2>/dev/null | wc -l)
  for j in $("$SB/squeue" -u $USER -h -t PD -o "%i" 2>/dev/null); do
    rn=$("$SB/scontrol" show job "$j" 2>/dev/null | grep -oE 'ReqNodeList=[^ ]+' | head -1)
    [[ "$rn" == *"$node"* ]] && pd=$((pd+1))
  done
  echo $((r + pd))
}
mapfile -t LINES < <(grep -vE '^\s*(#|$)' "$SPECS")
echo "[specs $(date +%H:%M:%S)] $SPECS: ${#LINES[@]} jobs to submit"
i=0
while [ $i -lt ${#LINES[@]} ]; do
  read_caps                         # re-read caps each iteration -> live reconfiguration
  spec="${LINES[$i]}"
  node=$(pick_node)
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
