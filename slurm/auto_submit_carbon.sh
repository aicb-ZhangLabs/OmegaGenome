#!/bin/bash
# Cap-respecting auto-submitter for the carbon-raw 18-task campaign (one task per job, mode=run inline).
# Keeps my (running+pending) jobs <= cap per node (laniakea 6, voyager 3), distributing across both,
# submitting as slots free. Usage: bash slurm/auto_submit_carbon.sh <config>  (default carbon-raw)
set -uo pipefail
cd /home/pengchx3/text-dna/OmegaGenome_Revise_202606/code_carbon
module load slurm 2>/dev/null||true
CONFIG="${1:-carbon-raw}"
HFSSD=/tmp/galaxy_srv_disk00/pengchx3/hf_cache_shared
LAN_CAP=6; VOY_CAP=3
# H3K27me3 is the validation job (already submitted) -> skip it here.
TASKS=(H3K36me3 H4K20me1 H2AFZ H3K27ac H3K4me1 H3K4me2 H3K4me3 H3K9ac H3K9me3 \
       promoter_all promoter_tata promoter_no_tata enhancers enhancers_types \
       splice_sites_all splice_sites_acceptors splice_sites_donors)
mine(){ squeue -u pengchx3 -w "$1" -h -t R,PD 2>/dev/null | wc -l; }
i=0
while [ $i -lt ${#TASKS[@]} ]; do
  t=${TASKS[$i]}
  lan=$(mine laniakea); voy=$(mine voyager)
  node=""; [ "$lan" -lt "$LAN_CAP" ] && node=laniakea || { [ "$voy" -lt "$VOY_CAP" ] && node=voyager; }
  if [ -n "$node" ]; then
    jid=$(sbatch --parsable --nodelist=$node --export=ALL,HF_OVERRIDE=$HFSSD \
          slurm/carbon_distill.sbatch "$CONFIG" --task-names "$t" --slurm-config.mode run 2>/dev/null)
    echo "[$(date +%H:%M:%S)] submitted $CONFIG/$t -> $node (job $jid)  [lan=$lan voy=$voy]"
    i=$((i+1)); sleep 8
  else
    sleep 90  # both nodes at cap; wait for a slot
  fi
done
echo "[$(date +%H:%M:%S)] all ${#TASKS[@]} $CONFIG tasks submitted"
