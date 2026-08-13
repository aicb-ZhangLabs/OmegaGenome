#!/bin/bash
# R1.3 relaunch (remaining 5 tasks). The first 10 (243606-243615) are already RUNNING; this submits
# the rest as laniakea/voyager slots free up, honoring fair-share caps (laniakea<=7, voyager<=4).
set -uo pipefail
module load slurm 2>/dev/null || true
SB=/pkg/slurm/22.05.3/bin
REPO=/home/pengchx3/text-dna/OmegaGenome_Revise_202606/code_carbon
RESULTS_CSV=/tmp/galaxy_srv_disk00/pengchx3/rebuttal_nt/run/results/r13_nt_embedding_results.csv

# task:node assignments for the remaining 5
declare -a JOBS=(
  "H3K4me3:laniakea"
  "enhancers_types:laniakea"
  "promoter_no_tata:laniakea"
  "splice_sites_donors:laniakea"
  "splice_sites_all:voyager"
)
cap_of() { [ "$1" = voyager ] && echo 4 || echo 7; }
run_on() { $SB/squeue -u "$(whoami)" -h -t RUNNING -w "$1" --format='%i' 2>/dev/null | grep -c .; }

for j in "${JOBS[@]}"; do
  task="${j%%:*}"; node="${j##*:}"; cap=$(cap_of "$node")
  while [ "$(run_on "$node")" -ge "$cap" ]; do
    echo "[$(date +%H:%M)] $node at cap $cap; waiting for $task"; sleep 90
  done
  echo -n "$node $task -> "
  $SB/sbatch --nodelist="$node" --export=ALL,TASK="$task",RESULTS_CSV="$RESULTS_CSV" "$REPO/slurm/run_nt_embedding.sh"
  sleep 5
done
echo "all remaining r13 tasks submitted"
