#!/bin/bash
# MULTI-NODE relaunch of the base-NT-2.5B embedding FROM-SCRATCH grid across galaxy+laniakea+voyager.
# THE FIX vs the original failure: pass each node its CORRECT SSD path explicitly (run_r13_matched.sh
# respects $SSD and skips its auto-detect, which otherwise grabbed the broken /srv/disk00/sshfs on the
# non-galaxy nodes). galaxy uses its LOCAL /srv/disk00/sshfs; laniakea/voyager use the (freshly
# remounted) /tmp/galaxy_srv_disk00 -- all the SAME underlying galaxy disk, so results land in one place.
# One job per task = all 4 arms sequentially into one per-task CSV. Job name encodes node for throttling.
export PATH=$PATH:/pkg/slurm/22.05.3/bin
REPO=/home/pengchx3/text-dna/OmegaGenome_Revise_202606
cd "$REPO" || exit 1
YAML="$REPO/rebuttal_infra/best_hp/best_hp_nt_fromscratch.yaml"
ARMS="onehot replace4_ntbase replaceK_ntbase latefuse_onehot_ntbase"
LOGD="$REPO/code_carbon/slurm/multinode_logs"; mkdir -p "$LOGD"
NODES="galaxy laniakea voyager"                    # priority: galaxy robust(local); then laniakea/voyager
declare -A CAP=( [galaxy]="${CAP_GALAXY:-4}" [laniakea]="${CAP_LANIAKEA:-4}" [voyager]="${CAP_VOYAGER:-2}" )
node_prefix(){ case "$1" in galaxy) echo /srv/disk00/sshfs/pengchx3 ;; *) echo /tmp/galaxy_srv_disk00/pengchx3 ;; esac; }
READ_PFX=/tmp/galaxy_srv_disk00/pengchx3            # this login node (laniakea) reads CSVs here
# small -> large so quick, complete rows land first
TASKS="promoter_tata promoter_no_tata promoter_all enhancers enhancers_types H3K4me3 H3K4me2 H3K4me1 H3K9ac H3K27ac H3K9me3 H3K27me3 H3K36me3 H4K20me1 H2AFZ splice_sites_donors splice_sites_acceptors splice_sites_all"

mycount(){ squeue -u pengchx3 -h -o '%j' 2>/dev/null | grep -c "^r13nbg_${1}_"; }   # my jobs targeting node $1
pick_node(){ local n c; for n in $NODES; do c=$(mycount "$n"); [ -z "$c" ] && c=99; [ "$c" -lt "${CAP[$n]:-0}" ] && { echo "$n"; return 0; }; done; echo ""; }
task_done(){ local csv="$READ_PFX/rebuttal_nt/run_fromscratch_ntbase/results/r13_ntbase_$1.csv"; [ -f "$csv" ] && [ "$(grep -c ',' "$csv" 2>/dev/null)" -ge 5 ]; }

echo "[$(date)] MULTINODE relaunch: NODES='$NODES' caps galaxy=${CAP[galaxy]} laniakea=${CAP[laniakea]} voyager=${CAP[voyager]}"
for task in $TASKS; do
  if task_done "$task"; then echo "[$(date)] SKIP $task (CSV complete)"; continue; fi
  node=""; while [ -z "$node" ]; do node=$(pick_node); [ -z "$node" ] && sleep 40; done
  pfx=$(node_prefix "$node"); cb="$pfx/rebuttal_nt/run_fromscratch_ntbase"; csv="$cb/results/r13_ntbase_$task.csv"
  jid=$(sbatch --parsable --nodelist="$node" --job-name="r13nbg_${node}_${task}" \
        --output="$LOGD/r13nbg_${node}_${task}_%j.out" \
        --export="ALL,SSD=$pfx,TASK=$task,ARMS=$ARMS,BEST_HP=$YAML,CACHE_BASE=$cb,RESULTS_CSV=$csv,PARAM_MATCHED=1" \
        "$REPO/code_carbon/slurm/run_r13_matched.sh" 2>&1)
  echo "[$(date)] submitted $task -> $node jid $jid (SSD=$pfx)"
  sleep 4
done
echo "[$(date)] all task submissions issued (multi-node)"
