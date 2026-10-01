#!/bin/bash
# GALAXY-ONLY relaunch of the base-NT-2.5B embedding FROM-SCRATCH grid.
# Robust by design: every job runs ON galaxy, where the 28TB SSD is LOCAL at /srv/disk00/sshfs/<user>
# (no sshfs to a remote host) -- so a galaxy reboot that breaks the laniakea/voyager sshfs mounts
# cannot break these jobs. One job per task runs all 4 arms sequentially into one per-task CSV.
# Throttled to <=CAP concurrent on galaxy (leave a GPU free). Submits from wherever this is launched.
export PATH=$PATH:/pkg/slurm/22.05.3/bin
REPO=${OG_WORKSPACE:-$PWD/..}
cd "$REPO" || exit 1
SSD=${OG_SCRATCH:-$PWD/output}                                   # galaxy-LOCAL SSD path
CACHE_BASE="$SSD/nt_runs/run_fromscratch_ntbase"
YAML="$REPO/analysis/best_hp/best_hp_nt_fromscratch.yaml"
ARMS="onehot replace4_ntbase replaceK_ntbase latefuse_onehot_ntbase"
CAP="${CAP:-5}"                                                  # galaxy has 6 GPUs; leave 1 free
LOGD="$REPO/code_carbon/slurm/galaxy_only_logs"; mkdir -p "$LOGD"
# small -> large so quick, complete table rows land first
TASKS="promoter_tata promoter_no_tata promoter_all enhancers enhancers_types H3K4me3 H3K4me2 H3K4me1 H3K9ac H3K27ac H3K9me3 H3K27me3 H3K36me3 H4K20me1 H2AFZ splice_sites_donors splice_sites_acceptors splice_sites_all"

running_count(){ squeue -u $USER -h -o '%j' 2>/dev/null | grep -c '^ntbaseg_'; }

echo "[$(date)] GALAXY-ONLY relaunch start; CAP=$CAP; CACHE_BASE=$CACHE_BASE"
for task in $TASKS; do
  # already have a complete 4-arm CSV for this task? skip (idempotent re-run safety)
  csv="$CACHE_BASE/results/ntbase_matched_${task}.csv"
  if [ -f "$csv" ] && [ "$(grep -c ',' "$csv" 2>/dev/null)" -ge 5 ]; then
    echo "[$(date)] SKIP $task (CSV already complete)"; continue
  fi
  # throttle
  n=$(running_count); [ -z "$n" ] && n=99
  while [ "$n" -ge "$CAP" ]; do sleep 45; n=$(running_count); [ -z "$n" ] && n=99; done
  jid=$(sbatch --parsable --nodelist=galaxy --job-name="ntbaseg_$task" \
        --output="$LOGD/ntbaseg_${task}_%j.out" \
        --export="ALL,TASK=$task,ARMS=$ARMS,BEST_HP=$YAML,CACHE_BASE=$CACHE_BASE,RESULTS_CSV=$csv,PARAM_MATCHED=1" \
        "$REPO/code_carbon/slurm/run_matched_capacity.sh" 2>&1)
  echo "[$(date)] submitted $task -> jid $jid (galaxy)"
  sleep 4
done
echo "[$(date)] all task submissions issued (galaxy-only)"
