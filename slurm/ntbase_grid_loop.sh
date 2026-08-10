#!/bin/bash
# R1.3 nt_base (BASE-foundation-model embedding) TASK-ORDERED throttle loop.
#
# One SLURM job per task = all 4 arms {onehot, replace4_ntbase, replaceK_ntbase, latefuse_onehot_ntbase}
# run SEQUENTIALLY (via run_r13_matched.sh) into ONE per-task CSV (r13_ntbase_<task>.csv). That IS the
# "each task's variations, then the next task" ordering the coordinator asked for.
#
# PHASE 1 (from-scratch, CE-only): BEST_HP=best_hp_nt_fromscratch.yaml, CACHE_BASE=run_fromscratch_ntbase.
# PHASE 2 (distilled, KD): BEST_HP=best_hp_nt.yaml, CACHE_BASE=run_distilled_ntbase.
# Phase 2 is submitted ONLY after ALL 18 Phase-1 task CSVs are complete (4 arm rows each).
#
# CAPS (MY r13nb jobs only, counted by job name -> robust to PENDING): galaxy<=4 (FIRST), laniakea<=4.
# voyager is AVOIDED (not in NODES). Never scancels. squeue errors are NOT suppressed (HARD lab rule):
# a squeue failure SKIPS the pass (never submits blind). Node is encoded in the job name r13nb_<node>_<task>
# so cap counting needs no `squeue -w` (which can miss PENDING --nodelist jobs).
#
# Launch (after the GPU smoke passes):
#   nohup env NODES="galaxy laniakea" bash slurm/ntbase_grid_loop.sh >> slurm/ntbase_grid_loop.log 2>&1 &
# DRYRUN=1 -> classify Phase-1 work and exit (no submits).
set -uo pipefail
SB=/pkg/slurm/22.05.3/bin
REPO=/home/pengchx3/text-dna/OmegaGenome_Revise_202606/code_carbon
RUN="$REPO/slurm/run_r13_matched.sh"
BEST_HP_DIR=/home/pengchx3/text-dna/OmegaGenome_Revise_202606/rebuttal_infra/best_hp
FS_YAML="$BEST_HP_DIR/best_hp_nt_fromscratch.yaml"
KD_YAML="$BEST_HP_DIR/best_hp_nt.yaml"
SSD=/srv/disk00/sshfs/pengchx3            # login node = galaxy mount (row/CSV checks)
RNT="$SSD/rebuttal_nt"
FS_DIR="$RNT/run_fromscratch_ntbase"
KD_DIR="$RNT/run_distilled_ntbase"
LOG="$REPO/slurm/ntbase_grid_loop.log"

# Tasks ordered small/fast -> large/slow so complete rows land early.
TASKS="promoter_tata promoter_no_tata promoter_all enhancers enhancers_types splice_sites_donors splice_sites_acceptors H3K4me3 splice_sites_all H3K4me2 H3K4me1 H3K9ac H3K9me3 H3K27ac H3K27me3 H3K36me3 H4K20me1 H2AFZ"
ARMS="onehot replace4_ntbase replaceK_ntbase latefuse_onehot_ntbase"

NODES="${NODES:-galaxy laniakea}"          # override to e.g. "laniakea" if galaxy OOMs on the smoke
declare -A CAP=( [galaxy]="${CAP_GALAXY:-4}" [laniakea]="${CAP_LANIAKEA:-4}" [voyager]="${CAP_VOYAGER:-0}" )
MAX_ATTEMPTS="${MAX_ATTEMPTS:-2}"          # per (phase,task); resubmit only after a job leaves the queue unfinished
POLL="${POLL:-90}"
DRYRUN="${DRYRUN:-0}"
declare -A ATTEMPTS=()
SNAP=""

log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

# One squeue snapshot per pass (job names of my RUNNING+PENDING jobs). rc!=0 -> caller skips the pass.
pass_snapshot(){ local out rc; out=$($SB/squeue -h -u pengchx3 -t RUNNING,PENDING -o '%j'); rc=$?
  if [ "$rc" -ne 0 ]; then log "WARN squeue rc=$rc -> skip pass (no blind submit)"; return 1; fi
  SNAP="$out"; return 0; }
node_count(){ printf '%s\n' "$SNAP" | grep -c "^r13nb_${1}_" || true; }         # my r13nb jobs targeting node $1
in_queue(){ printf '%s\n' "$SNAP" | grep -qE "^r13nb_[a-z]+_${1}$"; }           # task already queued (any node)
inflight_total(){ printf '%s\n' "$SNAP" | grep -c '^r13nb_' || true; }
pick_node(){ local n c; for n in $NODES; do c=$(node_count "$n"); [ "${c:-0}" -lt "${CAP[$n]:-0}" ] && { echo "$n"; return 0; }; done; echo ""; }

csv_for(){ echo "$1/results/r13_ntbase_$2.csv"; }                                # folder task
task_done(){ # folder task -> all 4 arm rows present (onehot + 3 nt_base fusions)
  local f; f=$(csv_for "$1" "$2"); [ -f "$f" ] || return 1
  local oh nb; oh=$(grep -c ',onehot,' "$f" 2>/dev/null || echo 0); nb=$(grep -c ',nt_base,mid,' "$f" 2>/dev/null || echo 0)
  [ "${oh:-0}" -ge 1 ] && [ "${nb:-0}" -ge 3 ]; }

submit_task(){ # node task folder yaml -> echoes jid
  local node="$1" task="$2" folder="$3" yaml="$4" csv
  csv=$(csv_for "$folder" "$task")
  $SB/sbatch --parsable --nodelist="$node" --job-name="r13nb_${node}_${task}" \
    --output="$REPO/slurm/slurm-r13nb-${task}-%j.out" \
    --export="ALL,TASK=$task,ARMS=$ARMS,BEST_HP=$yaml,CACHE_BASE=$folder,RESULTS_CSV=$csv,PARAM_MATCHED=1" \
    "$RUN" 2>>"$LOG"; }

classify(){ # folder -> print per-task status
  local folder="$1" t st done=0 rem=0
  pass_snapshot || return 1
  for t in $TASKS; do
    if task_done "$folder" "$t"; then st=DONE; done=$((done+1))
    elif in_queue "$t"; then st="INFLIGHT"; rem=$((rem+1))
    else st="TODO"; rem=$((rem+1)); fi
    printf '  %-24s %s\n' "$t" "$st" | tee -a "$LOG"
  done
  log "classify($(basename "$folder")): DONE=$done REMAINING=$rem of 18"
}

run_phase(){ # phase_name folder yaml
  local pname="$1" folder="$2" yaml="$3" stuck=0
  mkdir -p "$folder/results"
  log "=== PHASE $pname START: folder=$(basename "$folder") yaml=$(basename "$yaml") NODES='$NODES' caps galaxy=${CAP[galaxy]} laniakea=${CAP[laniakea]} ==="
  while :; do
    pass_snapshot || { sleep "$POLL"; continue; }
    local remaining=0 subs=0 t node jid key
    for t in $TASKS; do
      task_done "$folder" "$t" && continue
      remaining=$((remaining+1))
      in_queue "$t" && continue
      key="$pname:$t"; [ "${ATTEMPTS[$key]:-0}" -ge "$MAX_ATTEMPTS" ] && continue
      node=$(pick_node); [ -z "$node" ] && continue
      ATTEMPTS[$key]=$(( ${ATTEMPTS[$key]:-0} + 1 ))
      jid=$(submit_task "$node" "$t" "$folder" "$yaml")
      if [ -n "$jid" ]; then log "PHASE $pname SUBMIT $t -> node=$node jid=$jid (attempt ${ATTEMPTS[$key]}) csv=$(csv_for "$folder" "$t")"; subs=$((subs+1)); sleep 5;
      else log "PHASE $pname SUBMIT FAILED $t -> node=$node (see log)"; fi
      pass_snapshot || break
    done
    if [ "$remaining" -eq 0 ]; then log "=== PHASE $pname COMPLETE: all 18 tasks have 4-arm CSVs ==="; return 0; fi
    local infl; infl=$(inflight_total)
    log "PHASE $pname remaining=$remaining submitted=$subs inflight=$infl | galaxy=$(node_count galaxy)/${CAP[galaxy]} laniakea=$(node_count laniakea)/${CAP[laniakea]}"
    if [ "$subs" -eq 0 ] && [ "${infl:-0}" -eq 0 ] && [ "$remaining" -gt 0 ]; then
      stuck=$((stuck+1)); log "PHASE $pname STUCK pass $stuck/3 (nothing running, nothing submittable — likely maxed attempts)"
      [ "$stuck" -ge 3 ] && { log "ERROR PHASE $pname ABORT: $remaining task(s) unfinished, none in flight/submittable. Inspect logs."; return 1; }
    else stuck=0; fi
    sleep "$POLL"
  done
}

if [ "$DRYRUN" = "1" ]; then
  log "=== DRYRUN: classify Phase-1 (from-scratch) work; no submits ==="
  classify "$FS_DIR"; exit 0
fi

log "########## ntbase grid loop START (pid $$) ##########"
if run_phase FROMSCRATCH "$FS_DIR" "$FS_YAML"; then
  log ">>> Phase 1 (from-scratch) 100% complete -> queuing Phase 2 (distilled) <<<"
  run_phase DISTILLED "$KD_DIR" "$KD_YAML" && log "########## ALL PHASES COMPLETE ##########"
else
  log "########## STOPPED after Phase 1 abort; Phase 2 NOT started ##########"
fi
