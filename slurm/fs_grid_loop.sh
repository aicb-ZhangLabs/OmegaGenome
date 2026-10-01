#!/bin/bash
# Exp-1b self-throttling distribution loop (v2: CACHE-FILE gating + per-arm CSVs).
# Fills the remaining from-scratch (CE-only, no-KD) arms across galaxy/laniakea/voyager, one GPU
# free per server (caps galaxy<=5, laniakea<=7, voyager<=3; counts ALL my RUNNING+PENDING on a node).
#
# v2 change (max backfill speed): each arm writes a DISTINCT per-arm CSV (submit_fs_arm.sh:
# onehot->_onehot, replace4->_replace4, latefuse->_latefuse, replaceK->main), so there is NO
# shared-CSV append race and hence NO need to serialize dependents on the primer's ROW. Instead a
# new-task replace4/latefuse becomes submittable the moment that task's per-bp CACHE FILES exist
# (all splits' token_embeddings.npy + metadata.json) -- written at end-of-precompute (~1h in), a full
# 200-epoch training-cycle BEFORE the primer's row lands. All ready arms then compete for free GPUs.
#
# Gate:   ready(replaceK|onehot)=independent ;  ready(replace4|latefuse)=cache-files-present(task).
# Dedup:  skip if a job named fsmc_<task>_<arm> is RUNNING/PENDING,
#         and skip if the arm's completed row is present in its per-arm CSV OR the main CSV (covers the
#         already-done first-batch rows + the 3 in-flight replace4 launched under the old main-CSV scheme).
# Only ever SUBMITS; never scancels (non-fs jobs untouched).  DRYRUN=1 -> classify WORK and exit.
set -uo pipefail
SB=/pkg/slurm/22.05.3/bin
REPO=${OG_ROOT:-$PWD}
SUBMIT="$REPO/slurm/submit_fs_arm.sh"
SSD=${OG_SCRATCH:-$PWD/output}            # login node = galaxy mount, for row/cache-file checks
RESDIR="$SSD/nt_runs/run_fromscratch/results"
CACHEDIR="$SSD/nt_runs/run_fromscratch/data/cache_embedding/nt_adapters"
LOG="$REPO/slurm/fs_grid_loop.log"

BATCH1="splice_sites_donors splice_sites_all promoter_all promoter_tata promoter_no_tata H3K4me3"
NEW="H2AFZ H3K27ac H3K27me3 H3K36me3 H3K4me1 H3K4me2 H3K9ac H3K9me3 H4K20me1 enhancers enhancers_types splice_sites_acceptors"

declare -A CAP=( [galaxy]=5 [laniakea]=7 [voyager]=3 )
declare -A ATTEMPTS=()
MAX_ATTEMPTS=3
DRYRUN="${DRYRUN:-0}"

log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

main_csv(){ echo "$RESDIR/fromscratch_$1.csv"; }

# completed 200-epoch row present? checks per-arm CSV OR main CSV (old-scheme in-flight + done first batch).
have_row(){ # task arm
  local t="$1" a="$2" main; main="$(main_csv "$t")"
  case "$a" in
    onehot)   grep -q ",onehot," "$RESDIR/fromscratch_${t}_onehot.csv" 2>/dev/null ;;
    replace4) grep -q ",nt,mid,replace4," "$RESDIR/fromscratch_${t}_replace4.csv" 2>/dev/null \
              || grep -q ",nt,mid,replace4," "$main" 2>/dev/null ;;
    replaceK) grep -q ",nt,mid,replaceK," "$main" 2>/dev/null ;;
    latefuse) grep -q ",nt,mid,latefuse_onehot," "$RESDIR/fromscratch_${t}_latefuse.csv" 2>/dev/null \
              || grep -q ",nt,mid,latefuse_onehot," "$main" 2>/dev/null ;;
  esac
}

# all 3 splits' per-bp cache fully written? (token_embeddings.npy + metadata.json; metadata renamed LAST)
cache_present(){ # task
  local t="$1" s
  for s in train val test; do
    [ -f "$CACHEDIR/$t/${s}_mid_token_embeddings.npy" ] || return 1
    [ -f "$CACHEDIR/$t/${s}_mid_metadata.json" ]        || return 1
  done
  return 0
}

is_ready(){ # task arm
  case "$2" in
    onehot|replaceK) return 0 ;;
    replace4|latefuse) cache_present "$1" ;;
  esac
}

in_squeue(){ $SB/squeue -u $USER -h -o '%j' 2>/dev/null | grep -qx "fsmc_$1_$2"; }
node_count(){ $SB/squeue -u $USER -h -t RUNNING,PENDING -w "$1" -o '%i' 2>/dev/null | grep -c .; }
pick_node(){ local n c; for n in "$@"; do c=$(node_count "$n"); [ "$c" -lt "${CAP[$n]}" ] && { echo "$n"; return 0; }; done; echo ""; }

# Work list: 6 first-batch replace4 + 12 new x {replaceK,onehot,replace4,latefuse}
WORK=()
for t in $BATCH1;  do WORK+=("$t:replace4"); done
for t in $NEW; do for a in replaceK onehot replace4 latefuse; do WORK+=("$t:$a"); done; done

# Classification (used by DRYRUN and by the live loop's opening audit)
classify(){
  local done=0 inflight=0 ready=0 waiting=0 t a st
  for item in "${WORK[@]}"; do
    t="${item%%:*}"; a="${item##*:}"
    if   have_row "$t" "$a";  then st="DONE";     done=$((done+1))
    elif in_squeue "$t" "$a"; then st="INFLIGHT($($SB/squeue -u $USER -h -o '%i' -n fsmc_${t}_${a} 2>/dev/null))"; inflight=$((inflight+1))
    elif is_ready "$t" "$a";  then st="READY";    ready=$((ready+1))
    else                           st="WAIT(cache)"; waiting=$((waiting+1)); fi
    printf '  %-26s %-9s %s\n' "$t" "$a" "$st" | tee -a "$LOG"   # per-line tee keeps counters in main shell
  done
  log "CLASSIFY totals: DONE=$done INFLIGHT=$inflight READY=$ready WAIT=$waiting  (of ${#WORK[@]})"
}

log "=== fs_grid_loop v2 start (DRYRUN=$DRYRUN): ${#WORK[@]} arms; cache-file gate + per-arm CSVs ==="
classify
if [ "$DRYRUN" = "1" ]; then log "DRYRUN: no submits. exiting."; exit 0; fi

while :; do
  remaining=0; subs=0
  for phase in replaceK other; do
    for item in "${WORK[@]}"; do
      t="${item%%:*}"; a="${item##*:}"
      [ "$phase" = replaceK ] && [ "$a" != replaceK ] && continue
      [ "$phase" = other ]    && [ "$a" = replaceK ]  && continue
      have_row "$t" "$a" && continue
      remaining=$((remaining+1))
      in_squeue "$t" "$a" && continue
      is_ready "$t" "$a" || continue
      key="$t:$a"; [ "${ATTEMPTS[$key]:-0}" -ge "$MAX_ATTEMPTS" ] && continue
      if [ "$a" = onehot ]; then node=$(pick_node galaxy laniakea voyager)
      else                       node=$(pick_node galaxy); fi
      [ -z "$node" ] && continue
      ATTEMPTS[$key]=$(( ${ATTEMPTS[$key]:-0} + 1 ))
      jid=$(bash "$SUBMIT" "$t" "$a" "$node" 2>>"$LOG")
      [ -n "$jid" ] && { log "submitted $key -> node=$node jid=$jid (attempt ${ATTEMPTS[$key]})"; subs=$((subs+1)); sleep 3; }
    done
  done
  if [ "$remaining" -eq 0 ]; then log "=== ALL arms have completed rows. done. ==="; break; fi
  log "remaining=$remaining submitted_this_pass=$subs | galaxy=$(node_count galaxy)/5 laniakea=$(node_count laniakea)/7 voyager=$(node_count voyager)/3"
  sleep 90
done
