#!/bin/bash
# Exp-1b self-throttling distribution loop for the from-scratch (CE-only, no-KD) grid.
# Fills the 54 REMAINING arms (6 R7-tasks x replace4  +  12 new-tasks x {onehot,replaceK,replace4,latefuse})
# across galaxy/laniakea/voyager, leaving ONE GPU free per server via per-node caps.
#
# Design (why no SLURM afterok deps):
#   The loop gates each arm on ARTIFACT presence (a completed row in the shared CSV) -- "verify from
#   artifacts". This (a) enforces the per-task cache-primer ordering (new-task replace4/latefuse only
#   after that task's replaceK row lands = cache built), (b) SERIALIZES appends to the shared per-task
#   CSV (avoids the concurrent-append race), and (c) makes every running arm occupy exactly ONE cap
#   slot (a pending afterok chain would waste cap headroom). Each arm therefore occupies 1 GPU at a time.
#   onehot writes a DISTINCT _onehot.csv sibling so it never races the embedding arms.
#
# Caps (leave 1 GPU free/server): galaxy<=5 (of 6), laniakea<=7 (of 8), voyager<=3 (of 4).
# Counts ALL my RUNNING+PENDING jobs assigned to a node. Only ever SUBMITS/counts; never scancels.
set -uo pipefail
SB=/pkg/slurm/22.05.3/bin
REPO=/home/pengchx3/text-dna/OmegaGenome_Revise_202606/code_carbon
SUBMIT="$REPO/slurm/submit_fs_arm.sh"
RESDIR=/srv/disk00/sshfs/pengchx3/rebuttal_nt/run_fromscratch/results   # login node = galaxy mount, for row checks
LOG="$REPO/slurm/fs_grid_loop.log"

R7="splice_sites_donors splice_sites_all promoter_all promoter_tata promoter_no_tata H3K4me3"
NEW="H2AFZ H3K27ac H3K27me3 H3K36me3 H3K4me1 H3K4me2 H3K9ac H3K9me3 H4K20me1 enhancers enhancers_types splice_sites_acceptors"

declare -A CAP=( [galaxy]=5 [laniakea]=7 [voyager]=3 )
declare -A ATTEMPTS=()   # arm-key -> submit attempts (bound retries)
MAX_ATTEMPTS=3

log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

main_csv(){ echo "$RESDIR/r13_fromscratch_$1.csv"; }
oh_csv(){   echo "$RESDIR/r13_fromscratch_$1_onehot.csv"; }

# arm-row present? (a completed 200-epoch append)
have_row(){ # task arm
  local t="$1" a="$2"
  case "$a" in
    onehot)   grep -q ",onehot," "$(oh_csv "$t")" 2>/dev/null ;;
    replace4) grep -q ",nt,mid,replace4," "$(main_csv "$t")" 2>/dev/null ;;
    replaceK) grep -q ",nt,mid,replaceK," "$(main_csv "$t")" 2>/dev/null ;;
    latefuse) grep -q ",nt,mid,latefuse_onehot," "$(main_csv "$t")" 2>/dev/null ;;
  esac
}

# arm ready to submit? (predecessor artifact gate)
is_ready(){ # task arm
  local t="$1" a="$2"
  case "$a" in
    onehot|replaceK) return 0 ;;                         # independent / primer
    replace4)
      # R7 tasks: cache already exists -> ready. New tasks: need replaceK row first.
      if echo " $R7 " | grep -q " $t "; then return 0; fi
      have_row "$t" replaceK ;;
    latefuse) have_row "$t" replace4 ;;                  # serialize append after replace4
  esac
}

# a job for this arm already queued/running?
in_squeue(){ # task arm
  $SB/squeue -u pengchx3 -h -o '%j' 2>/dev/null | grep -qx "fs13_$1_$2"
}

# my RUNNING+PENDING jobs on a node
node_count(){ $SB/squeue -u pengchx3 -h -t RUNNING,PENDING -w "$1" -o '%i' 2>/dev/null | grep -c .; }

pick_node(){ # $1..=node pref order -> prints first node under cap, else empty
  local n c
  for n in "$@"; do c=$(node_count "$n"); [ "$c" -lt "${CAP[$n]}" ] && { echo "$n"; return 0; }; done
  echo ""
}

# Build the work list: "task:arm"
WORK=()
for t in $R7;  do WORK+=("$t:replace4"); done
for t in $NEW; do for a in replaceK onehot replace4 latefuse; do WORK+=("$t:$a"); done; done

log "=== fs_grid_loop start: ${#WORK[@]} arms to fill ==="

while :; do
  remaining=0; submitted_this_pass=0
  # Priority: primers (new replaceK) first so the 12 precomputes start ASAP; then the rest.
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
      # node preference: heavy primers favor voyager(H100); others favor galaxy then laniakea.
      if [ "$a" = replaceK ]; then node=$(pick_node voyager laniakea galaxy)
      else                         node=$(pick_node galaxy laniakea voyager); fi
      [ -z "$node" ] && continue
      ATTEMPTS[$key]=$(( ${ATTEMPTS[$key]:-0} + 1 ))
      jid=$(bash "$SUBMIT" "$t" "$a" "$node" 2>>"$LOG")
      if [ -n "$jid" ]; then log "submitted $key -> node=$node jid=$jid (attempt ${ATTEMPTS[$key]})"; submitted_this_pass=$((submitted_this_pass+1)); sleep 3; fi
    done
  done
  if [ "$remaining" -eq 0 ]; then log "=== ALL 54 arms have completed rows. done. ==="; break; fi
  g=$(node_count galaxy); l=$(node_count laniakea); v=$(node_count voyager)
  log "remaining=$remaining submitted_this_pass=$submitted_this_pass | node counts galaxy=$g/5 laniakea=$l/7 voyager=$v/3"
  sleep 90
done
