#!/bin/bash
# Capped submitter for the R1.3 base-NT-embedding suite. Leaves >=1 GPU FREE per node for others:
# submits a 1-GPU job to a node only when that node currently has >=2 free GPUs (all-user count), so
# after the job lands >=1 remains free. Manages Phase-1 (from-scratch) to completion, then Phase-2
# (distilled). Dedups on CSV-completion + already-queued. Skips down/drained nodes. NEVER scancels.
set -uo pipefail
SB=/pkg/slurm/22.05.3/bin
REPO=/home/pengchx3/text-dna/OmegaGenome_Revise_202606
RUN=$REPO/code_carbon/slurm/run_r13_matched.sh
LOG=$REPO/code_carbon/slurm/gpu_throttle.log
ARMS="onehot replace4_ntbase replaceK_ntbase latefuse_onehot_ntbase"
TASKS="promoter_tata H3K4me3 H3K4me2 splice_sites_donors splice_sites_acceptors promoter_all promoter_no_tata enhancers enhancers_types H3K9ac H3K27ac H3K4me1 H2AFZ H3K27me3 H3K36me3 H3K9me3 H4K20me1 splice_sites_all"
declare -A TOTGPU=( [galaxy]=6 [laniakea]=8 [voyager]=4 )
NODES="laniakea voyager galaxy"          # preference order; laniakea (49G) & voyager (H100) first
POLL="${POLL:-120}"
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

node_up(){ local s; s=$($SB/sinfo -h -n "$1" -o '%t' 2>/dev/null | head -1 | tr -d '[:space:]'); case "$s" in mix|idle|alloc|mix*|alloc*) return 0;; *) return 1;; esac; }
# free_eff = total - running GPU jobs on node (ALL users) - my PENDING jobs pinned to node.
# Subtracting my own pending (named *_<node>_*) is critical: a just-submitted job is PENDING, so
# counting only running would let one pass over-commit a node. Leaves >=1 free when caller needs >=2.
# SUM actual GPUs across all running jobs on the node (some jobs, e.g. arc3, request gpu:2) -- counting
# jobs would undercount multi-GPU jobs and over-report free GPUs.
run_on(){ $SB/squeue -h -t RUNNING -w "$1" -O 'tres-per-node:40' 2>/dev/null | grep -oE 'gpu:[0-9]+' | awk -F: '{s+=$2} END{print s+0}'; }
mypend_on(){ printf '%s\n' "$SNAP" | grep -c "_${1}_" | head -1 | tr -d '[:space:]'; }
free_gpu(){ local n="$1"; echo $(( ${TOTGPU[$n]:-0} - $(run_on "$n") - $(mypend_on "$n") )); }
# LEAVE-1 mode (from now on, per user): submit to a node only when it has >=2 free GPUs (free = total
# - all-user running - my pending), so after the job lands >=1 GPU stays free for others. This also
# naturally pre-queues one job when 2 free open up (grabbed within the poll), then holds at 1 free.
pick_node(){ local n f; for n in $NODES; do node_up "$n" || continue; f=$(free_gpu "$n"); [ "${f:-0}" -ge 2 ] && { echo "$n"; return 0; }; done; echo ""; }
csv_done(){ local f="$1" oh nb; [ -f "$f" ] || return 1
  oh=$(grep -c ',onehot,' "$f" 2>/dev/null | head -1 | tr -d '[:space:]'); nb=$(grep -c ',nt_base,mid,' "$f" 2>/dev/null | head -1 | tr -d '[:space:]')
  [ "${oh:-0}" -ge 1 ] && [ "${nb:-0}" -ge 3 ]; }
in_queue(){ printf '%s\n' "$SNAP" | grep -qE "^${1}_[a-z0-9]+_${2}$"; }   # prefix _<node>_ task

# Wedge-reaper: cancel MY <pfx>_* jobs whose log is stale >60min with 0 epochs logged (the DataLoader
# fork-wedge). csv_done stays false so the task is resubmitted fresh (workers=0). Only ever touches
# jobs named <pfx>_* -- never arc3/loom/rondo/others.
reap_wedged(){ local pfx="$1" jid task lg age ep
  for jid in $($SB/squeue -h -u pengchx3 -t RUNNING -o '%i %j' 2>/dev/null | grep -E "^[0-9]+ ${pfx}_" | awk '{print $1}'); do
    lg=$(ls -t "$REPO/code_carbon/slurm/slurm-${pfx}-"*"-${jid}.out" 2>/dev/null | head -1); [ -z "$lg" ] && continue
    age=$(( ($(date +%s) - $(stat -c %Y "$lg")) / 60 ))
    ep=$(tr '\r' '\n' < "$lg" 2>/dev/null | grep -ciE 'epoch [0-9]+/')
    # 75min threshold: the base-NT embedding PRECOMPUTE (embed ~27k seqs through the 2.5B model) is a
    # legit ~30-45min GPU step on 3090 with NO log output -- a 30min reaper false-killed it in a loop.
    if [ "$age" -gt 75 ] && [ "${ep:-0}" -eq 0 ]; then
      $SB/scancel "$jid" 2>/dev/null; log "REAP wedged $pfx jid=$jid (log stale ${age}min, 0 epochs) -> will resubmit fresh"
    fi
  done; }

run_phase(){ # pname jobprefix subdir yaml  (write path = node-local /tmp mount)
  local pname="$1" pfx="$2" subdir="$3" yaml="$4"
  local login="/srv/disk00/sshfs/pengchx3/rebuttal_nt/$subdir"          # read (login=galaxy mount)
  local write="/tmp/galaxy_srv_disk00/pengchx3/rebuttal_nt/$subdir"     # write (compute-node mount)
  mkdir -p "$login/results"
  log "PHASE $pname START subdir=$subdir yaml=$(basename "$yaml")"
  while :; do
    reap_wedged "$pfx"
    SNAP=$($SB/squeue -h -u pengchx3 -t RUNNING,PENDING -o '%j' 2>/dev/null) || { sleep "$POLL"; continue; }
    local remaining=0 t node jid
    for t in $TASKS; do
      csv_done "$login/results/r13_ntbase_$t.csv" && continue
      remaining=$((remaining+1))
      in_queue "$pfx" "$t" && continue
      node=$(pick_node); [ -z "$node" ] && continue
      jid=$($SB/sbatch --parsable --nodelist="$node" --gres=gpu:1 --mem=98304 \
        --job-name="${pfx}_${node}_${t}" --output="$REPO/code_carbon/slurm/slurm-${pfx}-${t}-%j.out" \
        --export="ALL,TASK=$t,ARMS=$ARMS,BEST_HP=$yaml,CACHE_BASE=$write,RESULTS_CSV=$write/results/r13_ntbase_$t.csv,PARAM_MATCHED=1,TEACHER_BS=32,NUM_WORKERS=4,WANDB_MODE=offline,SKIP_TEACHER_EVAL=1" \
        "$RUN" 2>>"$LOG")
      [ -n "$jid" ] && { log "PHASE $pname SUBMIT $t -> $node jid=$jid (free left >=1)"; SNAP="$SNAP"$'\n'"${pfx}_${node}_${t}"; sleep 8; }
    done
    [ "$remaining" -eq 0 ] && { log "PHASE $pname COMPLETE (18/18)"; return 0; }
    log "PHASE $pname remaining=$remaining | free: laniakea=$(free_gpu laniakea) voyager=$(free_gpu voyager) galaxy=$(free_gpu galaxy)"
    sleep "$POLL"
  done
}
log "=== gpu_throttle start (leave >=1 GPU free per node) ==="
run_phase FROMSCRATCH r13nb   run_fromscratch_ntbase "$REPO/rebuttal_infra/best_hp/best_hp_nt_fromscratch.yaml"
run_phase DISTILLED   r13nbKD run_distilled_ntbase   "$REPO/rebuttal_infra/best_hp/best_hp_nt.yaml"
log "=== ALL PHASES COMPLETE ==="
