#!/bin/bash
# Durable pull-sync of the FULL R1.3 artifact set (not just CSVs) off the ephemeral vast.ai H100 box
# to the lab galaxy SSD, so nothing valuable lives only on the box. Companion to
# sync_box_dnabert2_results.sh / sync_box_nt_latefuse_results.sh (those pull only *.csv). This loop
# additionally mirrors:
#   - result CSVs                          -> live/results/
#   - best_model/ (student.pt+metadata)    -> live/output/... (per completed run)
#   - all metadata.json / run structure    -> live/output/...  (per-epoch dirs give the val_mcc curve)
#   - offline wandb runs                   -> live/wandb/
#   - supervisor + per-job logs            -> live/logs/
# Runs FROM a machine that has BOTH the SSD mount and SSH to the box (the box cannot mount the SSD).
# Uses rsync --update (never clobbers a newer local copy) and is resilient to box SSH blips. In-progress
# runs are copied too, but only as snapshots -- rsync re-pulls each loop, so partial files self-heal on
# the next completed pass. Loops every INTERVAL seconds.
set -u
SSH_OPTS="-o ConnectTimeout=25 -o StrictHostKeyChecking=no -o ServerAliveInterval=15 -p 26925"
BOX=root@115.124.123.240
BOX_ROOT=/workspace/rebuttal_nt/run
DEST=/tmp/galaxy_srv_disk00/pengchx3/rebuttal_nt/box_r13_backup/live
INTERVAL="${INTERVAL:-600}"
LOG=/home/pengchx3/text-dna/OmegaGenome_Revise_202606/code_carbon/slurm/sync_box_r13_artifacts.log

mkdir -p "$DEST/results" "$DEST/output" "$DEST/wandb" "$DEST/logs"
echo "[$(date)] R1.3 full-artifact sync loop start -> $DEST (every ${INTERVAL}s)" >>"$LOG"
while true; do
  # 1. result CSVs
  rsync -a --update -e "ssh $SSH_OPTS" \
    --include='*.csv' --exclude='*' \
    "$BOX:$BOX_ROOT/results/" "$DEST/results/" >>"$LOG" 2>&1

  # 2. output tree: pull small artifacts only (best_model ckpts, ALL metadata.json, run_info)
  #    -> gives us every best_model + the per-epoch dir names (the val_mcc curve) without dragging
  #    the full 96MB-per-run of redundant per-epoch student.pt weights.
  rsync -a --update -m -e "ssh $SSH_OPTS" \
    --include='*/' \
    --include='best_model/**' \
    --include='metadata.json' \
    --include='run_info.txt' \
    --exclude='epoch_*/student.pt' \
    --exclude='*' \
    "$BOX:$BOX_ROOT/output/" "$DEST/output/" >>"$LOG" 2>&1

  # 3. offline wandb runs
  rsync -a --update -e "ssh $SSH_OPTS" \
    "$BOX:$BOX_ROOT/wandb/" "$DEST/wandb/" >>"$LOG" 2>&1

  # 4. supervisor + per-job logs (from /root on the box)
  rsync -a --update -e "ssh $SSH_OPTS" \
    --include='r13box*.log' --include='r13box_supervisor.log' \
    --include='r13box_logs/***' --include='r13box_nt_logs/***' \
    --exclude='*' \
    "$BOX:/root/" "$DEST/logs/" >>"$LOG" 2>&1

  rc=$?
  ncsv=$(ls "$DEST/results"/*.csv 2>/dev/null | wc -l)
  nbest=$(find "$DEST/output" -path '*best_model*student.pt' 2>/dev/null | wc -l)
  echo "[$(date)] pass done rc=$rc; csv=$ncsv best_model_ckpts=$nbest" >>"$LOG"
  sleep "$INTERVAL"
done
