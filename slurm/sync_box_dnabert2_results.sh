#!/bin/bash
# Durable pull-sync: copy the vast.ai box's R1.3 DNABERT-2 result CSVs to the lab galaxy SSD so the
# coordinator can read+record them. Runs FROM a machine that has BOTH the SSD mount and SSH to the box
# (the box cannot mount the lab SSD). Only pulls '*dnabert2*.csv' (distinct names from the lab's own
# CSVs -> never overwrites lab work). Loops every INTERVAL seconds; resilient to box SSH blips.
set -u
SSH_OPTS="-o ConnectTimeout=25 -o StrictHostKeyChecking=no -o ServerAliveInterval=15 -p 26925"
BOX=root@115.124.123.240
BOX_RESULTS=/workspace/rebuttal_nt/run/results
SSD_RESULTS=/tmp/galaxy_srv_disk00/pengchx3/rebuttal_nt/run/results
INTERVAL="${INTERVAL:-300}"
LOG=/home/pengchx3/text-dna/OmegaGenome_Revise_202606/code_carbon/slurm/sync_box_dnabert2.log

mkdir -p "$SSD_RESULTS"
echo "[$(date)] sync loop start: $BOX:$BOX_RESULTS/*dnabert2*.csv -> $SSD_RESULTS (every ${INTERVAL}s)" >>"$LOG"
while true; do
  # --update so we never clobber a newer copy; --ignore-missing-args harmless if no matches.
  rsync -a --update -e "ssh $SSH_OPTS" \
    --include='*dnabert2*.csv' --exclude='*' \
    "$BOX:$BOX_RESULTS/" "$SSD_RESULTS/" >>"$LOG" 2>&1
  rc=$?
  echo "[$(date)] rsync rc=$rc; files now: $(ls "$SSD_RESULTS"/*dnabert2*.csv 2>/dev/null | wc -l)" >>"$LOG"
  sleep "$INTERVAL"
done
