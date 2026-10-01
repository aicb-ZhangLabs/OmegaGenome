#!/bin/bash
# Exp-1b single-arm submitter for the from-scratch (CE-only, no-KD) 18-task x 4-fusion grid.
# Reuses slurm/run_matched_capacity.sh purely via env (no training-code change). One call = one arm.
#
# Usage:  submit_fs_arm.sh <task> <arm> <node>
#   <arm> in {onehot, replace4, replaceK, latefuse}
#   <node> in {galaxy, laniakea, voyager}
# Prints the submitted jobid on stdout (last line).
#
# Per-arm settings (verified against the completed first-batch 6-task from-scratch run):
#   onehot   -> ARMS=onehot, MODEL_SIZE=original (full C=64 ~121k), writes a DISTINCT _onehot.csv
#               sibling (avoids cross-node concurrent-append races with the embedding arms).
#   replace4 -> ARMS=replace4_nt,        PARAM_MATCHED=1 (per-arm emb_matched ~120k), main CSV.
#   replaceK -> ARMS=replaceK_nt,        PARAM_MATCHED=1, main CSV. (NT per-bp cache PRIMER.)
#   latefuse -> ARMS=latefuse_onehot_nt, PARAM_MATCHED=1, main CSV.
# All from-scratch arms use BEST_HP=best_hp_nt_fromscratch.yaml (weight_kl=weight_mse=0) and
# CACHE_BASE=<ssd>/nt_runs/run_fromscratch. NODE-PORTABLE: the shared galaxy SSD is mounted at
# ${OG_SCRATCH:-$PWD/output} on galaxy and ${OG_SCRATCH:-$PWD/output} on laniakea/voyager
# (SAME physical disk); we pick the prefix for the TARGET node so paths resolve wherever the job lands.
set -uo pipefail
SB=/pkg/slurm/22.05.3/bin
REPO=${OG_ROOT:-$PWD}
SCRIPT="$REPO/slurm/run_matched_capacity.sh"
BEST_HP=${OG_WORKSPACE:-$PWD/..}/analysis/best_hp/best_hp_nt_fromscratch.yaml

TASK="${1:?usage: submit_fs_arm.sh TASK ARM NODE}"
ARM="${2:?arm: onehot replace4 replaceK latefuse}"
NODE="${3:?node: galaxy laniakea voyager}"

# Node-appropriate prefix to the SAME shared disk.
if [ "$NODE" = "galaxy" ]; then SSD=${OG_SCRATCH:-$PWD/output}; else SSD=${OG_SCRATCH:-$PWD/output}; fi
RNT="$SSD/nt_runs"
CACHE_BASE="$RNT/run_fromscratch"
RESDIR="$CACHE_BASE/results"

# Each arm writes a DISTINCT per-arm CSV sibling (onehot/replace4/latefuse), so NO two arms ever
# append to the same file -> no concurrent-append race -> the loop needs no per-task serialization.
# replaceK keeps the main CSV (exactly one primer per task; nothing else writes it). The results
# reader must glob fromscratch_<task>{,_onehot,_replace4,_latefuse}.csv AND the main CSV.
case "$ARM" in
  onehot)   ARMS_TOK=onehot;             CSV="$RESDIR/fromscratch_${TASK}_onehot.csv";   SIZE_ENV="MODEL_SIZE=original" ;;
  replace4) ARMS_TOK=replace4_nt;        CSV="$RESDIR/fromscratch_${TASK}_replace4.csv"; SIZE_ENV="PARAM_MATCHED=1" ;;
  replaceK) ARMS_TOK=replaceK_nt;        CSV="$RESDIR/fromscratch_${TASK}.csv";          SIZE_ENV="PARAM_MATCHED=1" ;;
  latefuse) ARMS_TOK=latefuse_onehot_nt; CSV="$RESDIR/fromscratch_${TASK}_latefuse.csv"; SIZE_ENV="PARAM_MATCHED=1" ;;
  *) echo "[ERR] unknown arm '$ARM'" >&2; exit 2 ;;
esac

JOBNAME="fs13_${TASK}_${ARM}"
DEP=()
[ -n "${DEPENDENCY:-}" ] && DEP=( --dependency="$DEPENDENCY" )

jid=$($SB/sbatch --parsable --nodelist="$NODE" --job-name="$JOBNAME" "${DEP[@]}" \
      --export=ALL,TASK="$TASK",ARMS="$ARMS_TOK",RESULTS_CSV="$CSV",CACHE_BASE="$CACHE_BASE",BEST_HP="$BEST_HP",PATIENCE="${PATIENCE:-50}",NUM_WORKERS="${NUM_WORKERS:-4}",$SIZE_ENV \
      "$SCRIPT")
rc=$?
if [ "$rc" -ne 0 ] || [ -z "$jid" ]; then echo "[ERR] sbatch failed rc=$rc for $JOBNAME on $NODE" >&2; exit 1; fi
echo "[submit] $JOBNAME -> node=$NODE jid=$jid csv=$CSV" >&2
echo "$jid"
