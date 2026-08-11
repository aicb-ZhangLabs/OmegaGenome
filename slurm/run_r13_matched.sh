#!/bin/bash
# R1.3 MATCHED-PAIR (best-HP) driver: one SLURM job = one task's self-contained comparison.
#
# Per task, at that task's BEST-HP config (rebuttal_infra/best_hp/best_hp_nt.yaml) + the SAME strong
# NT-2.5B teacher (best-val staged adapter), this runs ALL arms into ONE per-task results CSV so the
# CSV is a self-contained "best-HP one-hot baseline vs best-HP pretrained-embedding" table:
#   (1) onehot                         <- BASELINE: must reproduce the paper's distilled student
#   (2) nt_embedding replaceK   nt     <- embedding variant (widened stem, no D->4 bottleneck)
#   (3) nt_embedding latefuse_onehot nt<- FAITHFUL: real one-hot stem + deep NT-embedding concat
#   (4) nt_embedding replaceK   dnabert2
#   (5) nt_embedding latefuse_onehot dnabert2
# All arms share the per-task per-bp embedding cache (cache_embedding/), computed once on the first
# nt arm and reused (the dnabert2 arms write a separate dnabert2-tagged cache once). embedding-layer
# = mid (driver default). The one-hot baseline does NOT touch the embedding cache.
#
# Usage:
#   sbatch --nodelist=voyager --export=ALL,TASK=splice_sites_all run_r13_matched.sh
#   SMOKE=1 ARMS="onehot replaceK_nt" EPOCHS=2 ...  (smoke: subset + few steps, verify config)
#
#SBATCH --job-name=r13-match
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --partition=zhanglab.p
#SBATCH --time=30-00:00:00
#SBATCH --output=slurm/slurm-r13-match-%j.out

set -uo pipefail
REPO="${REPO:-/home/pengchx3/text-dna/OmegaGenome_Revise_202606/code_carbon}"
cd "$REPO"
[ -f slurm/env_setup.sh ] && source slurm/env_setup.sh
PY="$REPO/.venv_carbon_portable/bin/python"

# Run entirely off the galaxy SSD (no /extra or /home runtime I/O). Override the whole HF family to
# the SSD-staged cache (NT-2.5B base + HF datasets). SSD root differs per node.
SSD="${SSD:-}"
if [ -z "$SSD" ]; then
  if   [ -d /srv/disk00/sshfs/pengchx3 ];      then SSD=/srv/disk00/sshfs/pengchx3
  elif [ -d /tmp/galaxy_srv_disk00/pengchx3 ]; then SSD=/tmp/galaxy_srv_disk00/pengchx3
  fi
fi
RNT="$SSD/rebuttal_nt"
# BULLETPROOF offline: this cluster's network to wandb.ai / huggingface.co hangs the loads, so force
# everything offline HERE in the launcher (env in --export was being overridden). W&B curves still
# save locally (syncable later); HF uses the local arrow/model cache only (datasets 4.2.0 fix).
export WANDB_MODE=offline WANDB_DISABLED=false
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_DATASETS_TRUST_REMOTE_CODE=1
export HF_HOME="$RNT/hf_cache"
export HF_HUB_CACHE="$HF_HOME/hub"
# NODE-LOCAL STAGING (per the lab SSD announcement: the /tmp/galaxy_srv_disk00 sshfs mount is slow
# under concurrent load). Bulk-copy the HF cache (dataset arrow + base NT-2.5B shards) to node-local
# /dev/shm ONCE per node, then read from RAM -- sequential cp is sshfs-robust where the driver's lazy
# random arrow/shard reads stall. mkdir-lock avoids concurrent-job races. Writes still go to the SSD.
STAGE_SHM="/dev/shm/$(whoami)/hf_cache"
if [ "${STAGE_LOCAL:-1}" = "1" ] && [ -d "$HF_HOME" ]; then
  if [ ! -e "$STAGE_SHM/.staged_ok" ]; then
    mkdir -p "$STAGE_SHM"
    if mkdir "$STAGE_SHM/.staging_lock" 2>/dev/null; then
      echo "[stage] copying HF cache ($(du -sh "$HF_HOME" 2>/dev/null|cut -f1)) -> $STAGE_SHM ..."
      if cp -rn "$HF_HOME"/. "$STAGE_SHM"/ 2>/dev/null; then touch "$STAGE_SHM/.staged_ok"; echo "[stage] done"; fi
      rmdir "$STAGE_SHM/.staging_lock" 2>/dev/null
    else
      echo "[stage] another job staging; waiting for cache..."
      for _i in $(seq 1 120); do [ -e "$STAGE_SHM/.staged_ok" ] && break; sleep 5; done
    fi
  fi
  if [ -e "$STAGE_SHM/.staged_ok" ]; then
    export HF_HOME="$STAGE_SHM"; export HF_HUB_CACHE="$STAGE_SHM/hub"
    echo "[stage] HF_HOME -> $STAGE_SHM (node-local RAM; no sshfs reads for dataset/model)"
  fi
fi
export HUGGINGFACE_HUB_CACHE="$HF_HUB_CACHE"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TRANSFORMERS_CACHE="$HF_HOME/hub"
export HF_ASSETS_CACHE="$HF_HOME/assets"

NT_PARENT="${NT_PARENT:-$RNT/nt_adapters}"
CACHE_BASE="${CACHE_BASE:-$RNT/run}"
# The per-bp embedding cache (~7.7G/task) + checkpoints are the HOT-PATH large I/O that stalls on the
# sshfs mount. Redirect CACHE_BASE (embedding cache + output) to node-local /dev/shm; RESULTS_CSV stays
# on the shared SSD (small appends) so the throttle still sees completions. /dev/shm has ~440G free.
if [ "${STAGE_LOCAL:-1}" = "1" ]; then
  _cbsub=$(basename "$CACHE_BASE")
  CACHE_BASE="/dev/shm/$(whoami)/r13cache/$_cbsub"
  mkdir -p "$CACHE_BASE/results"
  echo "[stage] CACHE_BASE (embedding cache + output) -> $CACHE_BASE (node-local RAM)"
fi
mkdir -p "$CACHE_BASE/results"

TASK="${TASK:?set TASK=<task_name> via --export}"
# Per-task self-contained CSV (best-HP one-hot vs best-HP embedding for THIS task).
RESULTS_CSV="${RESULTS_CSV:-$RNT/run/results/r13_matched_${TASK}.csv}"
BEST_HP="${BEST_HP:-/home/pengchx3/text-dna/OmegaGenome_Revise_202606/rebuttal_infra/best_hp/best_hp_nt.yaml}"

# Arms to run (default = full matched set). Override with ARMS="onehot replaceK_nt" for a smoke.
ARMS="${ARMS:-onehot replaceK_nt latefuse_onehot_nt replaceK_dnabert2 latefuse_onehot_dnabert2}"

COMMON=( --task-name "$TASK" --best-hp "$BEST_HP" --results-csv "$RESULTS_CSV"
         --nt-parent "$NT_PARENT" --cache-base "$CACHE_BASE"
         --output-dir "$CACHE_BASE/output/r13_matched"
         --teacher-batch-size "${TEACHER_BS:-8}" )
[ -n "${EPOCHS:-}" ] && COMMON+=( --epochs "$EPOCHS" )
[ -n "${PATIENCE:-}" ] && COMMON+=( --early-stop-patience "$PATIENCE" )
[ -n "${NUM_WORKERS:-}" ] && COMMON+=( --num-workers "$NUM_WORKERS" )
# Optional student-size override (e.g. MODEL_SIZE=original to force the 0.12M deployable student even
# when best_hp lists medium). Best-HP loss weights/T/lr/epochs are unchanged; only student width changes.
# In PARAM_MATCHED=1 mode this single MODEL_SIZE is IGNORED in favor of a PER-ARM matched size (below),
# because each NT-embedding front-end has a different param cost so each needs its own backbone width to
# hit TOTAL ~= the one-hot baseline 121,094.
[ -n "${MODEL_SIZE:-}" ] && [ "${PARAM_MATCHED:-0}" != "1" ] && COMMON+=( --model-size "$MODEL_SIZE" )
if [ "${SMOKE:-0}" = "1" ]; then
  COMMON+=( --smoke --max-steps "${MAX_STEPS:-20}" --wandb-mode offline )
fi

# R1.3 PER-ARM PARAM-MATCH: each embedding arm shrinks the backbone so backbone + front-end + classifier
# = TOTAL ~= one-hot 121,094. replaceK->emb_matched_replaceK (C=25, total 120,433); latefuse->
# emb_matched_latefuse (C=34, total 120,238); replace4->original_emb_matched (C=61, total 120,721,
# SECONDARY). The onehot arm stays at the full C=64 baseline (121,094) -- it is THE budget reference.
arm_matched_size() {
  case "$1" in
    *replaceK*)  echo emb_matched_replaceK ;;
    *latefuse*)  echo emb_matched_latefuse ;;
    *replace4*)  echo original_emb_matched ;;
    *)           echo "" ;;   # onehot: no override (full C=64 baseline)
  esac
}

echo "[$(date)] R1.3 MATCHED TASK=$TASK on $(hostname) GPU=$CUDA_VISIBLE_DEVICES"
echo "SSD=$SSD NT_PARENT=$NT_PARENT CACHE_BASE=$CACHE_BASE"
echo "RESULTS_CSV=$RESULTS_CSV  ARMS='$ARMS'  SMOKE=${SMOKE:-0} EPOCHS=${EPOCHS:-bestHP}"

# CSV signature of an arm's already-written result row (result-preserving skip on resubmit).
arm_csv_sig() {
  case "$1" in
    onehot)                   echo ',onehot,' ;;
    replace4_ntbase)          echo ',nt_base,mid,replace4,' ;;
    replaceK_ntbase)          echo ',nt_base,mid,replaceK,' ;;
    latefuse_onehot_ntbase)   echo ',nt_base,mid,latefuse_onehot,' ;;
    replace4_nt)              echo ',nt,mid,replace4,' ;;
    replaceK_nt)              echo ',nt,mid,replaceK,' ;;
    latefuse_onehot_nt)       echo ',nt,mid,latefuse_onehot,' ;;
    *)                        echo '' ;;
  esac
}

run_arm() {
  local mode="$1"; shift
  # SKIP_DONE_ARMS=1 (default): if this arm's result row already exists in RESULTS_CSV, skip re-training
  # it (result-preserving -- the completed arm's numbers are unchanged). Makes resubmits do only the
  # MISSING arms instead of re-running onehot/replace4 every time.
  if [ "${SKIP_DONE_ARMS:-1}" = "1" ]; then
    local sig; sig=$(arm_csv_sig "$mode")
    if [ -n "$sig" ] && [ -f "$RESULTS_CSV" ] && grep -qF "$sig" "$RESULTS_CSV" 2>/dev/null; then
      echo; echo "################## ARM: $mode -> already in CSV, SKIP (result-preserving) ##################"
      return 0
    fi
  fi
  local extra=()
  if [ "${PARAM_MATCHED:-0}" = "1" ]; then
    local sz; sz=$(arm_matched_size "$mode")
    [ -n "$sz" ] && extra+=( --model-size "$sz" )
    echo; echo "################## ARM: $mode (PARAM-MATCHED size=${sz:-baseline_C64}) ##################"
  else
    echo; echo "################## ARM: $mode ##################"
  fi
  "$PY" -m src.train.distill_nt_embedding "${COMMON[@]}" "${extra[@]}" "$@"
  echo "[$(date)] arm $mode rc=$?"
}

for arm in $ARMS; do
  case "$arm" in
    onehot)                    run_arm onehot --input-mode onehot ;;
    replace4_nt)               run_arm replace4_nt --input-mode nt_embedding --embedding-source nt       --fusion replace4 ;;
    replaceK_nt)               run_arm replaceK_nt --input-mode nt_embedding --embedding-source nt       --fusion replaceK ;;
    latefuse_onehot_nt)        run_arm latefuse_onehot_nt --input-mode nt_embedding --embedding-source nt       --fusion latefuse_onehot ;;
    replace4_ntbase)           run_arm replace4_ntbase --input-mode nt_embedding --embedding-source nt_base   --fusion replace4 ;;
    replaceK_ntbase)           run_arm replaceK_ntbase --input-mode nt_embedding --embedding-source nt_base   --fusion replaceK ;;
    latefuse_onehot_ntbase)    run_arm latefuse_onehot_ntbase --input-mode nt_embedding --embedding-source nt_base   --fusion latefuse_onehot ;;
    replaceK_dnabert2)         run_arm replaceK_dnabert2 --input-mode nt_embedding --embedding-source dnabert2 --fusion replaceK ;;
    latefuse_onehot_dnabert2)  run_arm latefuse_onehot_dnabert2 --input-mode nt_embedding --embedding-source dnabert2 --fusion latefuse_onehot ;;
    *) echo "[WARN] unknown arm '$arm' skipped" ;;
  esac
done

echo "[$(date)] done all arms for TASK=$TASK -> $RESULTS_CSV"
