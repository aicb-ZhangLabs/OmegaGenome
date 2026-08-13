#!/bin/bash
# R1.3 rebuttal: one SLURM job = one task's NT-embedding-input BPNet distillation.
#
# Reuses the shared carbon SLURM env (HF cache, portable venv, wandb, OUTPUT_PATH) then runs
# src.train.distill_nt_embedding for $TASK at that task's best-HP (rebuttal_infra/best_hp/best_hp_nt.yaml).
# Results auto-append to $RESULTS_CSV (shared across the 18 jobs). The one-hot baseline path is untouched.
#
# Usage (via sbatch, set TASK + optional RESULTS_CSV / extra args via --export):
#   sbatch --nodelist=laniakea --export=ALL,TASK=promoter_tata,RESULTS_CSV=/path/out.csv slurm/run_nt_embedding.sh
#   sbatch --nodelist=laniakea --export=ALL,TASK=H3K9me3,SMOKE=1 slurm/run_nt_embedding.sh   # smoke
#
#SBATCH --job-name=r13-ntemb
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --partition=zhanglab.p
#SBATCH --time=30-00:00:00
#SBATCH --output=slurm/slurm-r13-ntemb-%j.out

set -uo pipefail
# Under sbatch, BASH_SOURCE points at the /var/spool copy of this script, so deriving REPO from it
# breaks. Pin the repo path explicitly (env_setup.sh hardcodes the same REPO).
REPO="${REPO:-/home/pengchx3/text-dna/OmegaGenome_Revise_202606/code_carbon}"
cd "$REPO"

if [ -f slurm/env_setup.sh ]; then
  source slurm/env_setup.sh
fi
PY="$REPO/.venv_carbon_portable/bin/python"

# R1.3: run ENTIRELY off the galaxy SSD (no /extra runtime I/O). env_setup.sh pins HF caches to
# $REPO/.hf_cache; override the whole HF family to the SSD-staged cache (NT-2.5B base + dataset)
# so the teacher base model + HF datasets load from SSD. SSD root differs per node.
SSD="${SSD:-}"
if [ -z "$SSD" ]; then
  if   [ -d /srv/disk00/sshfs/pengchx3 ];        then SSD=/srv/disk00/sshfs/pengchx3
  elif [ -d /tmp/galaxy_srv_disk00/pengchx3 ];   then SSD=/tmp/galaxy_srv_disk00/pengchx3
  fi
fi
RNT="$SSD/rebuttal_nt"
export HF_HOME="$RNT/hf_cache"
export HF_HUB_CACHE="$HF_HOME/hub"
export HUGGINGFACE_HUB_CACHE="$HF_HUB_CACHE"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TRANSFORMERS_CACHE="$HF_HOME/hub"
export HF_ASSETS_CACHE="$HF_HOME/assets"

# SSD-staged NT LoRA adapters (best-per-task) + fresh recompute cache + run/output, all on SSD.
NT_PARENT="${NT_PARENT:-$RNT/nt_adapters}"
CACHE_BASE="${CACHE_BASE:-$RNT/run}"
mkdir -p "$CACHE_BASE"

TASK="${TASK:?set TASK=<task_name> via --export}"
RESULTS_CSV="${RESULTS_CSV:-$RNT/run/results/r13_nt_embedding_results.csv}"
BEST_HP="${BEST_HP:-/home/pengchx3/text-dna/OmegaGenome_Revise_202606/rebuttal_infra/best_hp/best_hp_nt.yaml}"
EXTRA=()
[ -n "${INPUT_MODE:-}" ] && EXTRA+=(--input-mode "$INPUT_MODE")
[ "${SMOKE:-0}" = "1" ] && EXTRA+=(--smoke --max-steps "${MAX_STEPS:-200}")
[ -n "${EPOCHS:-}" ] && EXTRA+=(--epochs "$EPOCHS")
[ -n "${MAX_STEPS:-}" ] && [ "${SMOKE:-0}" != "1" ] && EXTRA+=(--max-steps "$MAX_STEPS")
# Embedding front-end arm (R1.3 PI fix): replace4(default)/replaceK/latefuse. Reuses the cache.
[ -n "${FUSION:-}" ] && EXTRA+=(--fusion "$FUSION")
[ -n "${ADAPTER_WIDTH:-}" ] && EXTRA+=(--adapter-width "$ADAPTER_WIDTH")
[ -n "${FUSE_WIDTH:-}" ] && EXTRA+=(--fuse-width "$FUSE_WIDTH")
# Embedding hidden-state layer: -999=mid (driver default), -1=last, 0=raw token/byte embeddings, >=1 idx.
[ -n "${EMB_LAYER:-}" ] && EXTRA+=(--embedding-layer "$EMB_LAYER")

echo "[$(date)] R1.3 NT-embedding distill: TASK=$TASK on $(hostname) GPU=$CUDA_VISIBLE_DEVICES"
echo "SSD=$SSD HF_HOME=$HF_HOME NT_PARENT=$NT_PARENT CACHE_BASE=$CACHE_BASE"
echo "RESULTS_CSV=$RESULTS_CSV  EXTRA=${EXTRA[*]:-none}"

"$PY" -m src.train.distill_nt_embedding \
  --task-name "$TASK" \
  --best-hp "$BEST_HP" \
  --results-csv "$RESULTS_CSV" \
  --nt-parent "$NT_PARENT" \
  --cache-base "$CACHE_BASE" \
  --output-dir "$CACHE_BASE/output/r13_nt_embedding" \
  --teacher-batch-size "${TEACHER_BS:-8}" \
  "${EXTRA[@]}"

echo "[$(date)] done TASK=$TASK rc=$?"
