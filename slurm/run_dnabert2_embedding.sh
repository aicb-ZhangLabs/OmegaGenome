#!/bin/bash
# R1.3 rebuttal (FAST variant): DNABERT-2-embedding-input BPNet distillation, one task per job.
#
# Same NT-2.5B -> BPNet KD as the one-hot baseline and the NT-embedding R1.3 run, but the student's
# per-bp INPUT embeddings come from base DNABERT-2-117M (hidden 768) at a configurable layer (default
# MIDDLE) instead of NT-2.5B. The KD TARGETS (teacher logits/features) STILL come from the NT-2.5B
# teacher, so this isolates the input-representation effect. DNABERT-2 is tiny (117M) -> precompute is
# minutes, not the ~2.3h/task of NT-2.5B.
#
# Usage:
#   sbatch --nodelist=<node> --export=ALL,TASK=H3K4me3 slurm/run_dnabert2_embedding.sh
#   sbatch --nodelist=<node> --export=ALL,TASK=H3K4me3,SMOKE=1 slurm/run_dnabert2_embedding.sh
#
#SBATCH --job-name=r13-d2emb
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --partition=zhanglab.p
#SBATCH --time=30-00:00:00
#SBATCH --output=slurm/slurm-r13-d2emb-%j.out

set -uo pipefail
REPO="${REPO:-/home/pengchx3/text-dna/OmegaGenome_Revise_202606/code_carbon}"
cd "$REPO"

if [ -f slurm/env_setup.sh ]; then
  source slurm/env_setup.sh
fi
PY="$REPO/.venv_carbon_portable/bin/python"

# Run entirely off the galaxy SSD (no /extra runtime I/O). SSD root differs per node.
SSD="${SSD:-}"
if [ -z "$SSD" ]; then
  if   [ -d /srv/disk00/sshfs/pengchx3 ];        then SSD=/srv/disk00/sshfs/pengchx3
  elif [ -d /tmp/galaxy_srv_disk00/pengchx3 ];   then SSD=/tmp/galaxy_srv_disk00/pengchx3
  fi
fi
RNT="$SSD/rebuttal_nt"
# NT-2.5B base + HF datasets load from the SSD-staged HF cache (same as the NT-embedding run).
export HF_HOME="$RNT/hf_cache"
export HF_HUB_CACHE="$HF_HOME/hub"
export HUGGINGFACE_HUB_CACHE="$HF_HUB_CACHE"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TRANSFORMERS_CACHE="$HF_HOME/hub"
export HF_ASSETS_CACHE="$HF_HOME/assets"
# DNABERT-2 is loaded from the assembled LOCAL dir (offline); its trust_remote_code .py modules must
# be compiled to a WRITABLE modules cache (the bashrc default points at a read-only /srv path).
export HF_MODULES_CACHE="$RNT/hf_modules_cache"
mkdir -p "$HF_MODULES_CACHE"

NT_PARENT="${NT_PARENT:-$RNT/nt_adapters}"          # NT-2.5B per-task adapters (KD teacher)
CACHE_BASE="${CACHE_BASE:-$RNT/run}"                # logit/feature + embedding cache + outputs
DNABERT2_PATH="${DNABERT2_PATH:-$RNT/dnabert2_local}"  # assembled DNABERT-2 model+code dir
mkdir -p "$CACHE_BASE"

TASK="${TASK:?set TASK=<task_name> via --export}"
RESULTS_CSV="${RESULTS_CSV:-$RNT/run/results/r13_dnabert2_embedding_results.csv}"
BEST_HP="${BEST_HP:-/home/pengchx3/text-dna/OmegaGenome_Revise_202606/rebuttal_infra/best_hp/best_hp_nt.yaml}"
EMB_LAYER="${EMB_LAYER:--999}"   # -999=middle (default), -1=last, >=0 exact index

EXTRA=()
[ "${SMOKE:-0}" = "1" ] && EXTRA+=(--smoke --max-steps "${MAX_STEPS:-200}")
[ -n "${EPOCHS:-}" ] && EXTRA+=(--epochs "$EPOCHS")
[ -n "${MAX_STEPS:-}" ] && [ "${SMOKE:-0}" != "1" ] && EXTRA+=(--max-steps "$MAX_STEPS")
# Embedding front-end arm (R1.3 PI fix): replace4(default)/replaceK/latefuse. Reuses the cache.
[ -n "${FUSION:-}" ] && EXTRA+=(--fusion "$FUSION")
[ -n "${ADAPTER_WIDTH:-}" ] && EXTRA+=(--adapter-width "$ADAPTER_WIDTH")
[ -n "${FUSE_WIDTH:-}" ] && EXTRA+=(--fuse-width "$FUSE_WIDTH")

echo "[$(date)] R1.3 DNABERT-2-embedding distill: TASK=$TASK on $(hostname) GPU=$CUDA_VISIBLE_DEVICES"
echo "SSD=$SSD NT_PARENT=$NT_PARENT CACHE_BASE=$CACHE_BASE DNABERT2_PATH=$DNABERT2_PATH EMB_LAYER=$EMB_LAYER"
echo "RESULTS_CSV=$RESULTS_CSV  EXTRA=${EXTRA[*]:-none}"

"$PY" -m src.train.distill_nt_embedding \
  --task-name "$TASK" \
  --best-hp "$BEST_HP" \
  --results-csv "$RESULTS_CSV" \
  --nt-parent "$NT_PARENT" \
  --cache-base "$CACHE_BASE" \
  --output-dir "$CACHE_BASE/output/r13_dnabert2_embedding" \
  --embedding-source dnabert2 \
  --embedding-layer "$EMB_LAYER" \
  --dnabert2-path "$DNABERT2_PATH" \
  --teacher-batch-size "${TEACHER_BS:-8}" \
  "${EXTRA[@]}"

echo "[$(date)] done TASK=$TASK rc=$?"
