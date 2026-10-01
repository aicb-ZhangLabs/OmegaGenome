#!/bin/bash
# OOD -- TEACHER cross-task transfer matrix (INFERENCE ONLY, no training). One GPU job per
# foundation model: loads each of the 18 task-specific teachers and runs argmax inference over every
# label-compatible task's HF test set -> the full A x B MCC (+acc) matrix.
#
# Usage:
#   sbatch --nodelist=laniakea --export=ALL,MODEL=caduceus analysis/ood/run_teacher_cross_task_matrix.sh
#   SMOKE=1 N=256 TASKS=H2AFZ,H3K27ac ...   (subset + cap, quick config check)
#
#SBATCH --job-name=ood-teacher-xtask
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --partition=zhanglab.p
#SBATCH --time=30-00:00:00
#SBATCH --output=logs/slurm-ood-teacher-xtask-%j.out

set -uo pipefail
REPO="${REPO:-${OG_ROOT:-$PWD}}"
cd "$REPO"
[ -f slurm/env_setup.sh ] && source slurm/env_setup.sh
PY="$REPO/.venv_carbon_portable/bin/python"

export WANDB_MODE=disabled   # inference only

# Prefer an SSD-staged HF dataset cache if present (avoids sshfs-memmap); else env_setup's in-repo cache.
SSD=""
if   [ -d ${OG_SCRATCH:-$PWD/output} ];      then SSD=${OG_SCRATCH:-$PWD/output}
elif [ -d ${OG_SCRATCH:-$PWD/output} ]; then SSD=${OG_SCRATCH:-$PWD/output}
fi
if [ -n "$SSD" ] && [ -d "$SSD/nt_runs/hf_cache/datasets" ]; then
  export HF_DATASETS_CACHE="$SSD/nt_runs/hf_cache/datasets"
  echo "[env] using SSD HF_DATASETS_CACHE=$HF_DATASETS_CACHE"
fi

MODEL="${MODEL:-caduceus}"          # nt | dnabert2 | enformer | caduceus | carbon
N="${N:-0}"                          # 0 = full test sets
BATCH_SIZE="${BATCH_SIZE:-8}"
[ "${SMOKE:-0}" = "1" ] && N="${N:-256}"

# bf16 is baked into MODEL_SPECS for nt/carbon; this PYTORCH alloc knob just eases their large-teacher
# fragmentation. No-op for the small teachers.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

EXTRA=()
[ "${SMOKE:-0}" = "1" ] && EXTRA+=(--smoke)
[ -n "${TASKS:-}" ] && EXTRA+=(--tasks "$TASKS")
[ -n "${MAX_LEN:-}" ] && EXTRA+=(--max-len "$MAX_LEN")
[ -n "${STRICT_DIAG:-}" ] && EXTRA+=(--strict-diag)

echo "[run] model=$MODEL N=$N bs=$BATCH_SIZE host=$(hostname) py=$PY"
"$PY" -u analysis/ood/teacher_cross_task_matrix.py \
  --model "$MODEL" \
  --n "$N" \
  --batch-size "$BATCH_SIZE" \
  "${EXTRA[@]}" \
  --out "$REPO/analysis/ood"
echo "[run] exit=$?"
