#!/bin/bash
# R2.3 OOD / cross-task transfer matrix -- INFERENCE ONLY (no training). One tiny GPU job:
# loads each of the 18 distilled BPNet students (0.12M, one-hot input) and runs argmax inference
# over every label-compatible task's HF test set, scoring MCC + accuracy -> the full A x B matrix.
#
# Usage:
#   sbatch --nodelist=laniakea --export=ALL,TEACHER=enformer run_cross_task_matrix.sh
#   SMOKE=1 N=64 ...  (subset, quick config check)
#
#SBATCH --job-name=ood-xtask
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --partition=zhanglab.p
#SBATCH --time=30-00:00:00
#SBATCH --output=slurm/slurm-ood-xtask-%j.out

set -uo pipefail
REPO="${REPO:-/home/pengchx3/text-dna/OmegaGenome_Revise_202606/code_carbon}"
cd "$REPO"
[ -f slurm/env_setup.sh ] && source slurm/env_setup.sh
PY="$REPO/.venv_carbon_portable/bin/python"

# Inference only -- no wandb.
export WANDB_MODE=disabled

# Read the HF dataset cache node-local / from SSD (avoid sshfs-memmap). env_setup pins HF_HOME to the
# in-repo .hf_cache (already populated with the NT-revised dataset). If an SSD-staged copy exists,
# prefer it (faster). No memmap at start -- the loader copies splits into RAM.
SSD=""
if   [ -d /srv/disk00/sshfs/pengchx3 ];      then SSD=/srv/disk00/sshfs/pengchx3
elif [ -d /tmp/galaxy_srv_disk00/pengchx3 ]; then SSD=/tmp/galaxy_srv_disk00/pengchx3
fi
if [ -n "$SSD" ] && [ -d "$SSD/rebuttal_nt/hf_cache/datasets" ]; then
  export HF_DATASETS_CACHE="$SSD/rebuttal_nt/hf_cache/datasets"
  echo "[env] using SSD HF_DATASETS_CACHE=$HF_DATASETS_CACHE"
fi

TEACHER="${TEACHER:-enformer}"
STUDENTS_MODE="${STUDENTS_MODE:-distilled}"   # distilled | baseline | baseline-grid
BASELINE_MANIFEST="${BASELINE_MANIFEST:-$REPO/rebuttal_infra/ood/baseline_pureCE_manifest.csv}"
N="${N:-0}"
[ "${SMOKE:-0}" = "1" ] && N="${N:-64}"

EXTRA=()
[ "$STUDENTS_MODE" = "baseline-grid" ] && EXTRA=(--baseline-manifest "$BASELINE_MANIFEST")

echo "[run] teacher=$TEACHER mode=$STUDENTS_MODE N=$N host=$(hostname) py=$PY"
"$PY" -u rebuttal_infra/ood/cross_task_matrix.py \
  --teacher "$TEACHER" \
  --students-mode "$STUDENTS_MODE" \
  "${EXTRA[@]}" \
  --n "$N" \
  --out "$REPO/rebuttal_infra/ood"
echo "[run] exit=$?"
