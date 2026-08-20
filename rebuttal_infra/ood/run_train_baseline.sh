#!/bin/bash
# Train ONE missing from-scratch BPNet baseline (default H3K27ac) for the R2.3 OOD baseline matrix.
# Architecture-identical to the distilled student (original BPNet, one-hot, MAX_LEN=1024), PURE CE,
# no teacher. Saves model_final.pt into the existing from-scratch baseline tree so the 18x18 matrix
# picks up a complete 18/18 set.
#
# Usage: sbatch --nodelist=voyager --export=ALL,TASK=H3K27ac run_train_baseline.sh
#
#SBATCH --job-name=ood-train-base
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --partition=zhanglab.p
#SBATCH --time=30-00:00:00
#SBATCH --output=slurm/slurm-ood-train-base-%j.out

set -uo pipefail
REPO="${REPO:-/home/pengchx3/text-dna/OmegaGenome_Revise_202606/code_carbon}"
cd "$REPO"
[ -f slurm/env_setup.sh ] && source slurm/env_setup.sh
PY="$REPO/.venv_carbon_portable/bin/python"
export WANDB_MODE=disabled

# Read HF dataset cache node-local / from SSD (no sshfs-memmap).
SSD=""
if   [ -d /srv/disk00/sshfs/pengchx3 ];      then SSD=/srv/disk00/sshfs/pengchx3
elif [ -d /tmp/galaxy_srv_disk00/pengchx3 ]; then SSD=/tmp/galaxy_srv_disk00/pengchx3
fi
if [ -n "$SSD" ] && [ -d "$SSD/rebuttal_nt/hf_cache/datasets" ]; then
  export HF_DATASETS_CACHE="$SSD/rebuttal_nt/hf_cache/datasets"
fi

TASK="${TASK:-H3K27ac}"
EPOCHS="${EPOCHS:-200}"
# Save into the SAME from-scratch baseline tree (so cross_task_matrix.py's FROM_SCRATCH_ROOT sees 18/18).
OUT_DIR="${OUT_DIR:-/extra/zhanglab0/INDV/pengchx3/NT/nucleotide-transformer-student-results-3-11-v4-cnn-layer16-kernel-5-bpnet-4-21-onehot-laniakea}"

echo "[run] TASK=$TASK EPOCHS=$EPOCHS OUT_DIR=$OUT_DIR host=$(hostname)"
"$PY" -u rebuttal_infra/ood/train_from_scratch_baseline.py \
  --task "$TASK" --out-dir "$OUT_DIR" --epochs "$EPOCHS"
echo "[run] exit=$?"
