#!/bin/bash
# NTv3 multi-track distillation: local 100m_post teacher -> per-bp targets -> DilatedTrackNet student.
# Reuses src/train/ntv3_gen_targets.py (teacher) + src/train/distill_tracks.py (student); NO python edits.
# Parameterized via env so one script serves smoke / full and the 8M / 1M students:
#   N_TRAIN, N_TEST  (windows; smoke=24/8, full=2000/400)
#   HIDDEN, NBLOCKS  (8M = 352/16, 1M = 128/14)
#   TAG              (output subdir under <ssd>/ntv3_targets/)
#   MODEL            (teacher; default local 100m_post)
#   LOSS             (mse+pearson default — matches per-track Pearson eval)
# Usage:  N_TRAIN=24 N_TEST=8 TAG=smoke sbatch slurm/ntv3_distill_dilated_v1.sbatch
#         HIDDEN=352 NBLOCKS=16 TAG=dilated8M sbatch slurm/ntv3_distill_dilated_v1.sbatch
#SBATCH --job-name=ntv3-distill-dil
#SBATCH --nodes=1
#SBATCH --partition=zhanglab.p
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=32G
#SBATCH --nice=2000
#SBATCH --requeue
#SBATCH --time=06:00:00
#SBATCH --output=slurm-%x-%j.out
set -euo pipefail
REPO=/home/pengchx3/text-dna/OmegaGenome_Revise_202606/code_ntv3
cd "$REPO"
PY="$REPO/.venv_carbon_portable/bin/python"
[ -x "$PY" ] || PY=/home/pengchx3/text-dna/OmegaGenome_Revise_202606/code_carbon/.venv_carbon_portable/bin/python

# HF cache on home (model load); no gated download needed for the local teacher
HOME_HF=/home/pengchx3/text-dna/OmegaGenome_Revise_202606/code_carbon/.hf_cache
export HF_HOME="$HOME_HF" TOKENIZERS_PARALLELISM=false

# node-aware: teacher + outputs on the galaxy SSD if mounted here, else /extra + home
for _b in /tmp/galaxy_srv_disk00/pengchx3 /srv/disk00/sshfs/pengchx3; do [ -d "$_b/ntv3_local/100m_post" ] && { SSD="$_b"; break; }; done
MODEL="${MODEL:-${SSD:+$SSD/ntv3_local/100m_post}}"; MODEL="${MODEL:-/extra/zhanglab0/INDV/pengchx3/ntv3_local/100m_post}"
OUTBASE="${SSD:+$SSD/ntv3_targets}"; OUTBASE="${OUTBASE:-$REPO/.ntv3_run}"
TAG="${TAG:-dilated8M}"; OUT="$OUTBASE/$TAG"; mkdir -p "$OUT"

N_TRAIN="${N_TRAIN:-2000}"; N_TEST="${N_TEST:-400}"
HIDDEN="${HIDDEN:-352}"; NBLOCKS="${NBLOCKS:-16}"; LOSS="${LOSS:-mse+pearson}"
echo "teacher=$MODEL  out=$OUT  windows=$N_TRAIN/$N_TEST  student=dilated h$HIDDEN/nb$NBLOCKS  loss=$LOSS"

echo "=== 1. generate NTv3 100m_post per-bp targets ==="
"$PY" -m src.train.ntv3_gen_targets --model "$MODEL" --out "$OUT" \
      --window 16384 --n_train "$N_TRAIN" --n_test "$N_TEST" --batch_size 2

echo "=== 2. distill -> DilatedTrackNet (h$HIDDEN/nb$NBLOCKS) + per-track Pearson ==="
"$PY" -m src.train.distill_tracks --data "$OUT" --out "$OUT/student_h${HIDDEN}_nb${NBLOCKS}" \
      --student dilated --hidden "$HIDDEN" --n_blocks "$NBLOCKS" --loss "$LOSS" \
      --epochs 60 --lr 1e-3 --batch_size 8
echo "=== done -> $OUT/student_h${HIDDEN}_nb${NBLOCKS}/track_distill_result.json ==="
