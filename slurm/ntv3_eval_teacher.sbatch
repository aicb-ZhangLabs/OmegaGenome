#!/bin/bash
# Evaluate an NTv3 teacher vs ENCODE ground-truth signal (per-track Pearson) — reproduce the paper's
# teacher numbers + establish the distillation ceiling. Node-flexible. Model passed as $1.
#   sbatch slurm/ntv3_eval_teacher.sbatch InstaDeepAI/NTv3_100M_post
#   sbatch slurm/ntv3_eval_teacher.sbatch InstaDeepAI/NTv3_650M_post
#SBATCH --job-name=ntv3-eval-teacher
#SBATCH --nodes=1
#SBATCH --partition=zhanglab.p
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --nice=2000
#SBATCH --requeue
#SBATCH --time=12:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail
REPO=${OG_NTV3_ROOT:-$PWD}
cd "$REPO"
HOME_HF=${OG_ROOT:-$PWD}/.hf_cache
export HF_HOME="$HOME_HF" HF_HUB_CACHE="$HOME_HF/hub" TRANSFORMERS_CACHE="$HOME_HF/hub"
export HF_TOKEN="$(cat ${HF_TOKEN_FILE})"
PV=${OG_ROOT:-$PWD}/.venv_carbon_portable/bin/python

SSD=${OG_SCRATCH:-$PWD/output}
mountpoint -q "$SSD" 2>/dev/null || SSD=${OG_SCRATCH:-$PWD/output}   # galaxy-native path
MODEL="${1:-InstaDeepAI/NTv3_650M_post}"; [ $# -gt 0 ] && shift   # remaining args -> eval_teacher

"$PV" -m src.train.eval_teacher --model "$MODEL" \
      --bigwig_dir "$SSD/ntv3_targets/ground_truth_bigwigs" \
      --out "$SSD/ntv3_targets/teacher_eval" \
      --window 16384 --n_eval 200 --eval_chrom chr10 --batch_size 2 "$@"
echo "=== done: $SSD/ntv3_targets/teacher_eval/ ==="
