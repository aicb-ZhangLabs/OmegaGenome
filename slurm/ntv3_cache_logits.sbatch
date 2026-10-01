#!/bin/bash
# Build per-track generalist teacher LOGIT caches from ONE teacher pass (src.train.precompute_teacher_logits
# --save_per_track). Caching N tracks costs the SAME single 650M forward as caching one — the generalist head
# emits all 34 tracks per window. Writes gen/t{idx}.pt (no conflict with existing caches). Args pass through.
#   TRACKS=0,3,14,20 sbatch slurm/ntv3_cache_logits.sbatch
#SBATCH --job-name=ntv3logitcache
#SBATCH --nodes=1
#SBATCH --partition=zhanglab.p
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=slurm-%x-%j.out
set -euo pipefail
REPO=${OG_NTV3_ROOT:-$PWD}
cd "$REPO"
HOME_HF=${OG_ROOT:-$PWD}/.hf_cache
export HF_HOME="$HOME_HF" HF_HUB_CACHE="$HOME_HF/hub" TRANSFORMERS_CACHE="$HOME_HF/hub"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export HF_TOKEN="$(cat ${HF_TOKEN_FILE})"
export TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PV=${OG_ROOT:-$PWD}/.venv_carbon_portable/bin/python
SSD=${OG_SCRATCH:-$PWD/output}
mountpoint -q "$SSD" 2>/dev/null || SSD=${OG_SCRATCH:-$PWD/output}
TRACKS="${TRACKS:?set TRACKS=comma,sep,indices}"
# TEACHER/OUTSUB env-overridable: default = GENERALIST 34-track teacher -> gen/. For a SPECIALIST teacher set
# TEACHER=<spec ckpt>, OUTSUB=spec, SPEC_FLAG=--teacher_specialist (single-track head; no index-select).
TEACHER="${TEACHER:-$SSD/ntv3_targets/ntv3_ft_faithful_s0/best_model.pth}"
OUTSUB="${OUTSUB:-gen}"
SPEC_FLAG="${SPEC_FLAG:-}"
"$PV" -u -m src.train.precompute_teacher_logits \
      --data_dir "$SSD/ntv3_benchmark_data" --tokenizer "$SSD/ntv3_models/8m_pre" \
      --teacher "$TEACHER" $SPEC_FLAG \
      --track_subset "$TRACKS" --num_windows 64000 --amp --batch_size 4 \
      --save_per_track --out_dir "$SSD/ntv3_cache/teacher_logits/$OUTSUB" "$@"
echo "=== logit cache build done: $OUTSUB/t{$TRACKS}.pt ==="
