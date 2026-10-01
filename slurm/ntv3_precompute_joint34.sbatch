#!/bin/bash
# Precompute the JOINT 34-track teacher-logit MEMMAP cache in ONE 650M pass over the fixed training
# windows (num_windows 64000 -> ~63707 valid windows). Writes logits.npy (fp16, ~53GB) + meta.pt so the
# from-scratch distilled size sweep can train TEACHER-FREE via --cached_teacher_logits <this dir>.
# One big teacher-bound GPU job (~hours).
#   sbatch slurm/ntv3_precompute_joint34.sbatch
#SBATCH --job-name=loom_ntv3-teachercache
#SBATCH --nodes=1
#SBATCH --nodelist=laniakea
#SBATCH --partition=zhanglab.p
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --requeue
#SBATCH --time=1-00:00:00
#SBATCH --output=slurm-%x-%j.out
set -euo pipefail
REPO=${OG_NTV3_ROOT:-$PWD}
cd "$REPO"
SSD=${OG_SCRATCH:-$PWD/output}
mountpoint -q "$SSD" 2>/dev/null || SSD=${OG_SCRATCH:-$PWD/output}
# HF cache -> SSD archive (models--*/ at its root == the hub); audit-verified offline 650M resolve.
HOME_HF="$SSD/hf_cache_archive_from_code_carbon"
export HF_HOME="$HOME_HF" HF_HUB_CACHE="$HOME_HF" TRANSFORMERS_CACHE="$HOME_HF"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export HF_TOKEN="$(cat ${HF_TOKEN_FILE})"
export TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRITON_CACHE_DIR="/tmp/triton_${USER}_$(hostname -s)"
PV=${OG_ROOT:-$PWD}/.venv_carbon_portable/bin/python
OUT="$SSD/ntv3_targets/teacher_cache_joint34"

# --- PREFLIGHT (lab policy: assert past failure modes before burning GPU) ---
# 1) SSD free space >= 70GB (cache is ~53GB fp16; leave headroom).
avail=$(df -PB1 "$SSD" | awk 'NR==2{print $4}')
need=$((70*1024*1024*1024))
[ "$avail" -ge "$need" ] || { echo "FATAL: SSD free ${avail}B < required ${need}B (70GB)"; exit 1; }
echo "preflight: SSD free $(df -PBG "$SSD" | awk 'NR==2{print $4}') at $SSD"
# 2) teacher ckpt + data present.
[ -f "$SSD/ntv3_targets/ntv3_ft_faithful_s0/best_model.pth" ] || { echo "FATAL: teacher ckpt missing"; exit 1; }
[ -d "$SSD/ntv3_benchmark_data" ] || { echo "FATAL: benchmark data missing"; exit 1; }
# 3) offline HF resolve of the 650M teacher base (fail fast before the long run).
"$PV" - <<'PYCHK'
from huggingface_hub import snapshot_download
p = snapshot_download("InstaDeepAI/NTv3_650M_post", local_files_only=True)
print("preflight: offline 650M resolve OK ->", p)
PYCHK

# JOINT 34-track cache: OMIT --track_subset -> all native tracks, stored track_subset=None (matches a joint student).
"$PV" -u -m src.train.precompute_teacher_logits \
      --data_dir "$SSD/ntv3_benchmark_data" \
      --teacher "$SSD/ntv3_targets/ntv3_ft_faithful_s0/best_model.pth" \
      --tokenizer "$SSD/ntv3_models/8m_pre" \
      --num_windows 64000 --sequence_length 32768 --amp --batch_size 4 --num_workers 8 \
      --memmap_out "$OUT"
echo "=== teacher cache done: $OUT ==="
ls -la "$OUT"
du -sh "$OUT"
