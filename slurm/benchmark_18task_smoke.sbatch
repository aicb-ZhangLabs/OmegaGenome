#!/bin/bash
# SMOKE for the 18-task latency+memory sweep: 2 tasks, ALL 5 teachers (NT-2.5B, Caduceus,
# Enformer, DNABERT-2, Carbon-3B) + BPNet `original` student, few timed batches.
# Validates: Carbon-3B loads (LoRA adapter + base, <dna> prefix), per-task breakdown
# (n_test, native_seq_len, extrapolated total_time), gpu_name/cpu_name. laniakea (49G,
# GLIBC 2.35) so Caduceus + the big teachers all run in one process.
#   sbatch slurm/benchmark_18task_smoke.sbatch
#SBATCH --job-name=bench18-smoke
#SBATCH --nodes=1
#SBATCH --partition=zhanglab.p
#SBATCH --nodelist=laniakea
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --requeue
#SBATCH --time=4:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail
REPO=${OG_ROOT:-$PWD}
cd "$REPO"

HOME_HF="$REPO/.hf_cache"
export HF_HOME="$HOME_HF" HF_HUB_CACHE="$HOME_HF/hub" HUGGINGFACE_HUB_CACHE="$HOME_HF/hub"
export HF_DATASETS_CACHE="$HOME_HF/datasets" TRANSFORMERS_CACHE="$HOME_HF/hub" HF_ASSETS_CACHE="$HOME_HF/assets"
export HF_TOKEN="$(cat ${HF_TOKEN_FILE})"
export TOKENIZERS_PARALLELISM=false WANDB_MODE=offline
export HF_HUB_DISABLE_XET=1
export HF_XET_CACHE="$REPO/.hf_cache/xet"
export CARBON_TEACHER_DIR=${CARBON_TEACHER_DIR:-$PWD/data/finetuned_models}/carbon_3b_lora
# SSD output (galaxy SSD via sshfs on laniakea). NEVER /home (quota) or /extra-write.
export OUTPUT_PATH=${OG_SCRATCH:-$PWD/output}/latency_18task/smoke

PY="$REPO/.venv_carbon_portable/bin/python"

echo "=== node + GPU ==="; hostname; ldd --version | head -1
"$PY" -c "import torch; print('device:', torch.cuda.get_device_name(0))"

"$PY" -u -m src.train.benchmark_inference full_benchmark \
    --task-names H3K4me3 splice_sites_all \
    --teachers nt caduceus enformer dnabert2 carbon3b \
    --student-sizes original \
    --precision fp16 \
    --batch-size 16 \
    --max-length 1000 \
    --warmup-batches 2 \
    --num-batches 3 \
    --skip-on-error

echo "=== SMOKE DONE; CSVs under $OUTPUT_PATH/benchmark/ ==="
find "$OUTPUT_PATH/benchmark" -name "*.csv" 2>/dev/null | head
