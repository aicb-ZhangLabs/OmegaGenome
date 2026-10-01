#!/bin/bash
# Faithful NTv3 benchmark fine-tune — direct port of the official notebook
# 03_fine_tuning_posttrained_model_biwig.ipynb ("reproduce paper results"). Full-FT of 650M at
# 32,768 bp, eff. batch 32, ~20.9B tokens, ~28h on an H100 (voyager). Reference human mean Pearson ~0.605.
#   sbatch slurm/ntv3_finetune.sbatch                 # full faithful run (voyager H100)
#   sbatch slurm/ntv3_finetune.sbatch --use_lora      # cheap LoRA variant
# Extra args pass through to src.train.finetune_ntv3 (e.g. --smoke, --sequence_length 4096).
#SBATCH --job-name=ntv3-finetune
#SBATCH --nodes=1
#SBATCH --nodelist=voyager,laniakea
#SBATCH --partition=zhanglab.p
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --nice=0
#SBATCH --requeue
#SBATCH --time=30-00:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail
REPO=${OG_NTV3_ROOT:-$PWD}
cd "$REPO"
# HF cache: the InstaDeepAI/NTv3_* snapshots were archived off .hf_cache to the SSD archive dir; the
# models--*/ dirs sit at its root, so the archive dir IS the hub. Point HF there so the offline 650M
# teacher load resolves. Verified: NTv3_650M_post resolves and matches the live teacher.
SSD=${OG_SCRATCH:-$PWD/output}
mountpoint -q "$SSD" 2>/dev/null || SSD=${OG_SCRATCH:-$PWD/output}
HOME_HF="$SSD/hf_cache_archive_from_code_carbon"
export HF_HOME="$HOME_HF" HF_HUB_CACHE="$HOME_HF" TRANSFORMERS_CACHE="$HOME_HF"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1  # read the cached models, never WRITE to /home (quota)
export HF_TOKEN="$(cat ${HF_TOKEN_FILE})"
export TOKENIZERS_PARALLELISM=false
export TRITON_CACHE_DIR="/tmp/triton_${USER}_$(hostname -s)"  # node-local -> each node compiles its own kernels (galaxy GLIBC fix)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PV=${OG_ROOT:-$PWD}/.venv_carbon_portable/bin/python

DATA="$SSD/ntv3_benchmark_data"  # SSD resolved above (with the HF-cache archive)
# Keep wandb OFF /home (its per-user quota is what killed the wave of jobs: wandb's stdout-capture write
# raised OSError 122 "Disk quota exceeded"). Write run files to the SSD and disable console capture so a
# print() can never go through wandb's quota-bound writer.
export WANDB_DIR="$SSD/wandb_logs" WANDB_CACHE_DIR="$SSD/wandb_cache" WANDB_CONSOLE=off
mkdir -p "$WANDB_DIR" "$WANDB_CACHE_DIR"
STAGE="${SLURM_TMPDIR:-/tmp/ntv3_stage_$USER}"

# MODEL/RUNTAG env-overridable: default = 650M teacher reproduction; set MODEL=<8m_pre dir> +
# RUNTAG=ntv3_8m_baseline for the 8M-from-pretrain baseline (same faithful recipe, smaller backbone).
MODEL="${MODEL:-InstaDeepAI/NTv3_650M_post}"
RUNTAG="${RUNTAG:-ntv3_finetune_faithful}"
# torch.compile uses triton, whose cached .so needs GLIBC 2.34 (absent on galaxy). Set NO_COMPILE=1 to run
# eager (slower but works on galaxy). Default = compile (voyager/laniakea have the newer GLIBC).
COMPILE_FLAG="--compile"; [ "${NO_COMPILE:-0}" = "1" ] && COMPILE_FLAG=""
"$PV" -u -m src.train.finetune_ntv3 --data_dir "$DATA" --model "$MODEL" \
      --out "$SSD/ntv3_targets/$RUNTAG" --stage_dir "$STAGE" --wandb $COMPILE_FLAG "$@"
echo "=== done: $SSD/ntv3_targets/$RUNTAG/ntv3_finetune_result.json ==="
