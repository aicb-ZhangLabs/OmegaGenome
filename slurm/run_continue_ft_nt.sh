#!/bin/bash
# R2.1.b rebuttal: CONTINUE the no-LoRA FULL fine-tune of NT-2.5B on promoter_tata to test
# whether the teacher improves past eval MCC 0.8885 / test MCC 0.8774 with more epochs.
#
# Single-GPU full FT (no DeepSpeed): a 2.5B ESM + Adam + grad-checkpointing fits on one
# 49GB (laniakea 6000Ada) / 80GB (voyager H100) GPU. Functionally identical to the original
# 4-GPU ZeRO-3 run; queues far faster on a congested cluster. Output is a plain HF model
# loadable by rebuttal_infra/eval_fullft_nt.py.
#
# Usage:
#   sbatch --nodelist=laniakea slurm/run_continue_ft_nt.sh
#   sbatch --nodelist=voyager  --export=ALL,EPOCHS=20 slurm/run_continue_ft_nt.sh
#   sbatch --nodelist=laniakea --export=ALL,SMOKE=1 slurm/run_continue_ft_nt.sh   # 1-step smoke
#
#SBATCH --job-name=r21-ntcontft
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --partition=zhanglab.p
#SBATCH --time=30-00:00:00
#SBATCH --output=slurm/slurm-r21-ntcontft-%j.out

set -uo pipefail
REPO="${REPO:-/home/pengchx3/text-dna/OmegaGenome_Revise_202606/code_carbon}"
cd "$REPO"

if [ -f slurm/env_setup.sh ]; then
  source slurm/env_setup.sh
fi
PY="$REPO/.venv_carbon_portable/bin/python"

# Run entirely off /extra + node-agnostic HF cache (env_setup pins $REPO/.hf_cache). The dataset
# is tiny (promoter_tata: ~5k train / 212 test) so NFS I/O is a non-issue; we deliberately avoid
# the sshfs SSD here (memmap-over-sshfs hangs on laniakea/voyager).
echo "[$(date)] R2.1.b continue-FT NT-2.5B promoter_tata on $(hostname) GPU=$CUDA_VISIBLE_DEVICES"

EPOCHS="${EPOCHS:-15}"
RESUME_FROM="${RESUME_FROM:-/extra/zhanglab0/INDV/pengchx3/NT/2b5-multi-species_nucleotide-transformer-finetune-results-NO-LORA-MULTI-GPU-epoch10-3-22-revised-deepspeed/finetuned_models/promoter_tata_finetuned/final_model}"
OUTPUT_DIR="${OUTPUT_DIR:-/extra/zhanglab0/INDV/pengchx3/NT/2b5-multi-species_nucleotide-transformer-finetune-results-NO-LORA-MULTI-GPU-epoch10-3-22-revised-deepspeed-promoter_tata-continued/finetuned_models/promoter_tata_finetuned}"

EXTRA=()
[ "${SMOKE:-0}" = "1" ] && EXTRA+=(--smoke)

echo "RESUME_FROM=$RESUME_FROM"
echo "OUTPUT_DIR=$OUTPUT_DIR  EPOCHS=$EPOCHS  EXTRA=${EXTRA[*]:-none}"

"$PY" rebuttal_infra/continue_ft_nt.py \
  --task promoter_tata \
  --resume-from "$RESUME_FROM" \
  --output-dir "$OUTPUT_DIR" \
  --epochs "$EPOCHS" \
  "${EXTRA[@]}"

echo "[$(date)] done rc=$?"
