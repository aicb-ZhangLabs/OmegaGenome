#!/bin/bash
# (b) 10M distilled BPNet (C=621, ~10.5M) from CACHED NT-2.5B teacher targets (no teacher load).
# One SLURM job = one task. ce0.5/kl0.5/mse0.2 KD. Reuses distill_nt_embedding.py --teacher-target-dir.
#SBATCH --job-name=distill10m
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --partition=zhanglab.p
#SBATCH --time=2-00:00:00
#SBATCH --output=slurm/slurm-distill10m-%j.out
set -uo pipefail
REPO="${REPO:-/home/pengchx3/text-dna/OmegaGenome_Revise_202606/code_carbon}"
cd "$REPO"
PY="$REPO/.venv_carbon_portable/bin/python"
if   [ -d /srv/disk00/sshfs/pengchx3 ];      then SSD=/srv/disk00/sshfs/pengchx3
elif [ -d /tmp/galaxy_srv_disk00/pengchx3 ]; then SSD=/tmp/galaxy_srv_disk00/pengchx3; fi
RNT="$SSD/rebuttal_nt"
export PYTHONUNBUFFERED=1 SKIP_TEACHER_EVAL=1 WANDB_MODE=disabled WANDB_DISABLED=true
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_DATASETS_TRUST_REMOTE_CODE=1
export HF_HOME="$RNT/hf_cache" HF_HUB_CACHE="$RNT/hf_cache/hub" HUGGINGFACE_HUB_CACHE="$RNT/hf_cache/hub"
export HF_DATASETS_CACHE="$RNT/hf_cache/datasets" TRANSFORMERS_CACHE="$RNT/hf_cache/hub"
TASK="${TASK:?set TASK}"
YAML="${YAML:-/home/pengchx3/text-dna/OmegaGenome_Revise_202606/rebuttal_infra/best_hp/best_hp_nt_distill10m.yaml}"
TCACHE="${TCACHE:-/home/pengchx3/text-dna/OmegaGenome_Revise_202606/code_all_versions/OmegaGenome_1-27-clean/OmegaGenome/data/cache/2b5-multi-species_nucleotide-transformer-finetune-results-lora-epoch20-10-17-revised-r32-fix-num-label}"
WRITE="${WRITE:-$RNT/run_fromscratch_ntbase}"
RESULTS_CSV="${RESULTS_CSV:-$WRITE/results/distill10m_${TASK}.csv}"
echo "[$(date)] distill10m TASK=$TASK on $(hostname) GPU=$CUDA_VISIBLE_DEVICES"
$PY -m src.train.distill_nt_embedding --task-name "$TASK" --best-hp "$YAML" \
    --results-csv "$RESULTS_CSV" --nt-parent "$RNT/nt_adapters" --cache-base "$WRITE" \
    --output-dir "$WRITE/output/distill10m" --input-mode onehot --teacher-target-dir "$TCACHE" \
    --num-workers 0 ${PATIENCE:+--early-stop-patience $PATIENCE}
echo "[$(date)] done TASK=$TASK rc=$?"
