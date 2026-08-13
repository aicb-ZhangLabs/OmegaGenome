#!/bin/bash
# Minimal-NT linear probe: NT-2.5B FROZEN token (word) embedding -> mean-pool -> Linear, from scratch.
# One SLURM job = one task. Needs only the precomputed [vocab,2560] word-embedding tensor (WORDEMB) +
# the (cached) NT tokenizer + the (cached) task dataset -- NO 2.5B model load, so this is light.
#SBATCH --job-name=nttokprobe
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --partition=zhanglab.p
#SBATCH --time=1-00:00:00
#SBATCH --output=slurm/slurm-nttokprobe-%j.out

set -uo pipefail
REPO="${REPO:-/home/pengchx3/text-dna/OmegaGenome_Revise_202606/code_carbon}"
cd "$REPO"
PY="$REPO/.venv_carbon_portable/bin/python"

if   [ -d /srv/disk00/sshfs/pengchx3 ];      then SSD=/srv/disk00/sshfs/pengchx3
elif [ -d /tmp/galaxy_srv_disk00/pengchx3 ]; then SSD=/tmp/galaxy_srv_disk00/pengchx3
fi
RNT="$SSD/rebuttal_nt"
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_DATASETS_TRUST_REMOTE_CODE=1
export HF_HOME="$RNT/hf_cache" HF_HUB_CACHE="$RNT/hf_cache/hub" HUGGINGFACE_HUB_CACHE="$RNT/hf_cache/hub"
export HF_DATASETS_CACHE="$RNT/hf_cache/datasets" TRANSFORMERS_CACHE="$RNT/hf_cache/hub" HF_ASSETS_CACHE="$RNT/hf_cache/assets"

TASK="${TASK:?set TASK=<task_name> via --export}"
WORDEMB="${WORDEMB:-$RNT/nt2p5b_word_embedding.pt}"
RESULTS_CSV="${RESULTS_CSV:-$RNT/run_fromscratch_ntbase/results/${CSV_TAG:-nt_tokenemb_linear}_${TASK}.csv}"

# (e) FEATURE_CACHE_REL set -> probe the FINETUNED-NT mid-layer feature cache (resolved on THIS node's SSD,
# so the path is correct whether SSD is native /srv/disk00 or the sshfs mount). No word-embedding needed.
if [ -n "${FEATURE_CACHE_REL:-}" ]; then
    SRC_ARG="--feature-cache $SSD/$FEATURE_CACHE_REL/$TASK"
    echo "[$(date)] nt-feature-probe TASK=$TASK on $(hostname) GPU=$CUDA_VISIBLE_DEVICES"
    echo "FEATURE_CACHE=$SSD/$FEATURE_CACHE_REL/$TASK RESULTS_CSV=$RESULTS_CSV"
    [ -f "$SSD/$FEATURE_CACHE_REL/$TASK/train_mid_token_embeddings.npy" ] || { echo "ERROR: feature cache missing"; exit 2; }
else
    SRC_ARG="--wordemb $WORDEMB"
    echo "[$(date)] nt-tokenemb-probe TASK=$TASK on $(hostname) GPU=$CUDA_VISIBLE_DEVICES"
    echo "WORDEMB=$WORDEMB RESULTS_CSV=$RESULTS_CSV"
    [ -f "$WORDEMB" ] || { echo "ERROR: WORDEMB not found: $WORDEMB"; exit 2; }
fi

$PY -m src.train.nt_tokenemb_linear_probe \
    --task-name "$TASK" --results-csv "$RESULTS_CSV" $SRC_ARG \
    ${EPOCHS:+--epochs $EPOCHS} ${PATIENCE:+--patience $PATIENCE} ${LR:+--lr $LR} \
    ${SEED:+--seed $SEED} ${SMOKE:+--smoke} ${EXTRA:-}
echo "[$(date)] done TASK=$TASK rc=$?"
