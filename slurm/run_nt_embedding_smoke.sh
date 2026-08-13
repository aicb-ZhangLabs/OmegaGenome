#!/bin/bash
# R1.3 SMOKE: verify the compact per-token embedding cache writes+persists+reuses end-to-end.
# Runs a REAL (non-smoke) job with use_cache=True but caps training at a few optimizer steps so
# the embedding cache is actually written and a result row is appended quickly. Run twice: the 2nd
# run must print "[emb-cache HIT]" (no recompute). Same env as run_nt_embedding.sh.
#
#SBATCH --job-name=r13-smoke
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --partition=zhanglab.p
#SBATCH --time=04:00:00
#SBATCH --output=slurm/slurm-r13-smoke-%j.out

set -uo pipefail
REPO="${REPO:-/home/pengchx3/text-dna/OmegaGenome_Revise_202606/code_carbon}"
cd "$REPO"
[ -f slurm/env_setup.sh ] && source slurm/env_setup.sh
PY="$REPO/.venv_carbon_portable/bin/python"

SSD="${SSD:-}"
if [ -z "$SSD" ]; then
  if   [ -d /srv/disk00/sshfs/pengchx3 ];      then SSD=/srv/disk00/sshfs/pengchx3
  elif [ -d /tmp/galaxy_srv_disk00/pengchx3 ]; then SSD=/tmp/galaxy_srv_disk00/pengchx3
  fi
fi
RNT="$SSD/rebuttal_nt"
export HF_HOME="$RNT/hf_cache"
export HF_HUB_CACHE="$HF_HOME/hub"
export HUGGINGFACE_HUB_CACHE="$HF_HUB_CACHE"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TRANSFORMERS_CACHE="$HF_HOME/hub"
export HF_ASSETS_CACHE="$HF_HOME/assets"

NT_PARENT="${NT_PARENT:-$RNT/nt_adapters}"
CACHE_BASE="${CACHE_BASE:-$RNT/run}"
TASK="${TASK:-promoter_tata}"
RESULTS_CSV="${RESULTS_CSV:-$RNT/run/results/r13_smoke_results.csv}"
BEST_HP="${BEST_HP:-/home/pengchx3/text-dna/OmegaGenome_Revise_202606/rebuttal_infra/best_hp/best_hp_nt.yaml}"

echo "[$(date)] R1.3 SMOKE TASK=$TASK on $(hostname) GPU=$CUDA_VISIBLE_DEVICES SSD=$SSD"

for run in 1 2; do
  echo "===================== SMOKE RUN $run ====================="
  "$PY" -m src.train.distill_nt_embedding \
    --task-name "$TASK" \
    --best-hp "$BEST_HP" \
    --results-csv "$RESULTS_CSV" \
    --nt-parent "$NT_PARENT" \
    --cache-base "$CACHE_BASE" \
    --output-dir "$CACHE_BASE/output/r13_smoke" \
    --teacher-batch-size "${TEACHER_BS:-8}" \
    --max-steps 3 --epochs 2 --wandb-mode offline
  echo "[$(date)] SMOKE RUN $run rc=$?"
done
echo "[$(date)] SMOKE done. Expect run 2 to print [emb-cache HIT]."
