#!/bin/bash
# Fan out one SLURM job per NT task (online wandb), spread across free GPUs.
#   bash slurm/submit_all_tasks.sh [carbon_3b]
# Distributes laniakea (A6000 48G, bs8) / galaxy (3090 24G, bs4); avoids voyager.
# Jobs beyond the per-node fair-share cap simply queue and drain automatically.
set -euo pipefail
module load slurm 2>/dev/null || true
EXP="${1:-carbon_3b}"
REPO=${OG_ROOT:-$PWD}
cd "$REPO"

TASKS=(H3K4me3 H3K4me1 H3K4me2 H3K9ac H3K9me3 H3K27ac H3K27me3 H3K36me3 H4K20me1 H2AFZ \
       promoter_all promoter_tata promoter_no_tata enhancers enhancers_types \
       splice_sites_all splice_sites_acceptors splice_sites_donors)

i=0
submitted=0
for t in "${TASKS[@]}"; do
  # idempotent: skip if a job for this task is already queued/running
  if squeue -u "$USER" -h --name="ft-$t" 2>/dev/null | grep -q .; then
    echo "skip ft-$t (already in queue)"; i=$((i + 1)); continue
  fi
  if (( i % 2 == 0 )); then NODE=laniakea; BS=8; else NODE=galaxy; BS=4; fi
  sbatch --job-name="ft-$t" --nodelist="$NODE" \
         slurm/carbon_finetune.sbatch "$EXP" "$t" "$BS"
  submitted=$((submitted + 1)); i=$((i + 1))
done
echo "Submitted $submitted per-task jobs for '$EXP'."
