#!/bin/bash
# Full-parameter Carbon-3B fine-tune over all 18 NT tasks as a SLURM job array.
#
# Polite scheduling (so other users are never blocked) — via priority, not a hard cap:
#   --nice=2000   -> low priority; other users' jobs schedule ahead, and on this (preempting)
#                    cluster a low-priority running task yields when someone else needs the GPU.
#   --requeue     -> a preempted task auto-restarts instead of dying.
#   --array=0-17  -> no concurrency throttle; voyager's 4 H100 is the real ceiling. SLURM runs
#                    the next task automatically as a slot frees (dynamic, single queue entry).
# To cap concurrency instead (e.g. always leave H100 free), append %N: --array=0-17%2.
# Full-FT is H100-only (needs ~50GB optimizer/weights/grads), hence --nodelist=voyager.
#   sbatch slurm/carbon_fullft_array.sbatch
#SBATCH --job-name=ft-fullft
#SBATCH --array=0-17
#SBATCH --nodelist=voyager
#SBATCH --partition=zhanglab.p
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=110G
#SBATCH --nice=2000
#SBATCH --requeue
#SBATCH --time=30-00:00:00
#SBATCH --output=slurm-%x-%a-%j.out

set -euo pipefail
REPO=/home/pengchx3/text-dna/OmegaGenome_Revise_202606/code_carbon
cd "$REPO"
source slurm/env_setup.sh
export OMEGA_FINETUNE_LOCAL=1

TASKS=(H3K4me3 H3K4me1 H3K4me2 H3K9ac H3K9me3 H3K27ac H3K27me3 H3K36me3 H4K20me1 H2AFZ \
       promoter_all promoter_tata promoter_no_tata enhancers enhancers_types \
       splice_sites_all splice_sites_acceptors splice_sites_donors)
T="${TASKS[$SLURM_ARRAY_TASK_ID]}"
echo "array task $SLURM_ARRAY_TASK_ID -> full-FT $T"
"$PY" -m src.train.finetune_teacher carbon_3b_fullft --task-names "$T"
