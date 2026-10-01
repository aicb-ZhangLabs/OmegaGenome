#!/bin/bash
#SBATCH --job-name=ori-ext-solely
#SBATCH --partition=zhanglab.p
#SBATCH --nodes=1
#SBATCH --nodelist=laniakea
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=20GB
#SBATCH --time=17-00:00:00

source .venv/bin/activate


python -m src.train.distill_hyperparam nt_bpnet_original_hyperparam_extend_solely
