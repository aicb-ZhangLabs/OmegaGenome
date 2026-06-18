#!/bin/bash
#SBATCH --job-name=extra_s42
#SBATCH --partition=zhanglab.p
#SBATCH --nodes=1
#SBATCH --nodelist=laniakea
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=20GB
#SBATCH --time=17-00:00:00


python -m src.train.distill_size_comparison nt_extra_large_fix_only