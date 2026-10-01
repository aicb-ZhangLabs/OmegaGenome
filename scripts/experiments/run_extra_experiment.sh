#!/bin/bash
#SBATCH --job-name=extra_s42
#SBATCH --partition=zhanglab.p
#SBATCH --nodes=1
#SBATCH --nodelist=laniakea
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=20GB
#SBATCH --time=17-00:00:00

source .venv/bin/activate
python -m src.train.distill_nt \
    --task-name splice_sites_all \
    --model-size extra_large_fix \
    --seed 42 \
    --weight-ce 0.5 \
    --weight-kl 0.0 \
    --weight-mse 1.0 \
    --temperature 0.5 \
    --epochs 200 \
    --batch-size 16

python -m src.train.distill_hyperparam nt_bpnet_extra_large_fix_hyperparam_extend
python -m src.train.distill_size_comparison nt_extra_large_fix_only