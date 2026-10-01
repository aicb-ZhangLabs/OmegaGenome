#!/bin/bash
# PER-ASSAY multi-track KD-vs-baseline (multiple biological contexts): train ONE 8M student on all tracks of
# ONE assay family, both distilled-from-teacher (KD) and from-scratch (baseline, data-matched), for each
# of the 5 families. The exact recipe/hardening of slurm/ntv3_size_sweep_pertrack_t12{,_nokd}.sbatch,
# generalized from --track_subset 12 (1 track) to --track_subset "<assay idxs>" (N tracks).
#
# array idx -> (assay, mode, seed):  assay = idx%5, mode = (idx//5)%2 (0=kd,1=base), seed = idx//10.
#   assays: 0=atac(5) 1=histone(4) 2=rnaseq(5) 3=procap(10) 4=eclip(10).  30 jobs = 5x2x3.
# KD arm  : --cached_teacher_logits <assay cache .pt> (built by scripts/build_assay_cache_from_joint.py),
#           kd_w_ce .5 / kd_w_kl .5 / kd_w_mse 0, poisson_multinomial + standardized_mse (== per-track KD).
# BASE arm: NO teacher; --train_overlap 0 --max_train_samples 64000 => the SAME 63,707-window stream as KD
#           (data-matched, so the ONLY difference is teacher on/off). kd_w_* are inert on the no-teacher path.
#
# Submit (after caches are built + a dry-run passes):
#   sbatch --array=0-29%8 slurm/ntv3_assay_experiment.sbatch          # full: 5 assay x {kd,base} x 3 seed
#   sbatch --export=ALL,NTV3_DRY_RUN=1 --array=0 --time=03:00:00 slurm/ntv3_assay_experiment.sbatch  # dry
#SBATCH --job-name=loom_ntv3-assay
#SBATCH --array=0-29%14
#SBATCH --nodes=1
# galaxy INCLUDED: the 8M student at seq-32768 fits the 3090's 24G (dry-run verified 2026-08-17); its 6
# GPUs were idle. (The galaxy exclusion only applied to the larger 100M-300M size-ladder tiers.)
#SBATCH --partition=zhanglab.p
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --nice=200
#SBATCH --requeue
#SBATCH --time=30-00:00:00
#SBATCH --output=slurm-loom_ntv3-assay-%a-%A.out

REPO=${OG_NTV3_ROOT:-$PWD}
cd "$REPO"
source "$REPO/slurm/_mount_guard.sh"
source "$REPO/slurm/_hang_watchdog.sh"

# ---- decode array idx -> (assay, mode, seed) ----
T=${SLURM_ARRAY_TASK_ID:-0}
ASSAY_I=$(( T % 5 )); MODE_I=$(( (T / 5) % 2 )); SEED=$(( T / 10 ))
ASSAYS=(atac histone rnaseq procap eclip)
IDXS=("12,13,16,21,27" "22,24,30,33" "17,18,19,20,23" "0,1,2,3,4,5,25,26,31,32" "6,7,8,9,10,11,14,15,28,29")
ASSAY=${ASSAYS[$ASSAY_I]}; IDX=${IDXS[$ASSAY_I]}
[ "$MODE_I" -eq 0 ] && MODE=kd || MODE=base

SSD="$(resolve_ssd_mount ntv3_benchmark_data/human/genome.fasta)" || exit 1
export TMPDIR="/tmp/ntv3_sock_${USER}_${SLURM_JOB_ID:-$$}"; mkdir -p "$TMPDIR"
trap 'hang_watchdog_stop; rm -rf "$TMPDIR"' EXIT
HOME_HF="$SSD/hf_cache_archive_ntv3_assay"; mkdir -p "$HOME_HF"   # tokenizer/module cache dir (created at runtime)
export HF_HOME="$HOME_HF" HF_HUB_CACHE="$HOME_HF" TRANSFORMERS_CACHE="$HOME_HF"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export HF_TOKEN="$(cat ${HF_TOKEN_FILE})"
export TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRITON_CACHE_DIR="$SSD/triton_cache_${USER}_$(hostname -s)"; mkdir -p "$TRITON_CACHE_DIR"
export WANDB_MODE=offline WANDB_DIR="$SSD/wandb_logs" WANDB_CACHE_DIR="$SSD/wandb_cache"; mkdir -p "$WANDB_DIR" "$WANDB_CACHE_DIR"
PV=${OG_ROOT:-$PWD}/.venv_carbon_portable/bin/python

scontrol update JobId="${SLURM_JOB_ID}" JobName="loom_ntv3-assay-${ASSAY}-${MODE}-s${SEED}" 2>/dev/null || true
OUT="$SSD/ntv3_targets/assay_experiment/${ASSAY}/8m_${MODE}_s${SEED}"
CACHE="$SSD/ntv3_cache/teacher_logits/assay/${ASSAY}.pt"
[ -n "${NTV3_DRY_RUN:-}" ] && { OUT="${OUT%/*}/_dryrun/${ASSAY}_${MODE}_s${SEED}"; export NTV3_DRY_RUN; }
echo "=== [array $T] assay=$ASSAY idx=$IDX mode=$MODE seed=$SEED -> $OUT (host $(hostname -s)) ==="

GUARD_PATHS=("$SSD/ntv3_benchmark_data/human/genome.fasta"
             "$SSD/ntv3_benchmark_data/human/genome.fasta.fai"
             "$SSD/ntv3_benchmark_data/human/functional_tracks"
             "${OG_STORE:-$PWD/data}/ntv3_local/8m_pre")   # random-init 8M template (modeling+tokenizer)
if [ "$MODE" = kd ]; then
  [ -f "$CACHE" ] || { echo "FATAL: assay KD cache missing at $CACHE (build with build_assay_cache_from_joint.py)"; exit 1; }
  GUARD_PATHS+=("$CACHE")
fi
mount_preflight "${GUARD_PATHS[@]}" || exit 1

# common args
COMMON=(--data_dir "$SSD/ntv3_benchmark_data" --model ntv3-8m --track_subset "$IDX"
        --kd_gt_loss poisson_multinomial --kd_multinomial_weight 5.0
        --sequence_length 32768 --mini_batch_size 4 --num_accumulation_gradient 8
        --num_steps_training 19932 --num_steps_warmup 598 --num_workers 8 --seed "$SEED" --amp --out "$OUT")
if [ "$MODE" = kd ]; then
  MODE_ARGS=(--cached_teacher_logits "$CACHE" --kd_w_ce 0.5 --kd_w_kl 0.5 --kd_w_mse 0.0 --kd_distill_loss standardized_mse)
else
  MODE_ARGS=(--train_overlap 0 --max_train_samples 64000 --kd_w_ce 1.0 --kd_w_kl 0 --kd_w_mse 0)
fi
DRY=(); [ "${NTV3_DRY_RUN:-0}" = "1" ] && DRY=(--dry_run)   # 1 batch train->val->test, proves the queued config

hang_watchdog_start
set +e
"$PV" -u -m src.train.finetune_ntv3 "${COMMON[@]}" "${MODE_ARGS[@]}" "${DRY[@]}"
rc=$?
set -e
requeue_if_mount_failure "$rc" "${GUARD_PATHS[@]}" || exit "$rc"
echo "=== done $ASSAY/$MODE/s$SEED: $OUT/ntv3_finetune_result.json ==="
