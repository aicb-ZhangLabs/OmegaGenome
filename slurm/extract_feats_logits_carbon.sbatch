#!/bin/bash
# CARBON-3B panel: extract second_to_last FEATURES + class LOGITS for the Carbon-3B LoRA teacher
# + its OmegaGenome feature-aligned BPNet student + the SHARED from-scratch BPNet baseline on the FULL
# splice_sites_all test set. 5th teacher panel alongside enformer/nt/dnabert2/caduceus.
#   sbatch --nodelist=galaxy slurm/extract_feats_logits_carbon.sbatch
# Carbon-3B is ~3B params; loaded in bf16 (~6GB) -> fits galaxy 3090 (24G). The Llama hidden_states
# capture-wrappers are freed per batch (extract_feats_logits_multi._free_capture_wrappers) to avoid OOM.
#SBATCH --job-name=featlogit-carbon
#SBATCH --nodes=1
#SBATCH --nodelist=galaxy
#SBATCH --partition=zhanglab.p
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --requeue
#SBATCH --time=30-00:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail
REPO=${OG_ROOT:-$PWD}
cd "$REPO"

# Node-aware SSD root for the shared HF cache (Carbon-3B base weights). galaxy = local /srv/disk00/sshfs.
for _b in ${OG_SCRATCH:-$PWD/output} ${OG_SCRATCH:-$PWD/output}; do [ -d "$_b" ] && { _SSD="$_b"; break; }; done
if [ -n "${_SSD:-}" ] && [ -d "$_SSD/hf_cache_shared/hub" ]; then
  HOME_HF="$_SSD/hf_cache_shared"
else
  HOME_HF="$REPO/.hf_cache"
fi
export HF_HOME="$HOME_HF" HF_HUB_CACHE="$HOME_HF/hub" HUGGINGFACE_HUB_CACHE="$HOME_HF/hub"
export HF_DATASETS_CACHE="$HOME_HF/datasets" TRANSFORMERS_CACHE="$HOME_HF/hub" HF_ASSETS_CACHE="$HOME_HF/assets"
export HF_XET_CACHE="$HOME_HF/xet" HF_HUB_DISABLE_XET=1
export HF_TOKEN="$(cat ${HF_TOKEN_FILE})"
export TOKENIZERS_PARALLELISM=false WANDB_MODE=offline
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PY="$REPO/.venv_carbon_portable/bin/python"
N="${N:-100000}"

# Carbon-3B feature-aligned student (val MCC 0.885) + shared from-scratch BPNet baseline. The carbon
# panel has a single feature-aligned student -> reuse it for both --omega-ckpt and --dkd-ckpt (the panel
# renders the OmegaGenome row; the DKD row collapses to the same numbers, as for caduceus/dnabert2).
CARBON_STUDENT="${CARBON_STUDENT:-${OG_SCRATCH:-$PWD/output}/carbon_distillation/original/splice_sites_all/20260629_055105/a33a0bbb-3782-474d-b4f8-594587750084/distill_methodvanilla_dkd_alpha1.0_dkd_beta8.0_kl_methodkl_mse_normalizeFalse_temperature4.0_weight_ce0.5_weight_kl0.5_weight_mse1.0_zscoreFalse/best_model/student.pt}"
# galaxy resolves the SSD locally at /srv/disk00/sshfs; remap the sshfs-style path if needed.
if [ ! -f "$CARBON_STUDENT" ] && [ -f "${CARBON_STUDENT/\/tmp\/galaxy_srv_disk00//srv/disk00/sshfs}" ]; then
  CARBON_STUDENT="${CARBON_STUDENT/\/tmp\/galaxy_srv_disk00//srv/disk00/sshfs}"
fi
BASE="${BASE:-${OG_STORE:-$PWD/data}/NT/nucleotide-transformer-student-results-3-11-v4-cnn-layer16-kernel-5-bpnet-4-21-onehot-laniakea/splice_sites_all/model_final.pt}"

echo "=== node / GPU ==="; hostname
"$PY" -c "import torch; print('cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
echo "CARBON_STUDENT=$CARBON_STUDENT"
ls -la "$CARBON_STUDENT" "$BASE"

"$PY" -u analysis/extract_feats_logits_multi.py \
    --teacher carbon --omega-ckpt "$CARBON_STUDENT" --dkd-ckpt "$CARBON_STUDENT" \
    --baseline-ckpt "$BASE" --n "$N"

echo "=== render the carbon panel (CPU, same env) ==="
"$PY" -u analysis/feature_viz.py \
    --teacher carbon --scheme distinct5 \
    --npz "$REPO/analysis/figs/feature_logit_data_carbon.npz"

echo "=== DONE ==="
ls -la "$REPO/analysis/figs/feature_logit_data_carbon.npz" \
       "$REPO/analysis/figs/panel_omega_vs_teacher_carbon_distinct5_hist.png" \
       "$REPO/analysis/figs/cosine_table_panel_carbon.csv"
