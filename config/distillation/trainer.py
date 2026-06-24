from dataclasses import replace
from src.trainer.distill_trainer import DistillTrainerConfig
from ..env import output_path
from .paths import CARBON_OUTPUT_BASE

# Student-checkpoint base (galaxy SSD by default, /home fallback) — machine-specific value lives in
# ONE place (config/distillation/paths.py, env-overridable via CARBON_OUTPUT_BASE).
_CARBON_OUT = f"{CARBON_OUTPUT_BASE}/carbon_distillation"

trainer_config = DistillTrainerConfig(
    output_dir=f"{output_path}/distillation",
    wandb_project="omega_genome",
    epochs=100,
    batch_size=32,
    lr=1e-4,
    max_len=1024,
)

debug_trainer_config = replace(trainer_config, epochs=2, eval_every_n_epochs=1)

nt_trainer_config = DistillTrainerConfig(
    output_dir=f"{output_path}/nt_distillation/vanilla_original",
    wandb_project="OmegaGenome-NT",
    epochs=200,
    batch_size=16,
    lr=1e-4,
    max_len=1000,
)

nt_hyperparam_trainer_config = DistillTrainerConfig(
    output_dir=f"{output_path}/nt_distillation/hyperparam",
    wandb_project="OmegaGenome-NT-HyperParam-finetuned",
    epochs=200,  # Fewer epochs for hyperparameter search
    batch_size=16,
    max_len=1000,
)

nt_hyperparam_dkd_trainer_config = DistillTrainerConfig(
    output_dir=f"{output_path}/nt_distillation/hyperparam_dkd",
    wandb_project="OmegaGenome-NT-DKD-HyperParam",
    epochs=200,
    batch_size=16,
    max_len=1000,
)

nt_different_size_trainer_config = DistillTrainerConfig(
    output_dir=f"{output_path}/nt_distillation/different_size/",
    wandb_project="OmegaGenome-NT-Different-Size",
    epochs=200,
    batch_size=16,
    max_len=1000,
)

nt_different_size_original_trainer_config = DistillTrainerConfig(
    output_dir=f"{output_path}/nt_distillation/different_size/bpnet/original/",
    wandb_project="OmegaGenome-NT-Different-Size",
    epochs=200,
    batch_size=16,
    max_len=1000,
)
nt_debug_trainer_config = replace(nt_trainer_config, epochs=2, eval_every_n_epochs=1)

# Carbon-3B -> deploy_120k BPNet distillation (same 18-task NT-revised data, so max_len=1000).
# Student checkpoints land on the galaxy SSD (_CARBON_OUT), not the degraded /extra NFS.
carbon_trainer_config = DistillTrainerConfig(
    output_dir=f"{_CARBON_OUT}/deploy_120k",
    wandb_project="OmegaGenome-Carbon-Distill",
    epochs=200,
    batch_size=16,
    lr=1e-4,
    max_len=1000,
    cache_base_dir=CARBON_OUTPUT_BASE,  # precompute logits/features cache on SSD, not /extra
    teacher_batch_size=4,  # 3B teacher forward at seq~1000 needs a small batch (attn O(seq^2))
)
carbon_hyperparam_trainer_config = DistillTrainerConfig(
    output_dir=f"{_CARBON_OUT}/hyperparam",
    wandb_project="OmegaGenome-Carbon-Distill-HyperParam",
    epochs=200,
    batch_size=16,
    max_len=1000,
    cache_base_dir=CARBON_OUTPUT_BASE,
    teacher_batch_size=4,
)
carbon_debug_trainer_config = replace(carbon_trainer_config, epochs=2, eval_every_n_epochs=1)
# Carbon-3B distillation into the `original` BPNet student (full receptive field; the SAME student the
# NT/Enformer/Caduceus/DNABERT-2 distillations use). SEPARATE output leaf (`original`) so this re-search
# CANNOT collide with the existing `deploy_120k` ckpts/grid — final_summary.json does not record
# model_size, so a shared dir would corrupt resume/aggregation. Output dir is built from _CARBON_OUT
# (node-aware, resolved at import) — never hardcode the SSD path (galaxy vs sshfs differ). cache_base_dir
# is UNCHANGED so the expensive Carbon-3B teacher logit/feature cache is reused (teacher outputs are
# student-independent).
carbon_original_trainer_config = replace(
    carbon_trainer_config,
    output_dir=f"{_CARBON_OUT}/original",
    wandb_project="OmegaGenome-Carbon-Distill-Original",
)
# Caduceus trainer configurations
caduceus_trainer_config = DistillTrainerConfig(
    output_dir=f"{output_path}/caduceus_distillation/vanilla_original",
    wandb_project="OmegaGenome-Caduceus",
    epochs=200,
    batch_size=32,  # Caduceus can handle larger batch sizes
    lr=1e-4,
    max_len=1024,  # Caduceus uses 1024 sequence length
)

caduceus_hyperparam_trainer_config = DistillTrainerConfig(
    output_dir=f"{output_path}/caduceus_distillation/hyperparam",
    wandb_project="OmegaGenome-Caduceus-HyperParam",
    epochs=200,
    batch_size=32,
    max_len=1024,
)

caduceus_hyperparam_dkd_trainer_config = DistillTrainerConfig(
    output_dir=f"{output_path}/caduceus_distillation/hyperparam_dkd",
    wandb_project="OmegaGenome-Caduceus-DKD-HyperParam",
    epochs=200,
    batch_size=32,
    max_len=1024,
)
caduceus_debug_trainer_config = replace(caduceus_trainer_config, epochs=2, eval_every_n_epochs=1)

# Enformer trainer configurations
enformer_trainer_config = DistillTrainerConfig(
    output_dir=f"{output_path}/enformer_distillation/vanilla_original",
    wandb_project="OmegaGenome-Enformer",
    epochs=200,
    batch_size=16,  # Enformer may need smaller batch size due to memory
    lr=1e-4,
    max_len=1024,  # Enformer uses 1024 sequence length
)

enformer_hyperparam_trainer_config = DistillTrainerConfig(
    output_dir=f"{output_path}/enformer_distillation/hyperparam",
    wandb_project="OmegaGenome-Enformer-HyperParam",
    epochs=200,
    batch_size=16,
    max_len=1024,
)

enformer_debug_trainer_config = replace(enformer_trainer_config, epochs=2, eval_every_n_epochs=1)
