from dataclasses import replace
from ..env import output_path
import torch
from dataclasses import dataclass


@dataclass
class DistillTrainerConfig:
    output_dir: str
    wandb_project: str
    epochs: int = 100
    batch_size: int = 8
    lr: float = 1e-4
    max_len: int = 1024
    log_batch_every: int = 50
    eval_every_n_epochs: int = 5
    num_workers: int = 4
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


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
