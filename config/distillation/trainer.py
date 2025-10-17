from dataclasses import replace
from src.trainer.distill_trainer import DistillTrainerConfig
from ..env import output_path

trainer_config = DistillTrainerConfig(
    output_dir=f"{output_path}/distillation",
    wandb_project="omega_genome",
    epochs=100,
    batch_size=32,
    lr=1e-4,
    max_len=1024,
)

debug_trainer_config = replace(trainer_config, epochs=2, eval_every_n_epochs=1)
