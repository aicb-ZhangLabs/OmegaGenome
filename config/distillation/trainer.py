from src.trainer.distill_trainer import DistillTrainerConfig

trainer_config = DistillTrainerConfig(
    output_dir="models/distillation",
    wandb_project="distillation",
    epochs=100,
    batch_size=8,
    lr=1e-4,
    max_len=1024,
)
