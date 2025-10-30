import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import tyro
import json
import wandb
import csv

from datetime import datetime
from dataclasses import replace, asdict
from nntool.slurm import slurm_fn

from config.distillation.config import configs
from config.distillation.config_schema import DistillationExperimentConfig
from config.env import project_output_path
from accelerate.utils import set_seed

from ..model.bpnet_classifier import BPNetClassifier
from ..model.distillation import DistillationModel
from ..data.dataset import (
    get_num_labels,
    build_data_splits_from_huggingface,
)
from ..trainer.distill_trainer import (
    train_distill_task,
    create_run_directory,
    create_run_hyperparams_str,
)

import torch

from ..model.glm import (
    find_teacher_checkpoint,
    get_teacher_model,
    evaluate_and_log_teacher,
)


@slurm_fn
def distill(
    config: DistillationExperimentConfig,
    task_name: str,
):
    print(f"\n{'=' * 80}")
    print(f"=== Starting Distillation: {task_name} ===")
    print(f"{'=' * 80}")

    # Model type detection for display
    model_type = getattr(config, "model_type", "glm")
    print(f"Model Type: {model_type.upper()}")
    print(f"Teacher: {config.teacher_config.model_name_or_path}")
    print(
        f"Student: {config.student_config.model_type}-{config.student_config.model_size}"
    )
    print(f"Method: {config.distillation_config.distill_method}")
    print(f"{'=' * 80}\n")

    set_seed(config.random_state)

    # ===========================================================
    # MODIFIED SECTION: Use unified checkpoint finding
    # ===========================================================
    teacher_ckpt, score = find_teacher_checkpoint(config, task_name)

    if teacher_ckpt is None:
        print(f"[!] No teacher checkpoint found for {task_name}, skipping.")
        return

    print(f"Teacher checkpoint: {teacher_ckpt}")
    if score > 0:
        print(f"Teacher validation MCC: {score:.4f}")

    # ===========================================================
    # MODIFIED SECTION: Use unified teacher model loading
    # ===========================================================
    teacher_tokenizer, teacher_model = get_teacher_model(
        config, task_name, teacher_ckpt
    )
    teacher_model.eval()

    # ===========================================================
    # ORIGINAL CODE: Student model and distillation setup
    # ===========================================================
    # Build student model
    num_labels = get_num_labels(task_name)

    # Handle different teacher model types for hidden size
    if hasattr(teacher_model, "config"):
        teacher_hidden = teacher_model.config.hidden_size
    elif hasattr(teacher_model, "hidden_dim"):  # For wrapped Caduceus models
        teacher_hidden = teacher_model.hidden_dim if teacher_model.hidden_dim else 256
    else:
        teacher_hidden = 768  # Default fallback

    student_config = replace(
        config.student_config,
        num_labels=num_labels,
        teacher_hidden_size=teacher_hidden,
    )
    model = BPNetClassifier(student_config)

    # Build distillation model
    distillation_model = DistillationModel(
        config.distillation_config,
        teacher_model,
        model,
        config.trainer_config.device,
    )

    # Build data splits
    X_train, y_train, X_val, y_val, X_test, y_test = build_data_splits_from_huggingface(
        config.dataset_config
    )

    # Create run directory
    run_dir = create_run_directory(
        config.trainer_config.output_dir,
        task_name,
        config.distillation_config,
    )
    print(f"Run directory: {run_dir}")

    # WandB setup
    hyperparams_str = create_run_hyperparams_str(config.distillation_config)
    model_id = f"{config.student_config.model_type}_{config.student_config.model_size}"
    distill_method = config.distillation_config.distill_method
    run_name = f"{task_name}/{model_id}/{distill_method}/{hyperparams_str}"

    # Add model type to tags
    model_type_tag = getattr(config, "model_type", "glm")

    wandb.init(
        project=config.trainer_config.wandb_project,
        name=run_name,
        dir=project_output_path,
        config=asdict(config),
        notes=f"run_dir: {run_dir}",
        tags=[task_name, model_type_tag, model_id, distill_method],
    )
    wandb.watch(model, log="all", log_freq=100)

    # ===== EVALUATE TEACHER MODEL =====
    teacher_mcc = evaluate_and_log_teacher(
        teacher_model,
        teacher_tokenizer,
        X_test,
        y_test,
        task_name,
        config,
        run_dir,
        teacher_ckpt,
    )

    # Save hyperparameters
    hyperparams = asdict(config)
    hyperparams["run_dir"] = run_dir
    hyperparams["timestamp"] = datetime.now().isoformat()
    hyperparams["teacher_type"] = model_type_tag
    hyperparams["teacher_checkpoint"] = teacher_ckpt
    hyperparams["teacher_test_mcc"] = float(teacher_mcc)
    if score > 0:
        hyperparams["teacher_val_score"] = score

    with open(os.path.join(run_dir, "hyperparameters.json"), "w") as f:
        json.dump(hyperparams, f, indent=2)

    # Train
    train_distill_task(
        config.trainer_config,
        config.distillation_config,
        task_name,
        teacher_tokenizer,
        teacher_model,
        model,
        distillation_model,
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        run_dir,
        resume_from_checkpoint=config.resume_checkpoint,  # Read from config
        resume_from_epoch=config.resume_epoch,  # Read from config
    )

    wandb.finish()


def main(config: DistillationExperimentConfig):
    for task_name in config.task_names:
        # Update dataset config with current task
        distill_dataset_config = replace(config.dataset_config, task_name=task_name)
        distill_config = replace(config, dataset_config=distill_dataset_config)

        # Run distillation with SLURM if configured
        distill[distill_config.slurm_config](distill_config, task_name)


if __name__ == "__main__":
    config = tyro.extras.overridable_config_cli(configs, sort_subcommands=True)
    main(config)
