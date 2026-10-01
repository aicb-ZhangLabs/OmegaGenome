#!/usr/bin/env python
"""
Simple CLI for running NT distillation experiments.

Usage:
    python -m src.train.distill_nt --task-name splice_sites_all --model-size extra_large_fix --seed 42
"""

import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import json
import wandb

from accelerate.utils import set_seed

from config.env import project_path, project_output_path
from config.distillation.glm import nt_2b5
from config.distillation.experiments.nt import NT_PARENT_PATH

from src.model.bpnet_classifier import BPNetClassifier, BPNetClassifierConfig
from src.model.distillation import DistillationModel, DistillationModelConfig
from src.data.dataset import get_num_labels, build_data_splits_from_huggingface, DatasetConfig
from src.trainer.distill_trainer import (
    train_distill_task,
    create_run_directory,
    create_run_hyperparams_str,
)
from src.model.glm import (
    find_teacher_checkpoint,
    get_teacher_model,
    evaluate_and_log_teacher,
    GLMConfig,
)


def main():
    parser = argparse.ArgumentParser(description="Run NT distillation experiment")

    # Required
    parser.add_argument("--task-name", type=str, required=True)
    parser.add_argument("--model-size", type=str, default="extra_large_fix")
    parser.add_argument("--seed", type=int, default=42)

    # Hyperparameters
    parser.add_argument("--weight-ce", type=float, default=0.5)
    parser.add_argument("--weight-kl", type=float, default=0.0)
    parser.add_argument("--weight-mse", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--distill-method", type=str, default="vanilla")

    # Training
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max-len", type=int, default=1000)

    # Output
    parser.add_argument("--output-dir", type=str, default="")
    parser.add_argument("--wandb-project", type=str, default="OmegaGenome-ExtraLarge-Fix")

    args = parser.parse_args()

    # Setup
    set_seed(args.seed)
    task_name = args.task_name
    device = "cuda"

    print(f"\n{'=' * 60}")
    print(f"NT DISTILLATION: {task_name}")
    print(f"Model: bpnet-{args.model_size}, Seed: {args.seed}")
    print(
        f"Hyperparams: CE={args.weight_ce}, KL={args.weight_kl}, MSE={args.weight_mse}, T={args.temperature}"
    )
    print(f"{'=' * 60}\n")

    # Configs
    num_labels = get_num_labels(task_name)

    teacher_config = GLMConfig(
        model_name_or_path=nt_2b5.model_name_or_path,
        num_labels=num_labels,
        trust_remote_code=True,
        output_hidden_states=True,
    )

    distillation_config = DistillationModelConfig(
        weight_ce=args.weight_ce,
        weight_kl=args.weight_kl,
        weight_mse=args.weight_mse,
        temperature=args.temperature,
        distill_method=args.distill_method,
    )

    # Find teacher
    class Config:
        def __init__(self):
            self.teacher_config = teacher_config
            self.teacher_parent_dir = NT_PARENT_PATH
            self.model_type = "nt"

    config = Config()
    teacher_ckpt, score = find_teacher_checkpoint(config, task_name)

    if teacher_ckpt is None:
        print(f"No teacher checkpoint for {task_name}")
        return 1

    print(f"Teacher: {teacher_ckpt}")

    # Load teacher
    class FullConfig:
        def __init__(self):
            self.teacher_config = teacher_config
            self.teacher_parent_dir = NT_PARENT_PATH
            self.model_type = "nt"
            self.trainer_config = type("TC", (), {"device": device})()

    full_config = FullConfig()
    teacher_tokenizer, teacher_model, teacher_hidden = get_teacher_model(
        full_config, task_name, teacher_ckpt
    )
    teacher_model.eval()

    # Get actual feature dimension for MSE loss
    if args.weight_mse > 0:
        from src.trainer.utils import _get_cache_dir

        cache_dir = _get_cache_dir(project_path, NT_PARENT_PATH, task_name)
        metadata_path = cache_dir / "metadata.json"

        if metadata_path.exists():
            with open(metadata_path) as f:
                meta = json.load(f)
            if "features_shape" in meta and meta["features_shape"]:
                teacher_hidden = meta["features_shape"][-1]
                print(f"Feature dim from cache: {teacher_hidden}")

    # Data
    dataset_config = DatasetConfig(task_name=task_name)
    X_train, y_train, X_val, y_val, X_test, y_test = build_data_splits_from_huggingface(
        dataset_config
    )

    # Student
    student_config = BPNetClassifierConfig(
        num_labels=num_labels,
        model_type="bpnet",
        model_size=args.model_size,
        teacher_hidden_size=teacher_hidden,
    )
    model = BPNetClassifier(student_config)

    # Distillation model
    distillation_model = DistillationModel(distillation_config, teacher_model, model, device)

    # Output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = os.path.join(project_output_path, "nt", "bpnet", args.model_size, task_name)

    run_dir = create_run_directory(output_dir, task_name, distillation_config)
    print(f"Output: {run_dir}")

    # WandB
    hyperparams_str = create_run_hyperparams_str(distillation_config)
    run_name = f"{task_name}/bpnet_{args.model_size}/{args.distill_method}/{hyperparams_str}"

    wandb.init(
        project=args.wandb_project,
        name=run_name,
        dir=project_output_path,
        config=vars(args),
        tags=[task_name, "nt", f"bpnet_{args.model_size}", args.distill_method],
    )

    # Teacher eval
    teacher_mcc = evaluate_and_log_teacher(
        teacher_model,
        teacher_tokenizer,
        X_test,
        y_test,
        task_name,
        full_config,
        run_dir,
        teacher_ckpt,
    )

    # Save config
    with open(os.path.join(run_dir, "hyperparameters.json"), "w") as f:
        json.dump(
            {**vars(args), "teacher_ckpt": teacher_ckpt, "teacher_mcc": float(teacher_mcc)},
            f,
            indent=2,
        )

    # Train
    class TrainerConfig:
        def __init__(self):
            self.output_dir = output_dir
            self.wandb_project = args.wandb_project
            self.epochs = args.epochs
            self.batch_size = args.batch_size
            self.lr = args.lr
            self.max_len = args.max_len
            self.device = device

    trainer_config = TrainerConfig()

    train_distill_task(
        trainer_config,
        distillation_config,
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
        teacher_parent_dir=NT_PARENT_PATH,
        teacher_ckpt=teacher_ckpt,
    )

    wandb.finish()
    print(f"\nDone! Results in {run_dir}")
    return 0


if __name__ == "__main__":
    exit(main())
