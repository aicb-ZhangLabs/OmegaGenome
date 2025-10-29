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
from ..model.glm import build_glm, get_best_checkpoint
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
from ..trainer.utils import (
    evaluate_teacher_mcc,
    get_best_checkpoint as orig_get_best_checkpoint,
)

import torch
from torch.utils.data import DataLoader, Dataset


# ============================================================================
# CADUCEUS SUPPORT - Minimal additions
# ============================================================================


def get_teacher_model(config, task_name, teacher_ckpt):
    """
    Unified teacher model loading that supports GLM/DNABert2, NT, and Caduceus.

    This replaces the direct build_glm call to support multiple model types.
    """
    model_type = getattr(config, "model_type", "glm")  # Default to 'glm' for backward compatibility

    if model_type == "caduceus":
        # Import Caduceus utilities only when needed
        from ..trainer.utils import load_caduceus_model

        num_labels = get_num_labels(task_name)

        # For Caduceus, teacher_ckpt is the directory path
        # Check if there's a specific checkpoint file
        import re

        best_ckpt_file = None
        if os.path.isdir(teacher_ckpt):
            pattern = re.compile(r"epoch(\d+)_valmcc_(-?[0-9\.]+)\.pt")
            for filename in os.listdir(teacher_ckpt):
                if pattern.match(filename):
                    best_ckpt_file = os.path.join(teacher_ckpt, filename)
                    break

        wrapped_model, tokenizer, base_model = load_caduceus_model(
            teacher_ckpt, num_labels, config.trainer_config.device, best_ckpt_file
        )

        # Return in format compatible with existing code
        # wrapped_model has the feature extraction capabilities
        return tokenizer, wrapped_model

    else:  # Default GLM/NT path - unchanged
        teacher_config = replace(config.teacher_config, ckpt_path=teacher_ckpt)
        teacher_tokenizer, teacher_model = build_glm(teacher_config)
        return teacher_tokenizer, teacher_model


def find_teacher_checkpoint(config, task_name):
    """
    Find the best teacher checkpoint based on model type.

    Returns: (checkpoint_path, score)
    """
    model_type = getattr(config, "model_type", "glm")

    if model_type == "caduceus":
        # Import Caduceus utilities only when needed
        from ..trainer.utils import find_best_caduceus_checkpoint

        checkpoint_path, score, _ = find_best_caduceus_checkpoint(
            task_name, config.teacher_parent_dir
        )
        return checkpoint_path, score

    else:  # Default GLM/NT path
        # Determine if NT or GLM
        is_nt = "nucleotide" in config.teacher_config.model_name_or_path.lower()
        model_type_str = "NT" if is_nt else "GLM"

        if config.teacher_parent_dir:
            return get_best_checkpoint(
                config.teacher_parent_dir, task_name, model_type=model_type_str
            )
        else:
            teacher_ckpt = orig_get_best_checkpoint(config.trainer_config.output_dir, task_name)
            return teacher_ckpt, -1.0


# ============================================================================
# ORIGINAL FUNCTIONS - Unchanged
# ============================================================================


def evaluate_and_log_teacher(
    teacher_model,
    teacher_tokenizer,
    X_test,
    y_test,
    task_name,
    config,
    run_dir,
    teacher_ckpt,
):
    """
    Evaluate teacher model on test set and log results.

    Returns:
        float: Teacher test MCC score
    """
    print(f"\n{'=' * 60}")
    print("Evaluating Teacher Model on Test Set")
    print(f"{'=' * 60}")

    # Create test dataset for teacher
    class SimpleTextDataset(Dataset):
        def __init__(self, texts, labels):
            self.texts = texts
            self.labels = labels

        def __len__(self):
            return len(self.texts)

        def __getitem__(self, idx):
            return {"text": self.texts[idx], "label": self.labels[idx]}

    # Create dataset and dataloader
    test_dataset = SimpleTextDataset(X_test, y_test)

    # Collate function for teacher
    def collate_fn(batch):
        texts = [item["text"] for item in batch]
        labels = [item["label"] for item in batch]

        encoded = teacher_tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=config.trainer_config.max_len,
            return_tensors="pt",
        )

        # FIX: Handle missing attention_mask
        result = {
            "input_ids": encoded["input_ids"],
            "labels": torch.tensor(labels, dtype=torch.long),
        }

        # Only add attention_mask if it exists, otherwise create default
        if "attention_mask" in encoded:
            result["attention_mask"] = encoded["attention_mask"]
        else:
            result["attention_mask"] = torch.ones_like(encoded["input_ids"])

        return result

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.trainer_config.batch_size,
        shuffle=False,
        num_workers=config.trainer_config.num_workers,
        collate_fn=collate_fn,
    )

    # Evaluate
    teacher_mcc = evaluate_teacher_mcc(
        teacher_model,
        teacher_tokenizer,
        test_loader,
        config.trainer_config.device,
    )

    print(f"Teacher Test MCC: {teacher_mcc:.4f}")
    print(f"{'=' * 60}\n")

    # Save to file
    teacher_eval_file = os.path.join(run_dir, "teacher_evaluation.json")
    teacher_eval_data = {
        "task": task_name,
        "teacher_checkpoint": teacher_ckpt,
        "teacher_test_mcc": float(teacher_mcc),
        "timestamp": datetime.now().isoformat(),
    }

    with open(teacher_eval_file, "w") as f:
        json.dump(teacher_eval_data, f, indent=2)

    # Also append to a summary CSV for easy comparison across experiments
    summary_csv = os.path.join(config.trainer_config.output_dir, "teacher_scores_summary.csv")
    file_exists = os.path.exists(summary_csv)

    with open(summary_csv, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(
                [
                    "task",
                    "teacher_checkpoint",
                    "teacher_test_mcc",
                    "timestamp",
                    "run_dir",
                ]
            )
        writer.writerow(
            [
                task_name,
                teacher_ckpt,
                f"{teacher_mcc:.4f}",
                datetime.now().isoformat(),
                run_dir,
            ]
        )

    # Log to wandb
    wandb.log(
        {
            "teacher/test_mcc": teacher_mcc,
        }
    )

    return teacher_mcc


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
    print(f"Student: {config.student_config.model_type}-{config.student_config.model_size}")
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
    teacher_tokenizer, teacher_model = get_teacher_model(config, task_name, teacher_ckpt)
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
