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
    SeqDataset,
)
from ..trainer.distill_trainer import (
    train_distill_task,
    create_run_directory,
    create_run_hyperparams_str,
)
from ..trainer.utils import evaluate_teacher_mcc
from torch.utils.data import DataLoader


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
    print(f"Evaluating Teacher Model on Test Set")
    print(f"{'=' * 60}")

    # Create test dataset for teacher
    from torch.utils.data import Dataset

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

        import torch

        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "labels": torch.tensor(labels, dtype=torch.long),
        }

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
    summary_csv = os.path.join(
        config.trainer_config.output_dir, "teacher_scores_summary.csv"
    )
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
    resume_checkpoint: str = None,
    resume_epoch: int = 0,
):
    print(f"\n{'=' * 80}")
    print(f"=== Starting Distillation: {task_name} ===")
    print(f"{'=' * 80}")
    print(f"Teacher: {config.teacher_config.model_name_or_path}")
    print(
        f"Student: {config.student_config.model_type}-{config.student_config.model_size}"
    )
    print(f"Method: {config.distillation_config.distill_method}")
    print(f"{'=' * 80}\n")

    set_seed(config.random_state)

    # Determine model type (GLM/DNA-BERT or NT)
    is_nt = "nucleotide" in config.teacher_config.model_name_or_path.lower()
    model_type = "NT" if is_nt else "GLM"

    # Load teacher model checkpoint
    if config.teacher_parent_dir:
        teacher_ckpt, score = get_best_checkpoint(
            config.teacher_parent_dir, task_name, model_type=model_type
        )
    else:
        from ..trainer.utils import get_best_checkpoint as orig_get_best_checkpoint

        teacher_ckpt = orig_get_best_checkpoint(
            config.trainer_config.output_dir, task_name
        )
        score = -1.0

    if teacher_ckpt is None:
        print(f"[!] No teacher checkpoint found for {task_name}, skipping.")
        return

    print(f"Teacher checkpoint: {teacher_ckpt}")
    if score > 0:
        print(f"Teacher validation MCC: {score:.4f}")

    # Build teacher model
    teacher_config = replace(config.teacher_config, ckpt_path=teacher_ckpt)
    teacher_tokenizer, teacher_model = build_glm(teacher_config)
    teacher_model.eval()

    # Build student model
    num_labels = get_num_labels(task_name)
    teacher_hidden = teacher_model.config.hidden_size

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

    wandb.init(
        project=config.trainer_config.wandb_project,
        name=run_name,
        dir=project_output_path,
        config=asdict(config),
        notes=f"run_dir: {run_dir}",
        tags=[task_name, model_type, model_id, distill_method],
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
    hyperparams["teacher_type"] = model_type
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
