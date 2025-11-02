import os
import csv
import json
import wandb
import torch
from dataclasses import dataclass, replace
from typing import Optional
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel

from torch.utils.data import DataLoader, Dataset
from ..data.dataset import (
    get_num_labels,
)
from ..trainer.utils import (
    evaluate_teacher_mcc,
    orig_get_best_checkpoint,
)
from datetime import datetime


@dataclass
class GLMConfig:
    model_name_or_path: str  # Can be either full model path or LoRA adapter path
    num_labels: int = 2
    ckpt_path: Optional[str] = None
    output_hidden_states: bool = True  # For feature extraction
    trust_remote_code: bool = True

    # LoRA-specific fields
    base_model_path: Optional[str] = None  # Base model for LoRA adapters
    is_lora: Optional[bool] = None  # Auto-detect if None
    merge_lora: bool = True  # Merge adapter weights for faster inference

    def __post_init__(self):
        """Auto-detect if model is LoRA adapter if not specified."""
        if self.is_lora is None:
            import os

            adapter_config_path = os.path.join(
                self.model_name_or_path, "adapter_config.json"
            )
            self.is_lora = os.path.exists(adapter_config_path)

            # If LoRA and base_model_path not provided, try to read from adapter_config
            if self.is_lora and self.base_model_path is None:
                try:
                    import json

                    with open(adapter_config_path, "r") as f:
                        adapter_config = json.load(f)
                    self.base_model_path = adapter_config.get("base_model_name_or_path")
                    if self.base_model_path:
                        print(f"Auto-detected base model: {self.base_model_path}")
                except Exception as e:
                    print(
                        f"Warning: Could not read base model from adapter_config.json: {e}"
                    )


def build_glm(config: GLMConfig):
    """
    Build GLM model, handling both regular and LoRA fine-tuned models.
    """
    model_path = config.ckpt_path if config.ckpt_path else config.model_name_or_path

    if config.is_lora:
        print(f"Loading LoRA adapter from: {model_path}")

        if not config.base_model_path:
            raise ValueError(
                "LoRA adapter detected but base_model_path is not set. "
                "Please specify base_model_path in GLMConfig."
            )

        print(f"Loading base model: {config.base_model_path}")

        # Load tokenizer from base model
        tokenizer = AutoTokenizer.from_pretrained(
            config.base_model_path, trust_remote_code=config.trust_remote_code
        )

        # Load base model
        base_model = AutoModelForSequenceClassification.from_pretrained(
            config.base_model_path,
            num_labels=config.num_labels,
            output_hidden_states=config.output_hidden_states,
            trust_remote_code=config.trust_remote_code,
        )

        # Load LoRA adapter
        model = PeftModel.from_pretrained(base_model, model_path)

        # Merge adapter weights for faster inference
        if config.merge_lora:
            print("Merging LoRA weights into base model...")
            model = model.merge_and_unload()

    else:
        # Regular model loading (not LoRA)
        print(f"Loading regular model from: {model_path}")
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=config.trust_remote_code
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            num_labels=config.num_labels,
            output_hidden_states=config.output_hidden_states,
            trust_remote_code=config.trust_remote_code,
        )

    return tokenizer, model


def get_best_checkpoint(parent_path: str, task_name: str, model_type: str = "default"):
    """Find best checkpoint for a task - supports both GLM and NT directory structures"""
    import re

    if "NT" in model_type or "nucleotide" in parent_path.lower():
        # NT checkpoint structure: finetuned_models/{task}_finetuned/model-best*mcc_score*
        task_dir = os.path.join(
            parent_path, "finetuned_models", f"{task_name}_finetuned"
        )
        if not os.path.isdir(task_dir):
            return None, -1.0

        best_score = -1.0
        best_dir = None
        for d in os.listdir(task_dir):
            if d.startswith("model-best") and "mcc_score" in d:
                try:
                    match = re.search(r"mcc_score([\d.]+)", d)
                    if match:
                        score = float(match.group(1))
                        if score > best_score:
                            best_score, best_dir = score, os.path.join(task_dir, d)
                except (ValueError, AttributeError):
                    continue
        return best_dir, best_score
    else:
        # Original GLM checkpoint structure
        return orig_get_best_checkpoint(parent_path, task_name), -1.0


def get_teacher_model(config, task_name, teacher_ckpt):
    """
    Unified teacher model loading that supports GLM/DNABert2, NT, and Caduceus.

    This replaces the direct build_glm call to support multiple model types.

    Returns:
        tuple: (tokenizer, teacher_model, teacher_hidden)
    """
    model_type = getattr(
        config, "model_type", "glm"
    )  # Default to 'glm' for backward compatibility

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

        # Extract teacher hidden size
        if hasattr(wrapped_model, "hidden_dim") and wrapped_model.hidden_dim:
            teacher_hidden = wrapped_model.hidden_dim
        else:
            teacher_hidden = 256  # Default for Caduceus

        # Return in format compatible with existing code
        return tokenizer, wrapped_model, teacher_hidden

    else:  # Default GLM/NT path - unchanged
        teacher_config = replace(config.teacher_config, ckpt_path=teacher_ckpt)
        teacher_tokenizer, teacher_model = build_glm(teacher_config)

        # Extract teacher hidden size
        if hasattr(teacher_model, "config"):
            teacher_hidden = teacher_model.config.hidden_size
        else:
            teacher_hidden = 768  # Default fallback

        return teacher_tokenizer, teacher_model, teacher_hidden


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
            teacher_ckpt = orig_get_best_checkpoint(
                config.trainer_config.output_dir, task_name
            )
            return teacher_ckpt, -1.0


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
    # --- START: CACHE CHECK ---
    teacher_eval_file = os.path.join(run_dir, "teacher_evaluation.json")

    if os.path.exists(teacher_eval_file):
        try:
            with open(teacher_eval_file, "r") as f:
                teacher_eval_data = json.load(f)

            # Check if the cached checkpoint matches the current one
            if (
                teacher_eval_data.get("teacher_checkpoint") == teacher_ckpt
                and "teacher_test_mcc" in teacher_eval_data
            ):
                teacher_mcc = teacher_eval_data["teacher_test_mcc"]
                print(f"✓ Found cached teacher evaluation: {teacher_eval_file}")
                print(f"Cached Teacher Test MCC: {teacher_mcc:.4f}")
                print(f"{'=' * 60}\n")

                # Log to wandb (this is necessary as the original log is skipped)
                wandb.log(
                    {
                        "teacher/test_mcc": teacher_mcc,
                    }
                )
                return teacher_mcc
            else:
                print("Cached data is for a different checkpoint. Re-evaluating...")
        except Exception as e:
            print(
                f"Warning: Could not read cached teacher evaluation file. Re-evaluating. Error: {e}"
            )

    # --- END: CACHE CHECK ---
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
