"""
Standalone script to run a single distillation experiment from a JSON config file.

This script is designed to be called by sbatch. It loads experiment configuration
from a JSON file and runs the distillation training.

Usage:
    python -m src.train.run_single_experiment --config-path /path/to/config.json
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
import argparse
import wandb
from datetime import datetime
from dataclasses import replace, asdict

from accelerate.utils import set_seed

from config.env import project_output_path
from src.model.bpnet_classifier import BPNetClassifier, BPNetClassifierConfig
from src.model.distillation import DistillationModel, DistillationModelConfig
from src.data.dataset import (
    get_num_labels,
    build_data_splits_from_huggingface,
    DatasetConfig,
)
from src.trainer.distill_trainer import (
    train_distill_task,
    create_run_directory,
    create_run_hyperparams_str,
    DistillTrainerConfig,
)
from src.model.glm import (
    find_teacher_checkpoint,
    get_teacher_model,
    evaluate_and_log_teacher,
    GLMConfig,
)


def load_config_from_json(config_path: str) -> dict:
    """Load experiment configuration from JSON file."""
    with open(config_path, "r") as f:
        config = json.load(f)
    return config


def build_config_objects(config_dict: dict):
    """Build configuration dataclass objects from dictionary."""
    
    # Build GLMConfig (teacher config)
    teacher_config = GLMConfig(
        model_name_or_path=config_dict["teacher_config"]["model_name_or_path"],
        num_labels=config_dict["teacher_config"].get("num_labels", 2),
        trust_remote_code=config_dict["teacher_config"].get("trust_remote_code", True),
        output_hidden_states=config_dict["teacher_config"].get("output_hidden_states", True),
        is_lora=config_dict["teacher_config"].get("is_lora", None),
        base_model_path=config_dict["teacher_config"].get("base_model_path", None),
        # Carbon teacher needs the "<dna>" prefix + add_special_tokens=False + bf16 load.
        # No-op defaults ("", True, None) keep nt/dnabert2/caduceus byte-identical.
        input_prefix=config_dict["teacher_config"].get("input_prefix", ""),
        add_special_tokens=config_dict["teacher_config"].get("add_special_tokens", True),
        torch_dtype=config_dict["teacher_config"].get("torch_dtype", None),
    )
    
    # Build BPNetClassifierConfig (student config)
    student_config = BPNetClassifierConfig(
        num_labels=config_dict["student_config"].get("num_labels", 2),
        model_type=config_dict["student_config"].get("model_type", "bpnet"),
        model_size=config_dict["student_config"].get("model_size", "original"),
        teacher_hidden_size=config_dict["student_config"].get("teacher_hidden_size", None),
    )
    
    # Build DistillationModelConfig
    distillation_config = DistillationModelConfig(
        weight_ce=config_dict["distillation_config"].get("weight_ce", 0.5),
        weight_kl=config_dict["distillation_config"].get("weight_kl", 0.5),
        weight_mse=config_dict["distillation_config"].get("weight_mse", 0.0),
        temperature=config_dict["distillation_config"].get("temperature", 2.0),
        zscore=config_dict["distillation_config"].get("zscore", False),
        distill_method=config_dict["distillation_config"].get("distill_method", "vanilla"),
        dkd_alpha=config_dict["distillation_config"].get("dkd_alpha", 1.0),
        dkd_beta=config_dict["distillation_config"].get("dkd_beta", 8.0),
    )
    
    # Build DistillTrainerConfig
    trainer_config = DistillTrainerConfig(
        output_dir=config_dict["trainer_config"]["output_dir"],
        wandb_project=config_dict["trainer_config"].get("wandb_project", "OmegaGenome"),
        epochs=config_dict["trainer_config"].get("epochs", 200),
        batch_size=config_dict["trainer_config"].get("batch_size", 16),
        lr=config_dict["trainer_config"].get("lr", 1e-4),
        max_len=config_dict["trainer_config"].get("max_len", 1000),
        device=config_dict["trainer_config"].get("device", "cuda"),
    )
    
    # Build DatasetConfig
    dataset_config = DatasetConfig(
        task_name=config_dict["dataset_config"]["task_name"],
        data_path=config_dict["dataset_config"].get("data_path", ""),
        dataset_name=config_dict["dataset_config"].get(
            "dataset_name",
            "InstaDeepAI/nucleotide_transformer_downstream_tasks_revised"
        ),
    )
    
    return {
        "teacher_config": teacher_config,
        "student_config": student_config,
        "distillation_config": distillation_config,
        "trainer_config": trainer_config,
        "dataset_config": dataset_config,
        "teacher_parent_dir": config_dict.get("teacher_parent_dir", ""),
        "model_type": config_dict.get("model_type", "nt"),
        "random_state": config_dict.get("random_state", 42),
        "task_name": config_dict.get("task_name", config_dict["dataset_config"]["task_name"]),
    }


def run_experiment(config_path: str):
    """Run a single distillation experiment from config file."""
    
    print(f"\n{'=' * 80}")
    print("SBATCH-BASED DISTILLATION EXPERIMENT")
    print(f"{'=' * 80}")
    print(f"Config file: {config_path}")
    print(f"Start time: {datetime.now().isoformat()}")
    print(f"{'=' * 80}\n")
    
    # Load and parse config
    config_dict = load_config_from_json(config_path)
    config = build_config_objects(config_dict)
    
    task_name = config["task_name"]
    model_type = config["model_type"]
    
    print(f"Task: {task_name}")
    print(f"Model Type: {model_type.upper()}")
    print(f"Student: {config['student_config'].model_type}-{config['student_config'].model_size}")
    print(f"Method: {config['distillation_config'].distill_method}")
    print(f"Seed: {config['random_state']}")
    print(f"{'=' * 80}\n")
    
    # Set random seed
    set_seed(config["random_state"])
    
    # Get number of labels for the task
    num_labels = get_num_labels(task_name)
    teacher_config = replace(config["teacher_config"], num_labels=num_labels)
    
    # Create a minimal config object for find_teacher_checkpoint
    class MinimalConfig:
        def __init__(self, tc, tpd, mt):
            self.teacher_config = tc
            self.teacher_parent_dir = tpd
            self.model_type = mt
    
    mini_config = MinimalConfig(
        teacher_config,
        config["teacher_parent_dir"],
        model_type,
    )
    
    # Find teacher checkpoint
    teacher_ckpt, score = find_teacher_checkpoint(mini_config, task_name)
    
    if teacher_ckpt is None:
        print(f"[!] No teacher checkpoint found for {task_name}, exiting.")
        return
    
    print(f"Teacher checkpoint: {teacher_ckpt}")
    if score > 0:
        print(f"Teacher validation MCC: {score:.4f}")
    
    # Create trainer config for get_teacher_model.
    # Delegate every attribute to the real trainer_config so downstream teacher
    # evaluation can read batch_size / teacher_batch_size / max_len / num_workers / etc.
    class TrainerConfigWrapper:
        def __init__(self, trainer_config):
            self.__dict__["_tc"] = trainer_config
            self.device = getattr(trainer_config, "device", "cuda")

        def __getattr__(self, name):
            return getattr(self.__dict__["_tc"], name)

    class FullConfig:
        def __init__(self, tc, tpd, mt, trc):
            self.teacher_config = tc
            self.teacher_parent_dir = tpd
            self.model_type = mt
            self.trainer_config = trc

    full_config = FullConfig(
        teacher_config,
        config["teacher_parent_dir"],
        model_type,
        TrainerConfigWrapper(config["trainer_config"]),
    )
    
    # Load teacher model
    teacher_tokenizer, teacher_model, teacher_hidden = get_teacher_model(
        full_config, task_name, teacher_ckpt
    )
    print(f"Teacher hidden size: {teacher_hidden}")
    teacher_model.eval()
    
    # Build data splits
    X_train, y_train, X_val, y_val, X_test, y_test = build_data_splits_from_huggingface(
        config["dataset_config"]
    )
    
    # Determine actual teacher hidden size for MSE loss
    needs_features = config["distillation_config"].weight_mse > 0
    
    if needs_features:
        from config.env import project_path
        import torch
        from src.trainer.utils import _get_cache_dir
        
        actual_teacher_hidden = None
        
        # Try to read from cache metadata
        cache_dir = _get_cache_dir(project_path, config["teacher_parent_dir"], task_name)
        metadata_path = cache_dir / "metadata.json"
        
        if metadata_path.exists():
            try:
                with open(metadata_path, "r") as f:
                    metadata = json.load(f)
                if "features_shape" in metadata and metadata["features_shape"] is not None:
                    actual_teacher_hidden = metadata["features_shape"][-1]
                    print(f"✓ Read feature dimension from cache: {actual_teacher_hidden}")
            except Exception as e:
                print(f"Warning: Could not read cache metadata: {e}")
        
        # Fallback: do a forward pass
        if actual_teacher_hidden is None:
            print("Doing forward pass to determine feature dimension...")
            try:
                sample_seq = X_train[0] if len(X_train) > 0 else "ATCGATCG"
                tok = teacher_tokenizer(
                    sample_seq,
                    padding="max_length",
                    truncation=True,
                    max_length=config["trainer_config"].max_len,
                    return_tensors="pt",
                )
                input_ids = tok.input_ids.to(config["trainer_config"].device)
                attention_mask = (
                    tok.attention_mask.to(config["trainer_config"].device)
                    if hasattr(tok, "attention_mask")
                    else torch.ones_like(input_ids)
                )
                
                with torch.no_grad():
                    _, sample_features = teacher_model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        return_features=True,
                    )
                    actual_teacher_hidden = sample_features.shape[-1]
                    print(f"✓ Feature dimension from forward pass: {actual_teacher_hidden}")
            except Exception as e:
                print(f"Warning: Forward pass failed: {e}")
        
        if actual_teacher_hidden is not None and actual_teacher_hidden != teacher_hidden:
            print(f"Overriding teacher_hidden: {teacher_hidden} -> {actual_teacher_hidden}")
            teacher_hidden = actual_teacher_hidden
    
    # Build student model
    student_config = replace(
        config["student_config"],
        num_labels=num_labels,
        teacher_hidden_size=teacher_hidden,
    )
    model = BPNetClassifier(student_config)
    
    # Build distillation model
    distillation_model = DistillationModel(
        config["distillation_config"],
        teacher_model,
        model,
        config["trainer_config"].device,
    )
    
    # Create run directory
    run_dir = create_run_directory(
        config["trainer_config"].output_dir,
        task_name,
        config["distillation_config"],
    )
    print(f"Run directory: {run_dir}")
    
    # WandB setup
    hyperparams_str = create_run_hyperparams_str(config["distillation_config"])
    model_id = f"{student_config.model_type}_{student_config.model_size}"
    distill_method = config["distillation_config"].distill_method
    run_name = f"{task_name}/{model_id}/{distill_method}/{hyperparams_str}"
    
    wandb.init(
        project=config["trainer_config"].wandb_project,
        name=run_name,
        dir=project_output_path,
        config=config_dict,
        notes=f"run_dir: {run_dir}, sbatch submission",
        tags=[task_name, model_type, model_id, distill_method, "sbatch"],
    )
    wandb.watch(model, log="all", log_freq=100)
    
    # Evaluate teacher
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
    
    # Save hyperparameters
    hyperparams = config_dict.copy()
    hyperparams["run_dir"] = run_dir
    hyperparams["timestamp"] = datetime.now().isoformat()
    hyperparams["teacher_checkpoint"] = teacher_ckpt
    hyperparams["teacher_test_mcc"] = float(teacher_mcc)
    hyperparams["submission_method"] = "sbatch"
    
    with open(os.path.join(run_dir, "hyperparameters.json"), "w") as f:
        json.dump(hyperparams, f, indent=2, default=str)
    
    # Train
    train_distill_task(
        config["trainer_config"],
        config["distillation_config"],
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
        teacher_parent_dir=config["teacher_parent_dir"],
        teacher_ckpt=teacher_ckpt,
        # Thread the teacher input formatting (Carbon "<dna>"/no-special-tokens) and the
        # top-level seed so the teacher-logit precompute tokenizes correctly and the run is
        # reproducible. No-op for nt/dnabert2/caduceus (prefix="" / add_special_tokens=True).
        input_prefix=getattr(config["teacher_config"], "input_prefix", ""),
        add_special_tokens=getattr(config["teacher_config"], "add_special_tokens", True),
        random_state=config["random_state"],
    )

    wandb.finish()
    
    print(f"\n{'=' * 80}")
    print("EXPERIMENT COMPLETED")
    print(f"End time: {datetime.now().isoformat()}")
    print(f"{'=' * 80}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Run a single distillation experiment from a JSON config file."
    )
    parser.add_argument(
        "--config-path",
        type=str,
        required=True,
        help="Path to the JSON configuration file",
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.config_path):
        print(f"Error: Config file not found: {args.config_path}")
        return 1
    
    run_experiment(args.config_path)
    return 0


if __name__ == "__main__":
    exit(main())
