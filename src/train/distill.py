import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import tyro
import wandb

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
    import json

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
    num_labels = get_num_labels(task_name)
    config = replace(config, teacher_config=replace(config.teacher_config, num_labels=num_labels))
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
    teacher_tokenizer, teacher_model, teacher_hidden = get_teacher_model(
        config, task_name, teacher_ckpt
    )
    print(f"Initial teacher_hidden from model: {teacher_hidden}")
    teacher_model.eval()
    # Build data splits
    X_train, y_train, X_val, y_val, X_test, y_test = build_data_splits_from_huggingface(
        config.dataset_config
    )
    # ===========================================================
    # ORIGINAL CODE: Student model and distillation setup
    # ===========================================================
    # CRITICAL FIX: Determine actual teacher hidden size
    # This handles cases where config.d_model != actual feature dimension
    needs_features = config.distillation_config.weight_mse > 0

    if needs_features:
        from config.env import project_path
        import json
        import torch

        # Import the helper functions
        from src.trainer.utils import _get_cache_dir

        actual_teacher_hidden = None

        # Method 1: Try to read from existing cache metadata
        cache_dir = _get_cache_dir(project_path, config.teacher_parent_dir, task_name)
        metadata_path = cache_dir / "metadata.json"

        if metadata_path.exists():
            try:
                with open(metadata_path, "r") as f:
                    metadata = json.load(f)

                if "features_shape" in metadata and metadata["features_shape"] is not None:
                    actual_teacher_hidden = metadata["features_shape"][-1]  # Last dimension
                    print(f"✓ Read feature dimension from cache metadata: {actual_teacher_hidden}")
            except Exception as e:
                print(f"Warning: Could not read cache metadata: {e}")

        # Method 2: If no cache, do a quick single-sample forward pass
        if actual_teacher_hidden is None:
            print("No cache found, doing quick forward pass to determine feature dimension...")
            try:
                # Tokenize a single sample
                sample_seq = X_train[0] if len(X_train) > 0 else "ATCGATCG"
                tok = teacher_tokenizer(
                    sample_seq,
                    padding="max_length",
                    truncation=True,
                    max_length=config.trainer_config.max_len,
                    return_tensors="pt",
                )
                input_ids = tok.input_ids.to(config.trainer_config.device)
                attention_mask = (
                    tok.attention_mask.to(config.trainer_config.device)
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
                    print(
                        f"✓ Determined feature dimension from forward pass: {actual_teacher_hidden}"
                    )
            except Exception as e:
                print(f"Warning: Could not determine dimension from forward pass: {e}")

        # Update teacher_hidden if we found the actual dimension
        if actual_teacher_hidden is not None and actual_teacher_hidden != teacher_hidden:
            print(
                f"WARNING: Config says hidden_dim={teacher_hidden}, but actual features are {actual_teacher_hidden}-dimensional"
            )
            print(f"  Overriding to use actual dimension: {actual_teacher_hidden}")
            teacher_hidden = actual_teacher_hidden
        elif actual_teacher_hidden is not None:
            print(f"✓ Config hidden_dim={teacher_hidden} matches actual features")

    print(f"Final teacher_hidden for student model: {teacher_hidden}")

    # Build student model

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
        teacher_parent_dir=config.teacher_parent_dir,
        teacher_ckpt=teacher_ckpt,
        resume_from_checkpoint=config.resume_checkpoint,  # Read from config
        resume_from_epoch=config.resume_epoch,  # Read from config
        input_prefix=getattr(config.teacher_config, "input_prefix", ""),
        add_special_tokens=getattr(config.teacher_config, "add_special_tokens", True),
        random_state=config.random_state,
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
