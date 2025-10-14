import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import tyro
import json
import wandb

from datetime import datetime
from dataclasses import replace, asdict
from nntool.slurm import slurm_fn

from config.distillation.config import configs
from config.distillation.config_schema import DistillationExperimentConfig
from config.env import project_output_path
from accelerate.utils import set_seed
from ..model.glm import build_glm
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
from ..trainer.utils import get_best_checkpoint


@slurm_fn
def distill(config: DistillationExperimentConfig, task_name: str):
    print(f"\n=== Distilling {task_name} ===")
    set_seed(config.random_state)

    # load teacher model
    teacher_ckpt = get_best_checkpoint(config.teacher_parent_dir, task_name)
    if teacher_ckpt is None:
        print(f"[!] No teacher checkpoint for {task_name}, skipping.")
        return

    teacher_config = replace(config.teacher_config, ckpt_path=teacher_ckpt)
    teacher_tokenizer, teacher_model = build_glm(teacher_config)
    teacher_model.eval()

    # build student model
    num_labels = get_num_labels(task_name)
    teacher_hidden = teacher_model.config.hidden_size
    model = BPNetClassifier(
        replace(
            config.student_config,
            num_labels=num_labels,
            teacher_hidden_size=teacher_hidden,
        )
    )

    # build distillation model
    distillation_model = DistillationModel(
        config.distillation_config,
        teacher_model,
        model,
        config.trainer_config.device,
    )

    # build data splits
    X_train, y_train, X_val, y_val, X_test, y_test = build_data_splits_from_huggingface(
        config.dataset_config
    )

    # Create systematic run directory
    run_dir = create_run_directory(
        config.trainer_config.output_dir,
        task_name,
        config.distillation_config,
    )
    print(f"Run directory: {run_dir}")

    # WandB setup
    hyperparams_str = create_run_hyperparams_str(config.distillation_config)
    run_name = f"{task_name}/{hyperparams_str}"
    wandb.init(
        project=config.trainer_config.wandb_project,
        name=run_name,
        dir=project_output_path,
        config=asdict(config),
        notes=f"run_dir: {run_dir}",
    )
    wandb.watch(model, log="all", log_freq=100)

    # Save hyperparameters
    hyperparams = asdict(config)
    hyperparams["run_dir"] = run_dir
    hyperparams["timestamp"] = datetime.now().isoformat()
    with open(os.path.join(run_dir, "hyperparameters.json"), "w") as f:
        json.dump(hyperparams, f, indent=2)

    # train
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
    )

    # finish wandb
    wandb.finish()


def main(config: DistillationExperimentConfig):
    for task_name in config.task_names:
        # here we always set the task name to the current task name to avoid confusion
        distill_dataset_config_config = replace(config.dataset_config, task_name=task_name)
        distill_config = replace(config, dataset_config=distill_dataset_config_config)

        # here we use the slurm to run the distill function
        distill[distill_config.slurm_config](distill_config, task_name)


if __name__ == "__main__":
    config = tyro.extras.overridable_config_cli(configs, sort_subcommands=True)
    main(config)
