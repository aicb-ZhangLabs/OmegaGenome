import tyro

from dataclasses import replace

from ..model.glm import build_glm
from ..model.bpnet_classifier import BPNetClassifier
from ..data.dataset import (
    get_num_labels,
    build_data_splits,
)
from ..trainer.distill_trainer import train_distill_task, get_best_checkpoint
from config.distillation.config import configs
from config.distillation.config_schema import DistillationExperimentConfig


def distill(config: DistillationExperimentConfig):
    for task_name in config.dataset_config.task_name:
        print(f"\n=== Distilling {task_name} ===")

        teacher_ckpt = get_best_checkpoint(config.teacher_parent_dir, task_name)
        if teacher_ckpt is None:
            print(f"[!] No teacher checkpoint for {task_name}, skipping.")
            return

        teacher_config = replace(config.teacher_config, ckpt_path=teacher_ckpt)
        teacher_tokenizer, teacher_model = build_glm(teacher_config)
        teacher_model.eval()

        num_labels = get_num_labels(task_name)
        teacher_hidden = teacher_model.config.hidden_size
        model = BPNetClassifier(
            replace(
                config.student_config,
                num_labels=num_labels,
                teacher_hidden_size=teacher_hidden,
            )
        )

        # build data splits
        X_train, y_train, X_val, y_val, X_test, y_test = build_data_splits(
            config.dataset_config
        )

        # train
        train_distill_task(
            config.trainer_config,
            task_name,
            teacher_tokenizer,
            teacher_model,
            model,
            X_train,
            y_train,
            X_val,
            y_val,
            X_test,
            y_test,
        )


if __name__ == "__main__":
    config = tyro.extras.overridable_config_cli(configs, sort_subcommands=True)
    distill(config)
