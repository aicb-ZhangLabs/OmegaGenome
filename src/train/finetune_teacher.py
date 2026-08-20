"""Stage-1 teacher fine-tuning entrypoint.

Fine-tune a teacher gLM on the NT-benchmark tasks. Teacher / tasks are selected by
config, so the same command serves Carbon, AIDO.DNA, GENERATOR, ... :

    # whole 18-task benchmark
    python -m src.train.finetune_teacher carbon_3b
    # single-task smoke test
    python -m src.train.finetune_teacher carbon_3b_debug
    # override anything on the CLI (tyro), e.g. fewer tasks / epochs
    python -m src.train.finetune_teacher carbon_3b --config.epochs 5 --config.task-names H3K4me3 enhancers

Mirrors the dispatch style of ``src/train/distill.py`` (``@slurm_fn`` + tyro).
"""

import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import tyro
from dataclasses import replace
from nntool.slurm import slurm_fn

from accelerate.utils import set_seed

from config.distillation.teacher_finetune import TeacherFinetuneConfig, finetune_configs
from src.data.dataset import build_data_splits_from_huggingface, get_num_labels
from src.trainer.finetune_trainer import finetune_teacher_task


def run_task(config: TeacherFinetuneConfig, task_name: str):
    print(f"\n{'=' * 80}\n=== Fine-tuning teacher '{config.teacher_name}' on {task_name} ===")
    print(f"Teacher: {config.teacher_config.model_name_or_path}  (LoRA={config.use_lora})")
    print(f"{'=' * 80}\n")

    set_seed(config.seed)
    if config.use_wandb:
        os.environ.setdefault("WANDB_PROJECT", config.wandb_project)

    num_labels = get_num_labels(task_name)
    dataset_config = replace(config.dataset_config, task_name=task_name)
    splits = build_data_splits_from_huggingface(dataset_config)

    # Checkpoints land in output_dir/{task}_finetuned, the layout stage-2 distillation
    # discovers via teacher_parent_dir.
    finetune_teacher_task(
        config,
        config.teacher_config,
        task_name,
        num_labels,
        splits,
        run_dir=config.output_dir,
    )


@slurm_fn
def finetune(config: TeacherFinetuneConfig, task_name: str):
    run_task(config, task_name)


def main(config: TeacherFinetuneConfig):
    # Inside a SLURM allocation (sbatch sets OMEGA_FINETUNE_LOCAL=1) run inline on the
    # allocated GPU. From a login node, dispatch one SLURM job per task via @slurm_fn.
    run_local = os.environ.get("OMEGA_FINETUNE_LOCAL") == "1"
    for task_name in config.task_names:
        if run_local:
            run_task(config, task_name)
        else:
            finetune[config.slurm_config](config, task_name)


if __name__ == "__main__":
    config = tyro.extras.overridable_config_cli(finetune_configs, sort_subcommands=True)
    main(config)
