"""
Run extra_large_fix experiments with sbatch submission.

This script provides an alternative to nntool.slurm for submitting distillation jobs.
It uses sbatch directly to submit experiments to the SLURM cluster.

Usage:
    # Dry run (preview jobs without submitting)
    python -m src.train.run_extra_large_fix_sbatch --dry-run

    # Submit all jobs
    python -m src.train.run_extra_large_fix_sbatch

    # Submit to specific node
    python -m src.train.run_extra_large_fix_sbatch --node-list voyager

    # Custom seeds
    python -m src.train.run_extra_large_fix_sbatch --seeds 42 123 456

    # Hyperparameter search
    python -m src.train.run_extra_large_fix_sbatch --hyperparam-search
"""

import os
import argparse
from typing import List
from itertools import product

from config.env import project_path, output_path
from config.best_hyperparams import get_size_hyperparams
from src.train.sbatch_utils import (
    SbatchConfig,
    create_and_submit_experiment,
)


# Default experiment parameters
DEFAULT_TASKS = ["splice_sites_all"]
DEFAULT_SEEDS = [42, 123, 456, 789, 1024]
DEFAULT_MODEL_SIZE = "extra_large_fix"

# Hyperparameter search grid (for extra_large_fix)
HYPERPARAM_GRID = {
    "weight_ces": [0.5],
    "weight_kls": [0.0, 0.25, 0.5, 1.0],
    "weight_mses": [0.0, 1, 2, 5],
    "temperatures": [0.5, 1.0, 1.5, 2.0, 4.0],
}


def get_teacher_config():
    """Get NT teacher configuration."""
    from config.distillation.glm import nt_2b5
    from config.distillation.experiments.nt import NT_PARENT_PATH

    return {
        "model_name_or_path": nt_2b5.model_name_or_path,
        "num_labels": nt_2b5.num_labels,
        "trust_remote_code": nt_2b5.trust_remote_code,
        "output_hidden_states": nt_2b5.output_hidden_states,
        "is_lora": nt_2b5.is_lora,
        "base_model_path": nt_2b5.base_model_path,
    }, NT_PARENT_PATH


def build_experiment_config(
    task_name: str,
    model_size: str,
    seed: int,
    output_dir: str,
    wandb_project: str,
    # Hyperparameters
    weight_ce: float = 0.5,
    weight_kl: float = 0.0,
    weight_mse: float = 1.0,
    temperature: float = 0.5,
    distill_method: str = "vanilla",
) -> dict:
    """Build experiment configuration dictionary."""

    teacher_config, teacher_parent_dir = get_teacher_config()

    return {
        "task_name": task_name,
        "model_type": "nt",
        "teacher_parent_dir": teacher_parent_dir,
        "random_state": seed,
        "teacher_config": teacher_config,
        "student_config": {
            "num_labels": 2,  # Will be updated based on task
            "model_type": "bpnet",
            "model_size": model_size,
        },
        "distillation_config": {
            "weight_ce": weight_ce,
            "weight_kl": weight_kl,
            "weight_mse": weight_mse,
            "temperature": temperature,
            "zscore": False,
            "distill_method": distill_method,
            "dkd_alpha": 1.0,
            "dkd_beta": 8.0,
        },
        "trainer_config": {
            "output_dir": output_dir,
            "wandb_project": wandb_project,
            "epochs": 200,
            "batch_size": 16,
            "lr": 1e-4,
            "max_len": 1000,
            "device": "cuda",
        },
        "dataset_config": {
            "task_name": task_name,
            "data_path": "",
            "dataset_name": "InstaDeepAI/nucleotide_transformer_downstream_tasks_revised",
        },
    }


def run_standard_experiments(
    tasks: List[str],
    seeds: List[int],
    model_size: str,
    sbatch_config: SbatchConfig,
    base_output_dir: str,
    wandb_project: str,
    dry_run: bool = False,
):
    """
    Run standard experiments with best hyperparameters for each task.
    """
    total_experiments = len(tasks) * len(seeds)

    print(f"\n{'=' * 80}")
    print("EXTRA_LARGE_FIX STANDARD EXPERIMENTS")
    print(f"{'=' * 80}")
    print(f"Tasks: {tasks}")
    print(f"Seeds: {seeds}")
    print(f"Model size: {model_size}")
    print(f"Total experiments: {total_experiments}")
    print(f"Dry run: {dry_run}")
    print(f"{'=' * 80}\n")

    submitted_jobs = []

    for task_name in tasks:
        # Get best hyperparameters for this task
        hp = get_size_hyperparams("nt", task_name, model_size)

        print(f"\nTask: {task_name}")
        print(
            f"  Hyperparams: CE={hp.weight_ce}, KL={hp.weight_kl}, "
            f"MSE={hp.weight_mse}, T={hp.temperature}"
        )

        for seed in seeds:
            # Build output directory
            exp_output_dir = os.path.join(
                base_output_dir,
                "nt",
                "bpnet",
                model_size,
                task_name,
            )

            # Build experiment config
            config = build_experiment_config(
                task_name=task_name,
                model_size=model_size,
                seed=seed,
                output_dir=exp_output_dir,
                wandb_project=wandb_project,
                weight_ce=hp.weight_ce,
                weight_kl=hp.weight_kl,
                weight_mse=hp.weight_mse,
                temperature=hp.temperature,
                distill_method=hp.distill_method,
            )

            # Submit job
            job_id = create_and_submit_experiment(
                sbatch_config=sbatch_config,
                experiment_config=config,
                task_name=task_name,
                model_size=model_size,
                seed=seed,
                project_path=project_path,
                output_dir=exp_output_dir,
                dry_run=dry_run,
            )

            if job_id:
                submitted_jobs.append(
                    {
                        "job_id": job_id,
                        "task": task_name,
                        "seed": seed,
                    }
                )

    # Summary
    print(f"\n{'=' * 80}")
    print("SUBMISSION SUMMARY")
    print(f"{'=' * 80}")
    print(f"Total jobs submitted: {len(submitted_jobs)}")
    for job in submitted_jobs[:10]:  # Show first 10
        print(f"  Job {job['job_id']}: {job['task']} seed={job['seed']}")
    if len(submitted_jobs) > 10:
        print(f"  ... and {len(submitted_jobs) - 10} more")
    print(f"{'=' * 80}\n")

    return submitted_jobs


def run_hyperparam_search(
    tasks: List[str],
    model_size: str,
    sbatch_config: SbatchConfig,
    base_output_dir: str,
    wandb_project: str,
    dry_run: bool = False,
):
    """
    Run hyperparameter search experiments.
    """
    # Build all hyperparameter combinations
    weight_ces = HYPERPARAM_GRID["weight_ces"]
    weight_kls = HYPERPARAM_GRID["weight_kls"]
    weight_mses = HYPERPARAM_GRID["weight_mses"]
    temperatures = HYPERPARAM_GRID["temperatures"]

    # Filter redundant combinations where weight_kl=0
    kl_combinations = []
    seen_zero_kl = False
    for wkl, temp in product(weight_kls, temperatures):
        if wkl == 0.0:
            if not seen_zero_kl:
                kl_combinations.append((wkl, temp))
                seen_zero_kl = True
        else:
            kl_combinations.append((wkl, temp))

    all_combinations = list(
        product(
            tasks,
            weight_ces,
            weight_mses,
            kl_combinations,
        )
    )

    total_experiments = len(all_combinations)

    print(f"\n{'=' * 80}")
    print("EXTRA_LARGE_FIX HYPERPARAMETER SEARCH")
    print(f"{'=' * 80}")
    print(f"Tasks: {tasks}")
    print(f"Model size: {model_size}")
    print(f"Total hyperparameter combinations: {total_experiments}")
    print(f"  Weight CEs: {weight_ces}")
    print(f"  Weight KLs: {weight_kls}")
    print(f"  Weight MSEs: {weight_mses}")
    print(f"  Temperatures: {temperatures}")
    print(f"Dry run: {dry_run}")
    print(f"{'=' * 80}\n")

    submitted_jobs = []

    for idx, (task_name, weight_ce, weight_mse, (weight_kl, temperature)) in enumerate(
        all_combinations, 1
    ):
        print(
            f"\n[{idx}/{total_experiments}] Task: {task_name}, "
            f"CE={weight_ce}, KL={weight_kl}, MSE={weight_mse}, T={temperature}"
        )

        # Build output directory
        exp_output_dir = os.path.join(
            base_output_dir,
            "nt",
            "bpnet",
            model_size,
            "hyperparam",
            task_name,
        )

        # Build experiment config (use fixed seed for hyperparam search)
        seed = 42
        config = build_experiment_config(
            task_name=task_name,
            model_size=model_size,
            seed=seed,
            output_dir=exp_output_dir,
            wandb_project=f"{wandb_project}-HyperParam",
            weight_ce=weight_ce,
            weight_kl=weight_kl,
            weight_mse=weight_mse,
            temperature=temperature,
        )

        # Submit job
        job_id = create_and_submit_experiment(
            sbatch_config=sbatch_config,
            experiment_config=config,
            task_name=task_name,
            model_size=model_size,
            seed=seed,
            project_path=project_path,
            output_dir=exp_output_dir,
            dry_run=dry_run,
        )

        if job_id:
            submitted_jobs.append(
                {
                    "job_id": job_id,
                    "task": task_name,
                    "weight_ce": weight_ce,
                    "weight_kl": weight_kl,
                    "weight_mse": weight_mse,
                    "temperature": temperature,
                }
            )

    # Summary
    print(f"\n{'=' * 80}")
    print("HYPERPARAMETER SEARCH SUBMISSION SUMMARY")
    print(f"{'=' * 80}")
    print(f"Total jobs submitted: {len(submitted_jobs)}")
    print(f"{'=' * 80}\n")

    return submitted_jobs


def main():
    parser = argparse.ArgumentParser(
        description="Run extra_large_fix experiments with sbatch submission."
    )

    # Experiment type
    parser.add_argument(
        "--hyperparam-search",
        action="store_true",
        help="Run hyperparameter search instead of standard experiments",
    )

    # Tasks and seeds
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=DEFAULT_TASKS,
        help=f"Tasks to run (default: {DEFAULT_TASKS})",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=DEFAULT_SEEDS,
        help=f"Random seeds (default: {DEFAULT_SEEDS})",
    )
    parser.add_argument(
        "--model-size",
        default=DEFAULT_MODEL_SIZE,
        help=f"Model size (default: {DEFAULT_MODEL_SIZE})",
    )

    # SLURM options
    parser.add_argument(
        "--node-list",
        default=None,
        help="Specific node to submit to (e.g., voyager, laniakea)",
    )
    parser.add_argument(
        "--partition",
        default="zhanglab.p",
        help="SLURM partition (default: zhanglab.p)",
    )
    parser.add_argument(
        "--mail-user",
        default=None,
        help="Email for job notifications",
    )

    # Output options
    parser.add_argument(
        "--output-dir",
        default=os.path.join(output_path, "extra_large_fix_sbatch"),
        help="Base output directory",
    )
    parser.add_argument(
        "--wandb-project",
        default="OmegaGenome-ExtraLarge-Fix",
        help="WandB project name",
    )

    # Execution options
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing",
    )

    args = parser.parse_args()

    # Build sbatch config
    sbatch_config = SbatchConfig(
        job_name="og-xl-fix",
        partition=args.partition,
        node_list=args.node_list,
        cpus_per_task=4,
        gpus_per_task=1,
        mem="64GB",
        time="7-00:00:00",
        mail_user=args.mail_user,
    )

    # Run experiments
    if args.hyperparam_search:
        run_hyperparam_search(
            tasks=args.tasks,
            model_size=args.model_size,
            sbatch_config=sbatch_config,
            base_output_dir=args.output_dir,
            wandb_project=args.wandb_project,
            dry_run=args.dry_run,
        )
    else:
        run_standard_experiments(
            tasks=args.tasks,
            seeds=args.seeds,
            model_size=args.model_size,
            sbatch_config=sbatch_config,
            base_output_dir=args.output_dir,
            wandb_project=args.wandb_project,
            dry_run=args.dry_run,
        )


if __name__ == "__main__":
    main()
