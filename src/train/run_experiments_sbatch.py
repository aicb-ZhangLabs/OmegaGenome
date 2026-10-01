"""
General-purpose experiment runner using sbatch submission.

This script provides a unified interface for submitting various experiments
via sbatch, as an alternative to nntool.slurm.

Usage:
    # List available experiments
    python -m src.train.run_experiments_sbatch --help

    # Run extra_large_fix experiment with sbatch
    python -m src.train.run_experiments_sbatch extra_large_fix

    # Dry run (preview without submitting)
    python -m src.train.run_experiments_sbatch extra_large_fix --dry-run

    # Run hyperparameter search
    python -m src.train.run_experiments_sbatch extra_large_fix_hyperparam

    # Run with specific node
    python -m src.train.run_experiments_sbatch extra_large_fix --node-list voyager
"""

import os
import tyro
from dataclasses import dataclass, field
from typing import List, Optional, Literal
from itertools import product
from datetime import datetime

from config.env import project_path, output_path
from config.best_hyperparams import get_size_hyperparams
from src.train.sbatch_utils import (
    SbatchConfig,
    create_and_submit_experiment,
)


# ============================================================================
# CONFIGURATION CLASSES
# ============================================================================


@dataclass
class ExperimentConfig:
    """Base configuration for sbatch experiments."""

    # Model configuration
    model_type: Literal["nt", "caduceus", "enformer", "dnabert2"] = "nt"
    student_model_type: Literal["bpnet", "bilstm", "cnn"] = "bpnet"
    student_model_size: str = "extra_large_fix"

    # Tasks and seeds
    tasks: List[str] = field(default_factory=lambda: ["splice_sites_all"])
    seeds: List[int] = field(default_factory=lambda: [42, 123, 456, 789, 1024])

    # Training parameters
    epochs: int = 200
    batch_size: int = 16
    lr: float = 1e-4
    max_len: int = 1000

    # Output
    output_dir: str = ""
    wandb_project: str = "OmegaGenome-SBATCH"

    # SLURM configuration
    partition: str = "zhanglab.p"
    node_list: Optional[str] = None
    cpus_per_task: int = 4
    gpus_per_task: int = 1
    mem: str = "64GB"
    time: str = "7-00:00:00"
    mail_user: Optional[str] = None

    # Submission options
    dry_run: bool = False
    delay_between_jobs: float = 1.0  # seconds between job submissions

    # Python path (auto-detected if empty)
    python_path: str = ""

    def __post_init__(self):
        if not self.output_dir:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_dir = os.path.join(
                output_path,
                f"sbatch_{self.student_model_size}_{timestamp}",
            )


@dataclass
class HyperparamSearchConfig(ExperimentConfig):
    """Configuration for hyperparameter search experiments."""

    # Hyperparameter grid
    weight_ces: List[float] = field(default_factory=lambda: [0.5])
    weight_kls: List[float] = field(default_factory=lambda: [0.0, 0.25, 0.5, 1.0])
    weight_mses: List[float] = field(default_factory=lambda: [0.0, 1, 2, 5])
    temperatures: List[float] = field(default_factory=lambda: [0.5, 1.0, 1.5, 2.0, 4.0])

    # Use single seed for hyperparam search
    seeds: List[int] = field(default_factory=lambda: [42])

    def __post_init__(self):
        super().__post_init__()
        if "hyperparam" not in self.wandb_project.lower():
            self.wandb_project = f"{self.wandb_project}-HyperParam"


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def get_teacher_config(model_type: str):
    """Get teacher configuration for the specified model type."""
    from config.distillation.glm import nt_2b5, caduceus, enformer, dna_bert_v2
    from config.distillation.experiments.nt import NT_PARENT_PATH
    from config.distillation.experiments.caduceus import CADUCEUS_PARENT_PATH
    from config.distillation.experiments.enformer import ENFORMER_PARENT_PATH
    from config.distillation.experiments.dna_bert_v2 import DNABERT2_PARENT_PATH

    config_map = {
        "nt": (nt_2b5, NT_PARENT_PATH),
        "caduceus": (caduceus, CADUCEUS_PARENT_PATH),
        "enformer": (enformer, ENFORMER_PARENT_PATH),
        "dnabert2": (dna_bert_v2, DNABERT2_PARENT_PATH),
    }

    glm_config, parent_path = config_map[model_type]

    return {
        "model_name_or_path": glm_config.model_name_or_path,
        "num_labels": glm_config.num_labels,
        "trust_remote_code": glm_config.trust_remote_code,
        "output_hidden_states": glm_config.output_hidden_states,
        "is_lora": glm_config.is_lora,
        "base_model_path": glm_config.base_model_path,
    }, parent_path


def build_experiment_dict(
    config: ExperimentConfig,
    task_name: str,
    seed: int,
    weight_ce: float,
    weight_kl: float,
    weight_mse: float,
    temperature: float,
    distill_method: str = "vanilla",
) -> dict:
    """Build experiment configuration dictionary."""

    teacher_config, teacher_parent_dir = get_teacher_config(config.model_type)

    # Build task-specific output directory
    exp_output_dir = os.path.join(
        config.output_dir,
        config.model_type,
        config.student_model_type,
        config.student_model_size,
        task_name,
    )

    return {
        "task_name": task_name,
        "model_type": config.model_type,
        "teacher_parent_dir": teacher_parent_dir,
        "random_state": seed,
        "teacher_config": teacher_config,
        "student_config": {
            "num_labels": 2,
            "model_type": config.student_model_type,
            "model_size": config.student_model_size,
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
            "output_dir": exp_output_dir,
            "wandb_project": config.wandb_project,
            "epochs": config.epochs,
            "batch_size": config.batch_size,
            "lr": config.lr,
            "max_len": config.max_len,
            "device": "cuda",
        },
        "dataset_config": {
            "task_name": task_name,
            "data_path": "",
            "dataset_name": "InstaDeepAI/nucleotide_transformer_downstream_tasks_revised",
        },
    }, exp_output_dir


# ============================================================================
# MAIN EXECUTION FUNCTIONS
# ============================================================================


def run_standard_experiments(config: ExperimentConfig):
    """Run standard experiments with best hyperparameters."""
    import time

    total_experiments = len(config.tasks) * len(config.seeds)

    print(f"\n{'=' * 80}")
    print("SBATCH STANDARD EXPERIMENTS")
    print(f"{'=' * 80}")
    print(f"Model type: {config.model_type}")
    print(f"Student: {config.student_model_type}-{config.student_model_size}")
    print(f"Tasks: {config.tasks}")
    print(f"Seeds: {config.seeds}")
    print(f"Total experiments: {total_experiments}")
    print(f"Dry run: {config.dry_run}")
    print(f"Output: {config.output_dir}")
    print(f"{'=' * 80}\n")

    # Build sbatch config
    sbatch_config = SbatchConfig(
        job_name=f"og-{config.student_model_size[:8]}",
        partition=config.partition,
        node_list=config.node_list,
        cpus_per_task=config.cpus_per_task,
        gpus_per_task=config.gpus_per_task,
        mem=config.mem,
        time=config.time,
        mail_user=config.mail_user,
        python_path=config.python_path,
    )

    submitted_jobs = []

    for task_name in config.tasks:
        # Get best hyperparameters
        hp = get_size_hyperparams(config.model_type, task_name, config.student_model_size)

        print(f"\nTask: {task_name}")
        print(
            f"  Best hyperparams: CE={hp.weight_ce}, KL={hp.weight_kl}, "
            f"MSE={hp.weight_mse}, T={hp.temperature}"
        )

        for seed in config.seeds:
            # Build experiment config
            exp_dict, exp_output_dir = build_experiment_dict(
                config=config,
                task_name=task_name,
                seed=seed,
                weight_ce=hp.weight_ce,
                weight_kl=hp.weight_kl,
                weight_mse=hp.weight_mse,
                temperature=hp.temperature,
                distill_method=hp.distill_method,
            )

            # Submit job
            job_id = create_and_submit_experiment(
                sbatch_config=sbatch_config,
                experiment_config=exp_dict,
                task_name=task_name,
                model_size=config.student_model_size,
                seed=seed,
                project_path=project_path,
                output_dir=exp_output_dir,
                dry_run=config.dry_run,
            )

            if job_id:
                submitted_jobs.append(
                    {
                        "job_id": job_id,
                        "task": task_name,
                        "seed": seed,
                    }
                )

            # Delay between submissions
            if not config.dry_run and config.delay_between_jobs > 0:
                time.sleep(config.delay_between_jobs)

    print_summary(submitted_jobs)
    return submitted_jobs


def run_hyperparam_search(config: HyperparamSearchConfig):
    """Run hyperparameter search experiments."""
    import time

    # Build hyperparameter combinations (filter redundant weight_kl=0 cases)
    kl_combinations = []
    seen_zero_kl = False
    for wkl, temp in product(config.weight_kls, config.temperatures):
        if wkl == 0.0:
            if not seen_zero_kl:
                kl_combinations.append((wkl, temp))
                seen_zero_kl = True
        else:
            kl_combinations.append((wkl, temp))

    all_combinations = list(
        product(
            config.tasks,
            config.seeds,
            config.weight_ces,
            config.weight_mses,
            kl_combinations,
        )
    )

    total_experiments = len(all_combinations)

    print(f"\n{'=' * 80}")
    print("SBATCH HYPERPARAMETER SEARCH")
    print(f"{'=' * 80}")
    print(f"Model type: {config.model_type}")
    print(f"Student: {config.student_model_type}-{config.student_model_size}")
    print(f"Tasks: {config.tasks}")
    print(f"Total combinations: {total_experiments}")
    print(f"  Weight CEs: {config.weight_ces}")
    print(f"  Weight KLs: {config.weight_kls}")
    print(f"  Weight MSEs: {config.weight_mses}")
    print(f"  Temperatures: {config.temperatures}")
    print(f"Dry run: {config.dry_run}")
    print(f"{'=' * 80}\n")

    # Build sbatch config
    sbatch_config = SbatchConfig(
        job_name=f"og-hp-{config.student_model_size[:6]}",
        partition=config.partition,
        node_list=config.node_list,
        cpus_per_task=config.cpus_per_task,
        gpus_per_task=config.gpus_per_task,
        mem=config.mem,
        time=config.time,
        mail_user=config.mail_user,
        python_path=config.python_path,
    )

    submitted_jobs = []

    for idx, (task_name, seed, weight_ce, weight_mse, (weight_kl, temperature)) in enumerate(
        all_combinations, 1
    ):
        print(
            f"\n[{idx}/{total_experiments}] {task_name} s{seed}: "
            f"CE={weight_ce}, KL={weight_kl}, MSE={weight_mse}, T={temperature}"
        )

        # Build experiment config
        exp_dict, exp_output_dir = build_experiment_dict(
            config=config,
            task_name=task_name,
            seed=seed,
            weight_ce=weight_ce,
            weight_kl=weight_kl,
            weight_mse=weight_mse,
            temperature=temperature,
        )

        # Update output dir for hyperparam search
        exp_output_dir = os.path.join(exp_output_dir, "hyperparam")
        exp_dict["trainer_config"]["output_dir"] = exp_output_dir

        # Submit job
        job_id = create_and_submit_experiment(
            sbatch_config=sbatch_config,
            experiment_config=exp_dict,
            task_name=task_name,
            model_size=config.student_model_size,
            seed=seed,
            project_path=project_path,
            output_dir=exp_output_dir,
            dry_run=config.dry_run,
        )

        if job_id:
            submitted_jobs.append(
                {
                    "job_id": job_id,
                    "task": task_name,
                    "seed": seed,
                    "hyperparams": f"CE{weight_ce}_KL{weight_kl}_MSE{weight_mse}_T{temperature}",
                }
            )

        # Delay between submissions
        if not config.dry_run and config.delay_between_jobs > 0:
            time.sleep(config.delay_between_jobs)

    print_summary(submitted_jobs)
    return submitted_jobs


def print_summary(submitted_jobs: List[dict]):
    """Print submission summary."""
    print(f"\n{'=' * 80}")
    print("SUBMISSION SUMMARY")
    print(f"{'=' * 80}")
    print(f"Total jobs submitted: {len(submitted_jobs)}")

    if submitted_jobs:
        for job in submitted_jobs[:10]:
            info = f"Job {job['job_id']}: {job['task']}"
            if "seed" in job:
                info += f" seed={job['seed']}"
            if "hyperparams" in job:
                info += f" {job['hyperparams']}"
            print(f"  {info}")

        if len(submitted_jobs) > 10:
            print(f"  ... and {len(submitted_jobs) - 10} more")

    print(f"{'=' * 80}\n")


# ============================================================================
# PREDEFINED EXPERIMENT CONFIGURATIONS
# ============================================================================

experiment_configs = {
    # Standard extra_large_fix experiments
    "extra_large_fix": (
        "Run extra_large_fix model with best hyperparameters (5 seeds)",
        ExperimentConfig(
            model_type="nt",
            student_model_type="bpnet",
            student_model_size="extra_large_fix",
            tasks=["splice_sites_all"],
            seeds=[42, 123, 456, 789, 1024],
            wandb_project="OmegaGenome-ExtraLarge-Fix",
        ),
    ),
    # Single test run
    "extra_large_fix_single": (
        "Run single extra_large_fix experiment (for testing)",
        ExperimentConfig(
            model_type="nt",
            student_model_type="bpnet",
            student_model_size="extra_large_fix",
            tasks=["splice_sites_all"],
            seeds=[42],
            wandb_project="OmegaGenome-ExtraLarge-Fix-Test",
        ),
    ),
    # Hyperparameter search
    "extra_large_fix_hyperparam": (
        "Hyperparameter search for extra_large_fix model",
        HyperparamSearchConfig(
            model_type="nt",
            student_model_type="bpnet",
            student_model_size="extra_large_fix",
            tasks=["splice_sites_all"],
            seeds=[42],
            wandb_project="OmegaGenome-ExtraLarge-Fix-HyperParam",
        ),
    ),
    # Size comparison experiments
    "size_comparison": (
        "Compare different model sizes",
        ExperimentConfig(
            model_type="nt",
            student_model_type="bpnet",
            student_model_size="original",  # Will be overridden per size
            tasks=["splice_sites_all"],
            seeds=[42, 123, 456],
            wandb_project="OmegaGenome-Size-Comparison",
        ),
    ),
    # Method comparison
    "method_comparison": (
        "Compare different distillation methods",
        ExperimentConfig(
            model_type="nt",
            student_model_type="bpnet",
            student_model_size="original",
            tasks=["splice_sites_all"],
            seeds=[42, 123, 456, 789, 1024],
            wandb_project="OmegaGenome-Method-Comparison",
        ),
    ),
    # Scaling law verification
    "scaling_law": (
        "Scaling law verification (pico to xxlarge)",
        ExperimentConfig(
            model_type="nt",
            student_model_type="bpnet",
            student_model_size="pico",  # Will run multiple sizes
            tasks=["splice_sites_all"],
            seeds=[42, 123, 456, 789, 1024],
            wandb_project="OmegaGenome-Scaling-Law",
        ),
    ),
}


def main(config: ExperimentConfig):
    """Main entry point."""
    if isinstance(config, HyperparamSearchConfig):
        run_hyperparam_search(config)
    else:
        run_standard_experiments(config)


if __name__ == "__main__":
    config = tyro.extras.overridable_config_cli(
        experiment_configs,
        sort_subcommands=True,
    )
    main(config)
