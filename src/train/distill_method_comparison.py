"""
Experiment script for comparing different knowledge distillation methods
using task-specific best hyperparameters (from prior hyperparameter search)
and multiple random seeds.

Each (task, method) combination uses its own optimal hyperparameters.
Hyperparameters are defined in: config/best_hyperparams.py

Usage:
    python -m src.train.distill_method_comparison --help
    python -m src.train.distill_method_comparison nt_method_comparison
"""

import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import time
import tyro
from dataclasses import dataclass, field, replace
from typing import List, Literal
from itertools import product

from config.env import project_path, output_path
from config.slurm_manager import get_gpu_manager
from config.slurm import basic_distillation_slurm
from config.best_hyperparams import get_method_hyperparams, DistillHyperparams
from .distill import distill


# All 18 genomic tasks
ALL_TASKS = [
    "H2AFZ",
    "H3K27ac",
    "H3K27me3",
    "H3K36me3",
    "H3K4me1",
    "H3K4me2",
    "H3K4me3",
    "H3K9ac",
    "H3K9me3",
    "H4K20me1",
    "promoter_all",
    "promoter_tata",
    "promoter_no_tata",
    "enhancers",
    "enhancers_types",
    "splice_sites_all",
    "splice_sites_acceptors",
    "splice_sites_donors",
]

# Default seeds for reproducibility
DEFAULT_SEEDS = [24, 321, 654, 987, 4021]  # seed set 2
# [42, 123, 456, 789, 1024] #seed set 1


@dataclass
class MethodComparisonConfig:
    """Configuration for KD method comparison experiments."""

    # Teacher model type
    model_type: Literal["nt", "caduceus", "enformer", "dnabert2"] = "nt"

    # Methods to compare
    methods: List[Literal["vanilla", "logit_standard", "dkd", "dist"]] = field(
        default_factory=lambda: ["vanilla", "logit_standard", "dkd", "dist"]
    )

    # Tasks to run (empty = all tasks)
    tasks: List[str] = field(default_factory=list)

    # Random seeds
    seeds: List[int] = field(default_factory=lambda: DEFAULT_SEEDS.copy())

    # Student model
    student_model_type: Literal["bpnet", "bilstm", "cnn"] = "bpnet"
    student_model_size: str = "original"

    # Training parameters
    epochs: int = 200
    batch_size: int = 16
    lr: float = 1e-4
    max_len: int = 1000

    # Output
    output_dir: str = f"{output_path}/method_comparison-seedset2"
    wandb_project: str = "OmegaGenome-Method-Compariso-seedset2"

    # SLURM
    node_list: str = "voyager"


def get_teacher_config_and_path(model_type: str):
    """Get teacher configuration and checkpoint path."""
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

    return config_map[model_type]


def create_experiment_config(
    config: MethodComparisonConfig,
    task_name: str,
    method: str,
    seed: int,
):
    """Create a single experiment configuration with task-specific best hyperparameters."""
    from config.distillation.config_schema import DistillationExperimentConfig
    from config.distillation.data import nucletide_transformer_revised_benchmark
    from ..model.distillation import DistillationModelConfig
    from ..model.bpnet_classifier import BPNetClassifierConfig
    from ..trainer.distill_trainer import DistillTrainerConfig

    teacher_config, teacher_parent_dir = get_teacher_config_and_path(config.model_type)

    # Get task-specific best hyperparameters from config/best_hyperparams.py
    hp = get_method_hyperparams(config.model_type, task_name, method)

    # Create distillation config with task-specific hyperparameters
    distill_config = DistillationModelConfig(
        distill_method=method,
        weight_ce=hp.weight_ce,
        weight_kl=hp.weight_kl,
        weight_mse=hp.weight_mse,
        temperature=hp.temperature,
        zscore=hp.zscore,
        dkd_alpha=hp.dkd_alpha,
        dkd_beta=hp.dkd_beta,
    )

    # Create student config
    student_config = BPNetClassifierConfig(
        model_type=config.student_model_type,
        model_size=config.student_model_size,
    )

    # Create trainer config
    trainer_config = DistillTrainerConfig(
        output_dir=os.path.join(config.output_dir, config.model_type, method),
        wandb_project=config.wandb_project,
        epochs=config.epochs,
        batch_size=config.batch_size,
        lr=config.lr,
        max_len=config.max_len,
    )

    # Create dataset config
    dataset_config = replace(
        nucletide_transformer_revised_benchmark, task_name=task_name
    )

    # Create SLURM config
    slurm_config = replace(basic_distillation_slurm, node_list=config.node_list)

    # Create experiment config
    exp_config = DistillationExperimentConfig(
        task_names=[task_name],
        teacher_config=teacher_config,
        teacher_parent_dir=teacher_parent_dir,
        model_type=config.model_type,
        student_config=student_config,
        distillation_config=distill_config,
        trainer_config=trainer_config,
        dataset_config=dataset_config,
        slurm_config=slurm_config,
        random_state=seed,
    )

    return exp_config


def main(config: MethodComparisonConfig):
    """Run method comparison experiments with task-specific best hyperparameters."""

    # Determine tasks
    tasks = config.tasks if config.tasks else ALL_TASKS

    # Calculate total experiments
    total_experiments = len(tasks) * len(config.methods) * len(config.seeds)

    print(f"\n{'=' * 80}")
    print("METHOD COMPARISON EXPERIMENT")
    print("(using task-specific best hyperparameters from config/best_hyperparams.py)")
    print(f"{'=' * 80}")
    print(f"Teacher model: {config.model_type}")
    print(f"Student model: {config.student_model_type}-{config.student_model_size}")
    print(f"Methods: {config.methods}")
    print(f"Tasks: {len(tasks)} tasks")
    print(f"Seeds: {config.seeds}")
    print(f"Total experiments: {total_experiments}")

    # Print task-specific hyperparameters
    print(f"\nTask-specific best hyperparameters:")
    for task in tasks[:3]:  # Show first 3 tasks as example
        print(f"  {task}:")
        for method in config.methods:
            hp = get_method_hyperparams(config.model_type, task, method)
            print(
                f"    {method}: CE={hp.weight_ce}, KL={hp.weight_kl}, T={hp.temperature}"
            )
    if len(tasks) > 3:
        print(f"  ... and {len(tasks) - 3} more tasks")
    print(f"{'=' * 80}\n")

    # GPU manager
    gpu_manager = get_gpu_manager(
        node_limits={"voyager": 3, "laniakea": 6},
        node_capacity={"voyager": 4, "laniakea": 8},
    )

    # Run experiments
    experiment_idx = 0
    for task_name, method, seed in product(tasks, config.methods, config.seeds):
        experiment_idx += 1

        hp = get_method_hyperparams(config.model_type, task_name, method)

        print(f"\n{'=' * 60}")
        print(f"[Experiment {experiment_idx}/{total_experiments}]")
        print(f"Task: {task_name}, Method: {method}, Seed: {seed}")
        print(f"Hyperparams: CE={hp.weight_ce}, KL={hp.weight_kl}, T={hp.temperature}")
        if method == "dkd":
            print(f"  DKD: alpha={hp.dkd_alpha}, beta={hp.dkd_beta}")
        print(f"{'=' * 60}")

        # Wait for available GPU
        available_node = gpu_manager.wait_for_available_node(
            preferred_nodes=[config.node_list, "laniakea", "voyager"],
            check_interval=30,
            max_wait=3600,
        )

        if available_node is None:
            print(f"⚠️ Skipping - no available GPU")
            continue

        print(f"🚀 Submitting to node: {available_node}")

        # Create experiment config
        exp_config = create_experiment_config(config, task_name, method, seed)
        exp_config = replace(
            exp_config,
            slurm_config=replace(exp_config.slurm_config, node_list=available_node),
        )

        # Run distillation
        distill[exp_config.slurm_config](exp_config, task_name)

        # Wait for job registration
        time.sleep(3)
        for _ in range(10):
            current_jobs = gpu_manager.get_running_jobs_per_node()
            if current_jobs.get(available_node, 0) > 0:
                break
            time.sleep(1)

    print(f"\n{'=' * 80}")
    print("Method comparison experiments submitted!")
    print(f"{'=' * 80}\n")


# Pre-defined experiment configurations
experiment_configs = {
    "nt_method_comparison": (
        "Compare KD methods using NT teacher with task-specific best hyperparams",
        MethodComparisonConfig(
            model_type="nt",
            methods=["vanilla", "logit_standard", "dkd", "dist"],
            tasks=["splice_sites_all", "splice_sites_donors"],
            seeds=DEFAULT_SEEDS,
        ),
    ),
    "nt_method_comparison_all_tasks": (
        "Compare KD methods using NT teacher on all 18 tasks",
        MethodComparisonConfig(
            model_type="nt",
            methods=["vanilla", "logit_standard", "dkd", "dist"],
            tasks=[],  # All tasks
            seeds=[42, 123, 456],  # 3 seeds for all tasks
        ),
    ),
    "caduceus_method_comparison": (
        "Compare KD methods using Caduceus teacher",
        MethodComparisonConfig(
            model_type="caduceus",
            methods=["vanilla", "logit_standard", "dkd", "dist"],
            tasks=["promoter_all", "H3K4me3", "splice_sites_all"],
            seeds=[42, 123, 456, 789, 1024],
        ),
    ),
    "enformer_method_comparison": (
        "Compare KD methods using Enformer teacher",
        MethodComparisonConfig(
            model_type="enformer",
            methods=["vanilla", "logit_standard", "dkd", "dist"],
            tasks=["promoter_all", "H3K4me3", "splice_sites_all"],
            seeds=[42, 123, 456, 789, 1024],
        ),
    ),
}


if __name__ == "__main__":
    config = tyro.extras.overridable_config_cli(
        experiment_configs, sort_subcommands=True
    )
    main(config)
