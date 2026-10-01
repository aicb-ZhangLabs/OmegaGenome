"""
Experiment script for comparing different student model sizes
using task-specific best hyperparameters (from prior hyperparameter search)
and multiple random seeds.

Each (task, size) combination uses its own optimal hyperparameters.
Hyperparameters are defined in: config/best_hyperparams.py

Usage:
    python -m src.train.distill_size_comparison --help
    python -m src.train.distill_size_comparison nt_size_comparison
"""

import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import time
import tyro
from dataclasses import dataclass, field, replace
from typing import List, Literal
from itertools import product

from config.env import output_path
from config.slurm_manager import get_gpu_manager
from config.slurm import basic_distillation_slurm
from config.best_hyperparams import get_size_hyperparams
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

# Available model sizes (approximately sorted by parameter count)
MODEL_SIZES = [
    "pico",  # ~1.3k params
    "ultra_tiny",  # ~6.25k params
    "extra_tiny",  # ~25k params
    "tiny",  # ~50k params
    "small",  # ~200k params
    "medium_small",  # ~400k params
    "original",  # ~280k params (100k backbone)
    "medium",  # ~1M params
    "medium_large",  # ~0.4M params
    "extra_large",  # ~0.8M params (BROKEN - has dilation cap issue)
    "extra_large_fix",  # ~0.8M params (FIXED - no dilation cap)
    "large",  # ~5M params
    "xxlarge",  # ~3.6M params
]

# Model sizes for scaling law experiments (excludes broken extra_large)
MODEL_SIZES_SCALING = [
    "pico",  # ~1.3k params
    "ultra_tiny",  # ~6.25k params
    "extra_tiny",  # ~25k params
    "original",  # ~280k params
    "extra_large_fix",  # ~0.8M params (FIXED version)
    "large",  # ~5M params
    "xxlarge",  # ~3.6M params
]

MODEL_SIZES_SXL = [
    "pico",  # ~1.3k params
    "ultra_tiny",  # ~6.25k params
    "extra_tiny",
    "original",  # ~280k params
    "extra_large",  # ~0.8M params
    "large",  # ~5M params
    "xxlarge",  # ~3.6M params
]

# Model sizes for comparing extra_large vs extra_large_fix
MODEL_SIZES_EXTRA_LARGE_COMPARISON = [
    "original",  # Baseline
    "extra_large",  # Broken (dilation cap 6)
    "extra_large_fix",  # Fixed (no dilation cap)
]

# Default seeds for reproducibility
# DEFAULT_SEEDS = [42, 123, 456, 789, 1024]
DEFAULT_SEEDS = [42, 123, 456, 789, 1024, 24, 321, 654, 987, 4021]


@dataclass
class SizeComparisonConfig:
    """Configuration for model size comparison experiments."""

    # Teacher model type
    model_type: Literal["nt", "caduceus", "enformer", "dnabert2"] = "nt"

    # Student model sizes to compare
    sizes: List[str] = field(
        default_factory=lambda: ["pico", "tiny", "small", "original", "medium", "large"]
    )

    # Student model architecture
    student_model_type: Literal["bpnet", "bilstm", "cnn"] = "bpnet"

    # Tasks to run (empty = all tasks)
    tasks: List[str] = field(default_factory=list)

    # Random seeds
    seeds: List[int] = field(default_factory=lambda: DEFAULT_SEEDS.copy())

    # Training parameters
    epochs: int = 200
    batch_size: int = 16
    lr: float = 1e-4
    max_len: int = 1000

    # Output
    output_dir: str = f"{output_path}/size_comparison"
    wandb_project: str = "OmegaGenome-Size-Comparison"

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
    config: SizeComparisonConfig,
    task_name: str,
    model_size: str,
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
    # For extra_large_fix, use the same hyperparameters as extra_large initially
    lookup_size = model_size
    if model_size == "extra_large_fix":
        # Use extra_large hyperparams as starting point (can be tuned later)
        lookup_size = "extra_large"

    hp = get_size_hyperparams(config.model_type, task_name, lookup_size)

    # Create distillation config with task-specific hyperparameters
    distill_config = DistillationModelConfig(
        distill_method=hp.distill_method,
        weight_ce=hp.weight_ce,
        weight_kl=hp.weight_kl,
        weight_mse=hp.weight_mse,
        temperature=hp.temperature,
        zscore=hp.zscore,
        dkd_alpha=hp.dkd_alpha,
        dkd_beta=hp.dkd_beta,
    )

    # Create student config with specified size
    student_config = BPNetClassifierConfig(
        model_type=config.student_model_type,
        model_size=model_size,
    )

    # Create trainer config
    trainer_config = DistillTrainerConfig(
        output_dir=os.path.join(
            config.output_dir,
            config.model_type,
            config.student_model_type,
            model_size,
        ),
        wandb_project=config.wandb_project,
        epochs=config.epochs,
        batch_size=config.batch_size,
        lr=config.lr,
        max_len=config.max_len,
    )

    # Create dataset config
    dataset_config = replace(nucletide_transformer_revised_benchmark, task_name=task_name)

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


def main(config: SizeComparisonConfig):
    """Run size comparison experiments with task-specific best hyperparameters."""

    # Determine tasks
    tasks = config.tasks if config.tasks else ALL_TASKS

    # Validate sizes
    for size in config.sizes:
        if size not in MODEL_SIZES:
            print(f"Warning: Unknown model size '{size}'. Available: {MODEL_SIZES}")

    # Calculate total experiments
    total_experiments = len(tasks) * len(config.sizes) * len(config.seeds)

    print(f"\n{'=' * 80}")
    print("SIZE COMPARISON EXPERIMENT")
    print("(using task-specific best hyperparameters from config/best_hyperparams.py)")
    print(f"{'=' * 80}")
    print(f"Teacher model: {config.model_type}")
    print(f"Student architecture: {config.student_model_type}")
    print(f"Model sizes: {config.sizes}")
    print(f"Tasks: {len(tasks)} tasks")
    print(f"Seeds: {config.seeds}")
    print(f"Total experiments: {total_experiments}")

    # Print task-specific hyperparameters
    print("\nTask-specific best hyperparameters (examples):")
    for task in tasks[:2]:  # Show first 2 tasks as example
        print(f"  {task}:")
        for size in config.sizes[:3]:  # Show first 3 sizes
            lookup_size = size if size != "extra_large_fix" else "extra_large"
            hp = get_size_hyperparams(config.model_type, task, lookup_size)
            print(
                f"    {size}: method={hp.distill_method}, CE={hp.weight_ce}, "
                f"KL={hp.weight_kl}, T={hp.temperature}"
            )
        if len(config.sizes) > 3:
            print(f"    ... and {len(config.sizes) - 3} more sizes")
    if len(tasks) > 2:
        print(f"  ... and {len(tasks) - 2} more tasks")
    print(f"{'=' * 80}\n")

    # GPU manager
    gpu_manager = get_gpu_manager(
        node_limits={"voyager": 3, "laniakea": 6},
        node_capacity={"voyager": 4, "laniakea": 8},
    )

    # Run experiments
    experiment_idx = 0
    for task_name, model_size, seed in product(tasks, config.sizes, config.seeds):
        experiment_idx += 1

        lookup_size = model_size if model_size != "extra_large_fix" else "extra_large"
        hp = get_size_hyperparams(config.model_type, task_name, lookup_size)

        print(f"\n{'=' * 60}")
        print(f"[Experiment {experiment_idx}/{total_experiments}]")
        print(f"Task: {task_name}, Size: {model_size}, Seed: {seed}")
        print(
            f"Hyperparams: method={hp.distill_method}, CE={hp.weight_ce}, "
            f"KL={hp.weight_kl}, T={hp.temperature}"
        )
        print(f"{'=' * 60}")

        # Wait for available GPU
        available_node = gpu_manager.wait_for_available_node(
            preferred_nodes=[config.node_list, "laniakea", "voyager"],
            check_interval=30,
            max_wait=3600,
        )

        if available_node is None:
            print("⚠️ Skipping - no available GPU")
            continue

        print(f"🚀 Submitting to node: {available_node}")

        # Create experiment config
        exp_config = create_experiment_config(config, task_name, model_size, seed)
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
    print("Size comparison experiments submitted!")
    print(f"{'=' * 80}\n")


# Pre-defined experiment configurations
experiment_configs = {
    # ================================================================
    # MAIN EXPERIMENT: Compare extra_large vs extra_large_fix
    # ================================================================
    "nt_extra_large_fix_comparison": (
        "Compare extra_large (broken) vs extra_large_fix (fixed dilation) vs original",
        SizeComparisonConfig(
            model_type="nt",
            student_model_type="bpnet",
            sizes=MODEL_SIZES_EXTRA_LARGE_COMPARISON,
            tasks=["splice_sites_all"],
            seeds=DEFAULT_SEEDS,
            wandb_project="OmegaGenome-ExtraLarge-Fix",
        ),
    ),
    "nt_extra_large_fix_only": (
        "Run only extra_large_fix to verify the fix works",
        SizeComparisonConfig(
            model_type="nt",
            student_model_type="bpnet",
            sizes=["extra_large_fix"],
            tasks=["splice_sites_all"],
            seeds=DEFAULT_SEEDS,
            wandb_project="OmegaGenome-ExtraLarge-Fix",
        ),
    ),
    # ================================================================
    # SCALING LAW EXPERIMENTS: Verify proper scaling behavior
    # ================================================================
    "nt_scaling_law_verification": (
        "Verify scaling law with fixed models (excludes broken extra_large)",
        SizeComparisonConfig(
            model_type="nt",
            student_model_type="bpnet",
            sizes=MODEL_SIZES_SCALING,
            tasks=["splice_sites_all"],
            seeds=DEFAULT_SEEDS,
            wandb_project="OmegaGenome-Scaling-Law",
        ),
    ),
    # ================================================================
    # ORIGINAL EXPERIMENTS (preserved for backward compatibility)
    # ================================================================
    "nt_size_comparison-xxs-xxl": (
        "Compare BPNet sizes using NT teacher with task-specific best hyperparams",
        SizeComparisonConfig(
            model_type="nt",
            student_model_type="bpnet",
            sizes=MODEL_SIZES_SXL,
            tasks=["splice_sites_all"],
            seeds=DEFAULT_SEEDS,
        ),
    ),
    "nt_size_comparison_all": (
        "Compare all BPNet sizes using NT teacher",
        SizeComparisonConfig(
            model_type="nt",
            student_model_type="bpnet",
            sizes=MODEL_SIZES,
            tasks=["promoter_all", "splice_sites_all"],
            seeds=[42, 123, 456],
        ),
    ),
    "caduceus_size_comparison": (
        "Compare BPNet sizes using Caduceus teacher",
        SizeComparisonConfig(
            model_type="caduceus",
            student_model_type="bpnet",
            sizes=["pico", "tiny", "small", "original", "medium", "large"],
            tasks=["promoter_all", "H3K4me3", "splice_sites_all"],
            seeds=[42, 123, 456, 789, 1024],
        ),
    ),
    "enformer_size_comparison": (
        "Compare BPNet sizes using Enformer teacher",
        SizeComparisonConfig(
            model_type="enformer",
            student_model_type="bpnet",
            sizes=["pico", "tiny", "small", "original", "medium", "large"],
            tasks=["promoter_all", "H3K4me3", "splice_sites_all"],
            seeds=[42, 123, 456, 789, 1024],
        ),
    ),
    "nt_tiny_models": (
        "Focus on tiny models for edge deployment",
        SizeComparisonConfig(
            model_type="nt",
            student_model_type="bpnet",
            sizes=["pico", "ultra_tiny", "extra_tiny", "tiny"],
            tasks=ALL_TASKS,
            seeds=[42, 123, 456],
        ),
    ),
    # ================================================================
    # MULTI-TASK EXPERIMENTS with extra_large_fix
    # ================================================================
    "nt_extra_large_fix_all_tasks": (
        "Run extra_large_fix on all 18 genomic tasks",
        SizeComparisonConfig(
            model_type="nt",
            student_model_type="bpnet",
            sizes=["extra_large_fix"],
            tasks=ALL_TASKS,
            seeds=[42, 123, 456],
            wandb_project="OmegaGenome-ExtraLarge-Fix-AllTasks",
        ),
    ),
}


if __name__ == "__main__":
    config = tyro.extras.overridable_config_cli(experiment_configs, sort_subcommands=True)
    main(config)
