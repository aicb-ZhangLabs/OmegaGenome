"""
Experiment script for comparing different student model sizes
using size-specific best hyperparameters (from prior hyperparameter search)
and multiple random seeds.

Each model size uses its own optimal hyperparameters found from hyperparameter search.

Usage:
    python -m src.train.distill_size_comparison --help
    python -m src.train.distill_size_comparison nt_size_comparison
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import time
import tyro
from dataclasses import dataclass, field, replace
from typing import List, Literal, Dict
from itertools import product

from config.env import project_path, output_path
from config.slurm_manager import get_gpu_manager
from config.slurm import basic_distillation_slurm
from .distill import distill


# All 18 genomic tasks
ALL_TASKS = [
    "H2AFZ", "H3K27ac", "H3K27me3", "H3K36me3", "H3K4me1",
    "H3K4me2", "H3K4me3", "H3K9ac", "H3K9me3", "H4K20me1",
    "promoter_all", "promoter_tata", "promoter_no_tata",
    "enhancers", "enhancers_types",
    "splice_sites_all", "splice_sites_acceptors", "splice_sites_donors",
]

# Available model sizes (approximately sorted by parameter count)
MODEL_SIZES = [
    "pico",          # ~1.3k params
    "ultra_tiny",    # ~6.25k params
    "extra_tiny",    # ~25k params
    "tiny",          # ~50k params
    "small",         # ~200k params
    "medium_small",  # ~400k params
    "original",      # ~280k params
    "medium",        # ~1M params
    "medium_large",  # ~0.4M params
    "extra_large",   # ~0.8M params
    "large",         # ~5M params
    "xxlarge",       # ~3.6M params
]

# Default seeds for reproducibility
DEFAULT_SEEDS = [42, 123, 456, 789, 1024]


@dataclass
class SizeHyperparams:
    """Best hyperparameters for a specific model size (from hyperparameter search)."""
    weight_ce: float = 0.5
    weight_kl: float = 0.5
    weight_mse: float = 0.0
    temperature: float = 2.0
    zscore: bool = False
    distill_method: str = "vanilla"
    # DKD-specific (if using DKD method)
    dkd_alpha: float = 1.0
    dkd_beta: float = 8.0


# ============================================================================
# BEST HYPERPARAMETERS PER MODEL SIZE (from hyperparameter search)
# Update these based on your hyperparameter search results
# Key: (teacher_model_type, model_size) -> SizeHyperparams
# ============================================================================

# NT Teacher - Best hyperparameters per model size
NT_SIZE_HYPERPARAMS: Dict[str, SizeHyperparams] = {
    "pico": SizeHyperparams(
        weight_ce=0.5,
        weight_kl=0.5,
        weight_mse=0.0,
        temperature=2.0,
        distill_method="vanilla",
    ),
    "ultra_tiny": SizeHyperparams(
        weight_ce=0.5,
        weight_kl=0.5,
        weight_mse=0.0,
        temperature=2.0,
        distill_method="vanilla",
    ),
    "extra_tiny": SizeHyperparams(
        weight_ce=0.5,
        weight_kl=0.5,
        weight_mse=0.0,
        temperature=2.0,
        distill_method="vanilla",
    ),
    "tiny": SizeHyperparams(
        weight_ce=0.5,
        weight_kl=0.5,
        weight_mse=0.0,
        temperature=2.0,
        distill_method="vanilla",
    ),
    "small": SizeHyperparams(
        weight_ce=0.5,
        weight_kl=0.5,
        weight_mse=0.0,
        temperature=2.0,
        distill_method="vanilla",
    ),
    "medium_small": SizeHyperparams(
        weight_ce=0.5,
        weight_kl=0.5,
        weight_mse=0.0,
        temperature=2.0,
        distill_method="vanilla",
    ),
    "original": SizeHyperparams(
        weight_ce=0.5,
        weight_kl=0.5,
        weight_mse=0.0,
        temperature=2.0,
        distill_method="vanilla",
    ),
    "medium": SizeHyperparams(
        weight_ce=0.5,
        weight_kl=0.5,
        weight_mse=0.0,
        temperature=2.0,
        distill_method="vanilla",
    ),
    "medium_large": SizeHyperparams(
        weight_ce=0.5,
        weight_kl=0.5,
        weight_mse=0.0,
        temperature=2.0,
        distill_method="vanilla",
    ),
    "extra_large": SizeHyperparams(
        weight_ce=0.5,
        weight_kl=0.5,
        weight_mse=0.0,
        temperature=2.0,
        distill_method="vanilla",
    ),
    "large": SizeHyperparams(
        weight_ce=0.5,
        weight_kl=0.5,
        weight_mse=0.0,
        temperature=2.0,
        distill_method="vanilla",
    ),
    "xxlarge": SizeHyperparams(
        weight_ce=0.5,
        weight_kl=0.5,
        weight_mse=0.0,
        temperature=2.0,
        distill_method="vanilla",
    ),
}

# Caduceus Teacher - Best hyperparameters per model size
CADUCEUS_SIZE_HYPERPARAMS: Dict[str, SizeHyperparams] = {
    size: SizeHyperparams(
        weight_ce=0.5,
        weight_kl=0.5,
        weight_mse=0.0,
        temperature=2.0,
        distill_method="vanilla",
    )
    for size in MODEL_SIZES
}

# Enformer Teacher - Best hyperparameters per model size
ENFORMER_SIZE_HYPERPARAMS: Dict[str, SizeHyperparams] = {
    size: SizeHyperparams(
        weight_ce=0.5,
        weight_kl=0.5,
        weight_mse=0.0,
        temperature=2.0,
        distill_method="vanilla",
    )
    for size in MODEL_SIZES
}

# Mapping from model_type to size hyperparameters
MODEL_SIZE_HYPERPARAMS = {
    "nt": NT_SIZE_HYPERPARAMS,
    "caduceus": CADUCEUS_SIZE_HYPERPARAMS,
    "enformer": ENFORMER_SIZE_HYPERPARAMS,
    "dnabert2": NT_SIZE_HYPERPARAMS,  # Use NT defaults for DNABERT2
}


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


def get_size_hyperparams(model_type: str, model_size: str) -> SizeHyperparams:
    """Get best hyperparameters for a specific model size and teacher."""
    size_hyperparams = MODEL_SIZE_HYPERPARAMS.get(model_type, NT_SIZE_HYPERPARAMS)
    return size_hyperparams.get(model_size, SizeHyperparams())


def create_experiment_config(
    config: SizeComparisonConfig,
    task_name: str,
    model_size: str,
    seed: int,
):
    """Create a single experiment configuration with size-specific best hyperparameters."""
    from config.distillation.config_schema import DistillationExperimentConfig
    from config.distillation.data import nucletide_transformer_revised_benchmark
    from ..model.distillation import DistillationModelConfig
    from ..model.bpnet_classifier import BPNetClassifierConfig
    from ..trainer.distill_trainer import DistillTrainerConfig
    
    teacher_config, teacher_parent_dir = get_teacher_config_and_path(config.model_type)
    
    # Get size-specific best hyperparameters
    hp = get_size_hyperparams(config.model_type, model_size)
    
    # Create distillation config with size-specific hyperparameters
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
    """Run size comparison experiments with size-specific best hyperparameters."""
    
    # Determine tasks
    tasks = config.tasks if config.tasks else ALL_TASKS
    
    # Validate sizes
    for size in config.sizes:
        if size not in MODEL_SIZES:
            print(f"Warning: Unknown model size '{size}'. Available: {MODEL_SIZES}")
    
    # Calculate total experiments
    total_experiments = len(tasks) * len(config.sizes) * len(config.seeds)
    
    print(f"\n{'=' * 80}")
    print("SIZE COMPARISON EXPERIMENT (with size-specific best hyperparameters)")
    print(f"{'=' * 80}")
    print(f"Teacher model: {config.model_type}")
    print(f"Student architecture: {config.student_model_type}")
    print(f"Model sizes: {config.sizes}")
    print(f"Tasks: {len(tasks)} tasks")
    print(f"Seeds: {config.seeds}")
    print(f"Total experiments: {total_experiments}")
    
    # Print size-specific hyperparameters
    print(f"\nSize-specific best hyperparameters (from hyperparameter search):")
    for size in config.sizes:
        hp = get_size_hyperparams(config.model_type, size)
        print(f"  {size}:")
        print(f"    method={hp.distill_method}, weight_ce={hp.weight_ce}, "
              f"weight_kl={hp.weight_kl}, temperature={hp.temperature}")
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
        
        hp = get_size_hyperparams(config.model_type, model_size)
        
        print(f"\n{'=' * 60}")
        print(f"[Experiment {experiment_idx}/{total_experiments}]")
        print(f"Task: {task_name}, Size: {model_size}, Seed: {seed}")
        print(f"Hyperparams: method={hp.distill_method}, CE={hp.weight_ce}, "
              f"KL={hp.weight_kl}, T={hp.temperature}")
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
    "nt_size_comparison": (
        "Compare BPNet sizes using NT teacher with size-specific best hyperparams",
        SizeComparisonConfig(
            model_type="nt",
            student_model_type="bpnet",
            sizes=["pico", "tiny", "small", "original", "medium", "large"],
            tasks=["promoter_all", "H3K4me3", "splice_sites_all"],
            seeds=[42, 123, 456, 789, 1024],
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
}


if __name__ == "__main__":
    config = tyro.extras.overridable_config_cli(experiment_configs, sort_subcommands=True)
    main(config)
