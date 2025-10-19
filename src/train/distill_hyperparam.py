import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import tyro
import itertools

from dataclasses import replace
from config.distillation.config import hyperparam_configs
from config.distillation.config_schema import DistillationHyperparamExperimentConfig

from .distill import main as distill_main


def main(config: DistillationHyperparamExperimentConfig):
    """
    Run hyperparameter search over distillation parameters.

    This script performs a grid search over:
    - weight_ce, weight_kl, weight_mse (loss term weights)
    - temperature (for knowledge distillation)
    - zscore (normalization flag)

    It intelligently skips redundant combinations where weight_kl=0
    (since temperature only matters when KL divergence is used).
    """
    # Cartesian product of KL-related hyperparameters
    kl_combinations = list(
        itertools.product(
            config.weight_kls,
            config.temperatures,
            config.zscores,
        )
    )

    # Filter out redundant combinations where weight_kl=0
    # Keep only the first (weight_kl=0, any_temp, any_zscore) combination
    # since temperature and zscore don't matter when weight_kl=0
    to_skip = set()
    is_covered = False
    for i, (weight_kl, temperature, zscore) in enumerate(kl_combinations):
        if weight_kl == 0.0:
            if not is_covered:
                is_covered = True
            else:
                to_skip.add(i)
    kl_combinations = [
        combination for i, combination in enumerate(kl_combinations) if i not in to_skip
    ]

    # Cartesian product of other hyperparameters
    other_combinations = list(
        itertools.product(
            config.weight_ces,
            config.weight_mses,
        )
    )

    total_experiments = len(kl_combinations) * len(other_combinations)
    print(f"\n{'='*60}")
    print(f"Hyperparameter Search Configuration")
    print(f"{'='*60}")
    print(f"Tasks: {config.task_names}")
    print(f"Total experiments to run: {total_experiments}")
    print(f"  - CE weights: {config.weight_ces}")
    print(f"  - KL weights: {config.weight_kls}")
    print(f"  - MSE weights: {config.weight_mses}")
    print(f"  - Temperatures: {config.temperatures}")
    print(f"  - Z-scores: {config.zscores}")
    print(f"{'='*60}\n")

    # Node cycling for load balancing (2:1 ratio - voyager:laniakea)
    # voyager is ~2x faster, so we assign it 2x more jobs
    node_cycle = ["voyager", "voyager", "laniakea"]
    cycle_idx = 0

    experiment_count = 0
    for weight_ce, weight_mse in other_combinations:
        for weight_kl, temperature, zscore in kl_combinations:
            experiment_count += 1

            # Assign node based on cycle
            current_node = node_cycle[cycle_idx % len(node_cycle)]
            cycle_idx += 1

            print(
                f"\n[Experiment {experiment_count}/{total_experiments}] -> Node: {current_node}"
            )
            print(
                f"  CE={weight_ce}, KL={weight_kl}, MSE={weight_mse}, T={temperature}, zscore={zscore}"
            )

            # Create new distillation config with current hyperparameters
            new_distill_config = replace(
                config.distillation_config,
                weight_ce=weight_ce,
                weight_kl=weight_kl,
                weight_mse=weight_mse,
                temperature=temperature,
                zscore=zscore,
            )

            # Update slurm config with current node
            new_slurm_config = replace(config.slurm_config, node_list=current_node)

            # Create new experiment config with updated distillation and slurm configs
            new_experiment_config = replace(
                config,
                distillation_config=new_distill_config,
                slurm_config=new_slurm_config,  # <-- This distributes across nodes
            )

            # Run distillation for all tasks with this hyperparameter combination
            distill_main(new_experiment_config)

    print(f"\n{'='*60}")
    print(f"Hyperparameter search completed!")
    print(f"Total experiments run: {experiment_count}")
    print(
        f"Node distribution: Voyager={cycle_idx//3*2 + min(cycle_idx%3, 2)}, Laniakea={cycle_idx//3 + (1 if cycle_idx%3==2 else 0)}"
    )
    print(f"{'='*60}\n")


if __name__ == "__main__":
    # Use tyro to select from available hyperparam configs
    config = tyro.extras.overridable_config_cli(
        hyperparam_configs, sort_subcommands=True
    )
    main(config)
