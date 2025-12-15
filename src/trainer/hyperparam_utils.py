"""
Utilities for method-specific hyperparameter grids in distillation experiments.
"""

from typing import Dict, List
from dataclasses import asdict


def get_method_hyperparameter_grid(method: str, config) -> Dict[str, List]:
    """
    Get method-specific hyperparameter grid.

    Args:
        method: Distillation method name ('vanilla', 'logit_standard', 'dkd', 'dist')
        config: DistillationHyperparamExperimentConfig with base hyperparameters

    Returns:
        Dict mapping hyperparameter names to lists of values
    """
    # Common hyperparameters for all methods
    common = {
        "weight_ces": config.weight_ces,
        "weight_kls": config.weight_kls,
        "temperatures": config.temperatures,
        "zscores": config.zscores,
    }

    if method in ["vanilla", "logit_standard"]:
        # These methods support feature matching (MSE loss)
        return {
            **common,
            "weight_mses": config.weight_mses,
        }

    elif method == "dkd":
        # DKD has its own hyperparameters and doesn't use MSE
        return {
            **common,
            "weight_mses": [0.0],  # DKD doesn't use feature matching
            "dkd_alphas": getattr(config, "dkd_alphas", [0.5, 1.0, 2.0]),
            "dkd_betas": getattr(config, "dkd_betas", [4.0, 8.0, 16.0]),
        }

    elif method == "dist":
        # DIST uses correlation loss, not MSE
        return {
            **common,
            "weight_mses": [0.0],  # DIST doesn't use MSE
        }

    else:
        # Fallback to vanilla behavior
        return {
            **common,
            "weight_mses": config.weight_mses,
        }


def build_method_experiments(config, method: str):
    """
    Build all experiment configurations for a specific distillation method.

    Args:
        config: DistillationHyperparamExperimentConfig
        method: Distillation method name

    Returns:
        List of experiment dictionaries with task_name and hyperparam_config
    """
    import itertools
    from dataclasses import replace

    # Get method-specific hyperparameter grid
    hyperparam_grid = get_method_hyperparameter_grid(method, config)

    # Extract hyperparameters
    weight_ces = hyperparam_grid["weight_ces"]
    weight_kls = hyperparam_grid["weight_kls"]
    weight_mses = hyperparam_grid["weight_mses"]
    temperatures = hyperparam_grid["temperatures"]
    zscores = hyperparam_grid["zscores"]

    # DKD-specific parameters
    dkd_alphas = hyperparam_grid.get("dkd_alphas", [1.0])
    dkd_betas = hyperparam_grid.get("dkd_betas", [8.0])

    # Build KL-related combinations
    kl_combinations = list(
        itertools.product(
            weight_kls,
            temperatures,
            zscores,
        )
    )

    # Filter redundant combinations where weight_kl=0
    # (temperature and zscore don't matter when weight_kl=0)
    filtered_kl_combinations = []
    seen_zero_kl = False
    for weight_kl, temperature, zscore in kl_combinations:
        if weight_kl == 0.0:
            if not seen_zero_kl:
                filtered_kl_combinations.append((weight_kl, temperature, zscore))
                seen_zero_kl = True
        else:
            filtered_kl_combinations.append((weight_kl, temperature, zscore))

    # Build all experiments
    all_experiments = []
    experiment_count = 0

    for task_name in config.task_names:
        # Non-KL combinations
        if method == "dkd":
            # DKD has additional alpha/beta parameters
            other_combinations = list(
                itertools.product(
                    weight_ces,
                    weight_mses,
                    dkd_alphas,
                    dkd_betas,
                )
            )

            for weight_ce, weight_mse, dkd_alpha, dkd_beta in other_combinations:
                for weight_kl, temperature, zscore in filtered_kl_combinations:
                    experiment_count += 1

                    full_distill_config = replace(
                        config.distillation_config,
                        distill_method=method,
                        weight_ce=weight_ce,
                        weight_kl=weight_kl,
                        weight_mse=weight_mse,
                        temperature=temperature,
                        zscore=zscore,
                        dkd_alpha=dkd_alpha,
                        dkd_beta=dkd_beta,
                    )
                    hyperparam_config = asdict(full_distill_config)
                    all_experiments.append(
                        {
                            "task_name": task_name,
                            "hyperparam_config": hyperparam_config,
                            "experiment_count": experiment_count,
                        }
                    )

        else:
            # Standard methods (vanilla, logit_standard, dist)
            other_combinations = list(
                itertools.product(
                    weight_ces,
                    weight_mses,
                )
            )

            for weight_ce, weight_mse in other_combinations:
                for weight_kl, temperature, zscore in filtered_kl_combinations:
                    experiment_count += 1

                    full_distill_config = replace(
                        config.distillation_config,
                        distill_method=method,
                        weight_ce=weight_ce,
                        weight_kl=weight_kl,
                        weight_mse=weight_mse,
                        temperature=temperature,
                        zscore=zscore,
                    )
                    hyperparam_config = asdict(full_distill_config)
                    all_experiments.append(
                        {
                            "task_name": task_name,
                            "hyperparam_config": hyperparam_config,
                            "experiment_count": experiment_count,
                        }
                    )

    return all_experiments
