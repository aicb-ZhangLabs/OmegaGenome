import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import tyro
import itertools

from dataclasses import replace
from config.distillation.config import hyperparam_configs
from config.distillation.config_schema import DistillationHyperparamExperimentConfig

from .distill import main as distill_main


def main(config: DistillationHyperparamExperimentConfig):
    # Cartesian product
    kl_combinations = list(
        itertools.product(
            config.weight_kls,
            config.temperatures,
            config.zscores,
        )
    )

    # filter out the combinations where weight_kl is 0.0 and the first combination is not covered
    # this is to avoid running the same combination multiple times
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

    other_combinations = list(
        itertools.product(
            config.weight_ces,
            config.weight_mses,
        )
    )

    print(f"Running {len(kl_combinations) * len(other_combinations)} experiments...")
    for weight_ce, weight_mse in other_combinations:
        for weight_kl, temperature, zscore in kl_combinations:
            new_distill_config = replace(
                config.distillation_config,
                weight_ce=weight_ce,
                weight_kl=weight_kl,
                weight_mse=weight_mse,
                temperature=temperature,
                zscore=zscore,
            )

            new_experiment_config = replace(config, distillation_config=new_distill_config)
            distill_main(new_experiment_config)


if __name__ == "__main__":
    config = tyro.extras.overridable_config_cli(hyperparam_configs, sort_subcommands=True)
    main(config)
