import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import tyro
import itertools

from dataclasses import replace
from config.distillation.config import hyperparam_configs
from config.distillation.config_schema import DistillationHyperparamExperimentConfig

from .distill import main as distill_main


def main(config: DistillationHyperparamExperimentConfig):
    # Cartesian product of the three lists
    combinations = list(
        itertools.product(
            config.weight_ces,
            config.weight_kls,
            config.weight_mses,
            config.temperatures,
            config.zscores,
        )
    )

    print(f"Running {len(combinations)} experiments...")
    for combination in combinations:
        new_distill_config = replace(
            config.distillation_config,
            weight_ce=combination[0],
            weight_kl=combination[1],
            weight_mse=combination[2],
            temperature=combination[3],
            zscore=combination[4],
        )

        new_experiment_config = replace(config, distillation_config=new_distill_config)
        distill_main(new_experiment_config)


if __name__ == "__main__":
    config = tyro.extras.overridable_config_cli(
        hyperparam_configs, sort_subcommands=True
    )
    main(config)
