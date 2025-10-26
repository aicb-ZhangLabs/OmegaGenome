from dataclasses import replace
from ..config_schema import DistillationHyperparamExperimentConfig
from ..glm import caduceus
from ..bpnet import original_bpnet_classifier_config
from ..trainer import (
    caduceus_hyperparam_trainer_config,
    caduceus_hyperparam_dkd_trainer_config,
)
from ..data import nucletide_transformer_revised_benchmark
from ..distillation_model import (
    DistillationModelConfig,
    vanilla_distillation_model_config,
)
from ...env import output_path
from ...slurm import basic_distillation_slurm
from .caduceus import CADUCEUS_PARENT_PATH


experiment_configs = {
    "caduceus_hyperparam": (
        "Caduceus distillation hyperparameter search",
        DistillationHyperparamExperimentConfig(
            task_names=[
                "H3K27me3",
                "H3K36me3",
                "H4K20me1",
                "H2AFZ",
                "H3K27ac",
                "H3K4me1",
                "H3K4me2",
                "H3K4me3",
                "H3K9ac",
                "H3K9me3",
                "promoter_all",
                "promoter_tata",
                "promoter_no_tata",
                "enhancers",
                "enhancers_types",
                "splice_sites_all",
                "splice_sites_acceptors",
                "splice_sites_donors",
            ],
            dataset_config=nucletide_transformer_revised_benchmark,
            teacher_config=caduceus,
            teacher_parent_dir=CADUCEUS_PARENT_PATH,
            model_type="caduceus",
            student_config=original_bpnet_classifier_config,
            distillation_config=vanilla_distillation_model_config,
            trainer_config=caduceus_hyperparam_trainer_config,
            slurm_config=basic_distillation_slurm,
            # Grid search parameters
            weight_ces=[0.5, 0.7, 0.9],
            weight_kls=[0.0, 0.1, 0.3, 0.5],
            weight_mses=[0.0, 0.1, 0.2, 0.5],
            temperatures=[1.0, 2.0, 4.0, 8.0],
            zscores=[False],
        ),
    ),
    "caduceus_hyperparam_dkd": (
        "Caduceus distillation with DKD hyperparameter search",
        DistillationHyperparamExperimentConfig(
            task_names=["promoter_all", "H3K4me3", "H3K27me3", "enhancers"],
            dataset_config=nucletide_transformer_revised_benchmark,
            teacher_config=caduceus,
            teacher_parent_dir=CADUCEUS_PARENT_PATH,
            model_type="caduceus",
            student_config=original_bpnet_classifier_config,
            distillation_config=DistillationModelConfig(
                distill_method="dkd",
                weight_mse=0.0,  # DKD doesn't use feature matching
            ),
            trainer_config=caduceus_hyperparam_dkd_trainer_config,
            slurm_config=basic_distillation_slurm,
            # DKD-specific hyperparameters
            weight_ces=[0.5, 0.7, 0.9],
            weight_kls=[0.1, 0.3, 0.5],
            weight_mses=[0.0],  # DKD doesn't use MSE
            temperatures=[2.0, 4.0, 8.0],
            zscores=[False],
            # Additional DKD parameters (need to add to config schema if not present)
            # dkd_alphas=[0.5, 1.0, 2.0],
            # dkd_betas=[4.0, 8.0, 16.0],
        ),
    ),
    "caduceus_hyperparam_dist": (
        "Caduceus distillation with DIST hyperparameter search",
        DistillationHyperparamExperimentConfig(
            task_names=["promoter_all", "H3K4me3", "H3K27me3", "enhancers"],
            dataset_config=nucletide_transformer_revised_benchmark,
            teacher_config=caduceus,
            teacher_parent_dir=CADUCEUS_PARENT_PATH,
            model_type="caduceus",
            student_config=original_bpnet_classifier_config,
            distillation_config=DistillationModelConfig(
                distill_method="dist",
                weight_mse=0.0,
            ),
            trainer_config=replace(
                caduceus_hyperparam_trainer_config,
                output_dir=f"{output_path}/caduceus_distillation/hyperparam_dist",
            ),
            slurm_config=basic_distillation_slurm,
            weight_ces=[0.5, 0.7, 0.9],
            weight_kls=[0.1, 0.3, 0.5],
            weight_mses=[0.0],  # DIST doesn't use MSE
            temperatures=[1.0],  # DIST doesn't use temperature
            zscores=[False],
            # Additional DIST parameters (need to add to config schema if not present)
            # dist_betas=[1.0, 2.0, 4.0],
            # dist_gammas=[1.0, 2.0, 4.0],
        ),
    ),
}
