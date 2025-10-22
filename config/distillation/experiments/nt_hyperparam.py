from ..config_schema import DistillationHyperparamExperimentConfig
from ..glm import nt_2b5
from ..bpnet import original_bpnet_classifier_config
from ..trainer import (
    nt_hyperparam_trainer_config,
    nt_hyperparam_dkd_trainer_config,
)
from ..data import nucletide_transformer_revised_benchmark
from ..distillation_model import (
    DistillationModelConfig,
    vanilla_distillation_model_config,
)
from ...slurm import basic_distillation_slurm
from .nt import NT_PARENT_PATH


experiment_configs = {
    "nt_hyperparam": (
        "NT distillation hyperparameter search",
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
            teacher_config=nt_2b5,
            teacher_parent_dir=NT_PARENT_PATH,
            model_type="nt",
            student_config=original_bpnet_classifier_config,
            distillation_config=vanilla_distillation_model_config,
            trainer_config=nt_hyperparam_trainer_config,
            slurm_config=basic_distillation_slurm,
            # Grid search parameters
            weight_ces=[0.5],
            weight_kls=[0.0, 0.25, 0.5, 1.0],
            weight_mses=[0.0, 1, 2, 5],
            temperatures=[0.5, 1.0, 1.5, 2.0, 4.0],
            zscores=[False],
        ),
    ),
    "nt_hyperparam_dkd": (
        "NT distillation with DKD hyperparameter search",
        DistillationHyperparamExperimentConfig(
            task_names=["promoter_all", "H3K4me3"],
            dataset_config=nucletide_transformer_revised_benchmark,
            teacher_config=nt_2b5,
            teacher_parent_dir=NT_PARENT_PATH,
            model_type="nt",
            student_config=original_bpnet_classifier_config,
            distillation_config=DistillationModelConfig(
                distill_method="dkd",
                weight_mse=0.0,  # DKD doesn't use feature matching
            ),
            trainer_config=nt_hyperparam_dkd_trainer_config,
            slurm_config=basic_distillation_slurm,
            # DKD-specific hyperparameters
            weight_ces=[0.5],
            weight_kls=[0.0, 0.25, 0.5, 1.0],
            weight_mses=[0.0],  # DKD doesn't use MSE
            temperatures=[0.5, 1.0, 1.5, 2.0, 4.0],
            zscores=[False],
            # Additional DKD parameters (need to add to config schema)
            # dkd_alphas=[0.5, 1.0, 2.0],
            # dkd_betas=[4.0, 8.0, 16.0],
        ),
    ),
}
