"""
NT distillation method hyperparameter experiments.

This module provides hyperparameter search configurations for different
distillation methods using NT (Nucleotide Transformer) as teacher:
- Vanilla KD: Standard knowledge distillation
- Logit Standardization: Per-sample z-score normalization
- DKD: Decoupled knowledge distillation
- DIST: Correlation-based distillation
"""

from config.distillation.config_schema import DistillationHyperparamExperimentConfig
from config.distillation.glm import nt_2b5
from config.distillation.bpnet import original_bpnet_classifier_config
from config.distillation.trainer import DistillTrainerConfig
from config.distillation.data import nucletide_transformer_revised_benchmark
from config.distillation.distillation_model import DistillationModelConfig
from config.slurm import basic_distillation_slurm
from config.distillation.experiments.nt import NT_PARENT_PATH
from config.env import output_path

# Base trainer config for method experiments
method_hyperparam_trainer_config = DistillTrainerConfig(
    output_dir=f"{output_path}/nt_distillation/method_hyperparam",
    wandb_project="OmegaGenome-NT-Method-HyperParam",
    epochs=200,
    batch_size=16,
    max_len=1000,
)

# All 18 genomic tasks
ALL_TASKS = [
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
]
DIFF_METHOD_TASKS = [
    "splice_sites_donors",
]
# Common hyperparameter grid (for methods that use all parameters)
COMMON_HYPERPARAM_GRID = {
    "weight_ces": [0.5],
    "weight_kls": [0.0, 0.25, 0.5, 1.0],
    "weight_mses": [0.0, 1, 2, 5],
    "temperatures": [0.5, 1.0, 1.5, 2.0, 4.0],
    "zscores": [False],
}

# Reduced hyperparameter grid (for methods without feature matching)
REDUCED_HYPERPARAM_GRID = {
    "weight_ces": [0.5],
    "weight_kls": [0.0, 0.25, 0.5, 1.0],
    "weight_mses": [0.0],  # These methods don't use MSE
    "temperatures": [0.5, 1.0, 1.5, 2.0, 4.0],
    "zscores": [False],
}

# ===== Vanilla KD =====
vanilla_method_config = DistillationHyperparamExperimentConfig(
    task_names=DIFF_METHOD_TASKS,
    dataset_config=nucletide_transformer_revised_benchmark,
    teacher_config=nt_2b5,
    teacher_parent_dir=NT_PARENT_PATH,
    model_type="nt",
    student_config=original_bpnet_classifier_config,
    distillation_config=DistillationModelConfig(distill_method="vanilla"),
    trainer_config=method_hyperparam_trainer_config,
    slurm_config=basic_distillation_slurm,
    **COMMON_HYPERPARAM_GRID,
)

# ===== Logit Standardization =====
logit_standard_method_config = DistillationHyperparamExperimentConfig(
    task_names=DIFF_METHOD_TASKS,
    dataset_config=nucletide_transformer_revised_benchmark,
    teacher_config=nt_2b5,
    teacher_parent_dir=NT_PARENT_PATH,
    model_type="nt",
    student_config=original_bpnet_classifier_config,
    distillation_config=DistillationModelConfig(distill_method="logit_standard"),
    trainer_config=method_hyperparam_trainer_config,
    slurm_config=basic_distillation_slurm,
    **COMMON_HYPERPARAM_GRID,
)

# ===== DKD (Decoupled Knowledge Distillation) =====
dkd_method_config = DistillationHyperparamExperimentConfig(
    task_names=DIFF_METHOD_TASKS,
    dataset_config=nucletide_transformer_revised_benchmark,
    teacher_config=nt_2b5,
    teacher_parent_dir=NT_PARENT_PATH,
    model_type="nt",
    student_config=original_bpnet_classifier_config,
    distillation_config=DistillationModelConfig(
        distill_method="dkd",
        weight_mse=0.0,  # DKD doesn't use feature matching
    ),
    trainer_config=method_hyperparam_trainer_config,
    slurm_config=basic_distillation_slurm,
    **REDUCED_HYPERPARAM_GRID,
)

# ===== DIST (Correlation-based Distillation) =====
dist_method_config = DistillationHyperparamExperimentConfig(
    task_names=DIFF_METHOD_TASKS,
    dataset_config=nucletide_transformer_revised_benchmark,
    teacher_config=nt_2b5,
    teacher_parent_dir=NT_PARENT_PATH,
    model_type="nt",
    student_config=original_bpnet_classifier_config,
    distillation_config=DistillationModelConfig(
        distill_method="dist",
        weight_mse=0.0,  # DIST uses correlation loss instead
    ),
    trainer_config=method_hyperparam_trainer_config,
    slurm_config=basic_distillation_slurm,
    **REDUCED_HYPERPARAM_GRID,
)

# Export configurations
experiment_configs = {
    "nt_method_vanilla_hyperparam": (
        "NT distillation with Vanilla KD - method hyperparameter search",
        vanilla_method_config,
    ),
    "nt_method_logit_standard_hyperparam": (
        "NT distillation with Logit Standardization - method hyperparameter search",
        logit_standard_method_config,
    ),
    "nt_method_dkd_hyperparam": (
        "NT distillation with DKD - method hyperparameter search",
        dkd_method_config,
    ),
    "nt_method_dist_hyperparam": (
        "NT distillation with DIST - method hyperparameter search",
        dist_method_config,
    ),
}
