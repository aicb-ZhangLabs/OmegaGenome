from dataclasses import replace
from ..config_schema import DistillationExperimentConfig
from ..glm import enformer
from ..bpnet import original_bpnet_classifier_config
from ..trainer import enformer_trainer_config, enformer_debug_trainer_config
from ..data import nucletide_transformer_revised_benchmark
from ..distillation_model import (
    dist_model_config,
    vanilla_distillation_model_config,
    logits_standardization_model_config,
    dkd_model_config,
)
from ...env import project_path, output_path
from ...slurm import basic_distillation_slurm

# Enformer parent path for checkpoints
ENFORMER_PARENT_PATH = (
    f"{project_path}/data/finetuned_models/enformer_finetune_results/enformer_finetune_results/"
)

# Base Enformer experiment configuration
enformer_base_config = DistillationExperimentConfig(
    task_names=[
        "H3K4me2",
        "H3K4me3",
        # Add more tasks as needed
    ],
    teacher_config=enformer,
    teacher_parent_dir=ENFORMER_PARENT_PATH,
    model_type="enformer",  # Specify Enformer model type
    student_config=original_bpnet_classifier_config,
    distillation_config=vanilla_distillation_model_config,
    trainer_config=enformer_trainer_config,
    dataset_config=nucletide_transformer_revised_benchmark,
    slurm_config=basic_distillation_slurm,
)

# Different distillation methods
enformer_logit_standard = replace(
    enformer_base_config,
    distillation_config=logits_standardization_model_config,
    trainer_config=replace(
        enformer_base_config.trainer_config,
        output_dir=f"{output_path}/enformer_distillation/logit_standard",
    ),
)

enformer_dkd = replace(
    enformer_base_config,
    distillation_config=dkd_model_config,
    trainer_config=replace(
        enformer_base_config.trainer_config,
        output_dir=f"{output_path}/enformer_distillation/dkd",
    ),
)

enformer_dist = replace(
    enformer_base_config,
    distillation_config=dist_model_config,
    trainer_config=replace(
        enformer_base_config.trainer_config,
        output_dir=f"{output_path}/enformer_distillation/dist",
    ),
)

# Full task list configuration
enformer_all_tasks = replace(
    enformer_base_config,
    task_names=[
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
    ],
    trainer_config=replace(
        enformer_base_config.trainer_config,
        output_dir=f"{output_path}/enformer_distillation/all_tasks",
    ),
)

# Debug configuration
enformer_debug_config = replace(
    enformer_base_config,
    task_names=["promoter_all"],
    trainer_config=enformer_debug_trainer_config,
)

experiment_configs = {
    "enformer_vanilla": (
        "Enformer distillation with vanilla KD and original BPNet",
        enformer_base_config,
    ),
    "enformer_logit_standard": (
        "Enformer distillation with logit standardization",
        enformer_logit_standard,
    ),
    "enformer_dkd": (
        "Enformer distillation with decoupled KD",
        enformer_dkd,
    ),
    "enformer_dist": (
        "Enformer distillation with DIST method",
        enformer_dist,
    ),
    "enformer_all_tasks": (
        "Enformer distillation with all 18 tasks",
        enformer_all_tasks,
    ),
    "enformer_debug": (
        "Debug Mode for Enformer distillation",
        enformer_debug_config,
    ),
}
