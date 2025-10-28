from dataclasses import replace
from ..config_schema import DistillationExperimentConfig
from ..glm import caduceus
from ..bpnet import (
    original_bpnet_classifier_config,
)
from ..trainer import (
    caduceus_trainer_config,
)
from ..data import nucletide_transformer_revised_benchmark
from ..distillation_model import (
    dist_model_config,
    vanilla_distillation_model_config,
    logits_standardization_model_config,
    dkd_model_config,
)
from ...env import project_path, output_path
from ...slurm import basic_distillation_slurm

# Caduceus parent path for checkpoints
CADUCEUS_PARENT_PATH = f"{project_path}/data/finetuned_models/caduceus_finetune_results"

# Base Caduceus experiment configuration
caduceus_base_config = DistillationExperimentConfig(
    task_names=[
        "H3K4me2",
        "H3K4me3",
        "H3K9ac",
        "H3K9me3",
        "promoter_all",
        "promoter_tata",
    ],
    teacher_config=caduceus,
    teacher_parent_dir=CADUCEUS_PARENT_PATH,
    model_type="caduceus",  # Specify Caduceus model type
    student_config=original_bpnet_classifier_config,
    distillation_config=vanilla_distillation_model_config,
    trainer_config=caduceus_trainer_config,
    dataset_config=nucletide_transformer_revised_benchmark,
    slurm_config=basic_distillation_slurm,
)

# Different distillation methods
caduceus_logit_standard = replace(
    caduceus_base_config,
    distillation_config=logits_standardization_model_config,
    trainer_config=replace(
        caduceus_base_config.trainer_config,
        output_dir=f"{output_path}/caduceus_distillation/logit_standard",
    ),
)

caduceus_dkd = replace(
    caduceus_base_config,
    distillation_config=dkd_model_config,
    trainer_config=replace(
        caduceus_base_config.trainer_config,
        output_dir=f"{output_path}/caduceus_distillation/dkd",
    ),
)

caduceus_dist = replace(
    caduceus_base_config,
    distillation_config=dist_model_config,
    trainer_config=replace(
        caduceus_base_config.trainer_config,
        output_dir=f"{output_path}/caduceus_distillation/dist",
    ),
)

# Full task list configuration
caduceus_all_tasks = replace(
    caduceus_base_config,
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
        caduceus_base_config.trainer_config,
        output_dir=f"{output_path}/caduceus_distillation/all_tasks",
    ),
)

experiment_configs = {
    "caduceus_vanilla": (
        "Caduceus distillation with vanilla KD and original BPNet",
        caduceus_base_config,
    ),
    "caduceus_logit_standard": (
        "Caduceus distillation with logit standardization",
        caduceus_logit_standard,
    ),
    "caduceus_dkd": (
        "Caduceus distillation with decoupled KD",
        caduceus_dkd,
    ),
    "caduceus_dist": (
        "Caduceus distillation with DIST method",
        caduceus_dist,
    ),
    "caduceus_all_tasks": (
        "Caduceus distillation with all 18 tasks",
        caduceus_all_tasks,
    ),
}
