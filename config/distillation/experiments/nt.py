from dataclasses import replace
from ..config_schema import DistillationExperimentConfig
from ..glm import nt_2b5
from ..bpnet import (
    original_bpnet_classifier_config,
)
from ..trainer import nt_trainer_config, nt_debug_trainer_config
from ..data import nucletide_transformer_revised_benchmark
from ..distillation_model import (
    dist_model_config,
    vanilla_distillation_model_config,
    logits_standardization_model_config,
    dkd_model_config,
)
from ...env import project_path, output_path
from ...slurm import basic_distillation_slurm

# NT parent path for checkpoints
# NT_PARENT_PATH = f"{project_path}/data/finetuned_models/2b5-multi-species_nucleotide-transformer-finetune-results-lora-epoch10-3-22-revised-r32-fix-num-label-v2/2b5-multi-species_nucleotide-transformer-finetune-results-lora-epoch10-3-22-revised-r32-fix-num-label"
NT_PARENT_PATH = f"{project_path}/data/finetuned_models/2b5-multi-species_nucleotide-transformer-finetune-results-lora-epoch20-10-17-revised-r32-fix-num-label"
# Base NT experiment configuration
nt_base_config = DistillationExperimentConfig(
    task_names=[
        "H3K4me2",
        "H3K4me3",
        "H3K9ac",
        "H3K9me3",
        "promoter_all",
        "promoter_tata",
    ],
    teacher_config=nt_2b5,
    teacher_parent_dir=NT_PARENT_PATH,
    model_type="nt",  # Specify NT model type
    student_config=original_bpnet_classifier_config,
    distillation_config=vanilla_distillation_model_config,
    trainer_config=nt_trainer_config,
    dataset_config=nucletide_transformer_revised_benchmark,
    slurm_config=basic_distillation_slurm,
)

# Different distillation methods
nt_logit_standard = replace(
    nt_base_config,
    distillation_config=logits_standardization_model_config,
    trainer_config=replace(
        nt_base_config.trainer_config,
        output_dir=f"{output_path}/nt_distillation/logit_standard",
    ),
)

nt_dkd = replace(
    nt_base_config,
    distillation_config=dkd_model_config,
    trainer_config=replace(
        nt_base_config.trainer_config,
        output_dir=f"{output_path}/nt_distillation/dkd",
    ),
)

nt_dist = replace(
    nt_base_config,
    distillation_config=dist_model_config,
    trainer_config=replace(
        nt_base_config.trainer_config,
        output_dir=f"{output_path}/nt_distillation/dist",
    ),
)
nt_debug_config = DistillationExperimentConfig(
    task_names=["promoter_all"],
    dataset_config=nucletide_transformer_revised_benchmark,
    teacher_config=nt_2b5,
    model_type="NT",
    teacher_parent_dir=NT_PARENT_PATH,
    student_config=original_bpnet_classifier_config,
    distillation_config=vanilla_distillation_model_config,
    trainer_config=nt_debug_trainer_config,
    slurm_config=basic_distillation_slurm,
)
experiment_configs = {
    "nt_vanilla": (
        "NT distillation with vanilla KD and original BPNet",
        nt_base_config,
    ),
    "nt_logit_standard": (
        "NT distillation with logit standardization",
        nt_logit_standard,
    ),
    "nt_dkd": (
        "NT distillation with decoupled KD",
        nt_dkd,
    ),
    "nt_dist": (
        "NT distillation with DIST method",
        nt_dist,
    ),
    "nt_debug": (
        "Debug Mode for NT distillation with vanilla KD and original BPNet",
        nt_debug_config,
    ),
}
