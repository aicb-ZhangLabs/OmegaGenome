from dataclasses import replace
from ..config_schema import DistillationExperimentConfig
from ..glm import GLMConfig, nt_2b5
from ..bpnet import BPNetClassifierConfig, bpnet_classifier_config
from ..trainer import DistillTrainerConfig, trainer_config, debug_trainer_config
from ..data import DatasetConfig, nucletide_transformer_revised_benchmark
from ..distillation_model import DistillationModelConfig, distillation_model_config
from ...slurm import SlurmConfig
from ...env import project_path
from ...slurm import SlurmConfig, basic_distillation_slurm

# NT parent path for checkpoints
NT_PARENT_PATH = f"{project_path}/data/finetuned_models/2b5-multi-species_nucleotide-transformer-finetune-results-lora-epoch10-3-22-revised-r32-fix-num-label-v2/2b5-multi-species_nucleotide-transformer-finetune-results-lora-epoch10-3-22-revised-r32-fix-num-label"

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
    student_config=BPNetClassifierConfig(
        num_labels=2,
        model_type="bpnet",
        model_size="original",  # Default to original
    ),
    distillation_config=DistillationModelConfig(
        weight_ce=0.5,
        weight_kl=0.5,
        weight_mse=0.0,
        temperature=2.0,
        distill_method="vanilla",
    ),
    trainer_config=DistillTrainerConfig(
        output_dir=f"{project_path}/outputs/nt_distillation/vanilla_original",
        wandb_project="OmegaGenome-NT",
        epochs=200,
        batch_size=16,
        lr=1e-4,
        max_len=1000,
    ),
    dataset_config=nucletide_transformer_revised_benchmark,
    slurm_config=basic_distillation_slurm,
)

# Different distillation methods
nt_logit_standard = replace(
    nt_base_config,
    distillation_config=DistillationModelConfig(
        weight_ce=0.5,
        weight_kl=0.5,
        weight_mse=0.2,
        temperature=4.0,
        distill_method="logit_standard",
    ),
    trainer_config=replace(
        nt_base_config.trainer_config,
        output_dir=f"{project_path}/outputs/nt_distillation/logit_standard",
    ),
)

nt_dkd = replace(
    nt_base_config,
    distillation_config=DistillationModelConfig(
        weight_ce=0.5,
        weight_kl=0.5,
        weight_mse=0.0,  # DKD doesn't use feature matching
        temperature=4.0,
        distill_method="dkd",
        dkd_alpha=1.0,
        dkd_beta=8.0,
    ),
    trainer_config=replace(
        nt_base_config.trainer_config,
        output_dir=f"{project_path}/outputs/nt_distillation/dkd",
    ),
)

nt_dist = replace(
    nt_base_config,
    distillation_config=DistillationModelConfig(
        weight_ce=0.5,
        weight_kl=0.5,
        weight_mse=0.0,
        temperature=4.0,
        distill_method="dist",
    ),
    trainer_config=replace(
        nt_base_config.trainer_config,
        output_dir=f"{project_path}/outputs/nt_distillation/dist",
    ),
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
}
