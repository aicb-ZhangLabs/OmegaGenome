from dataclasses import replace
from ..config_schema import DistillationExperimentConfig
from ..glm import GLMConfig, nt_2b5
from ..bpnet import BPNetClassifierConfig
from ..trainer import DistillTrainerConfig
from ..data import DatasetConfig, nucletide_transformer_revised_benchmark
from ..distillation_model import DistillationModelConfig
from ...slurm import SlurmConfig, basic_distillation_slurm
from ...env import project_path
from .nt import NT_PARENT_PATH


# NT_PARENT_PATH = f"{project_path}/data/finetuned_models/2b5-multi-species_nucleotide-transformer-finetune-results-lora-epoch10-3-22-revised-r32-fix-num-label"

# Base configuration
base_config = DistillationExperimentConfig(
    task_names=["promoter_all", "promoter_tata", "H3K4me3", "H3K9ac"],
    teacher_config=nt_2b5,
    teacher_parent_dir=NT_PARENT_PATH,
    model_type="nt",
    distillation_config=DistillationModelConfig(
        weight_ce=0.5,
        weight_kl=0.5,
        weight_mse=0.2,
        temperature=4.0,
        distill_method="vanilla",
    ),
    trainer_config=DistillTrainerConfig(
        output_dir=f"{project_path}/outputs/nt_distillation/hyperparam",
        wandb_project="OmegaGenome-NT-HyperParam",
        epochs=200,  # Fewer epochs for hyperparameter search
        batch_size=16,
        max_len=1000,
    ),
    dataset_config=nucletide_transformer_revised_benchmark,
    slurm_config=basic_distillation_slurm,
)

# Different BPNet sizes
nt_bpnet_original = replace(
    base_config,
    student_config=BPNetClassifierConfig(
        num_labels=2,
        model_type="bpnet",
        model_size="original",
    ),
    trainer_config=DistillTrainerConfig(
        output_dir=f"{project_path}/outputs/nt_distillation/bpnet_original",
        wandb_project="OmegaGenome-NT-Sizes",
        epochs=200,
        batch_size=16,
        lr=1e-4,
        max_len=1000,
    ),
)

nt_bpnet_tiny = replace(
    nt_bpnet_original,
    student_config=BPNetClassifierConfig(
        num_labels=2,
        model_type="bpnet",
        model_size="tiny",
    ),
    trainer_config=replace(
        nt_bpnet_original.trainer_config,
        output_dir=f"{project_path}/outputs/nt_distillation/bpnet_tiny",
    ),
)

nt_bpnet_small = replace(
    nt_bpnet_original,
    student_config=BPNetClassifierConfig(
        num_labels=2,
        model_type="bpnet",
        model_size="small",
    ),
    trainer_config=replace(
        nt_bpnet_original.trainer_config,
        output_dir=f"{project_path}/outputs/nt_distillation/bpnet_small",
    ),
)

nt_bpnet_medium = replace(
    nt_bpnet_original,
    student_config=BPNetClassifierConfig(
        num_labels=2,
        model_type="bpnet",
        model_size="medium",
    ),
    trainer_config=replace(
        nt_bpnet_original.trainer_config,
        output_dir=f"{project_path}/outputs/nt_distillation/bpnet_medium",
    ),
)

nt_bpnet_large = replace(
    nt_bpnet_original,
    student_config=BPNetClassifierConfig(
        num_labels=2,
        model_type="bpnet",
        model_size="large",
    ),
    trainer_config=replace(
        nt_bpnet_original.trainer_config,
        output_dir=f"{project_path}/outputs/nt_distillation/bpnet_large",
        batch_size=8,  # Smaller batch size for large model
    ),
)

# Different architectures
nt_bilstm_small = replace(
    base_config,
    student_config=BPNetClassifierConfig(
        num_labels=2,
        model_type="bilstm",
        model_size="small",
    ),
    trainer_config=DistillTrainerConfig(
        output_dir=f"{project_path}/outputs/nt_distillation/bilstm_small",
        wandb_project="OmegaGenome-NT-Architectures",
        epochs=200,
        batch_size=16,
        lr=1e-4,
        max_len=1000,
    ),
)

nt_bilstm_medium = replace(
    nt_bilstm_small,
    student_config=BPNetClassifierConfig(
        num_labels=2,
        model_type="bilstm",
        model_size="medium",
    ),
    trainer_config=replace(
        nt_bilstm_small.trainer_config,
        output_dir=f"{project_path}/outputs/nt_distillation/bilstm_medium",
    ),
)

nt_bilstm_large = replace(
    nt_bilstm_small,
    student_config=BPNetClassifierConfig(
        num_labels=2,
        model_type="bilstm",
        model_size="large",
    ),
    trainer_config=replace(
        nt_bilstm_small.trainer_config,
        output_dir=f"{project_path}/outputs/nt_distillation/bilstm_large",
        batch_size=8,
    ),
)

nt_cnn_small = replace(
    base_config,
    student_config=BPNetClassifierConfig(
        num_labels=2,
        model_type="cnn",
        model_size="small",
    ),
    trainer_config=DistillTrainerConfig(
        output_dir=f"{project_path}/outputs/nt_distillation/cnn_small",
        wandb_project="OmegaGenome-NT-Architectures",
        epochs=200,
        batch_size=16,
        lr=1e-4,
        max_len=1000,
    ),
)

nt_cnn_medium = replace(
    nt_cnn_small,
    student_config=BPNetClassifierConfig(
        num_labels=2,
        model_type="cnn",
        model_size="medium",
    ),
    trainer_config=replace(
        nt_cnn_small.trainer_config,
        output_dir=f"{project_path}/outputs/nt_distillation/cnn_medium",
    ),
)

nt_cnn_large = replace(
    nt_cnn_small,
    student_config=BPNetClassifierConfig(
        num_labels=2,
        model_type="cnn",
        model_size="large",
    ),
    trainer_config=replace(
        nt_cnn_small.trainer_config,
        output_dir=f"{project_path}/outputs/nt_distillation/cnn_large",
        batch_size=8,
    ),
)

experiment_configs = {
    # BPNet different sizes
    "nt_bpnet_original": (
        "NT distillation with original BPNet (~280k params)",
        nt_bpnet_original,
    ),
    "nt_bpnet_tiny": (
        "NT distillation with tiny BPNet (~50k params)",
        nt_bpnet_tiny,
    ),
    "nt_bpnet_small": (
        "NT distillation with small BPNet (~200k params)",
        nt_bpnet_small,
    ),
    "nt_bpnet_medium": (
        "NT distillation with medium BPNet (~1M params)",
        nt_bpnet_medium,
    ),
    "nt_bpnet_large": (
        "NT distillation with large BPNet (~5M params)",
        nt_bpnet_large,
    ),
    # BiLSTM architectures
    "nt_bilstm_small": (
        "NT distillation with small BiLSTM",
        nt_bilstm_small,
    ),
    "nt_bilstm_medium": (
        "NT distillation with medium BiLSTM",
        nt_bilstm_medium,
    ),
    "nt_bilstm_large": (
        "NT distillation with large BiLSTM",
        nt_bilstm_large,
    ),
    # CNN architectures
    "nt_cnn_small": (
        "NT distillation with small CNN",
        nt_cnn_small,
    ),
    "nt_cnn_medium": (
        "NT distillation with medium CNN",
        nt_cnn_medium,
    ),
    "nt_cnn_large": (
        "NT distillation with large CNN",
        nt_cnn_large,
    ),
}
