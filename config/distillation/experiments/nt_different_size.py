from dataclasses import replace
from ..config_schema import (
    DistillationExperimentConfig,
    DistillationHyperparamExperimentConfig,
)
from ..glm import nt_2b5
from ..bpnet import (
    original_bpnet_classifier_config,
)
from ..trainer import (
    nt_different_size_trainer_config,
    nt_different_size_original_trainer_config,
    nt_hyperparam_trainer_config,
)
from ..data import nucletide_transformer_revised_benchmark
from ..distillation_model import (
    nt_different_size_model_config,
    vanilla_distillation_model_config,
)
from ...slurm import basic_distillation_slurm
from ...env import output_path
from .nt import NT_PARENT_PATH

# Base configuration
base_config = DistillationExperimentConfig(
    task_names=["splice_sites_all"],  # "promoter_all", "promoter_tata", "H3K4me3", "H3K9ac"
    teacher_config=nt_2b5,
    teacher_parent_dir=NT_PARENT_PATH,
    model_type="nt",
    distillation_config=nt_different_size_model_config,
    trainer_config=nt_different_size_trainer_config,
    dataset_config=nucletide_transformer_revised_benchmark,
    slurm_config=basic_distillation_slurm,
)

base_hyperparam_config = DistillationHyperparamExperimentConfig(
    task_names=[
        "splice_sites_all",
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
)


# Different BPNet sizes
nt_bpnet_original = replace(
    base_config,
    student_config=original_bpnet_classifier_config,
    trainer_config=nt_different_size_original_trainer_config,
)

nt_bpnet_tiny = replace(
    nt_bpnet_original,
    student_config=replace(
        nt_bpnet_original.student_config,
        model_size="tiny",
    ),
    trainer_config=replace(
        nt_bpnet_original.trainer_config,
        output_dir=f"{output_path}/nt_distillation/different_size/bpnet/tiny/",
    ),
)

nt_bpnet_small = replace(
    nt_bpnet_original,
    student_config=replace(
        nt_bpnet_original.student_config,
        model_size="small",
    ),
    trainer_config=replace(
        nt_bpnet_original.trainer_config,
        output_dir=f"{output_path}/nt_distillation/different_size/bpnet/small/",
    ),
)

nt_bpnet_small_hyperparam = replace(
    base_hyperparam_config,
    student_config=replace(
        nt_bpnet_original.student_config,
        model_size="small",
    ),
    trainer_config=replace(
        nt_bpnet_original.trainer_config,
        output_dir=f"{output_path}/nt_distillation/different_size/bpnet/small/hyperparam/",
    ),
)
nt_bpnet_medium = replace(
    nt_bpnet_original,
    student_config=replace(
        nt_bpnet_original.student_config,
        model_size="medium",
    ),
    trainer_config=replace(
        nt_bpnet_original.trainer_config,
        output_dir=f"{output_path}/nt_distillation/different_size/bpnet/medium/",
    ),
)
nt_bpnet_medium_hyperparam = replace(
    base_hyperparam_config,
    student_config=replace(
        nt_bpnet_original.student_config,
        model_size="medium",
    ),
    trainer_config=replace(
        nt_bpnet_original.trainer_config,
        output_dir=f"{output_path}/nt_distillation/different_size/bpnet/medium/hyperparam/",
    ),
)
nt_bpnet_large = replace(
    nt_bpnet_original,
    student_config=replace(
        nt_bpnet_original.student_config,
        model_size="large",
    ),
    trainer_config=replace(
        nt_bpnet_original.trainer_config,
        output_dir=f"{output_path}/nt_distillation/different_size/bpnet/large/",
        # batch_size=8,  # Smaller batch size for large model
    ),
)

nt_bpnet_large_hyperparam = replace(
    base_hyperparam_config,
    student_config=replace(
        nt_bpnet_original.student_config,
        model_size="large",
    ),
    trainer_config=replace(
        nt_bpnet_original.trainer_config,
        output_dir=f"{output_path}/nt_distillation/different_size/bpnet/large/hyperparam/",
        # batch_size=8,  # Smaller batch size for large model
    ),
)

# Different architectures
nt_bilstm_small = replace(
    base_config,
    student_config=replace(
        nt_bpnet_small.student_config,
        model_type="bilstm",
    ),
    trainer_config=replace(
        nt_bpnet_original.trainer_config,
        output_dir=f"{output_path}/nt_distillation/different_size/bilstm/small/",
        wandb_project="OmegaGenome-NT-Architectures",
    ),
)

nt_bilstm_medium = replace(
    nt_bilstm_small,
    student_config=replace(
        nt_bilstm_small.student_config,
        model_size="medium",
    ),
    trainer_config=replace(
        nt_bilstm_small.trainer_config,
        output_dir=f"{output_path}/nt_distillation/different_size/bilstm/medium/",
    ),
)

nt_bilstm_large = replace(
    nt_bilstm_small,
    student_config=replace(
        nt_bilstm_small.student_config,
        model_size="large",
    ),
    trainer_config=replace(
        nt_bilstm_small.trainer_config,
        output_dir=f"{output_path}/nt_distillation/different_size/bilstm/large/",
    ),
)

nt_cnn_small = replace(
    base_config,
    student_config=replace(
        nt_bpnet_small.student_config,
        model_type="cnn",
    ),
    trainer_config=replace(
        nt_bilstm_small.trainer_config,
        output_dir=f"{output_path}/nt_distillation/different_size/cnn/small/",
    ),
)

nt_cnn_medium = replace(
    nt_cnn_small,
    student_config=replace(
        nt_cnn_small.student_config,
        model_size="medium",
    ),
    trainer_config=replace(
        nt_cnn_small.trainer_config,
        output_dir=f"{output_path}/nt_distillation/different_size/cnn/medium/",
    ),
)

nt_cnn_large = replace(
    nt_cnn_small,
    student_config=replace(
        nt_cnn_small.student_config,
        model_size="large",
    ),
    trainer_config=replace(
        nt_cnn_small.trainer_config,
        output_dir=f"{output_path}/nt_distillation/different_size/cnn/large/",
        # batch_size=8,
    ),
)
small_medium_large_bpnet_experiment_configs = {
    "nt_bpnet_small_hyperparam": (
        "NT distillation with small BPNet (~200k params)",
        nt_bpnet_small_hyperparam,
    ),
    "nt_bpnet_medium_hyperparam": (
        "NT distillation with medium BPNet (~1M params)",
        nt_bpnet_medium_hyperparam,
    ),
    "nt_bpnet_large_hyperparam": (
        "NT distillation with large BPNet (~5M params)",
        nt_bpnet_large_hyperparam,
    ),
}
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
