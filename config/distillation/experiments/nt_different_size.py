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
    task_names=[
        "splice_sites_all"
    ],  # "promoter_all", "promoter_tata", "H3K4me3", "H3K9ac"
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
base_hyperparam_extend_config = DistillationHyperparamExperimentConfig(
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
    weight_mses=[0.0, 0.1, 0.25, 0.5, 1, 2, 5],
    temperatures=[0.5, 1.0, 1.5, 2.0, 4.0],
    zscores=[False],
)

base_hyperparam_extend_solely_config = DistillationHyperparamExperimentConfig(
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
    weight_mses=[0.0, 0.1, 0.25],
    temperatures=[0.5, 1.0, 1.5, 2.0, 4.0],
    zscores=[False],
)

different_size_hyperparam_config = replace(
    base_hyperparam_config,
    trainer_config=replace(
        nt_hyperparam_trainer_config, wandb_project="OmegaGenome-NT-Different-Size"
    ),
)

different_size_hyperparam_extend_config = replace(
    base_hyperparam_extend_config,
    trainer_config=replace(
        nt_hyperparam_trainer_config, wandb_project="OmegaGenome-NT-Different-Size"
    ),
)
different_size_hyperparam_extend_solely_config = replace(
    base_hyperparam_extend_solely_config,
    trainer_config=replace(
        nt_hyperparam_trainer_config, wandb_project="OmegaGenome-NT-Different-Size"
    ),
)

# Different BPNet sizes
nt_bpnet_original = replace(
    base_config,
    student_config=original_bpnet_classifier_config,
    trainer_config=nt_different_size_original_trainer_config,
)
nt_bpnet_original_hyperparam = replace(
    different_size_hyperparam_config,
    student_config=original_bpnet_classifier_config,
    trainer_config=replace(
        different_size_hyperparam_config.trainer_config,
        output_dir=f"{output_path}/nt_distillation/different_size/bpnet/original/hyperparam/",
        wandb_project="OmegaGenome-NT-Different-Size-Original",
    ),
)
nt_bpnet_original_hyperparam_extend = replace(
    different_size_hyperparam_extend_config,
    student_config=original_bpnet_classifier_config,
    trainer_config=replace(
        different_size_hyperparam_extend_config.trainer_config,
        output_dir=f"{output_path}/nt_distillation/different_size/bpnet/original/hyperparam/",
        wandb_project="OmegaGenome-NT-Different-Size-Original-Extend",
    ),
)
nt_bpnet_original_hyperparam_extend_solely = replace(
    different_size_hyperparam_extend_solely_config,
    student_config=original_bpnet_classifier_config,
    trainer_config=replace(
        different_size_hyperparam_extend_solely_config.trainer_config,
        output_dir=f"{output_path}/nt_distillation/different_size/bpnet/original/hyperparam/",
        wandb_project="OmegaGenome-NT-Different-Size-Original-Extend-Solely",
    ),
)

nt_bpnet_pico = replace(
    nt_bpnet_original,
    student_config=replace(
        nt_bpnet_original.student_config,
        model_size="pico",
    ),
    trainer_config=replace(
        nt_bpnet_original.trainer_config,
        output_dir=f"{output_path}/nt_distillation/different_size/bpnet/pico/",
    ),
)

nt_bpnet_pico_hyperparam = replace(
    different_size_hyperparam_config,
    student_config=replace(
        different_size_hyperparam_config.student_config,
        model_size="pico",
    ),
    trainer_config=replace(
        different_size_hyperparam_config.trainer_config,
        output_dir=f"{output_path}/nt_distillation/different_size/bpnet/pico/hyperparam/",
    ),
)
nt_bpnet_ultra_tiny = replace(
    nt_bpnet_original,
    student_config=replace(
        nt_bpnet_original.student_config,
        model_size="ultra_tiny",
    ),
    trainer_config=replace(
        nt_bpnet_original.trainer_config,
        output_dir=f"{output_path}/nt_distillation/different_size/bpnet/ultra_tiny/",
    ),
)

nt_bpnet_extra_tiny = replace(
    nt_bpnet_original,
    student_config=replace(
        nt_bpnet_original.student_config,
        model_size="extra_tiny",
    ),
    trainer_config=replace(
        nt_bpnet_original.trainer_config,
        output_dir=f"{output_path}/nt_distillation/different_size/bpnet/extra_tiny/",
    ),
)

nt_bpnet_medium_small = replace(
    nt_bpnet_original,
    student_config=replace(
        nt_bpnet_original.student_config,
        model_size="medium_small",
    ),
    trainer_config=replace(
        nt_bpnet_original.trainer_config,
        output_dir=f"{output_path}/nt_distillation/different_size/bpnet/medium_small/",
    ),
)

# Hyperparameter search variants
nt_bpnet_ultra_tiny_hyperparam = replace(
    different_size_hyperparam_config,
    student_config=replace(
        different_size_hyperparam_config.student_config,
        model_size="ultra_tiny",
    ),
    trainer_config=replace(
        different_size_hyperparam_config.trainer_config,
        output_dir=f"{output_path}/nt_distillation/different_size/bpnet/ultra_tiny/hyperparam/",
    ),
)

nt_bpnet_extra_tiny_hyperparam = replace(
    different_size_hyperparam_config,
    student_config=replace(
        different_size_hyperparam_config.student_config,
        model_size="extra_tiny",
    ),
    trainer_config=replace(
        different_size_hyperparam_config.trainer_config,
        output_dir=f"{output_path}/nt_distillation/different_size/bpnet/extra_tiny/hyperparam/",
    ),
)

nt_bpnet_medium_small_hyperparam = replace(
    different_size_hyperparam_config,
    student_config=replace(
        different_size_hyperparam_config.student_config,
        model_size="medium_small",
    ),
    trainer_config=replace(
        different_size_hyperparam_config.trainer_config,
        output_dir=f"{output_path}/nt_distillation/different_size/bpnet/medium_small/hyperparam/",
    ),
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
    different_size_hyperparam_config,
    student_config=replace(
        different_size_hyperparam_config.student_config,
        model_size="small",
    ),
    trainer_config=replace(
        different_size_hyperparam_config.trainer_config,
        output_dir=f"{output_path}/nt_distillation/different_size/bpnet/small/hyperparam/",
        wandb_project="OmegaGenome-NT-Different-Size",
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
    different_size_hyperparam_config,
    student_config=replace(
        different_size_hyperparam_config.student_config,
        model_size="medium",
    ),
    trainer_config=replace(
        different_size_hyperparam_config.trainer_config,
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
    different_size_hyperparam_config,
    student_config=replace(
        different_size_hyperparam_config.student_config,
        model_size="large",
    ),
    trainer_config=replace(
        different_size_hyperparam_config.trainer_config,
        output_dir=f"{output_path}/nt_distillation/different_size/bpnet/large/hyperparam/",
        # batch_size=8,  # Smaller batch size for large model
    ),
)

nt_bpnet_medium_large_hyperparam = replace(
    different_size_hyperparam_config,
    student_config=replace(
        different_size_hyperparam_config.student_config,
        model_size="medium_large",
    ),
    trainer_config=replace(
        different_size_hyperparam_config.trainer_config,
        output_dir=f"{output_path}/nt_distillation/different_size/bpnet/medium_large/hyperparam/",
    ),
)

nt_bpnet_extra_large_hyperparam = replace(
    different_size_hyperparam_config,
    student_config=replace(
        different_size_hyperparam_config.student_config,
        model_size="extra_large",
    ),
    trainer_config=replace(
        different_size_hyperparam_config.trainer_config,
        output_dir=f"{output_path}/nt_distillation/different_size/bpnet/extra_large/hyperparam/",
    ),
)

# ================================================================
# NEW: extra_large_fix experiment configs
# ================================================================
# The extra_large model has a dilation cap of 6 which limits receptive field
# and causes poor performance. extra_large_fix removes this cap.

nt_bpnet_extra_large_fix = replace(
    nt_bpnet_original,
    student_config=replace(
        nt_bpnet_original.student_config,
        model_size="extra_large_fix",
    ),
    trainer_config=replace(
        nt_bpnet_original.trainer_config,
        output_dir=f"{output_path}/nt_distillation/different_size/bpnet/extra_large_fix/",
        wandb_project="OmegaGenome-NT-ExtraLarge-Fix",
    ),
)

nt_bpnet_extra_large_fix_hyperparam = replace(
    different_size_hyperparam_config,
    student_config=replace(
        different_size_hyperparam_config.student_config,
        model_size="extra_large_fix",
    ),
    trainer_config=replace(
        different_size_hyperparam_config.trainer_config,
        output_dir=f"{output_path}/nt_distillation/different_size/bpnet/extra_large_fix/hyperparam/",
        wandb_project="OmegaGenome-NT-ExtraLarge-Fix-HyperParam",
    ),
)

nt_bpnet_extra_large_fix_hyperparam_extend = replace(
    different_size_hyperparam_extend_config,
    student_config=replace(
        different_size_hyperparam_extend_config.student_config,
        model_size="extra_large_fix",
    ),
    trainer_config=replace(
        different_size_hyperparam_extend_config.trainer_config,
        output_dir=f"{output_path}/nt_distillation/different_size/bpnet/extra_large_fix/hyperparam/",
        wandb_project="OmegaGenome-NT-ExtraLarge-Fix-HyperParam-Extend",
    ),
)

nt_bpnet_xxlarge_hyperparam = replace(
    different_size_hyperparam_config,
    student_config=replace(
        different_size_hyperparam_config.student_config,
        model_size="xxlarge",
    ),
    trainer_config=replace(
        different_size_hyperparam_config.trainer_config,
        output_dir=f"{output_path}/nt_distillation/different_size/bpnet/xxlarge/hyperparam/",
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

# ================================================================
# Hyperparameter experiment configs
# ================================================================
small_medium_large_bpnet_experiment_configs = {
    "nt_bpnet_pico_hyperparam": (
        "NT distillation with pico BPNet (~1.3k params)",
        nt_bpnet_pico_hyperparam,
    ),
    "nt_bpnet_ultra_tiny_hyperparam": (
        "NT distillation with ultra tiny BPNet (~6.25k params)",
        nt_bpnet_ultra_tiny_hyperparam,
    ),
    "nt_bpnet_extra_tiny_hyperparam": (
        "NT distillation with extra tiny BPNet (~25k params)",
        nt_bpnet_extra_tiny_hyperparam,
    ),
    "nt_bpnet_small_hyperparam": (
        "NT distillation with small BPNet (~200k params)",
        nt_bpnet_small_hyperparam,
    ),
    "nt_bpnet_medium_small_hyperparam": (
        "NT distillation with medium-small BPNet (~400k params)",
        nt_bpnet_medium_small_hyperparam,
    ),
    "nt_bpnet_medium_large_hyperparam": (
        "NT distillation with medium-large BPNet (~0.4M params)",
        nt_bpnet_medium_large_hyperparam,
    ),
    "nt_bpnet_original_hyperparam": (
        "NT distillation with original BPNet (~120k params)",
        nt_bpnet_original_hyperparam,
    ),
    "nt_bpnet_original_hyperparam_extend": (
        "NT distillation with original BPNet (~120k params)",
        nt_bpnet_original_hyperparam_extend,
    ),
    "nt_bpnet_original_hyperparam_extend_solely": (
        "NT distillation with original BPNet (~120k params)",
        nt_bpnet_original_hyperparam_extend_solely,
    ),
    "nt_bpnet_extra_large_hyperparam": (
        "NT distillation with extra-large BPNet (~0.8M params) - KNOWN ISSUE: dilation cap",
        nt_bpnet_extra_large_hyperparam,
    ),
    # NEW: extra_large_fix hyperparameter search
    "nt_bpnet_extra_large_fix_hyperparam": (
        "NT distillation with extra-large FIXED BPNet (~0.8M params) - Proper dilation",
        nt_bpnet_extra_large_fix_hyperparam,
    ),
    "nt_bpnet_extra_large_fix_hyperparam_extend": (
        "NT distillation with extra-large FIXED BPNet (~0.8M params) - Proper dilation, extended hyperparameter search for weight_mse",
        nt_bpnet_extra_large_fix_hyperparam_extend,
    ),
    "nt_bpnet_medium_hyperparam": (
        "NT distillation with medium BPNet (~1M params)",
        nt_bpnet_medium_hyperparam,
    ),
    "nt_bpnet_large_hyperparam": (
        "NT distillation with large BPNet (~5M params)",
        nt_bpnet_large_hyperparam,
    ),
    "nt_bpnet_xxlarge_hyperparam": (
        "NT distillation with xx-large BPNet (~3.6M params)",
        nt_bpnet_xxlarge_hyperparam,
    ),
}

# ================================================================
# Standard experiment configs
# ================================================================
experiment_configs = {
    "nt_bpnet_pico": (
        "NT distillation with pico BPNet (~1.3k params)",
        nt_bpnet_pico,
    ),
    # BPNet different sizes
    "nt_bpnet_original": (
        "NT distillation with original BPNet (~280k params)",
        nt_bpnet_original,
    ),
    "nt_bpnet_tiny": (
        "NT distillation with tiny BPNet (~50k params)",
        nt_bpnet_tiny,
    ),
    # NEW: Additional size variants
    "nt_bpnet_ultra_tiny": (
        "NT distillation with ultra tiny BPNet (~6.25k params)",
        nt_bpnet_ultra_tiny,
    ),
    "nt_bpnet_extra_tiny": (
        "NT distillation with extra tiny BPNet (~25k params)",
        nt_bpnet_extra_tiny,
    ),
    "nt_bpnet_small": (
        "NT distillation with small BPNet (~200k params)",
        nt_bpnet_small,
    ),
    "nt_bpnet_medium_small": (
        "NT distillation with medium-small BPNet (~400k params)",
        nt_bpnet_medium_small,
    ),
    "nt_bpnet_medium": (
        "NT distillation with medium BPNet (~1M params)",
        nt_bpnet_medium,
    ),
    "nt_bpnet_large": (
        "NT distillation with large BPNet (~5M params)",
        nt_bpnet_large,
    ),
    # NEW: extra_large_fix experiment
    "nt_bpnet_extra_large_fix": (
        "NT distillation with extra-large FIXED BPNet (~0.8M params) - Proper dilation for better scaling",
        nt_bpnet_extra_large_fix,
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
