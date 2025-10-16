from ..config_schema import DistillationHyperparamExperimentConfig
from ..glm import GLMConfig, nt_2b5
from ..bpnet import BPNetClassifierConfig
from ..trainer import DistillTrainerConfig
from ..data import DatasetConfig, nucletide_transformer_revised_benchmark
from ..distillation_model import DistillationModelConfig
from ...slurm import SlurmConfig, basic_distillation_slurm
from ...env import project_path
from .nt import NT_PARENT_PATH


# NT_PARENT_PATH = f"{project_path}/data/finetuned_models/2b5-multi-species_nucleotide-transformer-finetune-results-lora-epoch10-3-22-revised-r32-fix-num-label"

experiment_configs = {
    "nt_hyperparam": (
        "NT distillation hyperparameter search",
        DistillationHyperparamExperimentConfig(
            task_names=["promoter_all", "promoter_tata", "H3K4me3", "H3K9ac"],
            dataset_config=nucletide_transformer_revised_benchmark,
            teacher_config=nt_2b5,
            teacher_parent_dir=NT_PARENT_PATH,
            model_type="nt",
            student_config=BPNetClassifierConfig(
                num_labels=2,
                model_type="bpnet",
                model_size="original",  # Use original BPNet
            ),
            distillation_config=DistillationModelConfig(
                distill_method="vanilla",
            ),
            trainer_config=DistillTrainerConfig(
                output_dir=f"{project_path}/outputs/nt_distillation/hyperparam",
                wandb_project="OmegaGenome-NT-HyperParam",
                epochs=200,  # Fewer epochs for hyperparameter search
                batch_size=16,
                max_len=1000,
            ),
            slurm_config=basic_distillation_slurm,
            # Grid search parameters
            weight_ces=[0.3, 0.5, 0.7],
            weight_kls=[0.0, 0.25, 0.5, 0.75, 1.0],
            weight_mses=[0.0, 0.1, 0.2, 0.5],
            temperatures=[1.0, 2.0, 4.0, 8.0],
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
            student_config=BPNetClassifierConfig(
                num_labels=2,
                model_type="bpnet",
                model_size="original",
            ),
            distillation_config=DistillationModelConfig(
                distill_method="dkd",
                weight_mse=0.0,  # DKD doesn't use feature matching
            ),
            trainer_config=DistillTrainerConfig(
                output_dir=f"{project_path}/outputs/nt_distillation/hyperparam_dkd",
                wandb_project="OmegaGenome-NT-DKD-HyperParam",
                epochs=200,
                batch_size=16,
                max_len=1000,
            ),
            slurm_config=basic_distillation_slurm,
            # DKD-specific hyperparameters
            weight_ces=[0.3, 0.5, 0.7],
            weight_kls=[0.3, 0.5, 0.7],
            weight_mses=[0.0],  # DKD doesn't use MSE
            temperatures=[2.0, 4.0, 8.0],
            zscores=[False],
            # Additional DKD parameters (need to add to config schema)
            # dkd_alphas=[0.5, 1.0, 2.0],
            # dkd_betas=[4.0, 8.0, 16.0],
        ),
    ),
}
