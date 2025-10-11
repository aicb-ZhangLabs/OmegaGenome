from ..config_schema import DistillationExperimentConfig
from ..glm import dna_bert_v2
from ..bpnet import bpnet_classifier_config
from ..trainer import trainer_config, debug_trainer_config
from ..data import nucletide_transformer_revised_benchmark

from ...slurm import basic_distillation_slurm
from ...env import project_path

experiment_configs = {
    "dna_bert_v2": (
        "doing distillation",
        DistillationExperimentConfig(
            task_names=["promoter_all"],
            dataset_config=nucletide_transformer_revised_benchmark,
            teacher_config=dna_bert_v2,
            teacher_parent_dir=f"{project_path}/data/finetuned_models/dnabert2_output_shared/output",
            student_config=bpnet_classifier_config,
            trainer_config=trainer_config,
            slurm_config=basic_distillation_slurm,
        ),
    ),
    "dna_bert_v2_debug": (
        "doing distillation",
        DistillationExperimentConfig(
            task_names=["promoter_all"],
            dataset_config=nucletide_transformer_revised_benchmark,
            teacher_config=dna_bert_v2,
            teacher_parent_dir=f"{project_path}/data/finetuned_models/dnabert2_output_shared/output",
            student_config=bpnet_classifier_config,
            trainer_config=debug_trainer_config,
            slurm_config=basic_distillation_slurm,
        ),
    ),
}
