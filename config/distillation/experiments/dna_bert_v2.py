from ..config_schema import DistillationExperimentConfig
from ..glm import dna_bert2_config
from ..bpnet import bpnet_classifier_config
from ..trainer import trainer_config
from ..data import data
from ...slurm import basic_distillation_slurm

experiment_configs = {
    "dna_bert_v2": (
        "xxx",
        DistillationExperimentConfig(
            dataset_config=data,
            teacher_config=dna_bert2_config,
            teacher_parent_dir="models/dna_bert_v2",
            student_config=bpnet_classifier_config,
            trainer_config=trainer_config,
            slurm_config=basic_distillation_slurm,
        ),
    ),
    "dna_bert_v2_debug": (
        "xxx",
        DistillationExperimentConfig(
            dataset_config=data,
            teacher_config=dna_bert2_config,
            teacher_parent_dir="models/dna_bert_v2",
            student_config=bpnet_classifier_config,
            trainer_config=trainer_config,
            slurm_config=basic_distillation_slurm,
        ),
    ),
}
