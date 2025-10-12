from ..config_schema import DistillationHyperparamExperimentConfig
from ..glm import dna_bert_v2
from ..bpnet import bpnet_classifier_config
from ..trainer import trainer_config, debug_trainer_config
from ..data import nucletide_transformer_revised_benchmark
from ..distillation_model import distillation_model_config

from ...slurm import basic_distillation_slurm
from ...env import project_path

experiment_configs = {
    "dna_bert_v2": (
        "doing distillation",
        DistillationHyperparamExperimentConfig(
            task_names=["promoter_all"],
            dataset_config=nucletide_transformer_revised_benchmark,
            teacher_config=dna_bert_v2,
            teacher_parent_dir=f"{project_path}/data/finetuned_models/dnabert2_output_shared/output",
            student_config=bpnet_classifier_config,
            distillation_config=distillation_model_config,
            trainer_config=trainer_config,
            slurm_config=basic_distillation_slurm,
            weight_ces=[1.0],
            weight_kls=[0.0, 0.5, 1.0],
            weight_mses=[0.0, 0.5, 1.0],
            temperatures=[1.5, 2.0, 4.0],
        ),
    )
}
