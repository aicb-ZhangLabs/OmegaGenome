from dataclasses import dataclass, field
from typing import Literal, List
from src.model.glm import GLMConfig
from src.model.bpnet_classifier import BPNetClassifierConfig
from src.model.distillation import DistillationModelConfig
from src.data.dataset import (
    DatasetConfig,
)
from src.trainer.distill_trainer import DistillTrainerConfig
from nntool.slurm import SlurmConfig


@dataclass
class DistillationExperimentConfig:
    # this is for doing distillation for multiple tasks one by one
    task_names: List[
        Literal[
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
        ]
    ]
    dataset_config: DatasetConfig
    teacher_config: GLMConfig
    teacher_parent_dir: str
    student_config: BPNetClassifierConfig
    distillation_config: DistillationModelConfig
    trainer_config: DistillTrainerConfig
    slurm_config: SlurmConfig
    random_state: int = 42


@dataclass
class DistillationHyperparamExperimentConfig(DistillationExperimentConfig):
    weight_ces: List[float] = field(default_factory=lambda: [1.0])
    weight_kls: List[float] = field(default_factory=lambda: [0.1, 0.5, 1.0])
    weight_mses: List[float] = field(default_factory=lambda: [0.1, 0.5, 1.0])
    temperatures: List[float] = field(default_factory=lambda: [1.5, 2.0, 4.0])
    zscores: List[bool] = field(default_factory=lambda: [False])
