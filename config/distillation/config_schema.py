from dataclasses import dataclass
from src.model.glm import GLMConfig
from src.model.bpnet_classifier import BPNetClassifierConfig
from src.data.dataset import (
    DatasetConfig,
)
from src.trainer.distill_trainer import DistillTrainerConfig


@dataclass
class DistillationExperimentConfig:
    dataset_config: DatasetConfig
    teacher_config: GLMConfig
    teacher_parent_dir: str
    student_config: BPNetClassifierConfig
    trainer_config: DistillTrainerConfig
