from dataclasses import dataclass, field
from typing import Literal, List
from src.model.glm import GLMConfig
from src.model.bpnet_classifier import BPNetClassifierConfig
from src.model.distillation import DistillationModelConfig
from src.data.dataset import (
    DatasetConfig,
)
from nntool.slurm import SlurmConfig
from typing import Optional
from config.distillation.trainer import DistillTrainerConfig


@dataclass
class DistillationExperimentConfig:
    # task_names: List[str] = field(default_factory=lambda: ["H3K4me2", "H3K4me3"])
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
    teacher_config: GLMConfig = field(default_factory=GLMConfig)
    student_config: BPNetClassifierConfig = field(default_factory=BPNetClassifierConfig)
    distillation_config: DistillationModelConfig = field(
        default_factory=DistillationModelConfig
    )
    trainer_config: DistillTrainerConfig = field(default_factory=DistillTrainerConfig)
    dataset_config: DatasetConfig = field(default_factory=DatasetConfig)
    slurm_config: SlurmConfig = field(default_factory=SlurmConfig)
    random_state: int = 42

    # Extended for NT support
    teacher_parent_dir: str = ""  # Path to teacher checkpoints
    model_type: str = "glm"  # "glm" or "nt" to determine checkpoint structure
    # Resume support
    resume_checkpoint: Optional[str] = None
    resume_epoch: int = 0


@dataclass
class DistillationHyperparamExperimentConfig(DistillationExperimentConfig):
    weight_ces: List[float] = field(default_factory=lambda: [1.0])
    weight_kls: List[float] = field(default_factory=lambda: [0.0, 0.5, 1.0])
    weight_mses: List[float] = field(default_factory=lambda: [0.0, 0.5, 1.0])
    temperatures: List[float] = field(default_factory=lambda: [0.5, 1.0, 1.5, 2.0, 4.0])
    zscores: List[bool] = field(default_factory=lambda: [False, True])
