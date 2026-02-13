"""
Benchmark experiment configurations.
"""

from dataclasses import dataclass, field
from typing import List, Optional
from config.env import output_path


# All available tasks
ALL_TASKS = [
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


@dataclass
class BenchmarkExperimentConfig:
    """Configuration for inference benchmarking experiments."""

    # Tasks (single or multiple)
    task_names: List[str] = field(default_factory=lambda: ["splice_sites_all"])

    # Models to benchmark
    teachers: List[str] = field(default_factory=lambda: ["nt", "caduceus", "enformer"])
    student_sizes: List[str] = field(default_factory=lambda: ["original"])

    # Settings
    batch_size: int = 16
    max_length: int = 1000
    warmup_batches: int = 10
    num_batches: Optional[int] = 100

    # Output
    output_dir: str = f"{output_path}/benchmark"
    wandb_project: str = "OmegaGenome-Benchmark"


# Predefined configurations
TEACHER_BENCHMARK_SINGLE = BenchmarkExperimentConfig(
    task_names=["splice_sites_all"],
    teachers=["nt", "caduceus", "enformer"],
    student_sizes=[],
)

TEACHER_BENCHMARK_MULTI = BenchmarkExperimentConfig(
    task_names=["splice_sites_all", "promoter_all", "H3K4me3"],
    teachers=["nt", "caduceus", "enformer"],
    student_sizes=[],
)

STUDENT_SIZE_BENCHMARK = BenchmarkExperimentConfig(
    task_names=["splice_sites_all"],
    teachers=[],
    student_sizes=[
        "pico",
        "ultra_tiny",
        "extra_tiny",
        "tiny",
        "small",
        "original",
        "medium",
        "large",
        "xxlarge",
    ],
)

FULL_COMPARISON_BENCHMARK = BenchmarkExperimentConfig(
    task_names=["splice_sites_all"],
    teachers=["nt", "caduceus", "enformer"],
    student_sizes=["original"],
)

ALL_TASKS_BENCHMARK = BenchmarkExperimentConfig(
    task_names=ALL_TASKS,
    teachers=["nt", "caduceus", "enformer"],
    student_sizes=["original"],
    num_batches=50,  # Fewer batches for efficiency
)
