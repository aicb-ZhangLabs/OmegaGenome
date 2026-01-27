"""
Benchmark inference latency and memory usage for teacher models and BPNet student.

Usage:
    python -m src.train.benchmark_inference --help
    python -m src.train.benchmark_inference teacher_benchmark
    python -m src.train.benchmark_inference student_benchmark

    # Single task
    python -m src.train.benchmark_inference full_benchmark --task-names splice_sites_all

    # Multiple tasks
    python -m src.train.benchmark_inference full_benchmark --task-names '[splice_sites_all,promoter_all]'
"""

import gc
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import tyro
import wandb
import torch
from dataclasses import dataclass, field, replace
from typing import List, Literal, Optional, Union
from datetime import datetime

from torch.utils.data import DataLoader

from config.env import project_path, output_path

from ..data.dataset import (
    get_num_labels,
    build_data_splits_from_huggingface,
    SeqDataset,
)
from ..model.bpnet_classifier import BPNetClassifier, BPNetClassifierConfig
from ..model.glm import find_teacher_checkpoint, get_teacher_model
from ..trainer.benchmark_utils import (
    BenchmarkResult,
    benchmark_model_inference,
    benchmark_teacher_inference,
    save_benchmark_results_csv,
    log_benchmark_to_wandb,
    normalize_task_names,
)


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

# Known problematic models (e.g., Triton compatibility issues)
KNOWN_ISSUES = {
    "dnabert2": "DNABERT2 has known Triton flash attention compatibility issues with newer Triton versions. "
    "Consider using --skip-on-error or excluding dnabert2 from teachers list."
}


@dataclass
class BenchmarkConfig:
    """Configuration for inference benchmarking."""

    # Tasks to benchmark on (single task or list of tasks)
    task_names: List[str] = field(default_factory=lambda: ["splice_sites_all"])

    # Teachers to benchmark (empty = skip teacher benchmarking)
    # Note: dnabert2 excluded by default due to Triton compatibility issues
    teachers: List[Literal["nt", "caduceus", "enformer", "dnabert2"]] = field(
        default_factory=lambda: ["nt", "caduceus", "enformer"]
    )

    # Student sizes to benchmark (empty = skip student benchmarking)
    student_sizes: List[str] = field(default_factory=lambda: ["original"])

    # Benchmark settings
    batch_size: int = 16
    max_length: int = 1000
    warmup_batches: int = 10
    num_batches: Optional[int] = 100  # None = use all test data

    # Error handling
    skip_on_error: bool = True  # Continue with other models if one fails

    # Output settings
    output_dir: str = f"{output_path}/benchmark"
    wandb_project: str = "OmegaGenome-Benchmark"

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def get_teacher_config_and_path(model_type: str):
    """Get teacher configuration and checkpoint path."""
    from config.distillation.glm import nt_2b5, caduceus, enformer, dna_bert_v2
    from config.distillation.experiments.nt import NT_PARENT_PATH
    from config.distillation.experiments.caduceus import CADUCEUS_PARENT_PATH
    from config.distillation.experiments.enformer import ENFORMER_PARENT_PATH
    from config.distillation.experiments.dna_bert_v2 import DNABERT2_PARENT_PATH

    config_map = {
        "nt": (nt_2b5, NT_PARENT_PATH, "nt"),
        "caduceus": (caduceus, CADUCEUS_PARENT_PATH, "caduceus"),
        "enformer": (enformer, ENFORMER_PARENT_PATH, "enformer"),
        "dnabert2": (dna_bert_v2, DNABERT2_PARENT_PATH, "glm"),
    }

    return config_map.get(model_type)


def benchmark_teachers_on_task(
    config: BenchmarkConfig, task_name: str
) -> List[BenchmarkResult]:
    """Benchmark all specified teacher models on a single task."""
    from config.distillation.data import nucletide_transformer_revised_benchmark

    results = []
    errors = []

    # Load test data
    dataset_config = replace(
        nucletide_transformer_revised_benchmark, task_name=task_name
    )
    _, _, _, _, X_test, y_test = build_data_splits_from_huggingface(dataset_config)

    print(f"\n--- Benchmarking Teachers on: {task_name} ---")
    print(f"Test samples: {len(X_test)}")

    for teacher_type in config.teachers:
        print(f"\n  Benchmarking {teacher_type.upper()}...")

        # Check for known issues
        if teacher_type in KNOWN_ISSUES:
            print(f"  ⚠️  Warning: {KNOWN_ISSUES[teacher_type]}")

        try:
            teacher_info = get_teacher_config_and_path(teacher_type)
            if teacher_info is None:
                print(f"  Warning: Unknown teacher type {teacher_type}, skipping")
                continue

            teacher_config, teacher_parent_dir, model_type = teacher_info

            # Create a minimal config object for find_teacher_checkpoint
            @dataclass
            class MinimalConfig:
                teacher_config: any
                teacher_parent_dir: str
                model_type: str

            mini_config = MinimalConfig(
                teacher_config=teacher_config,
                teacher_parent_dir=teacher_parent_dir,
                model_type=model_type,
            )

            # Find checkpoint
            teacher_ckpt, score = find_teacher_checkpoint(mini_config, task_name)

            if teacher_ckpt is None:
                print(
                    f"  No checkpoint found for {teacher_type} on {task_name}, skipping"
                )
                continue

            # Create trainer config for get_teacher_model
            @dataclass
            class TrainerConfig:
                device: str

            @dataclass
            class FullConfig:
                teacher_config: any
                teacher_parent_dir: str
                model_type: str
                trainer_config: TrainerConfig

            full_config = FullConfig(
                teacher_config=teacher_config,
                teacher_parent_dir=teacher_parent_dir,
                model_type=model_type,
                trainer_config=TrainerConfig(device=config.device),
            )

            # Load teacher
            num_labels = get_num_labels(task_name)
            full_config.teacher_config = replace(
                full_config.teacher_config, num_labels=num_labels
            )

            tokenizer, teacher_model, _ = get_teacher_model(
                full_config, task_name, teacher_ckpt
            )

            # Benchmark
            result = benchmark_teacher_inference(
                model=teacher_model,
                tokenizer=tokenizer,
                sequences=X_test,
                device=config.device,
                model_name=teacher_type,
                task_name=task_name,
                batch_size=config.batch_size,
                max_length=config.max_length,
                warmup_batches=config.warmup_batches,
                num_batches=config.num_batches,
            )

            results.append(result)
            # Explicit cleanup (model already cleaned in benchmark function)
            torch.cuda.empty_cache()
            gc.collect()
            # Log to wandb
            log_benchmark_to_wandb(result)

            print(f"    ✓ Mean latency: {result.mean_latency_ms:.2f} ms")
            print(
                f"    ✓ Throughput: {result.throughput_samples_per_sec:.1f} samples/s"
            )
            print(f"    ✓ Peak memory: {result.peak_memory_mb:.1f} MB")
            print(f"    ✓ Parameters: {result.num_parameters:,}")

            # Clean up
            del teacher_model
            torch.cuda.empty_cache()

        except Exception as e:
            error_msg = str(e)
            errors.append((teacher_type, error_msg))

            # Provide more helpful error messages for known issues
            if "triton" in error_msg.lower() or "trans_b" in error_msg:
                print(f"  ✗ Error: Triton compatibility issue with {teacher_type}")
                print(
                    f"    This is a known issue with flash attention in newer Triton versions."
                )
                print(f"    Consider excluding {teacher_type} from the benchmark.")
            else:
                print(f"  ✗ Error benchmarking {teacher_type}: {error_msg[:200]}")

            if not config.skip_on_error:
                raise

            import traceback

            traceback.print_exc()
            continue

    # Report errors summary
    if errors:
        print(f"\n  ⚠️  {len(errors)} teacher(s) failed:")
        for teacher, err in errors:
            print(f"    - {teacher}: {err[:100]}...")

    return results


def benchmark_students_on_task(
    config: BenchmarkConfig, task_name: str
) -> List[BenchmarkResult]:
    """Benchmark BPNet student models on a single task."""
    from config.distillation.data import nucletide_transformer_revised_benchmark

    results = []
    errors = []

    # Load test data
    dataset_config = replace(
        nucletide_transformer_revised_benchmark, task_name=task_name
    )
    _, _, _, _, X_test, y_test = build_data_splits_from_huggingface(dataset_config)

    num_labels = get_num_labels(task_name)

    # Create test dataset
    test_ds = SeqDataset(X_test, y_test, config.max_length)
    test_loader = DataLoader(
        test_ds, batch_size=config.batch_size, shuffle=False, num_workers=4
    )

    print(f"\n--- Benchmarking Students on: {task_name} ---")
    print(f"Test samples: {len(X_test)}")

    for size in config.student_sizes:
        print(f"\n  Benchmarking BPNet {size}...")

        try:
            # Create student model
            student_config = BPNetClassifierConfig(
                num_labels=num_labels,
                model_type="bpnet",
                model_size=size,
            )
            model = BPNetClassifier(student_config)

            # Benchmark
            result = benchmark_model_inference(
                model=model,
                dataloader=test_loader,
                device=config.device,
                model_name=f"bpnet_{size}",
                model_type="student",
                task_name=task_name,
                warmup_batches=config.warmup_batches,
                num_batches=config.num_batches,
            )

            results.append(result)
            torch.cuda.empty_cache()
            gc.collect()
            # Log to wandb
            log_benchmark_to_wandb(result)

            print(f"    ✓ Mean latency: {result.mean_latency_ms:.2f} ms")
            print(
                f"    ✓ Throughput: {result.throughput_samples_per_sec:.1f} samples/s"
            )
            print(f"    ✓ Peak memory: {result.peak_memory_mb:.1f} MB")
            print(f"    ✓ Parameters: {result.num_parameters:,}")

            # Clean up
            del model
            torch.cuda.empty_cache()

        except Exception as e:
            error_msg = str(e)
            errors.append((size, error_msg))
            print(f"  ✗ Error benchmarking BPNet {size}: {error_msg[:200]}")

            if not config.skip_on_error:
                raise

            import traceback

            traceback.print_exc()
            continue

    # Report errors summary
    if errors:
        print(f"\n  ⚠️  {len(errors)} student size(s) failed:")
        for size, err in errors:
            print(f"    - {size}: {err[:100]}...")

    return results


def print_summary_table(results: List[BenchmarkResult]):
    """Print a formatted summary table of results."""
    if not results:
        print("\nNo benchmark results to display.")
        return

    print(f"\n{'=' * 100}")
    print("BENCHMARK SUMMARY")
    print(f"{'=' * 100}")
    print(
        f"{'Task':<25} {'Model':<20} {'Type':<10} {'Latency(ms)':<12} "
        f"{'Throughput':<12} {'Memory(MB)':<12} {'Params':<15}"
    )
    print("-" * 100)

    # Group by task
    tasks = sorted(set(r.task_name for r in results))
    for task in tasks:
        task_results = [r for r in results if r.task_name == task]
        for r in task_results:
            print(
                f"{r.task_name:<25} {r.model_name:<20} {r.model_type:<10} "
                f"{r.mean_latency_ms:<12.2f} {r.throughput_samples_per_sec:<12.1f} "
                f"{r.peak_memory_mb:<12.1f} {r.num_parameters:<15,}"
            )
        if task != tasks[-1]:
            print("-" * 100)

    print(f"{'=' * 100}")


def main(config: BenchmarkConfig):
    """Run inference benchmarks across all specified tasks."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Normalize task_names to list
    task_names = normalize_task_names(config.task_names)

    # Create output directory
    if len(task_names) == 1:
        run_dir = os.path.join(config.output_dir, task_names[0], timestamp)
    else:
        run_dir = os.path.join(config.output_dir, "multi_task", timestamp)
    os.makedirs(run_dir, exist_ok=True)

    # Print known issues warning
    for teacher in config.teachers:
        if teacher in KNOWN_ISSUES:
            print(f"\n⚠️  Warning for {teacher}: {KNOWN_ISSUES[teacher]}")

    # Initialize wandb
    wandb.init(
        project=config.wandb_project,
        name=f"benchmark_{'-'.join(task_names[:3])}_{timestamp}",
        config={
            "task_names": task_names,
            "teachers": config.teachers,
            "student_sizes": config.student_sizes,
            "batch_size": config.batch_size,
            "max_length": config.max_length,
            "num_batches": config.num_batches,
            "skip_on_error": config.skip_on_error,
        },
        tags=["benchmark"] + task_names[:5],  # Limit tags
    )

    all_results = []
    all_errors = []

    print(f"\n{'=' * 80}")
    print(f"INFERENCE BENCHMARK")
    print(f"{'=' * 80}")
    print(f"Tasks: {task_names}")
    print(f"Teachers: {config.teachers if config.teachers else 'None'}")
    print(f"Student sizes: {config.student_sizes if config.student_sizes else 'None'}")
    print(f"Batch size: {config.batch_size}")
    print(f"Max batches: {config.num_batches}")
    print(f"Skip on error: {config.skip_on_error}")
    print(f"{'=' * 80}\n")

    # Iterate through each task
    for task_idx, task_name in enumerate(task_names, 1):
        print(f"\n{'#' * 80}")
        print(f"# Task {task_idx}/{len(task_names)}: {task_name}")
        print(f"{'#' * 80}")

        task_results = []

        # Benchmark teachers on this task
        if config.teachers:
            teacher_results = benchmark_teachers_on_task(config, task_name)
            task_results.extend(teacher_results)
            all_results.extend(teacher_results)

        # Benchmark students on this task
        if config.student_sizes:
            student_results = benchmark_students_on_task(config, task_name)
            task_results.extend(student_results)
            all_results.extend(student_results)

        # Save per-task results
        if task_results:
            task_csv_path = os.path.join(run_dir, f"{task_name}_benchmark.csv")
            save_benchmark_results_csv(task_results, task_csv_path)

    # Save combined results
    if all_results:
        save_benchmark_results_csv(
            all_results,
            os.path.join(run_dir, "all_benchmark.csv"),
        )

        # Save teacher-only results
        teacher_results = [r for r in all_results if r.model_type == "teacher"]
        if teacher_results:
            save_benchmark_results_csv(
                teacher_results,
                os.path.join(run_dir, "teacher_benchmark.csv"),
            )

        # Save student-only results
        student_results = [r for r in all_results if r.model_type == "student"]
        if student_results:
            save_benchmark_results_csv(
                student_results,
                os.path.join(run_dir, "student_benchmark.csv"),
            )

        # Create summary table for wandb
        summary_table = wandb.Table(
            columns=[
                "task_name",
                "model_name",
                "model_type",
                "mean_latency_ms",
                "throughput",
                "peak_memory_mb",
                "num_parameters",
            ]
        )
        for r in all_results:
            summary_table.add_data(
                r.task_name,
                r.model_name,
                r.model_type,
                r.mean_latency_ms,
                r.throughput_samples_per_sec,
                r.peak_memory_mb,
                r.num_parameters,
            )
        wandb.log({"benchmark_summary": summary_table})

    # Print summary
    print_summary_table(all_results)
    print(f"\nResults saved to: {run_dir}")

    wandb.finish()


# Predefined experiment configurations
experiment_configs = {
    "teacher_benchmark": (
        "Benchmark teacher models (NT, Caduceus, Enformer) on splice_sites_all",
        BenchmarkConfig(
            task_names=["splice_sites_all"],
            teachers=["nt", "caduceus", "enformer"],  # Excludes dnabert2 by default
            student_sizes=[],
            batch_size=16,
            num_batches=100,
        ),
    ),
    "teacher_benchmark_all": (
        "Benchmark ALL teacher models including DNABERT2 (may have issues)",
        BenchmarkConfig(
            task_names=["splice_sites_all"],
            teachers=["nt", "caduceus", "enformer", "dnabert2"],
            student_sizes=[],
            batch_size=16,
            num_batches=100,
            skip_on_error=True,
        ),
    ),
    "teacher_benchmark_multi": (
        "Benchmark teacher models on multiple tasks",
        BenchmarkConfig(
            task_names=["splice_sites_all", "promoter_all", "H3K4me3"],
            teachers=["nt", "caduceus", "enformer"],
            student_sizes=[],
            batch_size=16,
            num_batches=100,
        ),
    ),
    "student_benchmark": (
        "Benchmark BPNet student models of different sizes",
        BenchmarkConfig(
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
            ],
            batch_size=16,
            num_batches=100,
        ),
    ),
    "student_benchmark_all_sizes": (
        "Benchmark all BPNet sizes on multiple tasks",
        BenchmarkConfig(
            task_names=["splice_sites_all", "promoter_all"],
            teachers=[],
            student_sizes=[
                "pico",
                "ultra_tiny",
                "extra_tiny",
                "tiny",
                "small",
                "medium_small",
                "original",
                "medium",
                "medium_large",
                "extra_large",
                "large",
                "xxlarge",
            ],
            batch_size=16,
            num_batches=100,
        ),
    ),
    "full_benchmark": (
        "Benchmark teachers (NT, Caduceus, Enformer) and BPNet original",
        BenchmarkConfig(
            task_names=["splice_sites_all"],
            teachers=["nt", "caduceus", "enformer", "dnabert2"],
            student_sizes=["original"],
            batch_size=16,
            num_batches=100,
        ),
    ),
    "full_benchmark_all_sample": (
        "Benchmark teachers (NT, Caduceus, Enformer) and BPNet original",
        BenchmarkConfig(
            task_names=["splice_sites_all"],
            teachers=["nt", "caduceus", "enformer", "dnabert2"],
            student_sizes=["original"],
            batch_size=16,
            num_batches=1875,
        ),
    ),
    "full_benchmark_multi": (
        "Benchmark teachers and BPNet original on multiple tasks",
        BenchmarkConfig(
            task_names=["splice_sites_all", "promoter_all", "H3K4me3"],
            teachers=["nt", "caduceus", "enformer"],
            student_sizes=["original"],
            batch_size=16,
            num_batches=100,
        ),
    ),
    "quick_benchmark": (
        "Quick benchmark with fewer batches (NT + BPNet original)",
        BenchmarkConfig(
            task_names=["splice_sites_all"],
            teachers=["nt"],
            student_sizes=["original"],
            batch_size=16,
            num_batches=20,
        ),
    ),
    "dnabert2_benchmark": (
        "Quick benchmark with fewer batches (NT + BPNet original)",
        BenchmarkConfig(
            task_names=["splice_sites_all"],
            teachers=["dnabert2"],
            student_sizes=[],
            batch_size=16,
            num_batches=100,
        ),
    ),
    "all_tasks_teacher": (
        "Benchmark teachers on all 18 genomic tasks",
        BenchmarkConfig(
            task_names=ALL_TASKS,
            teachers=["nt", "caduceus", "enformer"],
            student_sizes=[],
            batch_size=16,
            num_batches=50,
        ),
    ),
    "nt_only": (
        "Benchmark only NT teacher (most stable)",
        BenchmarkConfig(
            task_names=["splice_sites_all"],
            teachers=["nt"],
            student_sizes=[],
            batch_size=16,
            num_batches=100,
        ),
    ),
}


if __name__ == "__main__":
    config = tyro.extras.overridable_config_cli(
        experiment_configs, sort_subcommands=True
    )
    main(config)
