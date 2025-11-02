import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
import time
import tyro
import itertools

from dataclasses import replace, asdict
from config.distillation.config import hyperparam_configs
from config.distillation.config_schema import DistillationHyperparamExperimentConfig
from config.slurm_manager import get_gpu_manager
from ..trainer.utils import ExperimentTracker

from .distill import main as distill_main


def main(config: DistillationHyperparamExperimentConfig, resume: bool = True):
    """
    Run hyperparameter search over distillation parameters with resume support.

    This script performs a grid search over:
    - weight_ce, weight_kl, weight_mse (loss term weights)
    - temperature (for knowledge distillation)
    - zscore (normalization flag)

    Features:
    - Automatically detects and skips completed experiments (resume mode)
    - Monitors SLURM GPU usage per node
    - Waits for available GPUs before submitting jobs

    Args:
        config: Hyperparameter experiment configuration
        resume: If True (default), skip already completed experiments
    """
    # Initialize GPU manager with node limits
    gpu_manager = get_gpu_manager(
        node_limits={
            "voyager": 3,  # Max 2 concurrent GPU jobs YOU can run (shared server)
            "laniakea": 7,  # Max 4 concurrent GPU jobs YOU can run (shared server)
        },
        node_capacity={"voyager": 4, "laniakea": 8},
    )

    # Initialize experiment tracker for resume functionality
    tracker = ExperimentTracker(
        output_dir=config.trainer_config.output_dir, start_timestamp="20251019_000000"
    )  # Only track experiments after this)

    # ===== BUILD ALL EXPERIMENTS =====

    # Cartesian product of KL-related hyperparameters
    kl_combinations = list(
        itertools.product(
            config.weight_kls,
            config.temperatures,
            config.zscores,
        )
    )

    # Filter out redundant combinations where weight_kl=0
    to_skip = set()
    is_covered = False
    for i, (weight_kl, temperature, zscore) in enumerate(kl_combinations):
        if weight_kl == 0.0:
            if not is_covered:
                is_covered = True
            else:
                to_skip.add(i)
    kl_combinations = [
        combination for i, combination in enumerate(kl_combinations) if i not in to_skip
    ]

    # Cartesian product of other hyperparameters
    other_combinations = list(
        itertools.product(
            config.weight_ces,
            config.weight_mses,
        )
    )

    # Build complete list of all experiments
    all_experiments = []
    experiment_count = 0

    for task_name in config.task_names:
        for weight_ce, weight_mse in other_combinations:
            for weight_kl, temperature, zscore in kl_combinations:
                experiment_count += 1

                full_distill_config = replace(
                    config.distillation_config,
                    weight_ce=weight_ce,
                    weight_kl=weight_kl,
                    weight_mse=weight_mse,
                    temperature=temperature,
                    zscore=zscore,
                )
                hyperparam_config = asdict(full_distill_config)
                all_experiments.append(
                    {
                        "task_name": task_name,
                        "hyperparam_config": hyperparam_config,
                        "experiment_count": experiment_count,
                    }
                )

    # ===== CHECK COMPLETED EXPERIMENTS =====

    if resume:
        incomplete, completed, summary = tracker.generate_experiment_plan(
            all_experiments
        )
        tracker.print_summary_report(incomplete, completed, summary, save_to_file=True)

        if not incomplete:
            print("\n🎉 All experiments already completed! Nothing to run.")
            return

        print(f"\n{'=' * 80}")
        print(f"RESUME MODE: Running {len(incomplete)} incomplete experiments")
        print(f"{'=' * 80}\n")

        experiments_to_run = incomplete
    else:
        print(f"\n{'=' * 80}")
        print(
            f"FULL MODE: Running all {len(all_experiments)} experiments (ignoring completed)"
        )
        print(f"{'=' * 80}\n")
        experiments_to_run = all_experiments

    # ===== PRINT CONFIGURATION =====

    print(f"\n{'=' * 60}")
    print("Hyperparameter Search Configuration")
    print(f"{'=' * 60}")
    print(f"Tasks: {config.task_names}")
    print(f"Total experiments: {len(experiments_to_run)}")
    print(f"  - CE weights: {config.weight_ces}")
    print(f"  - KL weights: {config.weight_kls}")
    print(f"  - MSE weights: {config.weight_mses}")
    print(f"  - Temperatures: {config.temperatures}")
    print(f"  - Z-scores: {config.zscores}")
    print(f"{'=' * 60}")
    print("GPU Management (Per-User Limits on Shared Servers):")
    print("  - Voyager: You can use max 2 GPUs concurrently")
    print("  - Laniakea: You can use max 4 GPUs concurrently")
    print("  - Other users' jobs do NOT count toward your limits")
    print(f"{'=' * 60}\n")

    # Node preference order (voyager is ~2x faster, so prefer it)
    node_preference = ["voyager", "laniakea"]

    # ===== RUN EXPERIMENTS =====

    for idx, exp in enumerate(experiments_to_run, 1):
        task_name = exp["task_name"]
        hyperparam_config = exp["hyperparam_config"]

        weight_ce = hyperparam_config["weight_ce"]
        weight_kl = hyperparam_config["weight_kl"]
        weight_mse = hyperparam_config["weight_mse"]
        temperature = hyperparam_config["temperature"]
        zscore = hyperparam_config["zscore"]
        # Check if partial progress exists
        status = tracker.get_experiment_status(task_name, hyperparam_config)

        # Create configs
        new_distill_config = replace(config.distillation_config)  # , ...)
        new_slurm_config = replace(config.slurm_config)
        new_dataset_config = replace(config.dataset_config)

        # Create experiment config with resume info
        new_experiment_config = replace(
            config,
            task_names=[task_name],
            dataset_config=new_dataset_config,
            distillation_config=new_distill_config,
            slurm_config=new_slurm_config,
            resume_checkpoint=(
                status["latest_checkpoint"] if status["status"] == "partial" else None
            ),
            resume_epoch=status["latest_epoch"] if status["status"] == "partial" else 0,
        )

        print(f"\n{'=' * 60}")
        print(f"[Experiment {idx}/{len(experiments_to_run)}] Task: {task_name}")
        print(
            f"  CE={weight_ce}, KL={weight_kl}, MSE={weight_mse}, T={temperature}, zscore={zscore}"
        )
        print(f"{'=' * 60}")

        # Wait for an available node with GPU capacity
        available_node = gpu_manager.wait_for_available_node(
            preferred_nodes=node_preference,
            check_interval=30,  # Check every 30 seconds
            max_wait=3600,  # Wait up to 1 hour
        )

        if available_node is None:
            print(f"⚠️  Skipping experiment {idx} - no available GPU after timeout")
            continue

        print(f"🚀 Submitting to node: {available_node}")

        # Create new distillation config with current hyperparameters
        new_distill_config = replace(
            config.distillation_config,
            weight_ce=weight_ce,
            weight_kl=weight_kl,
            weight_mse=weight_mse,
            temperature=temperature,
            zscore=zscore,
        )

        # Update slurm config with the available node
        new_slurm_config = replace(config.slurm_config, node_list=available_node)

        # Update dataset config with current task
        new_dataset_config = replace(config.dataset_config, task_name=task_name)

        # Create new experiment config
        new_experiment_config = replace(
            config,
            task_names=[task_name],  # Single task
            dataset_config=new_dataset_config,
            distillation_config=new_distill_config,
            slurm_config=new_slurm_config,
        )

        # Run distillation
        distill_main(
            new_experiment_config,
        )
        # Wait until SLURM registers the new job (verify it appears in queue)

        # # Small delay to ensure SLURM registers the new job
        # import time
        # time.sleep(2)
        time.sleep(3)  # Initial delay
        for _ in range(10):  # Try up to 10 times (10 seconds max)
            current_jobs = gpu_manager.get_running_jobs_per_node()
            expected_jobs = current_jobs.get(available_node, 0)
            if expected_jobs > 0:  # Job registered
                break
            time.sleep(1)

    print(f"\n{'=' * 60}")
    print("Hyperparameter search completed!")
    print(f"Experiments run in this session: {len(experiments_to_run)}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    import sys

    # Check for --no-resume flag
    resume_mode = True
    if "--no-resume" in sys.argv:
        sys.argv.remove("--no-resume")
        resume_mode = False
        print(
            "⚠️  RESUME MODE DISABLED: Will run all experiments, ignoring completed ones\n"
        )
    else:
        print("✓ RESUME MODE ENABLED: Will skip completed experiments\n")
        print("  (Use --no-resume flag to run all experiments)\n")

    # Use tyro to select from available hyperparam configs
    config = tyro.extras.overridable_config_cli(
        hyperparam_configs, sort_subcommands=True
    )
    main(config, resume=resume_mode)
