import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
import time
import tyro

from dataclasses import replace
from config.distillation.config import hyperparam_configs
from config.distillation.config_schema import DistillationHyperparamExperimentConfig
from config.slurm_manager import get_gpu_manager
from ..trainer.utils import ExperimentTracker
from ..trainer.hyperparam_utils import build_method_experiments  # NEW IMPORT
from .distill import main as distill_main


def main(config: DistillationHyperparamExperimentConfig, resume: bool = True):
    """
    Run hyperparameter search over distillation parameters with resume support.

    Now supports method-specific hyperparameters:
    - Vanilla KD: weight_ce, weight_kl, weight_mse, temperature, zscore
    - Logit Standardization: same as vanilla
    - DKD: weight_ce, weight_kl, temperature, zscore, dkd_alpha, dkd_beta (no MSE)
    - DIST: weight_ce, weight_kl, temperature, zscore (no MSE)
    """
    # GPU manager setup
    max_gpu = {"voyager": 3, "laniakea": 7, "galaxy": 5}
    gpu_manager = get_gpu_manager(
        node_limits={
            "voyager": max_gpu["voyager"],
            "laniakea": max_gpu["laniakea"],
            "galaxy": max_gpu["galaxy"],
        },
        node_capacity={"voyager": 4, "laniakea": 8, "galaxy": 6},
    )

    # Initialize experiment tracker
    tracker = ExperimentTracker(
        output_dir=config.trainer_config.output_dir,
        start_timestamp="20251019_000000",
    )

    # ===== BUILD METHOD-SPECIFIC EXPERIMENTS =====
    distill_method = config.distillation_config.distill_method
    print(f"\n{'=' * 60}")
    print(f"Building experiments for method: {distill_method.upper()}")
    print(f"{'=' * 60}")

    all_experiments = build_method_experiments(config, distill_method)

    print(f"\nGenerated {len(all_experiments)} total experiments")

    # ===== CHECK COMPLETED EXPERIMENTS =====
    if resume:
        incomplete, completed, summary = tracker.generate_experiment_plan(all_experiments)
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
        print(f"FULL MODE: Running all {len(all_experiments)} experiments (ignoring completed)")
        print(f"{'=' * 80}\n")
        experiments_to_run = all_experiments

    # ===== PRINT CONFIGURATION =====
    print(f"\n{'=' * 60}")
    print("Hyperparameter Search Configuration")
    print(f"{'=' * 60}")
    print(f"Method: {distill_method}")
    print(f"Tasks: {config.task_names}")
    print(f"Total experiments: {len(experiments_to_run)}")
    print(f"  - CE weights: {config.weight_ces}")
    print(f"  - KL weights: {config.weight_kls}")

    if distill_method in ["vanilla", "logit_standard"]:
        print(f"  - MSE weights: {config.weight_mses}")
    elif distill_method == "dkd":
        print(f"  - DKD alphas: {getattr(config, 'dkd_alphas', [0.5, 1.0, 2.0])}")
        print(f"  - DKD betas: {getattr(config, 'dkd_betas', [4.0, 8.0, 16.0])}")
        print("  - MSE: disabled (DKD doesn't use feature matching)")
    elif distill_method == "dist":
        print("  - MSE: disabled (DIST uses correlation loss)")

    print(f"  - Temperatures: {config.temperatures}")
    print(f"  - Z-scores: {config.zscores}")
    print(f"{'=' * 60}\n")

    # Node preference order
    node_preference = ["voyager", "laniakea"]

    # ===== RUN EXPERIMENTS =====
    for idx, exp in enumerate(experiments_to_run, 1):
        task_name = exp["task_name"]
        hyperparam_config = exp["hyperparam_config"]

        # Extract hyperparameters for display
        weight_ce = hyperparam_config["weight_ce"]
        weight_kl = hyperparam_config["weight_kl"]
        weight_mse = hyperparam_config["weight_mse"]
        temperature = hyperparam_config["temperature"]
        zscore = hyperparam_config["zscore"]

        # Display string
        display_params = (
            f"CE={weight_ce}, KL={weight_kl}, MSE={weight_mse}, T={temperature}, zscore={zscore}"
        )

        # Add method-specific parameters
        if distill_method == "dkd":
            dkd_alpha = hyperparam_config.get("dkd_alpha", 1.0)
            dkd_beta = hyperparam_config.get("dkd_beta", 8.0)
            display_params += f", alpha={dkd_alpha}, beta={dkd_beta}"

        # Check for partial progress
        status = tracker.get_experiment_status(task_name, hyperparam_config)

        print(f"\n{'=' * 60}")
        print(f"[Experiment {idx}/{len(experiments_to_run)}] Task: {task_name}")
        print(f"  {display_params}")
        print(f"{'=' * 60}")

        # Wait for available node
        available_node = gpu_manager.wait_for_available_node(
            preferred_nodes=node_preference,
            check_interval=30,
            max_wait=3600 * 4,
        )

        if available_node is None:
            print(f"⚠️  Skipping experiment {idx} - no available GPU after timeout")
            continue

        print(f"🚀 Submitting to node: {available_node}")

        # Create distillation config with current hyperparameters
        # This preserves ALL fields from hyperparam_config
        new_distill_config = replace(
            config.distillation_config,
            **hyperparam_config,  # Unpack all hyperparameters
        )

        # Update slurm and dataset configs
        new_slurm_config = replace(config.slurm_config, node_list=available_node)
        new_dataset_config = replace(config.dataset_config, task_name=task_name)

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

        # Run distillation
        distill_main(new_experiment_config)

        # Wait for SLURM to register the job
        time.sleep(3)
        for _ in range(10):
            current_jobs = gpu_manager.get_running_jobs_per_node()
            if current_jobs.get(available_node, 0) > 0:
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
        print("⚠️  RESUME MODE DISABLED: Will run all experiments, ignoring completed ones\n")
    else:
        print("✓ RESUME MODE ENABLED: Will skip completed experiments\n")
        print("  (Use --no-resume flag to run all experiments)\n")

    # Use tyro to select from available hyperparam configs
    config = tyro.extras.overridable_config_cli(hyperparam_configs, sort_subcommands=True)
    main(config, resume=resume_mode)
