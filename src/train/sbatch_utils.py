"""
Utilities for sbatch-based job submission.

This module provides an alternative to nntool.slurm for submitting jobs via sbatch.
It generates sbatch scripts and submits them directly using the sbatch command.
"""

import os
import subprocess
import json
import tempfile
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any
from datetime import datetime
from pathlib import Path


@dataclass
class SbatchConfig:
    """Configuration for sbatch job submission."""
    
    job_name: str = "omega-genome"
    partition: str = "zhanglab.p"
    node_list: Optional[str] = None  # e.g., "voyager" or "laniakea"
    cpus_per_task: int = 4
    gpus_per_task: int = 1
    mem: str = "64GB"
    time: str = "7-00:00:00"  # 7 days
    mail_type: str = "BEGIN,END,FAIL"
    mail_user: Optional[str] = None
    
    # Python environment
    python_path: str = ""  # Will be auto-detected if empty
    conda_env: Optional[str] = None  # Conda environment name (alternative to python_path)
    
    # Output
    output_dir: str = ""  # Directory for slurm output logs
    
    def __post_init__(self):
        # Auto-detect python path if not set
        if not self.python_path:
            import sys
            self.python_path = sys.executable


@dataclass
class ExperimentJob:
    """Represents a single experiment job to submit."""
    
    task_name: str
    model_size: str
    seed: int
    config_json_path: str  # Path to JSON config file
    output_dir: str
    job_name_suffix: str = ""
    
    def get_job_name(self) -> str:
        """Generate unique job name."""
        base = f"og-{self.task_name[:8]}-{self.model_size}-s{self.seed}"
        if self.job_name_suffix:
            base += f"-{self.job_name_suffix}"
        return base[:64]  # SLURM job name limit


def generate_sbatch_script(
    sbatch_config: SbatchConfig,
    job: ExperimentJob,
    project_path: str,
) -> str:
    """
    Generate sbatch script content for a single experiment.
    
    Args:
        sbatch_config: Sbatch configuration
        job: Experiment job details
        project_path: Path to the project root
        
    Returns:
        String content of the sbatch script
    """
    
    # Build SBATCH directives
    directives = [
        "#!/bin/bash",
        f"#SBATCH --job-name={job.get_job_name()}",
        f"#SBATCH --partition={sbatch_config.partition}",
        f"#SBATCH --nodes=1",
        f"#SBATCH --cpus-per-task={sbatch_config.cpus_per_task}",
        f"#SBATCH --gres=gpu:{sbatch_config.gpus_per_task}",
        f"#SBATCH --mem={sbatch_config.mem}",
        f"#SBATCH --time={sbatch_config.time}",
    ]
    
    # Optional node list
    if sbatch_config.node_list:
        directives.append(f"#SBATCH --nodelist={sbatch_config.node_list}")
    
    # Mail notifications
    if sbatch_config.mail_user:
        directives.append(f"#SBATCH --mail-type={sbatch_config.mail_type}")
        directives.append(f"#SBATCH --mail-user={sbatch_config.mail_user}")
    
    # Output files
    slurm_log_dir = os.path.join(job.output_dir, "slurm_logs")
    directives.append(f"#SBATCH --output={slurm_log_dir}/%x_%j.out")
    directives.append(f"#SBATCH --error={slurm_log_dir}/%x_%j.err")
    
    # Build the script body
    script_lines = directives + [
        "",
        "# Print job info",
        'echo "=========================================="',
        f'echo "Job Name: {job.get_job_name()}"',
        'echo "Job ID: $SLURM_JOB_ID"',
        'echo "Node: $SLURM_NODELIST"',
        f'echo "Task: {job.task_name}"',
        f'echo "Model Size: {job.model_size}"',
        f'echo "Seed: {job.seed}"',
        'echo "Start Time: $(date)"',
        'echo "=========================================="',
        "",
        "# Change to project directory",
        f"cd {project_path}",
        "",
        "# Set environment variables",
        'export TOKENIZERS_PARALLELISM=false',
        "",
    ]
    
    # Conda activation if specified
    if sbatch_config.conda_env:
        script_lines.extend([
            "# Activate conda environment",
            f"source $(conda info --base)/etc/profile.d/conda.sh",
            f"conda activate {sbatch_config.conda_env}",
            "",
        ])
    
    # Python command
    script_lines.extend([
        "# Run experiment",
        f"{sbatch_config.python_path} -m src.train.run_single_experiment \\",
        f"    --config-path {job.config_json_path}",
        "",
        "# Print completion",
        'echo "=========================================="',
        'echo "End Time: $(date)"',
        'echo "=========================================="',
    ])
    
    return "\n".join(script_lines)


def save_experiment_config(
    config_dict: Dict[str, Any],
    output_dir: str,
    task_name: str,
    model_size: str,
    seed: int,
) -> str:
    """
    Save experiment configuration to a JSON file.
    
    Args:
        config_dict: Full experiment configuration dictionary
        output_dir: Directory to save the config
        task_name: Task name
        model_size: Model size
        seed: Random seed
        
    Returns:
        Path to the saved config file
    """
    # Create config directory
    config_dir = os.path.join(output_dir, "configs")
    os.makedirs(config_dir, exist_ok=True)
    
    # Generate unique filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config_filename = f"{task_name}_{model_size}_seed{seed}_{timestamp}.json"
    config_path = os.path.join(config_dir, config_filename)
    
    # Save config
    with open(config_path, "w") as f:
        json.dump(config_dict, f, indent=2, default=str)
    
    return config_path


def submit_sbatch_job(
    script_path: str,
    dry_run: bool = False,
) -> Optional[str]:
    """
    Submit an sbatch script.
    
    Args:
        script_path: Path to the sbatch script
        dry_run: If True, print the command but don't execute
        
    Returns:
        Job ID if successful, None otherwise
    """
    cmd = ["sbatch", script_path]
    
    if dry_run:
        print(f"[DRY RUN] Would execute: {' '.join(cmd)}")
        with open(script_path, "r") as f:
            print("Script content:")
            print(f.read())
        return "DRY_RUN_JOB_ID"
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
        )
        # Parse job ID from output like "Submitted batch job 12345"
        output = result.stdout.strip()
        if "Submitted batch job" in output:
            job_id = output.split()[-1]
            print(f"✓ Submitted job: {job_id}")
            return job_id
        else:
            print(f"Warning: Unexpected sbatch output: {output}")
            return None
            
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to submit job: {e.stderr}")
        return None


def create_and_submit_experiment(
    sbatch_config: SbatchConfig,
    experiment_config: Dict[str, Any],
    task_name: str,
    model_size: str,
    seed: int,
    project_path: str,
    output_dir: str,
    dry_run: bool = False,
) -> Optional[str]:
    """
    Create and submit a single experiment job.
    
    Args:
        sbatch_config: Sbatch configuration
        experiment_config: Full experiment configuration dictionary
        task_name: Task name
        model_size: Model size
        seed: Random seed
        project_path: Path to the project root
        output_dir: Output directory for this experiment
        dry_run: If True, don't actually submit
        
    Returns:
        Job ID if successful, None otherwise
    """
    # Create directories
    os.makedirs(output_dir, exist_ok=True)
    slurm_log_dir = os.path.join(output_dir, "slurm_logs")
    os.makedirs(slurm_log_dir, exist_ok=True)
    scripts_dir = os.path.join(output_dir, "sbatch_scripts")
    os.makedirs(scripts_dir, exist_ok=True)
    
    # Save experiment config
    config_path = save_experiment_config(
        experiment_config,
        output_dir,
        task_name,
        model_size,
        seed,
    )
    
    # Create job object
    job = ExperimentJob(
        task_name=task_name,
        model_size=model_size,
        seed=seed,
        config_json_path=config_path,
        output_dir=output_dir,
    )
    
    # Generate sbatch script
    script_content = generate_sbatch_script(
        sbatch_config,
        job,
        project_path,
    )
    
    # Save script
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    script_filename = f"{job.get_job_name()}_{timestamp}.sh"
    script_path = os.path.join(scripts_dir, script_filename)
    
    with open(script_path, "w") as f:
        f.write(script_content)
    
    # Make executable
    os.chmod(script_path, 0o755)
    
    print(f"\n{'=' * 60}")
    print(f"Job: {job.get_job_name()}")
    print(f"  Task: {task_name}")
    print(f"  Size: {model_size}")
    print(f"  Seed: {seed}")
    print(f"  Script: {script_path}")
    print(f"  Config: {config_path}")
    
    # Submit
    job_id = submit_sbatch_job(script_path, dry_run=dry_run)
    
    return job_id


def get_default_sbatch_config(
    node_list: Optional[str] = None,
    mail_user: Optional[str] = None,
) -> SbatchConfig:
    """Get default sbatch configuration for OmegaGenome experiments."""
    return SbatchConfig(
        job_name="omega-genome",
        partition="zhanglab.p",
        node_list=node_list,
        cpus_per_task=4,
        gpus_per_task=1,
        mem="64GB",
        time="7-00:00:00",
        mail_user=mail_user,
    )
