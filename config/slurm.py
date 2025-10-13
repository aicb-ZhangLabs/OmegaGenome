from nntool.slurm import SlurmConfig
from .env import env, output_path

exclude_code_folders = [
    "wandb",
    "outputs",
    "datasets",
    ".venv",
    ".vscode",
    "uv.lock",
    "tests",
]

basic_distillation_slurm = SlurmConfig(
    mode=env["slurm"]["mode"],
    job_name=env["slurm"]["job_name"],
    partition=env["slurm"]["partition"],
    node_list_exclude=env["slurm"]["node_list_exclude"],
    cpus_per_task=4,
    gpus_per_task=1,
    output_parent_path=output_path,
    mem="256GB",
    pack_code=True,
    use_packed_code=True,
)
