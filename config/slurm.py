from nntool.slurm import SlurmConfig
from .env import env, output_path

exclude_code_folders = [
    "wandb",
    "output",
    "output_original",
    "output_caduceus_hyperparam",
    "output_bpnet_medium",
    "outputs",
    "data",
    "datasets",
    ".venv_caduceus",
    ".venv",
    ".git",
    ".vscode",
    "uv.lock",
    "tests",
]

basic_distillation_slurm = SlurmConfig(
    mode=env["slurm"]["mode"],
    job_name=env["slurm"]["job_name"],
    partition=env["slurm"]["partition"],
    node_list=env["slurm"]["node_list"],
    cpus_per_task=4,
    gpus_per_task=1,
    output_parent_path=output_path,
    mem="64GB",
    pack_code=True,
    use_packed_code=True,
    exclude_code_folders=exclude_code_folders,
)
