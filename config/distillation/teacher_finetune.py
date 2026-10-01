"""Stage-1 teacher fine-tuning configs.

Adding a new teacher gLM is one entry here:
  1. define its ``GLMConfig`` in ``glm.py`` (model id + any input formatting),
  2. add a ``TeacherFinetuneConfig`` below and register it in ``finetune_configs``.

The CLI (``python -m src.train.finetune_teacher <name>``) and the engine
(``src/trainer/finetune_trainer.py``) are teacher-agnostic, so nothing else changes.
"""

from dataclasses import dataclass, field, replace
from typing import List, Optional, Union

from nntool.slurm import SlurmConfig
from src.data.dataset import DatasetConfig
from src.model.glm import GLMConfig

from .data import nucletide_transformer_revised_benchmark
from .glm import carbon_3b, carbon_8b
from ..env import output_path
from ..slurm import basic_distillation_slurm

# The full NT-revised benchmark (the 18 tasks evaluated in the paper).
ALL_18_TASKS: List[str] = [
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
class TeacherFinetuneConfig:
    """Everything needed to fine-tune one teacher across one or more tasks."""

    teacher_name: str
    teacher_config: GLMConfig
    output_dir: str
    wandb_project: str = "teacher_finetune"
    task_names: List[str] = field(default_factory=lambda: list(ALL_18_TASKS))
    dataset_config: DatasetConfig = field(
        default_factory=lambda: nucletide_transformer_revised_benchmark
    )
    slurm_config: SlurmConfig = field(default_factory=lambda: basic_distillation_slurm)

    # --- optimization ---
    epochs: int = 10
    batch_size: int = 8
    eval_batch_size: int = 16
    grad_accum: int = 1
    lr: float = 1e-4
    weight_decay: float = 0.0
    warmup_ratio: float = 0.1
    lr_scheduler_type: str = "cosine"
    max_len: int = 1024
    early_stopping_patience: int = 3  # in eval epochs; 0 disables
    optim: str = "adamw_torch"  # HF optimizer; use "paged_adamw_8bit" for full-FT on <80GB GPUs (needs bitsandbytes)

    # --- precision / memory (defaults suit billion-scale teachers) ---
    bf16: bool = True
    gradient_checkpointing: bool = True

    # --- LoRA (full fine-tune if use_lora=False) ---
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    # None lets peft infer; "all-linear" or an explicit list also work.
    lora_target_modules: Optional[Union[str, List[str]]] = None

    # --- misc ---
    log_every: int = 50
    num_workers: int = 4
    seed: int = 42
    use_wandb: bool = True


# ---------------------------------------------------------------------------
# Carbon (HuggingFaceBio, autoregressive, Apache-2.0). LoRA on all linear layers;
# bf16 + gradient checkpointing so Carbon-3B fits a single 24-48GB GPU.
# ---------------------------------------------------------------------------
carbon_3b_finetune = TeacherFinetuneConfig(
    teacher_name="carbon_3b",
    teacher_config=carbon_3b,
    output_dir=f"{output_path}/teacher_finetune/carbon_3b",
    wandb_project="carbon_teacher_finetune",
    lora_target_modules="all-linear",
)

# Quick single-task smoke test (1 short task, 1 epoch).
carbon_3b_debug = replace(
    carbon_3b_finetune,
    teacher_name="carbon_3b_debug",
    output_dir=f"{output_path}/teacher_finetune/carbon_3b_debug",
    task_names=["promoter_tata"],
    epochs=1,
    early_stopping_patience=0,
    use_wandb=False,
)

carbon_8b_finetune = replace(
    carbon_3b_finetune,
    teacher_name="carbon_8b",
    teacher_config=carbon_8b,
    output_dir=f"{output_path}/teacher_finetune/carbon_8b",
    batch_size=4,
    grad_accum=2,
)

# Full-parameter fine-tune (use_lora=False -> all 3B params train). Needs an 80GB H100 with
# plain adamw_torch (fp32 optimizer states ~48GB); on a 48GB A6000 set optim="paged_adamw_8bit"
# (requires bitsandbytes). Lower LR than LoRA, small batch + grad-accum for memory.
carbon_3b_fullft = replace(
    carbon_3b_finetune,
    teacher_name="carbon_3b_fullft",
    output_dir=f"{output_path}/teacher_finetune/carbon_3b_fullft",
    use_lora=False,
    batch_size=2,
    grad_accum=8,  # effective batch 16, matches the LoRA runs
    lr=2e-5,  # full-FT uses a lower LR than LoRA's 1e-4
    optim="adamw_torch",
)

# One-task timing probe for full-FT (enhancers = representative 400bp/30k task).
carbon_3b_fullft_debug = replace(
    carbon_3b_fullft,
    teacher_name="carbon_3b_fullft_debug",
    output_dir=f"{output_path}/teacher_finetune/carbon_3b_fullft_debug",
    task_names=["enhancers"],
    epochs=1,
    early_stopping_patience=0,
    use_wandb=False,
)

# Registry consumed by the CLI. To add a teacher (e.g. AIDO.DNA-7B, GENERATOR-1B):
#   aido_dna_7b_finetune = replace(carbon_3b_finetune, teacher_name="aido_dna_7b",
#                                  teacher_config=aido_dna_7b, output_dir=...)
# then add it to this dict.
finetune_configs = {
    "carbon_3b": ("Fine-tune Carbon-3B (LoRA) on the 18-task NT benchmark", carbon_3b_finetune),
    "carbon_3b_debug": ("Carbon-3B single-task smoke test", carbon_3b_debug),
    "carbon_8b": ("Fine-tune Carbon-8B (LoRA) on the 18-task NT benchmark", carbon_8b_finetune),
    "carbon_3b_fullft": ("Full-parameter fine-tune Carbon-3B (needs H100)", carbon_3b_fullft),
    "carbon_3b_fullft_debug": ("Full-FT timing probe on one task", carbon_3b_fullft_debug),
}
