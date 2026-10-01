"""Carbon-3B (LoRA teachers) -> deploy_120k BPNet distillation experiments.

Mirrors experiments/nt.py + nt_different_size.py, but with the Carbon-3B LoRA teachers (consolidated
on SSD) as the stage-1 teacher and the ~0.12M `deploy_120k` BPNet as the student.

Configs:
  * carbon_raw_config / carbon_l2norm_config  — the requested ce0.5/kl0.5/mse0.2 vanilla 18-task run,
    one with raw MSE and one with L2-normalized MSE (the raw-vs-norm comparison at full scale).
  * carbon_base_hyperparam_{raw,l2norm}_config — the `base` vanilla HP sweep (weight_mse in [0,1,2,5],
    weight_kl in [0,0.25,0.5,1.0], temperature in [0.5,1,1.5,2,4]); mse_normalize inherited from the
    distillation_config.
"""

from ..config_schema import (
    DistillationExperimentConfig,
    DistillationHyperparamExperimentConfig,
)
from ..glm import carbon_3b_lora, nt_2b5
from ..bpnet import deploy_120k_bpnet_config, original_bpnet_classifier_config
from .nt import NT_PARENT_PATH
from ..trainer import (
    carbon_trainer_config,
    carbon_hyperparam_trainer_config,
    carbon_debug_trainer_config,
    carbon_original_trainer_config,
)
from ..data import nucletide_transformer_revised_benchmark
from ..distillation_model import carbon_vanilla_mse_raw, carbon_vanilla_mse_l2norm
from ...slurm import basic_distillation_slurm, run_distillation_slurm
from ..paths import CARBON_TEACHER_DIR

# Parent dir of the 18 `{task}_finetuned/` LoRA adapters. Machine-specific value lives in ONE place
# (config/distillation/paths.py, env-overridable via CARBON_TEACHER_DIR) for easy reproduction.
CARBON_PARENT_PATH = CARBON_TEACHER_DIR

CARBON_TASKS = [
    "H3K27me3",
    "H3K36me3",
    "H4K20me1",
    "H2AFZ",
    "H3K27ac",
    "H3K4me1",
    "H3K4me2",
    "H3K4me3",
    "H3K9ac",
    "H3K9me3",
    "promoter_all",
    "promoter_tata",
    "promoter_no_tata",
    "enhancers",
    "enhancers_types",
    "splice_sites_all",
    "splice_sites_acceptors",
    "splice_sites_donors",
]

# --- single-config 18-task runs (ce0.5/kl0.5/mse0.2 vanilla): raw vs L2-norm MSE ---
carbon_raw_config = DistillationExperimentConfig(
    task_names=CARBON_TASKS,
    teacher_config=carbon_3b_lora,
    teacher_parent_dir=CARBON_PARENT_PATH,
    model_type="glm",
    student_config=deploy_120k_bpnet_config,
    distillation_config=carbon_vanilla_mse_raw,
    trainer_config=carbon_trainer_config,
    dataset_config=nucletide_transformer_revised_benchmark,
    slurm_config=basic_distillation_slurm,
)
carbon_l2norm_config = DistillationExperimentConfig(
    task_names=CARBON_TASKS,
    teacher_config=carbon_3b_lora,
    teacher_parent_dir=CARBON_PARENT_PATH,
    model_type="glm",
    student_config=deploy_120k_bpnet_config,
    distillation_config=carbon_vanilla_mse_l2norm,
    trainer_config=carbon_trainer_config,
    dataset_config=nucletide_transformer_revised_benchmark,
    slurm_config=basic_distillation_slurm,
)
# Carbon-3B distillation into the `original` BPNet student — the SAME student the NT/Enformer/Caduceus/
# DNABERT-2 distillations use (full receptive field, reproduces the published BPNet baseline). SEPARATE
# output leaf (carbon_original_trainer_config -> .../original) so it cannot collide with the existing
# deploy_120k grid/ckpts. Teacher cache is shared (same cache_base_dir).
carbon_raw_original_config = DistillationExperimentConfig(
    task_names=CARBON_TASKS,
    teacher_config=carbon_3b_lora,
    teacher_parent_dir=CARBON_PARENT_PATH,
    model_type="glm",
    student_config=original_bpnet_classifier_config,
    distillation_config=carbon_vanilla_mse_raw,
    trainer_config=carbon_original_trainer_config,
    dataset_config=nucletide_transformer_revised_benchmark,
    slurm_config=basic_distillation_slurm,
)

# --- `base` vanilla HP sweep (mse_normalize inherited from distillation_config) ---
# gen_hp_specs.py reads this grid (kl x mse x T x tasks), collapsing kl=0 x T (T is a no-op at kl=0).
# The carbon-raw-original re-search is STAGED BY mse to control queue load and let us review results
# between stages (edit weight_mses + rerun gen_hp_specs --cfg carbon-raw-original; resume is scoped to
# the `original` output leaf, so finished combos are skipped):
#   stage 1: weight_mses=[0.0, 1]   (current)
#   stage 2: weight_mses=[0.25]
#   stage 3: weight_mses=[2, 5]
# (The original deploy_120k grid used the full [0.0, 1, 2, 5] in one pass; it is complete and untouched.)
_GRID = dict(
    weight_ces=[0.5],
    weight_kls=[0.0, 0.25, 0.5, 1.0],
    weight_mses=[0.0, 1, 2, 5],  # FULL base grid (all stages) — resume skips finished combos
    temperatures=[0.5, 1.0, 1.5, 2.0, 4.0],
    zscores=[False],
)
carbon_base_hyperparam_raw_config = DistillationHyperparamExperimentConfig(
    task_names=CARBON_TASKS,
    teacher_config=carbon_3b_lora,
    teacher_parent_dir=CARBON_PARENT_PATH,
    model_type="glm",
    student_config=deploy_120k_bpnet_config,
    distillation_config=carbon_vanilla_mse_raw,
    trainer_config=carbon_hyperparam_trainer_config,
    dataset_config=nucletide_transformer_revised_benchmark,
    slurm_config=basic_distillation_slurm,
    **_GRID,
)
carbon_base_hyperparam_l2norm_config = DistillationHyperparamExperimentConfig(
    task_names=CARBON_TASKS,
    teacher_config=carbon_3b_lora,
    teacher_parent_dir=CARBON_PARENT_PATH,
    model_type="glm",
    student_config=deploy_120k_bpnet_config,
    distillation_config=carbon_vanilla_mse_l2norm,
    trainer_config=carbon_hyperparam_trainer_config,
    dataset_config=nucletide_transformer_revised_benchmark,
    slurm_config=basic_distillation_slurm,
    **_GRID,
)

# Focused grid (18 combos vs full 80) — explores the impactful knobs around the current default
# (T=2.0, w_mse=0.2, w_kl=0.5) incl. logits-only (w_mse=0) and feature-heavy (w_mse=1). 18x18=324 runs.
_GRID_FOCUSED = dict(
    weight_ces=[0.5],
    weight_kls=[0.5, 1.0],
    weight_mses=[0.0, 0.2, 1.0],
    temperatures=[1.0, 2.0, 4.0],
    zscores=[False],
)
carbon_hp_raw_focused_config = DistillationHyperparamExperimentConfig(
    task_names=CARBON_TASKS,
    teacher_config=carbon_3b_lora,
    teacher_parent_dir=CARBON_PARENT_PATH,
    model_type="glm",
    student_config=deploy_120k_bpnet_config,
    distillation_config=carbon_vanilla_mse_raw,
    trainer_config=carbon_hyperparam_trainer_config,
    dataset_config=nucletide_transformer_revised_benchmark,
    slurm_config=basic_distillation_slurm,
    **_GRID_FOCUSED,
)

# 1-task smoke (inline, 2 epochs) — validates teacher LoRA load -> precompute -> student train
# end-to-end before the full 36 + 1152-job launches. mode="run" so it runs in-process (no submit).
carbon_smoke_config = DistillationExperimentConfig(
    task_names=["H3K4me3"],
    teacher_config=carbon_3b_lora,
    teacher_parent_dir=CARBON_PARENT_PATH,
    model_type="glm",
    student_config=deploy_120k_bpnet_config,
    distillation_config=carbon_vanilla_mse_raw,
    trainer_config=carbon_debug_trainer_config,
    dataset_config=nucletide_transformer_revised_benchmark,
    slurm_config=run_distillation_slurm,
)

# ISOLATION SMOKE: an EXISTING teacher (NT-2.5B) through the exact shared code paths the Carbon
# changes touch (build_glm dtype/pad, get_best_checkpoint dispatch, tokenize_teacher_inputs, precompute
# + eval). NT's neutral config (no <dna>, add_special_tokens=True, fp32) must run end-to-end unaffected.
nt_iso_smoke_config = DistillationExperimentConfig(
    task_names=["H3K9me3"],
    teacher_config=nt_2b5,
    teacher_parent_dir=NT_PARENT_PATH,
    model_type="nt",
    student_config=deploy_120k_bpnet_config,
    distillation_config=carbon_vanilla_mse_raw,  # kl+mse>0 -> exercises both precompute paths
    trainer_config=carbon_debug_trainer_config,
    dataset_config=nucletide_transformer_revised_benchmark,
    slurm_config=run_distillation_slurm,
)

# Voyager (80GB) hedge of the smoke — DISTINCT cache_base_dir + output_dir so it can't collide with
# the laniakea carbon-smoke (same teacher+task would otherwise share the precompute cache path).
from dataclasses import replace as _replace  # noqa: E402
from ..paths import CARBON_OUTPUT_BASE as _COB  # noqa: E402
from ..trainer import _CARBON_OUT as _CO  # noqa: E402

carbon_smoke_vy_config = _replace(
    carbon_smoke_config,
    trainer_config=_replace(
        carbon_debug_trainer_config,
        output_dir=f"{_CO}/smoke_vy",
        cache_base_dir=f"{_COB}/smoke_vy",
    ),
)

# Registries (picked by name via tyro on the distill / distill_hyperparam CLIs).
# Values are (description, config) tuples — tyro's overridable_config_cli indexes [1] for the config.
experiment_configs = {
    "carbon-smoke": (
        "Carbon->deploy_120k 1-task inline smoke (H3K4me3, 2 epochs)",
        carbon_smoke_config,
    ),
    "carbon-smoke-vy": (
        "Voyager hedge of carbon-smoke (distinct cache/output paths)",
        carbon_smoke_vy_config,
    ),
    "nt-iso-smoke": (
        "NT-2.5B isolation smoke: proves Carbon changes don't break NT",
        nt_iso_smoke_config,
    ),
    "carbon-raw": (
        "Carbon->deploy_120k 18-task vanilla ce0.5/kl0.5/mse0.2, raw MSE",
        carbon_raw_config,
    ),
    "carbon-l2norm": (
        "Carbon->deploy_120k 18-task vanilla ce0.5/kl0.5/mse0.2, L2-norm MSE",
        carbon_l2norm_config,
    ),
    "carbon-raw-original": (
        "Carbon->ORIGINAL BPNet student (full RF, matches other teachers); SEPARATE output leaf",
        carbon_raw_original_config,
    ),
}
hyperparam_experiment_configs = {
    "carbon-base-raw": (
        "Carbon->deploy_120k base vanilla HP sweep (raw MSE)",
        carbon_base_hyperparam_raw_config,
    ),
    "carbon-base-l2norm": (
        "Carbon->deploy_120k base vanilla HP sweep (L2-norm MSE)",
        carbon_base_hyperparam_l2norm_config,
    ),
    "carbon-hp-raw": (
        "Carbon->deploy_120k FOCUSED HP sweep (raw, 18 combos x 18 tasks=324)",
        carbon_hp_raw_focused_config,
    ),
}
