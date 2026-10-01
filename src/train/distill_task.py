"""Per-task BATCH distillation CLI: load the Carbon-3B teacher ONCE, sweep HP configs.

WHY: each HP config of a task currently reloads the 3B base + merges the per-task LoRA
adapter + re-evaluates the teacher (~3-5 min of pure overhead), and a task has ~32
configs. This entry point loads the teacher exactly once via ``prepare_task`` and loops
the task's configs via ``train_student`` (see ``src.train.distill.distill_task_batch``),
amortizing the load across the whole sweep.

It DELIBERATELY does not touch the per-config ``python -m src.train.distill <spec>`` path
(used by SLURM), which stays 100% intact. Reuse via the same building blocks
(``prepare_task`` / ``train_student``) guarantees behavior parity.

Usage:
    python -m src.train.distill_task --task <name> --config-list <file> \
        [--base-experiment carbon-raw-original]

where ``<file>`` is a list of HP override lines, ONE config per line, each a string of
the SAME flags the per-config CLI accepts, e.g.:

    carbon-raw-original --task-names H3K27ac --distillation-config.weight-ce 0.5 \
        --distillation-config.weight-kl 0.25 --distillation-config.weight-mse 0.0 \
        --distillation-config.temperature 1.5 --trainer-config.early-stop-patience 70 \
        --slurm-config.mode run

Only the distillation-config HP flags (weight-ce/kl/mse, temperature) and random-state are
parsed into overrides; the leading experiment name + task-names + slurm/trainer flags on
each line are ignored (the task is fixed by --task, and the batch runs INLINE in one
process so slurm mode is irrelevant). This keeps the existing ``hp_original_stage1.txt``
spec files usable verbatim.
"""

import os

# FORK-SAFETY: set BEFORE any import that may import torch + call torch.cuda.is_available(), so the
# probe uses NVML (no cuInit) and never poisons a later fork. This is a process entry point for the
# batch/fork path, so setting it first here guarantees the var is in place regardless of internal
# import order. See src/train/distill.py for the full rationale.
os.environ.setdefault("PYTORCH_NVML_BASED_CUDA_CHECK", "1")

import argparse
import shlex
from dataclasses import replace

from config.distillation.experiments.carbon import experiment_configs
from .distill import distill_task_batch


# Map of CLI flag -> (DistillationModelConfig field) for the HP we sweep per task.
# These are the ONLY knobs that vary across a task's configs in the original stage specs.
_DISTILL_FLAG_TO_FIELD = {
    "--distillation-config.weight-ce": "weight_ce",
    "--distillation-config.weight-kl": "weight_kl",
    "--distillation-config.weight-mse": "weight_mse",
    "--distillation-config.temperature": "temperature",
}
# Allow the alternate underscore spelling too.
_DISTILL_FLAG_TO_FIELD.update(
    {k.replace("-", "_").replace("__", "--", 1): v for k, v in list(_DISTILL_FLAG_TO_FIELD.items())}
)

# Map of CLI flag -> (DistillTrainerConfig field, caster) for trainer-config knobs that the
# per-config (SLURM/tyro) path applies and that therefore MUST also be applied in batch mode to
# stay behavior-identical. ``--trainer-config.early-stop-patience`` is fixed (70) on every
# stage-1 spec line, but the base ``carbon_original_trainer_config`` leaves ``early_stop_patience``
# at the schema default ``None`` (= early stopping DISABLED). Dropping the flag in batch mode would
# therefore train every config to the full epoch budget instead of stopping at patience -> a real
# train-behavior divergence from the SLURM path. Parse it.
_TRAINER_FLAG_TO_FIELD = {
    "--trainer-config.early-stop-patience": ("early_stop_patience", int),
}
_TRAINER_FLAG_TO_FIELD.update(
    {k.replace("-", "_").replace("__", "--", 1): v for k, v in list(_TRAINER_FLAG_TO_FIELD.items())}
)


def parse_override_line(line: str) -> dict:
    """Parse one HP spec line into an override dict for ``distill_task_batch``.

    Extracts the distillation-config HP flags (weight-ce/kl/mse, temperature), the trainer-config
    knobs the SLURM path applies (``--trainer-config.early-stop-patience``), and an optional
    ``--random-state``; ignores the experiment name, --task-names, and slurm flags (the task is
    pinned by --task and the batch runs inline). Returns e.g.
    ``{"distillation_config": {"weight_ce": 0.5, ...}, "trainer_config": {"early_stop_patience": 70},
    "random_state": 42}``.
    """
    tokens = shlex.split(line.strip())
    dc: dict = {}
    tc: dict = {}
    top: dict = {}
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok in _DISTILL_FLAG_TO_FIELD and i + 1 < len(tokens):
            dc[_DISTILL_FLAG_TO_FIELD[tok]] = float(tokens[i + 1])
            i += 2
            continue
        if tok in _TRAINER_FLAG_TO_FIELD and i + 1 < len(tokens):
            field, cast = _TRAINER_FLAG_TO_FIELD[tok]
            tc[field] = cast(tokens[i + 1])
            i += 2
            continue
        if tok in ("--random-state", "--random_state") and i + 1 < len(tokens):
            top["random_state"] = int(tokens[i + 1])
            i += 2
            continue
        i += 1
    override: dict = {}
    if dc:
        override["distillation_config"] = dc
    if tc:
        override["trainer_config"] = tc
    override.update(top)
    return override


def load_config_list(path: str, task_name: str) -> list:
    """Read the spec file, keep lines for ``task_name``, parse each into an override dict."""
    overrides = []
    with open(path) as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            # Only keep this task's lines (spec files may interleave tasks).
            if f"--task-names {task_name} " not in (line + " "):
                continue
            ov = parse_override_line(line)
            if ov:
                overrides.append(ov)
    return overrides


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, help="Task name to distill (e.g. H3K27ac)")
    parser.add_argument(
        "--config-list",
        required=True,
        help="File of HP override spec lines (e.g. hp_original_stage1.txt)",
    )
    parser.add_argument(
        "--base-experiment",
        default="carbon-raw-original",
        help="Experiment name in carbon.experiment_configs to seed base_config",
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=1,
        help=(
            "Number of HP configs to train CONCURRENTLY per task (default 1 = the "
            "original serial path, unchanged). With >1, N workers share the idle GPU."
        ),
    )
    parser.add_argument(
        "--parallel-mode",
        default="fork",
        choices=("fork", "spawn"),
        help=(
            "How >1 parallel workers are fanned out. 'fork' (default): data + teacher "
            "logits/features loaded ONCE on CPU and SHARED across configs via Linux "
            "copy-on-write (memory ~1x/task; auto-falls back to spawn if the teacher "
            "could not be cache-skipped). 'spawn': fresh-Python workers each re-read the "
            "data + teacher cache from disk (memory ~Nx)."
        ),
    )
    args = parser.parse_args()

    if args.base_experiment not in experiment_configs:
        raise SystemExit(
            f"Unknown --base-experiment {args.base_experiment!r}; "
            f"choices: {sorted(experiment_configs)}"
        )
    base_config = experiment_configs[args.base_experiment][1]

    # Pin the task into dataset_config exactly as src.train.distill.main does.
    base_config = replace(
        base_config,
        dataset_config=replace(base_config.dataset_config, task_name=args.task),
    )

    overrides = load_config_list(args.config_list, args.task)
    if not overrides:
        raise SystemExit(f"No config lines for task={args.task!r} found in {args.config_list}")
    print(f"[distill_task] task={args.task}: {len(overrides)} configs from {args.config_list}")

    distill_task_batch(
        base_config,
        args.task,
        overrides,
        parallel=args.parallel,
        parallel_mode=args.parallel_mode,
    )


if __name__ == "__main__":
    main()
