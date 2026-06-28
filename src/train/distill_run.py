"""Unified, switchable entry point for Carbon-3B -> BPNet distillation: ``--mode {slurm,batch}``.

This is a THIN front door over the two existing dispatch paths -- it adds ONE obvious,
documented selector and routes to the SAME code the legacy entry points already use. It does
NOT replace them: ``python -m src.train.distill <spec...>`` (slurm) and
``python -m src.train.distill_task --task ... --config-list ... [--parallel N]`` (batch) keep
working byte-identically. See ``src.train.distill`` module docstring (the ``MODES`` block) for
the shared-CORE vs DISPATCH-layer split and when each mode applies.

Modes
-----
--mode slurm   ONE HP config per process; the scheduler does the fan-out (lab default).
               Everything AFTER ``--`` is the usual tyro config spec, e.g.:

                   python -m src.train.distill_run --mode slurm -- \
                       carbon-raw --task-names H3K27me3 --slurm-config.mode run

               which is forwarded verbatim to the same resolver+``main`` that
               ``python -m src.train.distill <spec...>`` uses.

--mode batch   Single box loads the teacher ONCE and sweeps a task's HP config list.
               ``--parallel 1`` (default) = serial loop; ``--parallel N`` (>1) = config-parallel.
               Same flags as ``src.train.distill_task``:

                   python -m src.train.distill_run --mode batch \
                       --task H3K27ac --config-list hp_original_stage1.txt [--parallel 8] \
                       [--base-experiment carbon-raw-original]
"""

import argparse
import sys

from .distill import run_distillation


def _run_slurm(spec_args):
    """Resolve a tyro config spec (the tokens after ``--``) and route it to ``mode='slurm'``.

    Mirrors ``src.train.distill.__main__`` exactly: resolve the overridable config CLI over
    the same ``configs`` registry, then dispatch via ``run_distillation(mode='slurm', ...)``
    (== the legacy ``main`` path, so per-config SLURM dispatch is unchanged).
    """
    import tyro
    from config.distillation.config import configs

    # tyro reads sys.argv; hand it ONLY the post-``--`` spec tokens so the unified flags
    # (--mode) are not seen as part of the experiment config spec.
    argv_backup = sys.argv
    try:
        sys.argv = [argv_backup[0]] + list(spec_args)
        config = tyro.extras.overridable_config_cli(configs, sort_subcommands=True)
    finally:
        sys.argv = argv_backup
    run_distillation("slurm", config=config)


def _run_batch(args):
    """Build the batch base_config + overrides exactly like ``distill_task.main`` and route them.

    Reuses ``src.train.distill_task``'s config/override loading so the batch behavior is
    identical to the legacy ``python -m src.train.distill_task`` invocation; only the front
    door differs.
    """
    from dataclasses import replace
    from config.distillation.experiments.carbon import experiment_configs
    from .distill_task import load_config_list

    if args.base_experiment not in experiment_configs:
        raise SystemExit(
            f"Unknown --base-experiment {args.base_experiment!r}; "
            f"choices: {sorted(experiment_configs)}"
        )
    base_config = experiment_configs[args.base_experiment][1]
    base_config = replace(
        base_config,
        dataset_config=replace(base_config.dataset_config, task_name=args.task),
    )
    overrides = load_config_list(args.config_list, args.task)
    if not overrides:
        raise SystemExit(
            f"No config lines for task={args.task!r} found in {args.config_list}"
        )
    print(
        f"[distill_run] mode=batch task={args.task}: {len(overrides)} configs "
        f"from {args.config_list} (parallel={args.parallel})"
    )
    run_distillation(
        "batch",
        base_config=base_config,
        task_name=args.task,
        config_overrides_list=overrides,
        parallel=args.parallel,
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the unified ``--mode {slurm,batch}`` parser (batch flags mirror distill_task)."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        required=True,
        choices=("slurm", "batch"),
        help=(
            "slurm = one HP config per process (scheduler fan-out; lab default); "
            "batch = one box loads the teacher ONCE and sweeps a task's config list "
            "(--parallel selects serial vs config-parallel)."
        ),
    )
    # Batch-mode flags (ignored in slurm mode, where the spec follows ``--``).
    parser.add_argument("--task", help="[batch] Task name to distill (e.g. H3K27ac)")
    parser.add_argument(
        "--config-list", help="[batch] File of HP override spec lines (e.g. hp_original_stage1.txt)"
    )
    parser.add_argument(
        "--base-experiment",
        default="carbon-raw-original",
        help="[batch] Experiment name in carbon.experiment_configs to seed base_config",
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=1,
        help=(
            "[batch] Number of HP configs to train CONCURRENTLY per task (default 1 = "
            "serial loop, unchanged). >1 = config-parallel (teacher freed after warming "
            "its cache; N spawn-workers share the idle GPU)."
        ),
    )
    return parser


def main(argv=None):
    """Parse ``--mode`` and route to the matching dispatch path via ``run_distillation``."""
    argv = list(sys.argv[1:] if argv is None else argv)
    # Split off the slurm tyro spec that follows a literal ``--`` so argparse never sees it.
    spec_args = []
    if "--" in argv:
        i = argv.index("--")
        argv, spec_args = argv[:i], argv[i + 1 :]

    parser = build_parser()
    args = parser.parse_args(argv)

    if args.mode == "slurm":
        _run_slurm(spec_args)
    else:  # batch
        missing = [n for n in ("task", "config_list") if getattr(args, n) is None]
        if missing:
            parser.error(
                "--mode batch requires --task and --config-list "
                f"(missing: {', '.join('--' + m.replace('_', '-') for m in missing)})"
            )
        _run_batch(args)


if __name__ == "__main__":
    main()
