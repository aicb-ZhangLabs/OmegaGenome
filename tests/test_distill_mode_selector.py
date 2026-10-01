"""Tests for the unified ``--mode {slurm,batch}`` selector (CPU-only, no GPU / no 3B teacher).

Proves the ONE documented switch routes each mode to the EXISTING dispatch path WITHOUT
changing it, and that the legacy entry points still dispatch as before:

  (1) run_distillation(mode='slurm', config=...) -> the single-config ``main`` path (and main
      itself drives the per-config SLURM dispatch ``distill[slurm_config]`` once per task);
  (2) run_distillation(mode='batch', ...) -> ``distill_task_batch`` with ``parallel`` HONORED
      (1 and >1 both plumbed through to the same function the H100 path calls);
  (3) an unknown mode / missing mode-args raise a clear ValueError;
  (4) BACK-COMPAT: the unified CLI ``distill_run --mode batch`` routes to the SAME
      ``distill_task_batch`` call the legacy ``distill_task.py --parallel N`` makes (parallel
      honored), and ``--mode slurm -- <spec>`` resolves a config and routes to ``main``;
  (5) BACK-COMPAT: the legacy ``distill_task.main()`` still calls ``distill_task_batch``
      (default parallel=1 and explicit --parallel N) -- the H100 invocation is untouched;
  (6) BACK-COMPAT: the legacy SLURM ``distill.main()`` still dispatches via
      ``distill[slurm_config]`` once per task name.

The heavy core (prepare_task/train_student) and the real teacher are never touched: every
dispatch target is monkeypatched to a recorder so this runs on CPU in ms.

Run: <venv>/bin/python -m tests.test_distill_mode_selector
"""

import sys
from dataclasses import dataclass, field

_n = 0


def ok(cond, msg):
    global _n
    assert cond, "FAIL: " + msg
    _n += 1
    print(f"  ok: {msg}")


@dataclass
class _FakeDistillCfg:
    weight_ce: float = 1.0
    weight_kl: float = 0.0
    weight_mse: float = 0.0
    temperature: float = 1.0


@dataclass
class _FakeDatasetCfg:
    task_name: str = ""


@dataclass
class _FakeExperimentCfg:
    distillation_config: _FakeDistillCfg = field(default_factory=_FakeDistillCfg)
    dataset_config: _FakeDatasetCfg = field(default_factory=_FakeDatasetCfg)
    random_state: int = 42
    task_names: tuple = ("H3K27ac",)


# ---------------------------------------------------------------------------
# (1) mode='slurm' routes to main(config); (2) mode='batch' -> distill_task_batch.
# ---------------------------------------------------------------------------
def test_run_distillation_routes_each_mode():
    import src.train.distill as d

    captured = {"main": [], "batch": []}
    orig = (d.main, d.distill_task_batch)
    try:
        d.main = lambda config: captured["main"].append(config)
        d.distill_task_batch = (
            lambda base_config, task_name, overrides, parallel=1, parallel_mode="fork": captured[
                "batch"
            ].append((task_name, len(overrides), parallel))
        )

        cfg = _FakeExperimentCfg()
        d.run_distillation("slurm", config=cfg)
        ok(captured["main"] == [cfg], "(1) mode='slurm' routes to the single-config main() path")
        ok(captured["batch"] == [], "(1) mode='slurm' does NOT touch the batch path")

        base = _FakeExperimentCfg()
        ovs = [
            {"distillation_config": {"temperature": 1.0}},
            {"distillation_config": {"temperature": 2.0}},
        ]
        d.run_distillation(
            "batch", base_config=base, task_name="H3K4me1", config_overrides_list=ovs
        )
        ok(
            captured["batch"] == [("H3K4me1", 2, 1)],
            "(2) mode='batch' routes to distill_task_batch (default parallel=1)",
        )

        d.run_distillation(
            "batch", base_config=base, task_name="H3K4me1", config_overrides_list=ovs, parallel=8
        )
        ok(captured["batch"][-1] == ("H3K4me1", 2, 8), "(2) mode='batch' HONORS parallel>1")
    finally:
        d.main, d.distill_task_batch = orig


def test_run_distillation_validation():
    """(3) bad mode / missing required args raise a clear ValueError."""
    import src.train.distill as d

    try:
        d.run_distillation("nope")
        ok(False, "unknown mode should raise")
    except ValueError as e:
        ok("Unknown mode" in str(e), "(3) unknown mode -> ValueError naming it")

    try:
        d.run_distillation("slurm")  # no config
        ok(False, "slurm without config should raise")
    except ValueError as e:
        ok("config" in str(e), "(3) mode='slurm' without config -> ValueError")

    try:
        d.run_distillation("batch", base_config=_FakeExperimentCfg(), task_name="T")  # no overrides
        ok(False, "batch without overrides should raise")
    except ValueError as e:
        ok("config_overrides_list" in str(e), "(3) mode='batch' missing args -> ValueError")


# ---------------------------------------------------------------------------
# (4) the unified CLI routes batch -> distill_task_batch (parallel honored) and
#     slurm -> main, identical to the legacy entry points.
# ---------------------------------------------------------------------------
def test_unified_cli_batch_routes_like_legacy():
    import src.train.distill_run as dr

    captured = {}
    orig = (dr.run_distillation, None)
    # Patch load_config_list + experiment_configs the batch front door uses.
    import src.train.distill_task as dt

    orig_load = dt.load_config_list
    import config.distillation.experiments.carbon as carbon

    orig_exp = carbon.experiment_configs
    try:

        def fake_run(mode, **kw):
            captured["mode"] = mode
            captured["kw"] = kw

        dr.run_distillation = fake_run
        dt.load_config_list = lambda path, task: [{"distillation_config": {"temperature": 1.0}}]
        carbon.experiment_configs = {"carbon-raw-original": (None, _FakeExperimentCfg())}

        # default parallel = 1
        dr.main(["--mode", "batch", "--task", "H3K4me1", "--config-list", "x.txt"])
        ok(captured["mode"] == "batch", "(4) CLI --mode batch -> run_distillation(mode='batch')")
        ok(captured["kw"]["parallel"] == 1, "(4) CLI batch default parallel=1 plumbed through")
        ok(captured["kw"]["task_name"] == "H3K4me1", "(4) CLI batch task plumbed through")

        # explicit parallel 8 (the H100 config-parallel invocation)
        dr.main(
            ["--mode", "batch", "--task", "H3K4me1", "--config-list", "x.txt", "--parallel", "8"]
        )
        ok(
            captured["kw"]["parallel"] == 8,
            "(4) CLI batch --parallel 8 plumbed through (H100 config-parallel)",
        )
    finally:
        dr.run_distillation = orig[0]
        dt.load_config_list = orig_load
        carbon.experiment_configs = orig_exp


def test_unified_cli_slurm_routes_to_main():
    """(4) --mode slurm -- <spec> resolves a config and routes to run_distillation(mode='slurm')."""
    import src.train.distill_run as dr

    captured = {}
    orig_run = dr.run_distillation
    # Patch the tyro resolver so we don't depend on the real config registry / CLI.
    import tyro

    orig_tyro = tyro.extras.overridable_config_cli
    try:
        sentinel = _FakeExperimentCfg()

        def fake_resolve(configs, **kw):
            # tyro reads sys.argv; assert the post-`--` spec is what it sees.
            captured["argv"] = list(sys.argv[1:])
            return sentinel

        tyro.extras.overridable_config_cli = fake_resolve

        def fake_run(mode, **kw):
            captured["mode"] = mode
            captured["config"] = kw.get("config")

        dr.run_distillation = fake_run

        dr.main(
            [
                "--mode",
                "slurm",
                "--",
                "carbon-raw",
                "--task-names",
                "H3K27me3",
                "--slurm-config.mode",
                "run",
            ]
        )
        ok(captured["mode"] == "slurm", "(4) CLI --mode slurm -> run_distillation(mode='slurm')")
        ok(captured["config"] is sentinel, "(4) resolved config forwarded to main path")
        ok(
            captured["argv"]
            == ["carbon-raw", "--task-names", "H3K27me3", "--slurm-config.mode", "run"],
            "(4) only the post-`--` spec is handed to tyro (unified flags hidden)",
        )
    finally:
        dr.run_distillation = orig_run
        tyro.extras.overridable_config_cli = orig_tyro


def test_unified_cli_batch_requires_task_and_list():
    """--mode batch without --task/--config-list errors clearly (argparse SystemExit)."""
    import src.train.distill_run as dr

    try:
        dr.main(["--mode", "batch"])
        ok(False, "batch without --task/--config-list should error")
    except SystemExit:
        ok(True, "(4) --mode batch without --task/--config-list -> parser error (SystemExit)")


# ---------------------------------------------------------------------------
# (5) BACK-COMPAT: legacy distill_task.main() still calls distill_task_batch
#     with default + explicit parallel -- the H100 invocation is untouched.
# ---------------------------------------------------------------------------
def test_legacy_distill_task_cli_unchanged():
    import src.train.distill_task as dt

    captured = {}
    orig = (dt.distill_task_batch, dt.load_config_list, dt.experiment_configs)
    try:
        dt.experiment_configs = {"carbon-raw-original": (None, _FakeExperimentCfg())}
        dt.load_config_list = lambda path, task: [{"distillation_config": {"temperature": 1.0}}]
        dt.distill_task_batch = (
            lambda base_config, task, overrides, parallel=1, parallel_mode="fork": captured.update(
                parallel=parallel, task=task
            )
        )

        sys.argv = ["prog", "--task", "H3K4me1", "--config-list", "x.txt"]
        dt.main()
        ok(
            captured["parallel"] == 1,
            "(5) legacy distill_task.main default parallel=1 (H100 serial)",
        )

        sys.argv = ["prog", "--task", "H3K4me1", "--config-list", "x.txt", "--parallel", "8"]
        dt.main()
        ok(
            captured["parallel"] == 8,
            "(5) legacy distill_task.main --parallel 8 plumbed through (H100 config-parallel)",
        )
    finally:
        dt.distill_task_batch, dt.load_config_list, dt.experiment_configs = orig


# ---------------------------------------------------------------------------
# (6) BACK-COMPAT: legacy SLURM distill.main() still dispatches via
#     distill[slurm_config] once per task.
# ---------------------------------------------------------------------------
def test_legacy_slurm_main_dispatch_unchanged():
    import src.train.distill as d

    # main() does: distill[distill_config.slurm_config](distill_config, task_name)
    # Build a tiny config whose distill[...] is indexable to a recorder.
    dispatched = []

    class _SlurmDispatch:
        def __getitem__(self, slurm_config):
            def run(config, task_name):
                dispatched.append((slurm_config, task_name, config.dataset_config.task_name))

            return run

    @dataclass
    class _Cfg:
        task_names: tuple = ("H3K27ac", "H3K4me1")
        slurm_config: str = "local"
        dataset_config: _FakeDatasetCfg = field(default_factory=_FakeDatasetCfg)

    orig = d.distill
    try:
        d.distill = _SlurmDispatch()
        d.main(_Cfg())
        ok(len(dispatched) == 2, "(6) legacy slurm main() dispatches once per task name")
        ok(
            dispatched == [("local", "H3K27ac", "H3K27ac"), ("local", "H3K4me1", "H3K4me1")],
            "(6) each dispatch uses distill[slurm_config] with the task pinned into dataset_config",
        )
    finally:
        d.distill = orig


def main():
    for name, fn in [
        ("test_run_distillation_routes_each_mode", test_run_distillation_routes_each_mode),
        ("test_run_distillation_validation", test_run_distillation_validation),
        ("test_unified_cli_batch_routes_like_legacy", test_unified_cli_batch_routes_like_legacy),
        ("test_unified_cli_slurm_routes_to_main", test_unified_cli_slurm_routes_to_main),
        (
            "test_unified_cli_batch_requires_task_and_list",
            test_unified_cli_batch_requires_task_and_list,
        ),
        ("test_legacy_distill_task_cli_unchanged", test_legacy_distill_task_cli_unchanged),
        ("test_legacy_slurm_main_dispatch_unchanged", test_legacy_slurm_main_dispatch_unchanged),
    ]:
        print(name)
        fn()
    print(f"\nALL PASSED ({_n} assertions)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
