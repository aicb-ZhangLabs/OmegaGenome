"""Structure test for the per-task BATCH distillation refactor (CPU-only, no teacher load).

Proves the throughput fix WITHOUT any GPU / Carbon-3B load: it monkeypatches
``prepare_task`` and ``train_student`` to record call counts + the configs they receive,
then asserts ``distill_task_batch``:

  (1) calls prepare_task ONCE for a task (teacher-load amortization), and
  (2) calls train_student N times (once per HP config), and
  (3) applies each per-config HP override + random_state correctly onto base_config.

Also checks the distill_task CLI line parser turns a real hp_original_stage1.txt-style spec
line into the right override dict.

Run: python -m tests.test_distill_task_batch
"""

import sys
from dataclasses import dataclass, field

_n = 0


def ok(cond, msg):
    global _n
    assert cond, "FAIL: " + msg
    _n += 1
    print(f"  ok: {msg}")


# --- Minimal fakes mirroring the real config shape used by _apply_override / batch ---
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


def test_apply_override():
    """_apply_override merges nested distillation_config + top-level fields, no mutation."""
    from src.train.distill import _apply_override

    base = _FakeExperimentCfg()
    ov = {
        "distillation_config": {"weight_ce": 0.5, "weight_kl": 0.25, "temperature": 2.0},
        "random_state": 7,
    }
    out = _apply_override(base, ov)
    ok(out.distillation_config.weight_ce == 0.5, "weight_ce overridden")
    ok(out.distillation_config.weight_kl == 0.25, "weight_kl overridden")
    ok(out.distillation_config.temperature == 2.0, "temperature overridden")
    ok(out.distillation_config.weight_mse == 0.0, "weight_mse untouched (default)")
    ok(out.random_state == 7, "random_state overridden")
    # base unchanged (shared teacher/data context stays valid)
    ok(base.distillation_config.weight_ce == 1.0, "base config NOT mutated")
    ok(base.random_state == 42, "base random_state NOT mutated")


def test_batch_call_counts():
    """distill_task_batch -> prepare_task ONCE, train_student N times, overrides applied."""
    import src.train.distill as d

    calls = {"prepare": 0, "train": []}

    sentinel_ctx = object()

    def fake_prepare(config, task_name):
        calls["prepare"] += 1
        return sentinel_ctx

    def fake_train(config, task_name, task_ctx):
        # record the per-config HP + that the shared ctx was passed through
        calls["train"].append(
            {
                "task": task_name,
                "ctx_is_sentinel": task_ctx is sentinel_ctx,
                "weight_ce": config.distillation_config.weight_ce,
                "weight_kl": config.distillation_config.weight_kl,
                "weight_mse": config.distillation_config.weight_mse,
                "temperature": config.distillation_config.temperature,
                "random_state": config.random_state,
            }
        )

    # Patch the module-level functions distill_task_batch resolves by name.
    orig_prepare, orig_train = d.prepare_task, d.train_student
    # wandb is referenced in the batch loop's finally block; stub it out.
    orig_wandb = d.wandb

    class _FakeWandb:
        run = None

        @staticmethod
        def finish(*a, **k):
            pass

    try:
        d.prepare_task = fake_prepare
        d.train_student = fake_train
        d.wandb = _FakeWandb

        base = _FakeExperimentCfg(dataset_config=_FakeDatasetCfg(task_name="H3K27ac"))
        overrides = [
            {
                "distillation_config": {
                    "weight_ce": 0.5,
                    "weight_kl": 0.25,
                    "weight_mse": 0.0,
                    "temperature": 1.5,
                }
            },
            {
                "distillation_config": {
                    "weight_ce": 0.5,
                    "weight_kl": 0.25,
                    "weight_mse": 1.0,
                    "temperature": 0.5,
                }
            },
            {
                "distillation_config": {
                    "weight_ce": 1.0,
                    "weight_kl": 0.0,
                    "weight_mse": 0.0,
                    "temperature": 1.0,
                },
                "random_state": 99,
            },
        ]

        d.distill_task_batch(base, "H3K27ac", overrides)

        ok(
            calls["prepare"] == 1,
            f"prepare_task called ONCE (got {calls['prepare']}) -> teacher amortized",
        )
        ok(len(calls["train"]) == 3, f"train_student called N=3 times (got {len(calls['train'])})")
        ok(
            all(c["ctx_is_sentinel"] for c in calls["train"]),
            "every train_student got the SHARED ctx",
        )
        ok(all(c["task"] == "H3K27ac" for c in calls["train"]), "task pinned for all configs")

        c0, c1, c2 = calls["train"]
        ok(
            c0["weight_kl"] == 0.25 and c0["temperature"] == 1.5 and c0["weight_mse"] == 0.0,
            "config 0 HP applied",
        )
        ok(c1["weight_mse"] == 1.0 and c1["temperature"] == 0.5, "config 1 HP applied (mse on)")
        ok(c2["weight_ce"] == 1.0 and c2["weight_kl"] == 0.0, "config 2 HP applied")
        ok(c0["random_state"] == 42 and c2["random_state"] == 99, "per-config random_state applied")
    finally:
        d.prepare_task, d.train_student, d.wandb = orig_prepare, orig_train, orig_wandb


def test_batch_one_bad_config_does_not_abort():
    """A failing config is caught+logged; the rest still run."""
    import src.train.distill as d

    trained = []
    orig_prepare, orig_train, orig_wandb = d.prepare_task, d.train_student, d.wandb

    class _FakeWandb:
        run = None

        @staticmethod
        def finish(*a, **k):
            pass

    def fake_prepare(config, task_name):
        return object()

    def fake_train(config, task_name, task_ctx):
        if config.distillation_config.temperature == 0.0:
            raise RuntimeError("boom")
        trained.append(config.distillation_config.temperature)

    try:
        d.prepare_task = fake_prepare
        d.train_student = fake_train
        d.wandb = _FakeWandb
        base = _FakeExperimentCfg()
        overrides = [
            {"distillation_config": {"temperature": 1.0}},
            {"distillation_config": {"temperature": 0.0}},  # this one raises
            {"distillation_config": {"temperature": 2.0}},
        ]
        d.distill_task_batch(base, "T", overrides)
        ok(trained == [1.0, 2.0], f"good configs ran despite one failure (got {trained})")
    finally:
        d.prepare_task, d.train_student, d.wandb = orig_prepare, orig_train, orig_wandb


def test_cli_line_parser():
    """The distill_task CLI parses a real spec line into the right override dict."""
    from src.train.distill_task import parse_override_line

    line = (
        "carbon-raw-original --task-names H3K27ac "
        "--distillation-config.weight-ce 0.5 --distillation-config.weight-kl 0.25 "
        "--distillation-config.weight-mse 0.0 --distillation-config.temperature 1.5 "
        "--trainer-config.early-stop-patience 70 --slurm-config.mode run"
    )
    ov = parse_override_line(line)
    dc = ov["distillation_config"]
    ok(dc["weight_ce"] == 0.5, "parsed weight_ce")
    ok(dc["weight_kl"] == 0.25, "parsed weight_kl")
    ok(dc["weight_mse"] == 0.0, "parsed weight_mse")
    ok(dc["temperature"] == 1.5, "parsed temperature")
    ok("random_state" not in ov, "no random_state flag -> not in override")


def main():
    print("test_apply_override")
    test_apply_override()
    print("test_batch_call_counts")
    test_batch_call_counts()
    print("test_batch_one_bad_config_does_not_abort")
    test_batch_one_bad_config_does_not_abort()
    print("test_cli_line_parser")
    test_cli_line_parser()
    print(f"\nALL PASSED ({_n} assertions)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
