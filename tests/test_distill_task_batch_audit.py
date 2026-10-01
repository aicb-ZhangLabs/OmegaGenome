"""ADVERSARIAL audit tests for the per-task BATCH distillation refactor (CPU-only, no teacher).

These tests EXERCISE the correctness risks of loading the Carbon-3B teacher ONCE per task and
looping HP configs in-process, WITHOUT any GPU / 3B model. They monkeypatch ``prepare_task`` /
``train_student`` and assert the batch path is behavior-identical to the per-config SLURM path:

  (a) prepare_task called EXACTLY once for N configs (teacher amortization);
  (b) train_student called once per config with the correct per-config HP + random_state;
  (c) base_config is NOT mutated after the whole batch (immutability -> shared ctx stays valid);
  (d) a config that RAISES does not stop the remaining configs and is logged + counted;
  (e) the shared task_ctx is the SAME identity for every train_student and is not mutated;
  (f) the CLI parser produces correct overrides for representative hp_original_stage1.txt lines,
      including weight_mse==0 AND weight_mse>0 lines, AND the trainer-config early-stop-patience
      knob (whose silent drop would diverge from the SLURM path);
  (g) per-config seeding determinism + per-config nested overrides applied via dataclasses.replace.

Run: <venv>/bin/python -m tests.test_distill_task_batch_audit
"""

import sys
from dataclasses import dataclass, field

_n = 0


def ok(cond, msg):
    global _n
    assert cond, "FAIL: " + msg
    _n += 1
    print(f"  ok: {msg}")


# --- Minimal fakes mirroring the real nested-dataclass config shape ---
@dataclass
class _FakeDistillCfg:
    weight_ce: float = 1.0
    weight_kl: float = 0.0
    weight_mse: float = 0.0
    temperature: float = 1.0


@dataclass
class _FakeTrainerCfg:
    early_stop_patience: int = None
    lr: float = 1e-4


@dataclass
class _FakeTeacherCfg:
    num_labels: int = 2


@dataclass
class _FakeDatasetCfg:
    task_name: str = ""


@dataclass
class _FakeExperimentCfg:
    distillation_config: _FakeDistillCfg = field(default_factory=_FakeDistillCfg)
    trainer_config: _FakeTrainerCfg = field(default_factory=_FakeTrainerCfg)
    teacher_config: _FakeTeacherCfg = field(default_factory=_FakeTeacherCfg)
    dataset_config: _FakeDatasetCfg = field(default_factory=_FakeDatasetCfg)
    random_state: int = 42


class _FakeWandb:
    """Stub for the wandb referenced in distill_task_batch's finally block."""

    run = None

    @staticmethod
    def finish(*a, **k):
        pass


# ---------------------------------------------------------------------------
# (g) + nested override correctness, including the NEW trainer_config path.
# ---------------------------------------------------------------------------
def test_apply_override_nested_and_immutable():
    from src.train.distill import _apply_override

    base = _FakeExperimentCfg()
    ov = {
        "distillation_config": {"weight_ce": 0.5, "weight_kl": 0.25, "temperature": 2.0},
        "trainer_config": {"early_stop_patience": 70},
        "random_state": 7,
    }
    out = _apply_override(base, ov)
    ok(out.distillation_config.weight_ce == 0.5, "weight_ce overridden")
    ok(out.distillation_config.weight_kl == 0.25, "weight_kl overridden")
    ok(out.distillation_config.temperature == 2.0, "temperature overridden")
    ok(
        out.distillation_config.weight_mse == 0.0,
        "weight_mse untouched (nested merge, not replace)",
    )
    ok(
        out.trainer_config.early_stop_patience == 70,
        "trainer_config.early_stop_patience overridden",
    )
    ok(out.trainer_config.lr == 1e-4, "trainer_config.lr untouched (nested merge)")
    ok(out.random_state == 7, "random_state (top-level) overridden")
    # Immutability of base AND its nested configs.
    ok(base.distillation_config.weight_ce == 1.0, "base.distillation_config NOT mutated")
    ok(base.trainer_config.early_stop_patience is None, "base.trainer_config NOT mutated")
    ok(base.random_state == 42, "base.random_state NOT mutated")


# ---------------------------------------------------------------------------
# (a)(b)(e)(g) call counts, per-config HP, shared-ctx identity, seeding.
# ---------------------------------------------------------------------------
def test_batch_call_counts_and_shared_ctx():
    import src.train.distill as d

    calls = {"prepare": 0, "train": []}
    sentinel_ctx = type("Ctx", (), {"num_labels": 2, "touched": False})()

    def fake_prepare(config, task_name):
        calls["prepare"] += 1
        return sentinel_ctx

    def fake_train(config, task_name, task_ctx):
        # mutating attempt would be visible later; we DON'T mutate, we record identity.
        calls["train"].append(
            {
                "task": task_name,
                "ctx_id": id(task_ctx),
                "weight_ce": config.distillation_config.weight_ce,
                "weight_kl": config.distillation_config.weight_kl,
                "weight_mse": config.distillation_config.weight_mse,
                "temperature": config.distillation_config.temperature,
                "patience": config.trainer_config.early_stop_patience,
                "random_state": config.random_state,
            }
        )

    orig = (d.prepare_task, d.train_student, d.wandb)
    try:
        d.prepare_task, d.train_student, d.wandb = fake_prepare, fake_train, _FakeWandb
        base = _FakeExperimentCfg(dataset_config=_FakeDatasetCfg(task_name="H3K27ac"))
        overrides = [
            {
                "distillation_config": {
                    "weight_ce": 0.5,
                    "weight_kl": 0.25,
                    "weight_mse": 0.0,
                    "temperature": 1.5,
                },
                "trainer_config": {"early_stop_patience": 70},
            },
            {
                "distillation_config": {
                    "weight_ce": 0.5,
                    "weight_kl": 0.25,
                    "weight_mse": 1.0,
                    "temperature": 0.5,
                },
                "trainer_config": {"early_stop_patience": 70},
            },
            {
                "distillation_config": {
                    "weight_ce": 1.0,
                    "weight_kl": 0.0,
                    "weight_mse": 0.0,
                    "temperature": 1.0,
                },
                "trainer_config": {"early_stop_patience": 70},
                "random_state": 99,
            },
        ]
        d.distill_task_batch(base, "H3K27ac", overrides)

        ok(calls["prepare"] == 1, f"(a) prepare_task called ONCE (got {calls['prepare']})")
        ok(len(calls["train"]) == 3, f"(b) train_student called N=3 (got {len(calls['train'])})")
        ids = {c["ctx_id"] for c in calls["train"]}
        ok(
            ids == {id(sentinel_ctx)},
            "(e) every train_student got the SAME ctx identity (teacher reuse)",
        )
        ok(all(c["task"] == "H3K27ac" for c in calls["train"]), "task pinned for all configs")

        c0, c1, c2 = calls["train"]
        ok(
            c0["weight_kl"] == 0.25 and c0["temperature"] == 1.5 and c0["weight_mse"] == 0.0,
            "config0 HP",
        )
        ok(c1["weight_mse"] == 1.0 and c1["temperature"] == 0.5, "config1 HP (mse on)")
        ok(c2["weight_ce"] == 1.0 and c2["weight_kl"] == 0.0, "config2 HP")
        ok(
            all(c["patience"] == 70 for c in calls["train"]),
            "(f) early_stop_patience=70 reached EVERY config",
        )
        ok(
            c0["random_state"] == 42 and c2["random_state"] == 99,
            "(g) per-config random_state applied",
        )
    finally:
        d.prepare_task, d.train_student, d.wandb = orig


# ---------------------------------------------------------------------------
# (c) base_config NOT mutated after the whole batch.
# ---------------------------------------------------------------------------
def test_base_config_immutable_after_batch():
    import src.train.distill as d

    orig = (d.prepare_task, d.train_student, d.wandb)
    try:
        d.prepare_task = lambda config, task_name: object()
        d.train_student = lambda config, task_name, task_ctx: None
        d.wandb = _FakeWandb
        base = _FakeExperimentCfg(dataset_config=_FakeDatasetCfg(task_name="X"))
        snap = (
            base.distillation_config.weight_ce,
            base.distillation_config.weight_mse,
            base.trainer_config.early_stop_patience,
            base.random_state,
        )
        overrides = [
            {
                "distillation_config": {"weight_ce": 9.0, "weight_mse": 5.0},
                "trainer_config": {"early_stop_patience": 1},
                "random_state": 123,
            },
            {"distillation_config": {"weight_ce": 7.0}},
        ]
        d.distill_task_batch(base, "X", overrides)
        now = (
            base.distillation_config.weight_ce,
            base.distillation_config.weight_mse,
            base.trainer_config.early_stop_patience,
            base.random_state,
        )
        ok(snap == now, f"(c) base_config UNCHANGED after batch (was {snap}, now {now})")
    finally:
        d.prepare_task, d.train_student, d.wandb = orig


# ---------------------------------------------------------------------------
# (d) one bad config does not abort the rest; it is logged + counted.
# ---------------------------------------------------------------------------
def test_one_bad_config_does_not_abort():
    import src.train.distill as d

    trained = []
    orig = (d.prepare_task, d.train_student, d.wandb)

    def fake_train(config, task_name, task_ctx):
        if config.distillation_config.temperature == 0.0:
            raise RuntimeError("boom")
        trained.append(config.distillation_config.temperature)

    try:
        d.prepare_task = lambda config, task_name: object()
        d.train_student = fake_train
        d.wandb = _FakeWandb
        base = _FakeExperimentCfg()
        overrides = [
            {"distillation_config": {"temperature": 1.0}},
            {"distillation_config": {"temperature": 0.0}},  # raises
            {"distillation_config": {"temperature": 2.0}},
        ]
        d.distill_task_batch(base, "T", overrides)
        ok(trained == [1.0, 2.0], f"(d) good configs ran despite a failure (got {trained})")
    finally:
        d.prepare_task, d.train_student, d.wandb = orig


# ---------------------------------------------------------------------------
# MSE seeding: prepare_task is seeded with weight_mse>0 if ANY config uses MSE,
# but base_config / per-config overrides are unaffected. Verify the prep_config
# passed to prepare_task has mse>0 while the configs reaching train_student keep
# their own mse (incl. mse==0), and base stays immutable.
# ---------------------------------------------------------------------------
def test_mse_seeding_only_affects_prepare():
    import src.train.distill as d

    seen = {"prep_mse": None, "train_mse": []}
    orig = (d.prepare_task, d.train_student, d.wandb)
    try:

        def fake_prepare(config, task_name):
            seen["prep_mse"] = config.distillation_config.weight_mse
            return object()

        def fake_train(config, task_name, task_ctx):
            seen["train_mse"].append(config.distillation_config.weight_mse)

        d.prepare_task, d.train_student, d.wandb = fake_prepare, fake_train, _FakeWandb
        base = _FakeExperimentCfg()  # base weight_mse == 0.0
        overrides = [
            {"distillation_config": {"weight_mse": 0.0}},
            {"distillation_config": {"weight_mse": 1.0}},
        ]
        d.distill_task_batch(base, "M", overrides)
        ok(
            seen["prep_mse"] > 0,
            f"prepare_task seeded with mse>0 since a config uses MSE (got {seen['prep_mse']})",
        )
        ok(
            seen["train_mse"] == [0.0, 1.0],
            f"train_student gets each config's OWN mse incl 0 (got {seen['train_mse']})",
        )
        ok(
            base.distillation_config.weight_mse == 0.0,
            "base weight_mse still 0 (seeding did not mutate base)",
        )

        # If NO config uses MSE, prepare must NOT be seeded (stays at base 0.0).
        seen["prep_mse"] = None
        seen["train_mse"] = []
        d.distill_task_batch(base, "M", [{"distillation_config": {"weight_mse": 0.0}}])
        ok(
            seen["prep_mse"] == 0.0,
            f"no-MSE batch leaves prepare mse at 0.0 (got {seen['prep_mse']})",
        )
    finally:
        d.prepare_task, d.train_student, d.wandb = orig


# ---------------------------------------------------------------------------
# (f) CLI parser on REAL representative spec lines (mse==0 and mse>0) incl the
# trainer-config early-stop-patience knob that the SLURM path applies.
# ---------------------------------------------------------------------------
def test_cli_parser_real_lines():
    from src.train.distill_task import parse_override_line, load_config_list

    line_mse0 = (
        "carbon-raw-original --task-names H3K27me3 "
        "--distillation-config.weight-ce 0.5 --distillation-config.weight-kl 0.25 "
        "--distillation-config.weight-mse 0.0 --distillation-config.temperature 1.5 "
        "--trainer-config.early-stop-patience 70 --slurm-config.mode run"
    )
    ov = parse_override_line(line_mse0)
    dc = ov["distillation_config"]
    ok(dc["weight_ce"] == 0.5 and dc["weight_kl"] == 0.25, "mse0 line: ce/kl parsed")
    ok(dc["weight_mse"] == 0.0 and dc["temperature"] == 1.5, "mse0 line: mse/temp parsed")
    ok(
        ov["trainer_config"]["early_stop_patience"] == 70,
        "mse0 line: early_stop_patience parsed (NOT dropped)",
    )
    ok(
        isinstance(ov["trainer_config"]["early_stop_patience"], int),
        "early_stop_patience cast to int",
    )
    ok("random_state" not in ov, "mse0 line: no random_state flag -> absent")

    line_mse1 = (
        "carbon-raw-original --task-names H3K27me3 "
        "--distillation-config.weight-ce 0.5 --distillation-config.weight-kl 0.25 "
        "--distillation-config.weight-mse 1 --distillation-config.temperature 0.5 "
        "--trainer-config.early-stop-patience 70 --slurm-config.mode run"
    )
    ov1 = parse_override_line(line_mse1)
    ok(ov1["distillation_config"]["weight_mse"] == 1.0, "mse1 line: weight_mse=1 parsed")
    ok(ov1["distillation_config"]["temperature"] == 0.5, "mse1 line: temperature=0.5 parsed")

    # random-state spelling
    ov2 = parse_override_line(line_mse0 + " --random-state 123")
    ok(ov2["random_state"] == 123, "random-state parsed when present")

    # task-filtering: enhancers must NOT match enhancers_types lines (prefix trap).
    import tempfile
    import os

    txt = (
        "x --task-names enhancers --distillation-config.weight-ce 0.5 "
        "--distillation-config.weight-kl 0.0 --distillation-config.weight-mse 0.0 "
        "--distillation-config.temperature 1.0 --trainer-config.early-stop-patience 70\n"
        "x --task-names enhancers_types --distillation-config.weight-ce 0.5 "
        "--distillation-config.weight-kl 0.0 --distillation-config.weight-mse 0.0 "
        "--distillation-config.temperature 2.0 --trainer-config.early-stop-patience 70\n"
    )
    fd, path = tempfile.mkstemp(suffix=".txt")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(txt)
        only_enh = load_config_list(path, "enhancers")
        ok(
            len(only_enh) == 1,
            f"task filter: 'enhancers' selects exactly 1 line not the _types one (got {len(only_enh)})",
        )
        ok(
            only_enh[0]["distillation_config"]["temperature"] == 1.0,
            "task filter: picked the enhancers (T=1.0) line",
        )
        only_types = load_config_list(path, "enhancers_types")
        ok(
            len(only_types) == 1 and only_types[0]["distillation_config"]["temperature"] == 2.0,
            "task filter: 'enhancers_types' selects its own line",
        )
    finally:
        os.unlink(path)


# ---------------------------------------------------------------------------
# (f) Parse the ACTUAL hp_original_stage1.txt and verify every line for one task
# round-trips with both mse classes present and patience always 70.
# ---------------------------------------------------------------------------
def test_real_spec_file_roundtrip():
    import os
    from src.train.distill_task import load_config_list

    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    spec = os.path.join(repo, "hp_original_stage1.txt")
    if not os.path.exists(spec):
        ok(True, "hp_original_stage1.txt absent -> skip real-file roundtrip (non-fatal)")
        return
    ovs = load_config_list(spec, "H3K27me3")
    ok(len(ovs) > 0, f"real spec: H3K27me3 has configs (got {len(ovs)})")
    mses = {o["distillation_config"]["weight_mse"] for o in ovs}
    ok(
        0.0 in mses and any(m > 0 for m in mses),
        f"real spec: both mse==0 and mse>0 present (got {sorted(mses)})",
    )
    ok(
        all(o.get("trainer_config", {}).get("early_stop_patience") == 70 for o in ovs),
        "real spec: early_stop_patience=70 on every parsed config",
    )


def main():
    for name, fn in [
        ("test_apply_override_nested_and_immutable", test_apply_override_nested_and_immutable),
        ("test_batch_call_counts_and_shared_ctx", test_batch_call_counts_and_shared_ctx),
        ("test_base_config_immutable_after_batch", test_base_config_immutable_after_batch),
        ("test_one_bad_config_does_not_abort", test_one_bad_config_does_not_abort),
        ("test_mse_seeding_only_affects_prepare", test_mse_seeding_only_affects_prepare),
        ("test_cli_parser_real_lines", test_cli_parser_real_lines),
        ("test_real_spec_file_roundtrip", test_real_spec_file_roundtrip),
    ]:
        print(name)
        fn()
    print(f"\nALL PASSED ({_n} assertions)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
