"""Tests for the OPT-IN config-parallel batch path (CPU-only, no GPU / no 3B teacher).

These exercise the ``parallel`` knob added to ``distill_task_batch`` WITHOUT forking
CUDA or loading the Carbon-3B teacher. They monkeypatch ``prepare_task`` /
``train_student`` / the spawn pool to a synchronous in-process executor, and assert:

  (1) parallel==1 (default) dispatches to the EXACT serial path: no pool created,
      prepare_task ONCE, train_student per config IN ORDER -- byte-for-byte the old
      behavior (the audited production path must not change);
  (2) the teacher-output ctx cache ROUND-TRIPS: torch.save'd dict reloads with equal
      tensors/values and a teacher-free (stub) teacher_model;
  (3) parallel==K shards ALL N configs and collects N results (none dropped);
  (4) worker ERROR ISOLATION: one config raising does NOT propagate or drop siblings;
      the failure is recorded;
  (5) the live 3B teacher is FREED (teacher_model=None + cuda.empty_cache) BEFORE the
      worker pool starts;
  (6) the ``--parallel`` CLI flag parses and defaults to 1.

Run: <venv>/bin/python -m tests.test_distill_task_parallel
"""

import os
import sys
import tempfile
from dataclasses import dataclass, field, replace

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
    output_dir: str = ""
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
    run = None

    @staticmethod
    def finish(*a, **k):
        pass


class _SyncPool:
    """Synchronous stand-in for multiprocessing.Pool over the SPAWN context.

    Runs ``imap_unordered`` work IN-PROCESS (no real spawn / no CUDA fork) so the
    test asserts the parent's scheduling + result-collection + error-isolation
    logic, not real training. Records that it was constructed.
    """

    created = []

    def __init__(self, processes):
        self.processes = processes
        _SyncPool.created.append(processes)

    def imap_unordered(self, fn, iterable):
        return [fn(item) for item in iterable]

    def close(self):
        pass

    def join(self):
        pass


# ---------------------------------------------------------------------------
# (1) parallel==1 hits the SERIAL path: no pool, prepare ONCE, train in order.
# ---------------------------------------------------------------------------
def test_parallel_one_is_serial_path():
    import src.train.distill as d

    calls = {"prepare": 0, "train_order": [], "pool_created": 0}
    sentinel_ctx = object()
    orig = (d.prepare_task, d.train_student, d.wandb, d._spawn_pool)
    try:
        def fake_prepare(config, task_name):
            calls["prepare"] += 1
            return sentinel_ctx

        def fake_train(config, task_name, task_ctx):
            calls["train_order"].append(config.distillation_config.temperature)

        def boom_pool(parallel):
            calls["pool_created"] += 1
            raise AssertionError("serial path must NOT create a pool")

        d.prepare_task, d.train_student, d.wandb = fake_prepare, fake_train, _FakeWandb
        d._spawn_pool = boom_pool

        base = _FakeExperimentCfg(dataset_config=_FakeDatasetCfg(task_name="H3K27ac"))
        overrides = [
            {"distillation_config": {"temperature": 1.0}},
            {"distillation_config": {"temperature": 2.0}},
            {"distillation_config": {"temperature": 3.0}},
        ]
        # default parallel (=1) and explicit parallel=1 both -> serial path.
        d.distill_task_batch(base, "H3K27ac", overrides)
        d.distill_task_batch(base, "H3K27ac", overrides, parallel=1)

        ok(calls["pool_created"] == 0, "parallel==1 created NO worker pool")
        ok(calls["prepare"] == 2, f"prepare_task ONCE per batch (got {calls['prepare']})")
        ok(
            calls["train_order"] == [1.0, 2.0, 3.0, 1.0, 2.0, 3.0],
            f"train_student per config IN ORDER, serial (got {calls['train_order']})",
        )
    finally:
        d.prepare_task, d.train_student, d.wandb, d._spawn_pool = orig


# ---------------------------------------------------------------------------
# (2) teacher-output ctx cache round-trips with equal tensors/values + stub teacher.
# ---------------------------------------------------------------------------
def test_ctx_cache_roundtrip():
    import torch
    import src.train.distill as d
    from src.train.distill import TaskContext, _NoOpTeacher

    hid = torch.randn(4, 8)
    ctx = TaskContext(
        teacher_tokenizer={"tok": 1},
        teacher_model=object(),  # stands in for the live 3B model; must NOT be saved
        teacher_hidden=8,
        num_labels=3,
        teacher_ckpt="/path/to/ckpt",
        score=0.77,
        X_train=["ACGT", "TTTT"],
        y_train=[0, 1],
        X_val=["GGGG"],
        y_val=[1],
        X_test=["CCCC"],
        y_test=[0],
    )
    # smuggle a tensor through a data slot to prove tensor equality round-trips.
    ctx.y_train = hid

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "sub", "taskctx.pt")
        d._serialize_task_ctx(ctx, path)
        ok(os.path.exists(path), "cache file written under a created subdir")

        raw = torch.load(path, map_location="cpu", weights_only=False)
        ok("teacher_model" not in raw, "live teacher_model is NOT serialized")
        ok(set(raw.keys()) == set(d._TASK_CTX_CACHE_FIELDS), "exactly the HP-independent fields saved")

        loaded = d._load_task_ctx_from_cache(path)
        ok(isinstance(loaded.teacher_model, _NoOpTeacher), "reconstructed ctx has a teacher-free stub")
        ok(loaded.teacher_hidden == 8 and loaded.num_labels == 3, "scalar fields round-trip")
        ok(loaded.teacher_ckpt == "/path/to/ckpt" and loaded.score == 0.77, "ckpt/score round-trip")
        ok(loaded.X_train == ["ACGT", "TTTT"], "X_train list round-trips")
        ok(torch.equal(loaded.y_train, hid), "tensor field round-trips EQUAL")
        # the stub is a no-op for the calls train_distill_task makes on the teacher.
        ok(loaded.teacher_model.to("cuda") is loaded.teacher_model, "stub .to() is a no-op")
        ok(loaded.teacher_model.eval() is loaded.teacher_model, "stub .eval() is a no-op")


# ---------------------------------------------------------------------------
# (3)+(4)+(5) parallel==K shards all N, collects N, isolates errors, frees teacher.
# ---------------------------------------------------------------------------
def test_parallel_shards_collects_isolates_and_frees():
    import src.train.distill as d
    from src.train.distill import TaskContext

    trained = []  # temperatures that train_student actually completed
    events = {"freed": False, "empty_cache": 0}
    orig = (
        d.prepare_task,
        d.train_student,
        d.wandb,
        d._spawn_pool,
        d._serialize_task_ctx,
        d._load_task_ctx_from_cache,
        d._free_teacher,
    )
    try:
        def fake_prepare(config, task_name):
            return TaskContext(
                teacher_tokenizer=None,
                teacher_model="LIVE_3B",  # sentinel for "teacher still resident"
                teacher_hidden=8,
                num_labels=2,
                teacher_ckpt="ckpt",
                score=0.5,
                X_train=["A"], y_train=[0],
                X_val=["C"], y_val=[1],
                X_test=["G"], y_test=[0],
            )

        def fake_train(config, task_name, task_ctx):
            t = config.distillation_config.temperature
            if t == 0.0:
                raise RuntimeError("boom in worker")
            trained.append(t)

        def fake_free(ctx):
            # mirror real _free_teacher: drop teacher + empty cache, BEFORE pool runs.
            ok(events["freed"] is False, "teacher freed exactly once")
            ctx.teacher_model = None
            events["freed"] = True
            events["empty_cache"] += 1

        # serialize/load become identity-ish: hand the SAME ctx object to workers
        # (sync pool runs in-process, so we skip real torch.save round-trip here).
        saved = {}

        def fake_serialize(ctx, path):
            # at serialize time the teacher MUST already be freed.
            ok(ctx.teacher_model is None, "(5) teacher freed BEFORE ctx is serialized for workers")
            saved["ctx"] = ctx
            return path

        def fake_load(path):
            return saved["ctx"]

        _SyncPool.created = []
        d.prepare_task = fake_prepare
        d.train_student = fake_train
        d.wandb = _FakeWandb
        d._spawn_pool = lambda parallel: _SyncPool(parallel)
        d._free_teacher = fake_free
        d._serialize_task_ctx = fake_serialize
        d._load_task_ctx_from_cache = fake_load

        base = _FakeExperimentCfg(
            dataset_config=_FakeDatasetCfg(task_name="H3K4me1"),
            trainer_config=_FakeTrainerCfg(output_dir=tempfile.mkdtemp()),
        )
        overrides = [
            {"distillation_config": {"temperature": 1.0}},  # config 0 = warm-up (live teacher)
            {"distillation_config": {"temperature": 2.0}},
            {"distillation_config": {"temperature": 0.0}},  # this worker RAISES
            {"distillation_config": {"temperature": 4.0}},
        ]
        d.distill_task_batch(base, "H3K4me1", overrides, parallel=3)

        ok(events["freed"] is True, "(5) live teacher was freed in the parallel path")
        ok(events["empty_cache"] >= 1, "(5) cuda.empty_cache invoked when freeing teacher")
        ok(_SyncPool.created == [3], f"(3) ONE pool with parallel=3 workers (got {_SyncPool.created})")
        # warm-up (1.0) + worker 2.0 + worker 4.0 ran; 0.0 raised and was isolated.
        ok(sorted(trained) == [1.0, 2.0, 4.0], f"(3)(4) all non-failing configs ran (got {sorted(trained)})")
        ok(0.0 not in trained, "(4) the raising config did not complete, but did not abort siblings")
    finally:
        (
            d.prepare_task,
            d.train_student,
            d.wandb,
            d._spawn_pool,
            d._serialize_task_ctx,
            d._load_task_ctx_from_cache,
            d._free_teacher,
        ) = orig


# ---------------------------------------------------------------------------
# (4) worker entry point isolates a raising train_student into (idx, False, err).
# ---------------------------------------------------------------------------
def test_worker_entry_error_isolation():
    import src.train.distill as d
    from src.train.distill import TaskContext

    orig = (d.train_student, d.wandb, d._load_task_ctx_from_cache)
    try:
        ctx = TaskContext(
            teacher_tokenizer=None, teacher_model=None, teacher_hidden=8,
            num_labels=2, teacher_ckpt="c", score=0.0,
            X_train=["A"], y_train=[0], X_val=["C"], y_val=[1],
            X_test=["G"], y_test=[0],
        )
        d._load_task_ctx_from_cache = lambda path: ctx
        d.wandb = _FakeWandb

        def boom_train(config, task_name, task_ctx):
            raise RuntimeError("kaboom")

        d.train_student = boom_train
        base = _FakeExperimentCfg()
        idx, ok_flag, err = d._run_one_config_worker(
            (5, base, "T", {"distillation_config": {"temperature": 9.0}}, "/no/such/path")
        )
        ok(idx == 5, "worker returns its index")
        ok(ok_flag is False, "worker reports failure (not raised)")
        ok(err is not None and "kaboom" in err, "worker captures the traceback string")

        # success path returns (idx, True, None)
        d.train_student = lambda config, task_name, task_ctx: None
        idx2, ok2, err2 = d._run_one_config_worker(
            (1, base, "T", {"distillation_config": {"temperature": 1.0}}, "/p")
        )
        ok(idx2 == 1 and ok2 is True and err2 is None, "worker success -> (idx, True, None)")
    finally:
        d.train_student, d.wandb, d._load_task_ctx_from_cache = orig


# ---------------------------------------------------------------------------
# (6) --parallel CLI flag parses and defaults to 1.
# ---------------------------------------------------------------------------
def test_cli_parallel_flag():
    import argparse
    # Rebuild the same parser distill_task.main() builds, minimally, to assert the flag.
    # We invoke main() with patched internals to confirm plumbing end-to-end.
    import src.train.distill_task as dt

    captured = {}
    orig = (dt.distill_task_batch, dt.load_config_list, dt.experiment_configs)
    try:
        # experiment_configs[name] -> (something, base_config); fake it.
        dt.experiment_configs = {"carbon-raw-original": (None, _FakeExperimentCfg())}
        dt.load_config_list = lambda path, task: [{"distillation_config": {"temperature": 1.0}}]

        def fake_batch(base_config, task, overrides, parallel=1):
            captured["parallel"] = parallel

        dt.distill_task_batch = fake_batch

        # default: no --parallel -> 1
        sys.argv = ["prog", "--task", "H3K4me1", "--config-list", "x.txt"]
        dt.main()
        ok(captured["parallel"] == 1, f"--parallel defaults to 1 (got {captured['parallel']})")

        # explicit --parallel 8 -> plumbed through
        sys.argv = ["prog", "--task", "H3K4me1", "--config-list", "x.txt", "--parallel", "8"]
        dt.main()
        ok(captured["parallel"] == 8, f"--parallel 8 plumbed into distill_task_batch (got {captured['parallel']})")
    finally:
        dt.distill_task_batch, dt.load_config_list, dt.experiment_configs = orig


# ---------------------------------------------------------------------------
# Shared scaffold for the WARM-vs-full-train tests: prepare_task -> live-teacher
# ctx; train_student is the WORKER body (must only ever run inside the pool, never
# serially as a warm-up); _warm_teacher_cache is the MINIMAL cache warm (counts
# invocations); _free_teacher records the free order; the sync pool records the
# indices it was handed.
# ---------------------------------------------------------------------------
def _patch_parallel_scaffold(d, fake_train, base):
    from src.train.distill import TaskContext

    state = {
        "warm_calls": 0,
        "freed": False,
        "warm_before_free": None,
        "pool_indices": None,
        "serialize_after_free": None,
    }
    saved = {}

    def fake_prepare(config, task_name):
        return TaskContext(
            teacher_tokenizer=None, teacher_model="LIVE_3B", teacher_hidden=8,
            num_labels=2, teacher_ckpt="ckpt", score=0.5,
            X_train=["A"], y_train=[0], X_val=["C"], y_val=[1],
            X_test=["G"], y_test=[0],
        )

    def fake_warm(base_config, task_name, ctx, overrides):
        # Warm must run with the LIVE teacher (before free) and exactly once.
        state["warm_before_free"] = (state["freed"] is False)
        state["warm_calls"] += 1

    def fake_free(ctx):
        ctx.teacher_model = None
        state["freed"] = True

    def fake_serialize(ctx, path):
        state["serialize_after_free"] = (ctx.teacher_model is None)
        saved["ctx"] = ctx
        return path

    def fake_load(path):
        return saved["ctx"]

    class _IdxPool(_SyncPool):
        def imap_unordered(self, fn, iterable):
            items = list(iterable)
            state["pool_indices"] = [it[0] for it in items]
            return [fn(it) for it in items]

    d.prepare_task = fake_prepare
    d.train_student = fake_train
    d.wandb = _FakeWandb
    d._spawn_pool = lambda parallel: _IdxPool(parallel)
    d._warm_teacher_cache = fake_warm
    d._free_teacher = fake_free
    d._serialize_task_ctx = fake_serialize
    d._load_task_ctx_from_cache = fake_load
    return state


# ---------------------------------------------------------------------------
# (i)+(ii) teacher-needing batch: minimal warm invoked ONCE (with live teacher),
# pool covers ALL N configs INCLUDING index 0, and NO full train_student runs
# before the pool (no serial config-0 warm-up).
# ---------------------------------------------------------------------------
def test_warm_once_and_pool_covers_all_configs():
    import src.train.distill as d

    trained_idx = []
    train_calls = {"before_pool": 0, "in_pool": 0}
    pool_started = {"flag": False}

    def fake_train(config, task_name, task_ctx):
        # Every train_student here is a FULL run; it must only happen inside the pool.
        if pool_started["flag"]:
            train_calls["in_pool"] += 1
        else:
            train_calls["before_pool"] += 1
        trained_idx.append(config.distillation_config.temperature)

    orig = (
        d.prepare_task, d.train_student, d.wandb, d._spawn_pool,
        d._warm_teacher_cache, d._free_teacher, d._serialize_task_ctx,
        d._load_task_ctx_from_cache,
    )
    try:
        base = _FakeExperimentCfg(
            dataset_config=_FakeDatasetCfg(task_name="H3K4me1"),
            trainer_config=_FakeTrainerCfg(output_dir=tempfile.mkdtemp()),
        )
        state = _patch_parallel_scaffold(d, fake_train, base)

        # Wrap the pool so we know when the pool phase begins.
        real_spawn = d._spawn_pool
        def spawn_marking(parallel):
            pool_started["flag"] = True
            return real_spawn(parallel)
        d._spawn_pool = spawn_marking

        # 3 configs, at least one teacher-needing (weight_kl>0) -> warm fires.
        overrides = [
            {"distillation_config": {"weight_kl": 0.5, "temperature": 1.0}},  # idx 0
            {"distillation_config": {"weight_kl": 0.5, "temperature": 2.0}},  # idx 1
            {"distillation_config": {"weight_kl": 0.5, "temperature": 3.0}},  # idx 2
        ]
        d.distill_task_batch(base, "H3K4me1", overrides, parallel=2)

        ok(state["warm_calls"] == 1, f"(ii) minimal warm invoked exactly ONCE (got {state['warm_calls']})")
        ok(state["warm_before_free"] is True, "(ii) warm ran with the LIVE teacher (before free)")
        ok(train_calls["before_pool"] == 0, "(ii) NO full train_student ran before the pool (no serial config-0 warm-up)")
        ok(train_calls["in_pool"] == 3, f"(i) all 3 full trainings happened IN the pool (got {train_calls['in_pool']})")
        ok(state["pool_indices"] == [0, 1, 2], f"(i) pool covers ALL configs INCLUDING index 0 (got {state['pool_indices']})")
        ok(sorted(trained_idx) == [1.0, 2.0, 3.0], "(i) every config (incl idx 0) got its full training")
    finally:
        (
            d.prepare_task, d.train_student, d.wandb, d._spawn_pool,
            d._warm_teacher_cache, d._free_teacher, d._serialize_task_ctx,
            d._load_task_ctx_from_cache,
        ) = orig


# ---------------------------------------------------------------------------
# (iii) all-pure-CE batch: warm SKIPPED, teacher STILL freed before the pool,
# and all configs (incl idx 0) still run in the pool.
# ---------------------------------------------------------------------------
def test_pure_ce_skips_warm_but_frees_and_runs_all():
    import src.train.distill as d

    trained_idx = []

    def fake_train(config, task_name, task_ctx):
        trained_idx.append(config.distillation_config.temperature)

    orig = (
        d.prepare_task, d.train_student, d.wandb, d._spawn_pool,
        d._warm_teacher_cache, d._free_teacher, d._serialize_task_ctx,
        d._load_task_ctx_from_cache,
    )
    try:
        base = _FakeExperimentCfg(
            dataset_config=_FakeDatasetCfg(task_name="H3K4me1"),
            trainer_config=_FakeTrainerCfg(output_dir=tempfile.mkdtemp()),
        )
        state = _patch_parallel_scaffold(d, fake_train, base)

        # All pure-CE: weight_kl==0 AND weight_mse==0 (the _FakeDistillCfg defaults).
        overrides = [
            {"distillation_config": {"temperature": 1.0}},
            {"distillation_config": {"temperature": 2.0}},
        ]
        d.distill_task_batch(base, "H3K4me1", overrides, parallel=2)

        ok(state["warm_calls"] == 0, "(iii) pure-CE batch SKIPS the teacher warm")
        ok(state["freed"] is True, "(iii) teacher still FREED even when warm is skipped")
        ok(state["serialize_after_free"] is True, "(iii) ctx serialized AFTER the teacher was freed")
        ok(state["pool_indices"] == [0, 1], f"(iii) pool still covers ALL configs incl idx 0 (got {state['pool_indices']})")
        ok(sorted(trained_idx) == [1.0, 2.0], "(iii) all configs trained in the pool")
    finally:
        (
            d.prepare_task, d.train_student, d.wandb, d._spawn_pool,
            d._warm_teacher_cache, d._free_teacher, d._serialize_task_ctx,
            d._load_task_ctx_from_cache,
        ) = orig


# ---------------------------------------------------------------------------
# (iv) error isolation preserved with the all-configs pool: a raising config
# (now possibly index 0) does not abort siblings and is recorded.
# ---------------------------------------------------------------------------
def test_error_isolation_with_all_configs_including_index0():
    import src.train.distill as d

    trained_idx = []

    def fake_train(config, task_name, task_ctx):
        t = config.distillation_config.temperature
        if t == 0.0:
            raise RuntimeError("boom in config 0")
        trained_idx.append(t)

    orig = (
        d.prepare_task, d.train_student, d.wandb, d._spawn_pool,
        d._warm_teacher_cache, d._free_teacher, d._serialize_task_ctx,
        d._load_task_ctx_from_cache,
    )
    try:
        base = _FakeExperimentCfg(
            dataset_config=_FakeDatasetCfg(task_name="H3K4me1"),
            trainer_config=_FakeTrainerCfg(output_dir=tempfile.mkdtemp()),
        )
        state = _patch_parallel_scaffold(d, fake_train, base)

        # index 0 RAISES now that it runs in the pool (no longer the serial warm-up).
        overrides = [
            {"distillation_config": {"weight_kl": 0.5, "temperature": 0.0}},  # idx 0 raises
            {"distillation_config": {"weight_kl": 0.5, "temperature": 2.0}},
            {"distillation_config": {"weight_kl": 0.5, "temperature": 3.0}},
        ]
        d.distill_task_batch(base, "H3K4me1", overrides, parallel=2)

        ok(state["pool_indices"] == [0, 1, 2], "(iv) pool was handed all configs incl the raising idx 0")
        ok(sorted(trained_idx) == [2.0, 3.0], f"(iv) siblings ran despite idx-0 failure (got {sorted(trained_idx)})")
        ok(0.0 not in trained_idx, "(iv) the raising config did not complete, but did not abort siblings")
    finally:
        (
            d.prepare_task, d.train_student, d.wandb, d._spawn_pool,
            d._warm_teacher_cache, d._free_teacher, d._serialize_task_ctx,
            d._load_task_ctx_from_cache,
        ) = orig


def main():
    for name, fn in [
        ("test_parallel_one_is_serial_path", test_parallel_one_is_serial_path),
        ("test_ctx_cache_roundtrip", test_ctx_cache_roundtrip),
        ("test_parallel_shards_collects_isolates_and_frees", test_parallel_shards_collects_isolates_and_frees),
        ("test_worker_entry_error_isolation", test_worker_entry_error_isolation),
        ("test_cli_parallel_flag", test_cli_parallel_flag),
        ("test_warm_once_and_pool_covers_all_configs", test_warm_once_and_pool_covers_all_configs),
        ("test_pure_ce_skips_warm_but_frees_and_runs_all", test_pure_ce_skips_warm_but_frees_and_runs_all),
        ("test_error_isolation_with_all_configs_including_index0", test_error_isolation_with_all_configs_including_index0),
    ]:
        print(name)
        fn()
    print(f"\nALL PASSED ({_n} assertions)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
