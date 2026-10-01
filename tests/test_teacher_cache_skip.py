"""Tests for the skip-the-3B-teacher-load fast path in ``prepare_task`` (CPU-only, ms).

The optimization: when the HP-independent teacher caches (logits/features for whatever
the run needs + teacher_evaluation.json) already exist and validate against
``(X_train, teacher_ckpt, max_len)``, ``prepare_task`` must NOT call ``get_teacher_model``
(the ~7GB Carbon-3B load + LoRA merge). When any required cache is missing/stale it MUST
load as before. These tests mock the teacher load to BLOW UP if invoked, and mock the
cache-validity checks + filesystem so they run on CPU in milliseconds.

Asserts:
  (a) caches present+valid -> NO get_teacher_model; returns a usable teacher-free ctx
      (stub teacher) with the right data + ckpt.
  (b) a required cache missing -> DOES load (mocked) + proceeds.
  (c) cache key MISMATCH (stale) -> does NOT skip; loads + rebuilds.
  (d) the needed-outputs gate is CONSERVATIVE: a features-needing run won't skip on a
      logits-only cache; and the single-config (SLURM) path treats BOTH as needed.

Run: <venv>/bin/python -m tests.test_teacher_cache_skip
"""

import os
import json
import tempfile
from dataclasses import dataclass, field

_n = 0


def ok(cond, msg):
    global _n
    assert cond, "FAIL: " + msg
    _n += 1
    print(f"  ok: {msg}")


# --- Minimal fakes mirroring the real nested-dataclass config shape -------------------
@dataclass
class _FakeDistillCfg:
    weight_ce: float = 1.0
    weight_kl: float = 0.0
    weight_mse: float = 0.0
    temperature: float = 1.0
    distill_method: str = "vanilla_kl"


@dataclass
class _FakeTrainerCfg:
    output_dir: str = ""
    max_len: int = 1024
    cache_base_dir: str = None
    device: str = "cpu"


@dataclass
class _FakeTeacherCfg:
    num_labels: int = 2
    model_name_or_path: str = "carbon-3b"


@dataclass
class _FakeStudentCfg:
    model_type: str = "bpnet"
    model_size: str = "small"
    teacher_hidden_size: int = 999


@dataclass
class _FakeDatasetCfg:
    task_name: str = ""


@dataclass
class _FakeExperimentCfg:
    distillation_config: _FakeDistillCfg = field(default_factory=_FakeDistillCfg)
    trainer_config: _FakeTrainerCfg = field(default_factory=_FakeTrainerCfg)
    teacher_config: _FakeTeacherCfg = field(default_factory=_FakeTeacherCfg)
    student_config: _FakeStudentCfg = field(default_factory=_FakeStudentCfg)
    dataset_config: _FakeDatasetCfg = field(default_factory=_FakeDatasetCfg)
    teacher_parent_dir: str = "/parent"
    random_state: int = 42


def _boom_get_teacher(*a, **k):
    raise AssertionError(
        "get_teacher_model was called -- the 3B teacher load should have been SKIPPED"
    )


class _StubTeacher:
    """Stand-in for a LOADED 3B teacher: supports the .eval() prepare_task calls on it."""

    def __init__(self, tag="LIVE_3B_TEACHER"):
        self.tag = tag

    def eval(self):
        return self


def _common_patches(d, *, teacher_ckpt="/parent/H3K27ac_finetuned"):
    """Patch the HP-independent, teacher-free deps of prepare_task. Returns the originals.

    Patches: get_num_labels, find_teacher_checkpoint, build_data_splits_from_huggingface.
    Caller separately patches get_teacher_model and the cache-validity helpers.
    """
    orig = (
        d.get_num_labels,
        d.find_teacher_checkpoint,
        d.build_data_splits_from_huggingface,
    )
    d.get_num_labels = lambda task_name: 2
    d.find_teacher_checkpoint = lambda config, task_name: (teacher_ckpt, 0.83)
    d.build_data_splits_from_huggingface = lambda dataset_config: (
        ["ACGT", "TTTT"],
        [0, 1],  # X_train, y_train
        ["GGGG"],
        [1],  # X_val, y_val
        ["CCCC"],
        [0],  # X_test, y_test
    )
    return orig


# ---------------------------------------------------------------------------
# (a) caches present+valid -> NO get_teacher_model; usable teacher-free ctx.
# ---------------------------------------------------------------------------
def test_skips_load_when_caches_valid():
    import src.train.distill as d
    from src.train.distill import _NoOpTeacher

    orig_common = _common_patches(d)
    orig = (d.get_teacher_model, d._teacher_caches_valid_for_task)
    try:
        d.get_teacher_model = _boom_get_teacher
        # caches valid -> skip the load.
        d._teacher_caches_valid_for_task = lambda *a, **k: True

        base = _FakeExperimentCfg(dataset_config=_FakeDatasetCfg(task_name="H3K27ac"))
        # batch sweep: one KL config.
        overrides = [{"distillation_config": {"weight_kl": 1.0}}]
        ctx = d.prepare_task(base, "H3K27ac", config_overrides_list=overrides)

        ok(ctx is not None, "ctx returned (not skipped as missing-teacher)")
        ok(
            isinstance(ctx.teacher_model, _NoOpTeacher),
            "(a) teacher is the no-op stub (no 3B load)",
        )
        ok(ctx.teacher_tokenizer is None, "(a) no tokenizer needed in the skip path")
        ok(ctx.X_train == ["ACGT", "TTTT"], "(a) data splits built + carried")
        ok(ctx.X_test == ["CCCC"] and ctx.num_labels == 2, "(a) test split + num_labels carried")
        ok(ctx.teacher_ckpt == "/parent/H3K27ac_finetuned", "(a) teacher_ckpt resolved")
        ok(ctx.score == 0.83, "(a) score carried")
        # stub is a no-op for the calls train_distill_task makes.
        ok(ctx.teacher_model.to("cuda") is ctx.teacher_model, "(a) stub .to() no-op")
        ok(ctx.teacher_model.eval() is ctx.teacher_model, "(a) stub .eval() no-op")
    finally:
        (d.get_num_labels, d.find_teacher_checkpoint, d.build_data_splits_from_huggingface) = (
            orig_common
        )
        (d.get_teacher_model, d._teacher_caches_valid_for_task) = orig


# ---------------------------------------------------------------------------
# (b) a required cache MISSING -> DOES load (mocked) + proceeds.
# ---------------------------------------------------------------------------
def test_loads_when_cache_missing():
    import src.train.distill as d

    orig_common = _common_patches(d)
    orig = (d.get_teacher_model, d._teacher_caches_valid_for_task)
    loaded = {"flag": False}
    try:
        teacher = _StubTeacher()

        def fake_load(config, task_name, teacher_ckpt):
            loaded["flag"] = True
            return ("TOKENIZER", teacher, 256)

        d.get_teacher_model = fake_load
        # caches NOT valid -> must load.
        d._teacher_caches_valid_for_task = lambda *a, **k: False

        base = _FakeExperimentCfg(dataset_config=_FakeDatasetCfg(task_name="H3K27ac"))
        overrides = [{"distillation_config": {"weight_kl": 1.0}}]
        ctx = d.prepare_task(base, "H3K27ac", config_overrides_list=overrides)

        ok(loaded["flag"] is True, "(b) get_teacher_model WAS called when a cache is missing")
        ok(ctx is not None and ctx.teacher_model is teacher, "(b) ctx carries the loaded teacher")
        ok(ctx.teacher_tokenizer == "TOKENIZER", "(b) ctx carries the loaded tokenizer")
        ok(ctx.teacher_hidden == 256, "(b) teacher_hidden from the loaded model")
    finally:
        (d.get_num_labels, d.find_teacher_checkpoint, d.build_data_splits_from_huggingface) = (
            orig_common
        )
        (d.get_teacher_model, d._teacher_caches_valid_for_task) = orig


# ---------------------------------------------------------------------------
# (c) cache key MISMATCH (stale) -> does NOT skip; loads + rebuilds.
# Exercises the REAL teacher_cache_is_valid against an on-disk stale metadata.
# ---------------------------------------------------------------------------
def test_stale_cache_key_forces_load():
    from src.trainer.utils import (
        teacher_cache_is_valid,
        _get_cache_dir,
        _save_cache,
        _compute_cache_key,
    )
    import numpy as np

    with tempfile.TemporaryDirectory() as proj:
        teacher_parent = os.path.join(proj, "data", "finetuned_models", "carbon_3b_lora")
        os.makedirs(teacher_parent, exist_ok=True)
        task = "H3K27ac"
        teacher_ckpt = "/ckpt/H3K27ac"
        seqs = ["ACGT", "TTTT", "GGGG"]
        max_len = 1024

        cache_dir = _get_cache_dir(proj, teacher_parent, task)
        # Write a VALID logits cache for the CURRENT seqs first.
        good_key = _compute_cache_key(seqs, teacher_ckpt, max_len)
        _save_cache(
            cache_dir,
            np.zeros((3, 2), dtype=np.float32),
            None,
            {
                "num_samples": 3,
                "max_length": max_len,
                "teacher_checkpoint": teacher_ckpt,
                "cache_key": good_key,
                "logits_computed": True,
                "features_computed": False,
                "logits_shape": [3, 2],
                "features_shape": None,
            },
        )
        ok(
            teacher_cache_is_valid(
                seqs,
                teacher_ckpt,
                max_len,
                needs_logits=True,
                needs_features=False,
                project_path=proj,
                teacher_parent_dir=teacher_parent,
                task_name=task,
            )
            is True,
            "(c) fresh matching cache validates",
        )
        # Now the training set CHANGED (different seqs) -> key mismatches -> NOT valid.
        ok(
            teacher_cache_is_valid(
                ["AAAA", "CCCC", "TTTT"],
                teacher_ckpt,
                max_len,
                needs_logits=True,
                needs_features=False,
                project_path=proj,
                teacher_parent_dir=teacher_parent,
                task_name=task,
            )
            is False,
            "(c) stale cache (different X_train) is INVALID -> forces a real load",
        )
        # Different teacher_ckpt also invalidates.
        ok(
            teacher_cache_is_valid(
                seqs,
                "/ckpt/OTHER",
                max_len,
                needs_logits=True,
                needs_features=False,
                project_path=proj,
                teacher_parent_dir=teacher_parent,
                task_name=task,
            )
            is False,
            "(c) different teacher_ckpt is INVALID",
        )
        # Missing-entirely cache (different task) is invalid.
        ok(
            teacher_cache_is_valid(
                seqs,
                teacher_ckpt,
                max_len,
                needs_logits=True,
                needs_features=False,
                project_path=proj,
                teacher_parent_dir=teacher_parent,
                task_name="OTHERTASK",
            )
            is False,
            "(c) absent cache is INVALID",
        )


# ---------------------------------------------------------------------------
# (c2) END-TO-END: a stale cache in prepare_task forces get_teacher_model.
# ---------------------------------------------------------------------------
def test_prepare_task_stale_cache_loads():
    import src.train.distill as d

    orig_common = _common_patches(d)
    orig = (d.get_teacher_model, d._teacher_caches_valid_for_task)
    loaded = {"flag": False}
    try:
        teacher = _StubTeacher("LIVE")

        def fake_load(config, task_name, teacher_ckpt):
            loaded["flag"] = True
            return ("TOK", teacher, 256)

        d.get_teacher_model = fake_load
        # Simulate the real check returning False because the on-disk key is stale.
        d._teacher_caches_valid_for_task = lambda *a, **k: False

        base = _FakeExperimentCfg(dataset_config=_FakeDatasetCfg(task_name="H3K27ac"))
        ctx = d.prepare_task(
            base, "H3K27ac", config_overrides_list=[{"distillation_config": {"weight_kl": 1.0}}]
        )
        ok(
            loaded["flag"] is True,
            "(c2) stale cache -> prepare_task loaded the teacher (no wrong skip)",
        )
        ok(ctx.teacher_model is teacher, "(c2) loaded teacher carried")
    finally:
        (d.get_num_labels, d.find_teacher_checkpoint, d.build_data_splits_from_huggingface) = (
            orig_common
        )
        (d.get_teacher_model, d._teacher_caches_valid_for_task) = orig


# ---------------------------------------------------------------------------
# (d) needed-outputs gate is CONSERVATIVE.
# ---------------------------------------------------------------------------
def test_needed_outputs_gate_conservative():
    import src.train.distill as d

    base = _FakeExperimentCfg()

    # Batch sweep: KL-only -> needs logits, NOT features.
    nl, nf = d._resolve_needed_teacher_outputs(
        base, [{"distillation_config": {"weight_kl": 1.0, "weight_mse": 0.0}}]
    )
    ok(nl is True and nf is False, "(d) KL-only sweep needs logits not features")

    # Batch sweep: MSE in any config -> needs features (union across configs).
    nl, nf = d._resolve_needed_teacher_outputs(
        base,
        [
            {"distillation_config": {"weight_kl": 1.0, "weight_mse": 0.0}},
            {"distillation_config": {"weight_kl": 0.0, "weight_mse": 1.0}},
        ],
    )
    ok(nl is True and nf is True, "(d) sweep with an MSE config needs BOTH (union)")

    # Pure-CE sweep -> neither.
    nl, nf = d._resolve_needed_teacher_outputs(
        base, [{"distillation_config": {"weight_kl": 0.0, "weight_mse": 0.0}}]
    )
    ok(nl is False and nf is False, "(d) pure-CE sweep needs neither logits nor features")

    # Single-config (SLURM) path: no sweep -> CONSERVATIVE both needed.
    nl, nf = d._resolve_needed_teacher_outputs(base, None)
    ok(nl is True and nf is True, "(d) single-config path conservatively needs BOTH")


# ---------------------------------------------------------------------------
# (d2) feature-needing run won't skip on a LOGITS-ONLY cache (real helper).
# ---------------------------------------------------------------------------
def test_features_run_wont_skip_on_logits_only_cache():
    from src.trainer.utils import (
        teacher_cache_is_valid,
        _get_cache_dir,
        _save_cache,
        _compute_cache_key,
    )
    import numpy as np

    with tempfile.TemporaryDirectory() as proj:
        teacher_parent = os.path.join(proj, "data", "finetuned_models", "carbon_3b_lora")
        os.makedirs(teacher_parent, exist_ok=True)
        task = "H3K27ac"
        teacher_ckpt = "/ckpt/H3K27ac"
        seqs = ["ACGT", "TTTT", "GGGG"]
        max_len = 1024

        cache_dir = _get_cache_dir(proj, teacher_parent, task)
        # Cache has ONLY logits (no train_features.npy).
        _save_cache(
            cache_dir,
            np.zeros((3, 2), dtype=np.float32),
            None,  # no features
            {
                "num_samples": 3,
                "max_length": max_len,
                "teacher_checkpoint": teacher_ckpt,
                "cache_key": _compute_cache_key(seqs, teacher_ckpt, max_len),
                "logits_computed": True,
                "features_computed": False,
                "logits_shape": [3, 2],
                "features_shape": None,
            },
        )
        # logits-only request: valid.
        ok(
            teacher_cache_is_valid(
                seqs,
                teacher_ckpt,
                max_len,
                needs_logits=True,
                needs_features=False,
                project_path=proj,
                teacher_parent_dir=teacher_parent,
                task_name=task,
            )
            is True,
            "(d2) logits-only request validates on a logits-only cache",
        )
        # features-needing request: INVALID (no features file) -> won't skip.
        ok(
            teacher_cache_is_valid(
                seqs,
                teacher_ckpt,
                max_len,
                needs_logits=True,
                needs_features=True,
                project_path=proj,
                teacher_parent_dir=teacher_parent,
                task_name=task,
            )
            is False,
            "(d2) features-needing request does NOT skip on a logits-only cache",
        )


# ---------------------------------------------------------------------------
# (e) teacher_eval_cache_is_valid: present+matching -> valid; mismatch/absent -> not.
# ---------------------------------------------------------------------------
def test_teacher_eval_cache_validity():
    from src.model.glm import teacher_eval_cache_is_valid

    base = _FakeExperimentCfg()
    with tempfile.TemporaryDirectory() as ckpt_dir:
        # absent -> invalid.
        ok(
            teacher_eval_cache_is_valid(base, ckpt_dir) is False,
            "(e) absent teacher_evaluation.json is INVALID",
        )
        eval_file = os.path.join(ckpt_dir, "teacher_evaluation.json")
        # matching checkpoint + has mcc -> valid.
        with open(eval_file, "w") as f:
            json.dump({"teacher_checkpoint": ckpt_dir, "teacher_test_mcc": 0.71}, f)
        ok(
            teacher_eval_cache_is_valid(base, ckpt_dir) is True,
            "(e) matching teacher_evaluation.json is VALID",
        )
        # mismatched checkpoint -> invalid.
        with open(eval_file, "w") as f:
            json.dump({"teacher_checkpoint": "/some/other/ckpt", "teacher_test_mcc": 0.71}, f)
        ok(
            teacher_eval_cache_is_valid(base, ckpt_dir) is False,
            "(e) mismatched-checkpoint teacher_evaluation.json is INVALID",
        )
        # missing mcc -> invalid.
        with open(eval_file, "w") as f:
            json.dump({"teacher_checkpoint": ckpt_dir}, f)
        ok(
            teacher_eval_cache_is_valid(base, ckpt_dir) is False,
            "(e) teacher_evaluation.json without teacher_test_mcc is INVALID",
        )


# ---------------------------------------------------------------------------
# (f) missing teacher eval cache alone blocks the skip (e2e through real helper).
# ---------------------------------------------------------------------------
def test_missing_teacher_eval_blocks_skip():
    import src.train.distill as d
    import numpy as np
    from src.trainer.utils import _get_cache_dir, _save_cache, _compute_cache_key

    orig_common = _common_patches(d, teacher_ckpt=None)  # set below to a real temp dir
    orig = (d.get_teacher_model,)
    loaded = {"flag": False}
    try:
        with tempfile.TemporaryDirectory() as proj:
            teacher_ckpt = os.path.join(proj, "ckpt")
            os.makedirs(teacher_ckpt, exist_ok=True)
            teacher_parent = os.path.join(proj, "data", "finetuned_models", "carbon_3b_lora")
            os.makedirs(teacher_parent, exist_ok=True)
            seqs = ["ACGT", "TTTT"]
            max_len = 1024

            # Valid logits+features cache, but NO teacher_evaluation.json.
            cache_dir = _get_cache_dir(proj, teacher_parent, "H3K27ac")
            _save_cache(
                cache_dir,
                np.zeros((2, 2), dtype=np.float32),
                np.zeros((2, 8), dtype=np.float32),
                {
                    "num_samples": 2,
                    "max_length": max_len,
                    "teacher_checkpoint": teacher_ckpt,
                    "cache_key": _compute_cache_key(seqs, teacher_ckpt, max_len),
                    "logits_computed": True,
                    "features_computed": True,
                    "logits_shape": [2, 2],
                    "features_shape": [2, 8],
                },
            )

            d.find_teacher_checkpoint = lambda config, task_name: (teacher_ckpt, 0.5)
            d.build_data_splits_from_huggingface = lambda dataset_config: (
                seqs,
                [0, 1],
                ["GG"],
                [1],
                ["CC"],
                [0],
            )

            teacher = _StubTeacher("LIVE")

            def fake_load(config, task_name, tckpt):
                loaded["flag"] = True
                return ("TOK", teacher, 8)

            d.get_teacher_model = fake_load

            base = _FakeExperimentCfg(
                dataset_config=_FakeDatasetCfg(task_name="H3K27ac"),
                teacher_parent_dir=teacher_parent,
                trainer_config=_FakeTrainerCfg(cache_base_dir=proj, max_len=max_len),
            )
            ctx = d.prepare_task(
                base,
                "H3K27ac",
                config_overrides_list=[
                    {"distillation_config": {"weight_kl": 1.0, "weight_mse": 1.0}}
                ],
            )
            ok(
                loaded["flag"] is True,
                "(f) valid logits/features but MISSING teacher-eval -> still loads",
            )
            ok(ctx.teacher_model is teacher, "(f) loaded teacher carried")

            # Now WRITE a matching teacher_evaluation.json -> skip kicks in.
            loaded["flag"] = False
            d.get_teacher_model = _boom_get_teacher
            with open(os.path.join(teacher_ckpt, "teacher_evaluation.json"), "w") as f:
                json.dump({"teacher_checkpoint": teacher_ckpt, "teacher_test_mcc": 0.5}, f)
            ctx2 = d.prepare_task(
                base,
                "H3K27ac",
                config_overrides_list=[
                    {"distillation_config": {"weight_kl": 1.0, "weight_mse": 1.0}}
                ],
            )
            from src.train.distill import _NoOpTeacher

            ok(
                isinstance(ctx2.teacher_model, _NoOpTeacher),
                "(f) once ALL caches valid -> skips the 3B load",
            )
            ok(
                ctx2.teacher_hidden == 8,
                "(f) teacher_hidden read from features_shape in cache metadata",
            )
    finally:
        (d.get_num_labels, d.find_teacher_checkpoint, d.build_data_splits_from_huggingface) = (
            orig_common
        )
        (d.get_teacher_model,) = orig


if __name__ == "__main__":
    test_skips_load_when_caches_valid()
    test_loads_when_cache_missing()
    test_stale_cache_key_forces_load()
    test_prepare_task_stale_cache_loads()
    test_needed_outputs_gate_conservative()
    test_features_run_wont_skip_on_logits_only_cache()
    test_teacher_eval_cache_validity()
    test_missing_teacher_eval_blocks_skip()
    print(f"\nALL PASSED ({_n} assertions)")
