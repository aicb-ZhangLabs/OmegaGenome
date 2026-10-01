"""Carbon-3B -> BPNet distillation: shared CORE + a thin, switchable DISPATCH layer.

This module is deliberately split into two layers so the two ways we fan a task's HP
sweep out across processes are cleanly separable and switchable behind ONE selector.

SHARED CORE (mode-independent; byte-identical for every execution strategy)
    ``prepare_task(config, task_name) -> TaskContext``
        The once-per-task expensive setup: find teacher ckpt, load Carbon-3B + merge the
        per-task LoRA, build data splits, resolve the teacher feature dim. ~3-5 min.
    ``train_student(config, task_name, task_ctx)``
        The per-HP-config build + distill-train + eval + save. Cheap vs the teacher load.
    EVERY mode below ultimately calls exactly these two functions with the same configs;
    the modes differ ONLY in WHO loads the teacher and HOW the configs are fanned out.

DISPATCH LAYER (the only thing that varies between modes) -- see ``MODES``.

    MODES = the three execution strategies, selectable via ``run_distillation(..., mode=...)``
    (and the ``--mode`` flag of ``src.train.distill_run``):

      mode="slurm"  -> ONE HP config per process. ``main()`` resolves a single config from
                       the CLI/SLURM spec and runs ``distill`` (== prepare_task + train_student)
                       for it. The cluster scheduler does the fan-out: one sbatch per config.
                       USE WHEN a scheduler (SLURM) owns the fan-out -- the lab default.
                       Back-compat entry: ``python -m src.train.distill <spec...>`` (unchanged).

      mode="batch" + parallel == 1  -> SERIAL single-box batch. ``distill_task_batch`` loads
                       the teacher ONCE (prepare_task) then loops the task's whole config list
                       calling train_student per config. Amortizes the 3B load across ~32 configs.
                       USE WHEN one fat box has no per-config scheduler (e.g. a single rented GPU box).

      mode="batch" + parallel > 1   -> CONFIG-PARALLEL single-box batch. Same single teacher
                       load, but the teacher outputs are warmed into the on-disk cache once, the
                       3B teacher is freed from GPU, and ``parallel`` spawn-workers each run the
                       SAME train_student over the configs (the tiny BPNet students share the
                       idle GPU). USE WHEN a single box has GPU headroom for concurrent students.

    Back-compat entry for both batch sub-modes: ``python -m src.train.distill_task
    --task <t> --config-list <f> [--parallel N]`` (unchanged). The ``parallel`` knob picks
    serial vs config-parallel WITHIN batch mode exactly as before.

The dispatch functions (``distill`` / ``main`` for slurm; ``distill_task_batch`` and its
``_distill_task_batch_serial`` / ``_distill_task_batch_parallel`` helpers for batch) are
thin: they decide teacher-load + fan-out only, then delegate to the shared core. Adding a
mode means adding a dispatch function, never touching the core. ``run_distillation`` is the
single documented selector that routes a ``mode`` value to the right dispatch function.
"""

import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# FORK-SAFETY (must be set BEFORE torch's first `cuda.is_available()` call): make
# `torch.cuda.is_available()` use the NVML-based probe instead of the CUDA Runtime API
# (`cudaGetDeviceCount` -> `cuInit`). The default Runtime-API probe INITIALIZES the CUDA driver
# context as a side effect, which POISONS any later `fork` (children then die with "Cannot
# re-initialize CUDA in forked subprocess") AND is invisible to `torch.cuda.is_initialized()`
# (which only tracks PyTorch's Python-level lazy init, not the driver `cuInit`). The NVML path
# does NOT call `cuInit`, so the import-time `DistillTrainerConfig.device` default and every other
# pre-fork `is_available()` stay fork-safe. Set unconditionally here (idempotent; honors an
# existing user value if already set to something truthy). See `_distill_task_batch_fork`.
os.environ.setdefault("PYTORCH_NVML_BASED_CUDA_CHECK", "1")

import tyro
import wandb

from datetime import datetime
from dataclasses import dataclass, replace, asdict
from typing import Any, List, Optional
from nntool.slurm import slurm_fn

from config.distillation.config import configs
from config.distillation.config_schema import DistillationExperimentConfig
from config.env import project_output_path
from accelerate.utils import set_seed

from ..model.bpnet_classifier import BPNetClassifier
from ..model.distillation import DistillationModel
from ..data.dataset import (
    get_num_labels,
    build_data_splits_from_huggingface,
)
from ..trainer.distill_trainer import (
    train_distill_task,
    create_run_directory,
    create_run_hyperparams_str,
)


from ..model.glm import (
    find_teacher_checkpoint,
    get_teacher_model,
    evaluate_and_log_teacher,
    teacher_eval_cache_is_valid,
)


@dataclass
class TaskContext:
    """Bundle of the once-per-task expensive setup shared across a task's HP configs.

    WHY: the teacher checkpoint, the loaded+merged Carbon-3B teacher model, the
    data splits, and the resolved teacher feature dimension depend ONLY on the
    task (and the teacher_config that selects it), NOT on the distillation
    hyperparameters (weight_ce/kl/mse, temperature, lr, random_state). Loading
    the 3B base + merging the per-task LoRA adapter + re-evaluating the teacher
    costs ~3-5 min and is identical for every config of a task, so we compute it
    ONCE in ``prepare_task`` and reuse it for all of the task's ~32 HP configs.
    """

    teacher_tokenizer: Any
    teacher_model: Any
    teacher_hidden: int
    num_labels: int
    teacher_ckpt: str
    score: float
    # Data splits (identical across the task's configs).
    X_train: List[Any]
    y_train: Any
    X_val: List[Any]
    y_val: Any
    X_test: List[Any]
    y_test: Any
    # SHARED-MEMORY (fork) fast path ONLY: teacher logits/features pre-loaded ONCE into CPU arrays
    # by the fork-parallel parent so forked children read them via Linux COW (shared, not re-loaded
    # per child). Default None on EVERY other path (serial/SLURM/spawn) -> train_student forwards
    # None and train_distill_task reads the on-disk cache exactly as before (byte-identical). These
    # are NOT serialized into the spawn ctx cache (spawn workers each read the disk cache); they
    # exist purely to back the fork COW sharing.
    teacher_logits: Any = None
    teacher_features: Any = None


def _resolve_needed_teacher_outputs(
    config: DistillationExperimentConfig,
    config_overrides_list: Optional[List[dict]],
) -> tuple:
    """Decide which teacher outputs (logits, features) this task's RUN can need.

    The teacher is forwarded ONLY for KL logits (``weight_kl > 0``) or MSE features
    (``weight_mse > 0``) — mirrors ``train_distill_task``'s gate. Returns the UNION of
    what ANY config in the run needs, so the skip-the-load fast path NEVER drops an
    output a later config will read uncached:

      - Batch path: ``config_overrides_list`` is the full HP sweep -> take the exact
        union across every resolved config (precise, can skip features for a KL-only
        sweep and vice-versa).
      - SLURM/single-config path: ``config_overrides_list is None`` -> we only see ONE
        resolved config, so we CONSERVATIVELY treat BOTH logits AND features as needed.
        That way we never skip the load when this lone config (or a sibling sharing the
        same teacher+task cache dir) would later require an output we hadn't validated.

    Returns ``(needs_logits, needs_features)``.
    """
    if config_overrides_list is None:
        return True, True
    resolved = [_apply_override(config, ov) for ov in config_overrides_list]
    needs_logits = any(cfg.distillation_config.weight_kl > 0 for cfg in resolved)
    needs_features = any(cfg.distillation_config.weight_mse > 0 for cfg in resolved)
    return needs_logits, needs_features


def _teacher_caches_valid_for_task(
    config: DistillationExperimentConfig,
    task_name: str,
    teacher_ckpt: str,
    X_train: List[Any],
    needs_logits: bool,
    needs_features: bool,
) -> bool:
    """True iff EVERY teacher output this run needs is already cached + valid on disk.

    Reuses ``teacher_cache_is_valid`` (logits/features key+path, shared with
    ``precompute_teacher_logits``) and ``teacher_eval_cache_is_valid`` (the
    teacher_evaluation.json the per-config ``evaluate_and_log_teacher`` always reads).
    A True result means the whole run can proceed off-cache, so the 3B teacher load is
    pure waste and can be skipped. Conservative by construction — any missing/stale
    cache returns False and forces a real load (self-correcting on first run).

    Note ``train_distill_task`` always calls ``evaluate_and_log_teacher`` (so the
    teacher-eval cache is required UNCONDITIONALLY), but only forwards the teacher for
    logits/features when ``needs_logits``/``needs_features``; pure-CE runs that need
    neither still require a valid teacher-eval cache to skip the load.
    """
    from src.trainer.utils import teacher_cache_is_valid
    from config.env import project_path

    tcfg = config.trainer_config
    cache_base = getattr(tcfg, "cache_base_dir", None) or project_path

    if not teacher_eval_cache_is_valid(config, teacher_ckpt):
        return False

    if needs_logits or needs_features:
        if not teacher_cache_is_valid(
            X_train,
            teacher_ckpt,
            tcfg.max_len,
            needs_logits=needs_logits,
            needs_features=needs_features,
            project_path=cache_base,
            teacher_parent_dir=config.teacher_parent_dir,
            task_name=task_name,
        ):
            return False
    return True


def _teacher_hidden_from_cache_or_config(
    config: DistillationExperimentConfig,
    task_name: str,
    teacher_ckpt: str,
    needs_features: bool,
) -> Optional[int]:
    """Resolve ``teacher_hidden`` WITHOUT loading the teacher, for the skip-load path.

    When MSE features are in play the student builds a teacher->student projection sized
    by the teacher feature dim, so this MUST match what the loaded path resolves. The
    loaded path reads exactly this dim from the cache metadata's ``features_shape`` when
    a valid features cache exists (the SAME ``metadata.json`` we just validated), so we
    read it here too — no forward pass needed.

    When MSE is NOT needed, the student never builds the projection and ``teacher_hidden``
    is unused downstream; we return the config's declared teacher hidden size (falling
    back to None), purely as ctx metadata. This can never change a student that has no
    MSE term.
    """
    if needs_features:
        from config.env import project_path
        from src.trainer.utils import _get_cache_dir
        import json

        tcfg = config.trainer_config
        cache_base = getattr(tcfg, "cache_base_dir", None) or project_path
        metadata_path = (
            _get_cache_dir(cache_base, config.teacher_parent_dir, task_name) / "metadata.json"
        )
        try:
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            fshape = metadata.get("features_shape")
            if fshape:
                return int(fshape[-1])
        except Exception as e:  # pragma: no cover - validated cache makes this rare
            print(f"[prepare_task] Warning: could not read features_shape from cache: {e}")
    # MSE not needed (dim unused) -> harmless config-declared value (or None).
    return getattr(config.student_config, "teacher_hidden_size", None)


def prepare_task(
    config: DistillationExperimentConfig,
    task_name: str,
    config_overrides_list: Optional[List[dict]] = None,
) -> Optional[TaskContext]:
    """Run the once-per-task expensive setup and return a reusable ``TaskContext``.

    WHY: this is the throughput bottleneck — it finds the teacher checkpoint,
    loads the Carbon-3B base model and merges the per-task LoRA adapter, builds
    the data splits, and resolves the actual teacher feature dimension. None of
    this depends on the distillation hyperparameters, so it is shared verbatim
    across all of a task's HP configs (see ``distill_task_batch``).

    FAST PATH (additive, behavior-preserving): when the HP-independent teacher
    caches (logits/features for whatever this run needs + teacher_evaluation.json)
    already exist and validate against ``(X_train, teacher_ckpt, max_len)``, the 3B
    teacher load is pure waste — every downstream read hits the cache. In that case we
    SKIP ``get_teacher_model`` entirely and return a ``TaskContext`` whose
    ``teacher_model`` is the ``_NoOpTeacher`` stub (so ``.to()/.eval()`` are no-ops and
    a real forward raises loudly). Any missing/stale cache forces the original full
    load + rebuild (self-correcting). ``config_overrides_list`` (the batch sweep, when
    available) makes the "which outputs are needed" gate exact; without it we gate
    conservatively (treat both logits+features as needed) so a later config can never
    need an output we did not validate.

    Returns None (after logging a skip) when no teacher checkpoint exists for
    the task — byte-for-byte the same skip behavior as the original ``distill``.
    """
    print(f"\n{'=' * 80}")
    print(f"=== Preparing Task (teacher load + data): {task_name} ===")
    print(f"{'=' * 80}")

    # Model type detection for display
    model_type = getattr(config, "model_type", "glm")
    print(f"Model Type: {model_type.upper()}")
    print(f"Teacher: {config.teacher_config.model_name_or_path}")
    print(f"Student: {config.student_config.model_type}-{config.student_config.model_size}")
    print(f"Method: {config.distillation_config.distill_method}")
    print(f"{'=' * 80}\n")

    # ===========================================================
    # MODIFIED SECTION: Use unified checkpoint finding
    # ===========================================================
    num_labels = get_num_labels(task_name)
    config = replace(config, teacher_config=replace(config.teacher_config, num_labels=num_labels))
    teacher_ckpt, score = find_teacher_checkpoint(config, task_name)

    if teacher_ckpt is None:
        print(f"[!] No teacher checkpoint found for {task_name}, skipping.")
        return None

    print(f"Teacher checkpoint: {teacher_ckpt}")
    if score > 0:
        print(f"Teacher validation MCC: {score:.4f}")

    # Build data splits FIRST (needed for the cache-key check below AND for the ctx).
    # These don't need the teacher model, so resolving them here lets us decide whether
    # the 3B load can be skipped BEFORE paying for it.
    X_train, y_train, X_val, y_val, X_test, y_test = build_data_splits_from_huggingface(
        config.dataset_config
    )

    # ===========================================================
    # FAST PATH: skip the 3B teacher load when its output caches are already valid.
    # ===========================================================
    needs_logits, needs_features = _resolve_needed_teacher_outputs(config, config_overrides_list)
    if _teacher_caches_valid_for_task(
        config, task_name, teacher_ckpt, X_train, needs_logits, needs_features
    ):
        print(f"[prepare_task] teacher cache valid for {task_name} — skipping 3B teacher load")
        # teacher_hidden comes from the cache metadata (features) when MSE is in play;
        # otherwise the student never reads it (no MSE projection), so a sentinel of the
        # config's declared hidden size is correct. This matches what the loaded path
        # would resolve, WITHOUT the 3B forward.
        teacher_hidden = _teacher_hidden_from_cache_or_config(
            config, task_name, teacher_ckpt, needs_features
        )
        return TaskContext(
            teacher_tokenizer=None,
            teacher_model=_NoOpTeacher(),
            teacher_hidden=teacher_hidden,
            num_labels=num_labels,
            teacher_ckpt=teacher_ckpt,
            score=score,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            X_test=X_test,
            y_test=y_test,
        )

    # ===========================================================
    # MODIFIED SECTION: Use unified teacher model loading
    # ===========================================================
    teacher_tokenizer, teacher_model, teacher_hidden = get_teacher_model(
        config, task_name, teacher_ckpt
    )
    print(f"Initial teacher_hidden from model: {teacher_hidden}")
    teacher_model.eval()
    # ===========================================================
    # ORIGINAL CODE: Student model and distillation setup
    # ===========================================================
    # CRITICAL FIX: Determine actual teacher hidden size
    # This handles cases where config.d_model != actual feature dimension
    needs_features = config.distillation_config.weight_mse > 0

    if needs_features:
        from config.env import project_path
        import json
        import torch

        # Import the helper functions
        from src.trainer.utils import _get_cache_dir

        actual_teacher_hidden = None

        # Method 1: Try to read from existing cache metadata
        cache_dir = _get_cache_dir(project_path, config.teacher_parent_dir, task_name)
        metadata_path = cache_dir / "metadata.json"

        if metadata_path.exists():
            try:
                with open(metadata_path, "r") as f:
                    metadata = json.load(f)

                if "features_shape" in metadata and metadata["features_shape"] is not None:
                    actual_teacher_hidden = metadata["features_shape"][-1]  # Last dimension
                    print(f"✓ Read feature dimension from cache metadata: {actual_teacher_hidden}")
            except Exception as e:
                print(f"Warning: Could not read cache metadata: {e}")

        # Method 2: If no cache, do a quick single-sample forward pass
        if actual_teacher_hidden is None:
            print("No cache found, doing quick forward pass to determine feature dimension...")
            try:
                # Tokenize a single sample
                sample_seq = X_train[0] if len(X_train) > 0 else "ATCGATCG"
                tok = teacher_tokenizer(
                    sample_seq,
                    padding="max_length",
                    truncation=True,
                    max_length=config.trainer_config.max_len,
                    return_tensors="pt",
                )
                input_ids = tok.input_ids.to(config.trainer_config.device)
                attention_mask = (
                    tok.attention_mask.to(config.trainer_config.device)
                    if hasattr(tok, "attention_mask")
                    else torch.ones_like(input_ids)
                )

                with torch.no_grad():
                    _, sample_features = teacher_model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        return_features=True,
                    )
                    actual_teacher_hidden = sample_features.shape[-1]
                    print(
                        f"✓ Determined feature dimension from forward pass: {actual_teacher_hidden}"
                    )
            except Exception as e:
                print(f"Warning: Could not determine dimension from forward pass: {e}")

        # Update teacher_hidden if we found the actual dimension
        if actual_teacher_hidden is not None and actual_teacher_hidden != teacher_hidden:
            print(
                f"WARNING: Config says hidden_dim={teacher_hidden}, but actual features are {actual_teacher_hidden}-dimensional"
            )
            print(f"  Overriding to use actual dimension: {actual_teacher_hidden}")
            teacher_hidden = actual_teacher_hidden
        elif actual_teacher_hidden is not None:
            print(f"✓ Config hidden_dim={teacher_hidden} matches actual features")

    print(f"Final teacher_hidden for student model: {teacher_hidden}")

    return TaskContext(
        teacher_tokenizer=teacher_tokenizer,
        teacher_model=teacher_model,
        teacher_hidden=teacher_hidden,
        num_labels=num_labels,
        teacher_ckpt=teacher_ckpt,
        score=score,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
    )


def _config_already_done(config, task_name) -> bool:
    """Resume-safety: True iff a ``final_summary.json`` for this EXACT config already exists
    under the config's own ``output_dir``.

    Keyed on the full config identity — ``(task, weight_ce, weight_kl, weight_mse, temperature,
    random_state)`` — so it skips a re-run of an already-completed grid config but NEVER skips an
    intentional different-seed re-run (3-seed uses distinct ``random_state``). Scoped to
    ``config.trainer_config.output_dir`` so the raw/l2norm/deploy searches stay separate, and
    ``mse_normalizeTrue`` (l2norm) runs are excluded — matching ``gen_hp_specs.done_combos``.
    Run output dirs are randomized (timestamp+uuid), so we glob the task subtree rather than stat a
    fixed path. Defensive: any unreadable/partial summary is ignored (treated as not-a-match).
    """
    import glob
    import json

    dc = config.distillation_config
    target = (
        task_name,
        float(dc.weight_ce),
        float(dc.weight_kl),
        float(dc.weight_mse),
        float(dc.temperature),
        int(config.random_state),
    )
    pattern = os.path.join(config.trainer_config.output_dir, task_name, "**", "final_summary.json")
    for f in glob.iglob(pattern, recursive=True):
        if "mse_normalizeTrue" in f:  # l2norm variant has its own search; never count it here
            continue
        try:
            with open(f) as fh:
                s = json.load(fh)
            hp = s["hyperparameters"]
            if (
                s.get("task"),
                float(hp["weight_ce"]),
                float(hp["weight_kl"]),
                float(hp["weight_mse"]),
                float(hp["temperature"]),
                int(s.get("random_state", 42)),
            ) == target:
                return True
        except Exception:
            continue
    return False


def train_student(
    config: DistillationExperimentConfig,
    task_name: str,
    task_ctx: TaskContext,
):
    """Run the per-config student build + distillation train + eval + save.

    WHY: everything here varies per distillation HP config (student model,
    DistillationModel weights, run dir, wandb run, seed) and is cheap relative to
    the teacher load. It reuses the pre-loaded teacher + data from ``task_ctx``
    so a task's ~32 configs amortize the single 3B-load+LoRA-merge in
    ``prepare_task``.

    NOTE: ``set_seed(config.random_state)`` runs HERE (per config), not once per
    task, so every config gets its own deterministic seeding exactly as the
    original per-config ``distill`` did.
    """
    # Resume-safety (OPT-IN, default off): skip a config whose exact result already exists, so a
    # stale specs file (or a re-submitted grid) cannot waste compute re-running done work. Enabled
    # only by the grid-search launchers via CARBON_SKIP_IF_DONE=1; never set for 3-seed/intentional
    # re-runs (and even if it were, the seed-aware key would not skip a new-seed run).
    if os.environ.get("CARBON_SKIP_IF_DONE") == "1" and _config_already_done(config, task_name):
        dc = config.distillation_config
        print(
            f"[skip-if-done] {task_name} CE{dc.weight_ce}/KL{dc.weight_kl}/MSE{dc.weight_mse}/"
            f"T{dc.temperature}/seed{config.random_state} already has a result — skipping (resume-safe)."
        )
        return

    import json

    # Per-config deterministic seeding (matches original distill: set per config,
    # never amortized across a task's configs).
    set_seed(config.random_state)

    # Mirror the original distill(): stamp the task's num_labels onto teacher_config
    # BEFORE asdict(config) is recorded, so hyperparameters.json + the wandb config
    # log the task's label count (not the base-config default). teacher_config is
    # otherwise unused downstream here (train uses input_prefix/add_special_tokens),
    # so this is purely metadata fidelity vs the per-config path.
    config = replace(
        config,
        teacher_config=replace(config.teacher_config, num_labels=task_ctx.num_labels),
    )

    # Unpack the shared, pre-loaded task context.
    teacher_tokenizer = task_ctx.teacher_tokenizer
    teacher_model = task_ctx.teacher_model
    teacher_hidden = task_ctx.teacher_hidden
    num_labels = task_ctx.num_labels
    teacher_ckpt = task_ctx.teacher_ckpt
    score = task_ctx.score
    X_train, y_train = task_ctx.X_train, task_ctx.y_train
    X_val, y_val = task_ctx.X_val, task_ctx.y_val
    X_test, y_test = task_ctx.X_test, task_ctx.y_test

    # Build student model

    student_config = replace(
        config.student_config,
        num_labels=num_labels,
        teacher_hidden_size=teacher_hidden,
    )
    model = BPNetClassifier(student_config)

    # Build distillation model
    distillation_model = DistillationModel(
        config.distillation_config,
        teacher_model,
        model,
        config.trainer_config.device,
    )

    # Create run directory
    run_dir = create_run_directory(
        config.trainer_config.output_dir,
        task_name,
        config.distillation_config,
    )
    print(f"Run directory: {run_dir}")

    # WandB setup
    hyperparams_str = create_run_hyperparams_str(config.distillation_config)
    model_id = f"{config.student_config.model_type}_{config.student_config.model_size}"
    distill_method = config.distillation_config.distill_method
    run_name = f"{task_name}/{model_id}/{distill_method}/{hyperparams_str}"

    # Add model type to tags
    model_type_tag = getattr(config, "model_type", "glm")

    wandb.init(
        project=config.trainer_config.wandb_project,
        name=run_name,
        dir=project_output_path,
        config=asdict(config),
        notes=f"run_dir: {run_dir}",
        tags=[task_name, model_type_tag, model_id, distill_method],
    )
    wandb.watch(model, log="all", log_freq=100)

    # ===== EVALUATE TEACHER MODEL =====
    teacher_mcc = evaluate_and_log_teacher(
        teacher_model,
        teacher_tokenizer,
        X_test,
        y_test,
        task_name,
        config,
        run_dir,
        teacher_ckpt,
    )

    # Save hyperparameters
    hyperparams = asdict(config)
    hyperparams["run_dir"] = run_dir
    hyperparams["timestamp"] = datetime.now().isoformat()
    hyperparams["teacher_type"] = model_type_tag
    hyperparams["teacher_checkpoint"] = teacher_ckpt
    hyperparams["teacher_test_mcc"] = float(teacher_mcc)
    if score > 0:
        hyperparams["teacher_val_score"] = score

    with open(os.path.join(run_dir, "hyperparameters.json"), "w") as f:
        json.dump(hyperparams, f, indent=2)

    # Train
    train_distill_task(
        config.trainer_config,
        config.distillation_config,
        task_name,
        teacher_tokenizer,
        teacher_model,
        model,
        distillation_model,
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        run_dir,
        teacher_parent_dir=config.teacher_parent_dir,
        teacher_ckpt=teacher_ckpt,
        resume_from_checkpoint=config.resume_checkpoint,  # Read from config
        resume_from_epoch=config.resume_epoch,  # Read from config
        input_prefix=getattr(config.teacher_config, "input_prefix", ""),
        add_special_tokens=getattr(config.teacher_config, "add_special_tokens", True),
        random_state=config.random_state,
        # Fork-shared teacher arrays (None on every non-fork path -> precompute reads the disk cache
        # exactly as before). getattr keeps this safe for any TaskContext built without the fields.
        preloaded_teacher_logits=getattr(task_ctx, "teacher_logits", None),
        preloaded_teacher_features=getattr(task_ctx, "teacher_features", None),
    )

    wandb.finish()


@slurm_fn
def distill(
    config: DistillationExperimentConfig,
    task_name: str,
):
    """Thin wrapper preserving the original per-config entry point.

    Behavior-identical to the pre-refactor ``distill``: it runs the once-per-task
    setup (``prepare_task``) then the per-config train/eval/save
    (``train_student``). The refactor is a pure extraction so the existing
    ``python -m src.train.distill <spec>`` / SLURM path is unchanged.
    """
    print(f"\n{'=' * 80}")
    print(f"=== Starting Distillation: {task_name} ===")
    print(f"{'=' * 80}")

    ctx = prepare_task(config, task_name)
    if ctx is None:
        return
    train_student(config, task_name, ctx)


def distill_task_batch(
    base_config: DistillationExperimentConfig,
    task_name: str,
    config_overrides_list: List[dict],
    parallel: int = 1,
    parallel_mode: str = "fork",
):
    """Load a task's teacher ONCE and train every HP config against it.

    WHY: ``prepare_task`` (3B base load + per-task LoRA merge + teacher eval +
    data splits) is ~3-5 min and identical for all of a task's ~32 HP configs.
    Running one process per task and looping the configs in-process amortizes
    that single load across the whole sweep instead of paying it per config.

    Args:
        base_config: the seed experiment config (e.g. ``carbon-raw-original``)
            with ``dataset_config.task_name`` set for this task.
        task_name: the task to distill.
        config_overrides_list: list of dicts; each is a set of attribute paths
            to override on ``base_config`` for one HP config. Supported keys:
            ``distillation_config`` (a dict of DistillationModelConfig field
            overrides) and any top-level DistillationExperimentConfig field
            (e.g. ``random_state``). Each override is applied via
            ``dataclasses.replace`` so the teacher/data context is untouched.
        parallel: number of HP configs to train CONCURRENTLY (default ``1`` =
            the original serial path, byte-for-byte unchanged). When ``> 1`` the
            teacher outputs are warmed into the on-disk cache, the 3B teacher is
            freed from GPU, and ``parallel`` spawn-based workers each run the SAME
            ``train_student`` on a teacher-free ``TaskContext`` (the tiny 317K
            BPNet students share the otherwise-idle GPU). Each config's result is
            identical to the serial run; only the concurrency differs.
        parallel_mode: how the ``parallel`` workers are fanned out (ignored when
            ``parallel <= 1``):

              ``"fork"`` (default) -- prepare_task loads the data + teacher
                logits/features ONCE on CPU (the skip-load fast path keeps the 3B
                teacher OFF the GPU so NO CUDA context exists), then forks workers
                with the ``fork`` start method. Linux copy-on-write means the big
                read-only CPU arrays are SHARED, not copied, so memory stays ~1×
                (data + features) per task regardless of ``parallel``. Each child
                inits CUDA fresh post-fork (safe) and trains its config. Falls back
                to ``"spawn"`` automatically (logged) if CUDA is already initialized
                in the parent (e.g. the teacher could NOT be cache-skipped), since
                forking a CUDA-initialized process is unsafe.

              ``"spawn"`` -- the original config-parallel path: warm the on-disk
                teacher cache, free the teacher, serialize a teacher-free ctx, and
                spawn fresh Python workers that EACH re-read the data + teacher
                cache from disk (memory ~N×). Use when fork is unavailable.

    Per-config exceptions are caught + logged so one bad config does not abort
    the rest of the task.
    """
    if parallel is None or parallel <= 1:
        return _distill_task_batch_serial(base_config, task_name, config_overrides_list)
    if parallel_mode == "fork":
        return _distill_task_batch_fork(base_config, task_name, config_overrides_list, parallel)
    if parallel_mode == "spawn":
        return _distill_task_batch_parallel(base_config, task_name, config_overrides_list, parallel)
    raise ValueError(f"Unknown parallel_mode {parallel_mode!r}; expected 'fork' or 'spawn'.")


def _distill_task_batch_serial(
    base_config: DistillationExperimentConfig,
    task_name: str,
    config_overrides_list: List[dict],
):
    """The ORIGINAL serial batch path (parallel == 1). Kept byte-for-byte.

    WHY isolated: the serial path is audited + in production. The new ``parallel``
    knob must not alter it, so ``parallel == 1`` dispatches straight here with no
    pool, no teacher-cache serialization, and no teacher free.
    """
    import gc
    import traceback

    try:
        import torch
    except Exception:
        torch = None

    print(f"\n{'#' * 80}")
    print(f"### BATCH distillation for task={task_name}: {len(config_overrides_list)} configs ###")
    print(f"{'#' * 80}")

    # Ensure teacher_hidden (the MSE feature dim) is resolved in prepare_task if
    # ANY config in this batch uses MSE. prepare_task only resolves it when its
    # config has weight_mse > 0; the dim is task-intrinsic so resolving once for
    # all configs is correct. Seed base_config's weight_mse from the overrides.
    prep_config = _seed_prep_config(base_config, config_overrides_list)

    # Pass the FULL sweep so prepare_task's skip-the-load gate is EXACT (union of
    # logits/features needed across all configs) rather than conservative.
    ctx = _call_prepare_task(prep_config, task_name, config_overrides_list)
    if ctx is None:
        print(f"[!] prepare_task returned None for {task_name}; nothing to run.")
        return

    n_ok, n_fail = 0, 0
    for i, override in enumerate(config_overrides_list):
        print(f"\n{'-' * 80}")
        print(f"--- [{task_name}] config {i + 1}/{len(config_overrides_list)}: {override} ---")
        print(f"{'-' * 80}")
        try:
            cfg = _apply_override(base_config, override)
            train_student(cfg, task_name, ctx)
            n_ok += 1
        except Exception:
            n_fail += 1
            print(f"[!] config {i + 1} FAILED for {task_name}:")
            traceback.print_exc()
        finally:
            # Free the per-config student (teacher stays resident in ctx).
            try:
                if wandb.run is not None:
                    wandb.finish(exit_code=0)
            except Exception:
                pass
            gc.collect()
            if torch is not None and torch.cuda.is_available():
                torch.cuda.empty_cache()

    print(f"\n{'#' * 80}")
    print(f"### BATCH done for task={task_name}: {n_ok} ok, {n_fail} failed ###")
    print(f"{'#' * 80}")


def _call_prepare_task(
    prep_config: DistillationExperimentConfig,
    task_name: str,
    config_overrides_list: List[dict],
):
    """Invoke ``prepare_task``, passing the sweep ONLY if its signature accepts it.

    The batch dispatchers want to hand ``prepare_task`` the full HP sweep so its
    skip-the-3B-load gate is EXACT (the precise union of logits/features any config
    needs). But tests (and any caller) may monkeypatch ``prepare_task`` with the legacy
    ``(config, task_name)`` signature; passing an unexpected kwarg would break them. We
    introspect the live ``prepare_task`` and forward ``config_overrides_list`` only when
    it is a supported parameter — otherwise we fall back to the 2-arg call (conservative
    gate). Module-level lookup of ``prepare_task`` keeps monkeypatching honored.
    """
    import inspect

    fn = prepare_task
    try:
        accepts = "config_overrides_list" in inspect.signature(fn).parameters
    except (TypeError, ValueError):
        accepts = False
    if accepts:
        return fn(prep_config, task_name, config_overrides_list=config_overrides_list)
    return fn(prep_config, task_name)


def _seed_prep_config(
    base_config: DistillationExperimentConfig,
    config_overrides_list: List[dict],
) -> DistillationExperimentConfig:
    """Return the config to hand ``prepare_task`` so the MSE feature dim resolves.

    ``prepare_task`` only resolves the actual teacher feature dimension when its
    config has ``weight_mse > 0``. The dim is task-intrinsic, so if ANY config in
    the batch uses MSE we seed ``weight_mse=1.0`` for the (single) prepare call;
    ``base_config`` and every per-config override are unaffected. Extracted so the
    serial and parallel paths seed prepare_task IDENTICALLY.
    """
    any_mse = any(
        (ov.get("distillation_config", {}) or {}).get(
            "weight_mse", base_config.distillation_config.weight_mse
        )
        > 0
        for ov in config_overrides_list
    )
    if any_mse and base_config.distillation_config.weight_mse <= 0:
        return replace(
            base_config,
            distillation_config=replace(base_config.distillation_config, weight_mse=1.0),
        )
    return base_config


# ---------------------------------------------------------------------------
# Config-parallel path (opt-in via parallel > 1). Additive; never touched when
# parallel == 1.
# ---------------------------------------------------------------------------

# Fields of TaskContext that are HP-INDEPENDENT and must be serialized for the
# workers. EVERY field except ``teacher_model`` (the live 3B model, which is NEVER
# sent to a worker -- workers get a no-op stub and read the already-warm teacher
# logits/features + teacher-eval from the on-disk cache instead).
_TASK_CTX_CACHE_FIELDS = (
    "teacher_tokenizer",
    "teacher_hidden",
    "num_labels",
    "teacher_ckpt",
    "score",
    "X_train",
    "y_train",
    "X_val",
    "y_val",
    "X_test",
    "y_test",
)


class _NoOpTeacher:
    """Teacher placeholder for parallel workers; NEVER the real 3B model.

    ``train_distill_task`` calls ``teacher_model.to(device)``/``.eval()`` and passes
    the teacher to ``precompute_teacher_logits``/``evaluate_and_log_teacher``. In the
    parallel path those caches are already WARM (populated once while the real
    teacher was alive), so each of those calls short-circuits to the cached values
    BEFORE invoking any teacher forward. This stub provides the no-op ``.to()``/
    ``.eval()`` surface so the worker never needs -- and never loads -- the 3B teacher.
    """

    def to(self, *args, **kwargs):
        return self

    def eval(self, *args, **kwargs):
        return self

    def __call__(self, *args, **kwargs):  # pragma: no cover - cache makes this dead
        raise RuntimeError(
            "_NoOpTeacher was invoked: the teacher-output cache was expected to be "
            "warm in the parallel path but a teacher forward was requested. This is a "
            "bug -- a worker must never run the 3B teacher."
        )


def _serialize_task_ctx(ctx: "TaskContext", cache_path: str) -> str:
    """torch.save the HP-independent TaskContext fields to ``cache_path``.

    Saves EXACTLY ``_TASK_CTX_CACHE_FIELDS`` (everything a worker's ``train_student``
    reads off the ctx EXCEPT the live 3B ``teacher_model``). Returns ``cache_path``.
    """
    import torch

    payload = {f: getattr(ctx, f) for f in _TASK_CTX_CACHE_FIELDS}
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    torch.save(payload, cache_path)
    return cache_path


def _load_task_ctx_from_cache(cache_path: str) -> "TaskContext":
    """Reconstruct a teacher-free ``TaskContext`` from a ``_serialize_task_ctx`` file.

    ``teacher_model`` is a ``_NoOpTeacher`` stub -- workers MUST NOT load the 3B
    teacher. All other fields round-trip from disk verbatim.
    """
    import torch

    payload = torch.load(cache_path, map_location="cpu", weights_only=False)
    return TaskContext(teacher_model=_NoOpTeacher(), **payload)


def _run_one_config_worker(args):
    """Worker entry: build the config for ONE override + run train_student.

    Loads the cached teacher-free ctx from disk (so it gets its OWN ctx copy -- no
    shared-state race when train_student re-stamps teacher_config.num_labels), seeds
    via the SAME per-config path as serial (set_seed inside train_student), and runs
    the identical ``train_student``. Returns ``(index, ok, error_repr)`` so the
    parent can record successes/failures WITHOUT one bad config killing siblings.
    """
    index, base_config, task_name, override, cache_path = args
    try:
        import torch
    except Exception:
        torch = None
    try:
        ctx = _load_task_ctx_from_cache(cache_path)
        cfg = _apply_override(base_config, override)
        train_student(cfg, task_name, ctx)
        return (index, True, None)
    except Exception:
        import traceback

        return (index, False, traceback.format_exc())
    finally:
        try:
            if wandb.run is not None:
                wandb.finish(exit_code=0)
        except Exception:
            pass
        import gc

        gc.collect()
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()


def _free_teacher(ctx: "TaskContext"):
    """Drop the live 3B teacher from ``ctx`` + GPU before workers spawn.

    Sets ``ctx.teacher_model=None``, gc-collects, and empties the CUDA cache so the
    teacher's ~7GB is released before the worker pool starts (workers never need it
    -- only its already-cached outputs).
    """
    import gc

    try:
        import torch
    except Exception:
        torch = None
    ctx.teacher_model = None
    gc.collect()
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()


def _spawn_pool(parallel: int):
    """Return a ``multiprocessing`` Pool over the SPAWN context (never fork).

    Spawn (not fork) is mandatory: forking a process that has already initialized
    CUDA corrupts the child's CUDA context. Isolated so tests can monkeypatch it
    with a synchronous executor.
    """
    import multiprocessing as mp

    ctx = mp.get_context("spawn")
    return ctx.Pool(processes=parallel)


def _config_needs_teacher(cfg: DistillationExperimentConfig) -> bool:
    """True iff a resolved config requires teacher outputs (KL logits or MSE features).

    Mirrors ``train_distill_task``'s ``needs_logits/needs_features`` gate: the teacher
    is only ever forwarded when ``weight_kl > 0`` (logits) or ``weight_mse > 0``
    (features). A pure-CE config (both zero) never touches the teacher, so the parallel
    warm can be skipped for it.
    """
    dc = cfg.distillation_config
    return dc.weight_kl > 0 or dc.weight_mse > 0


def _warm_teacher_cache(
    base_config: DistillationExperimentConfig,
    task_name: str,
    ctx: "TaskContext",
    config_overrides_list: List[dict],
) -> None:
    """Populate the HP-independent on-disk teacher caches ONCE, WITHOUT full training.

    WHY: the previous parallel path ran config-0's FULL ``train_student`` (up to 200
    epochs) serially just to warm the cache, blocking all parallelism for hours -- and
    when config-0 was pure-CE it populated NOTHING (precompute is skipped for
    weight_kl==0 AND weight_mse==0). This replaces that with the minimal correct warm:
    a direct ``precompute_teacher_logits`` (teacher logits/features) plus
    ``evaluate_and_log_teacher`` (teacher-eval json) call using the LIVE teacher on
    ``ctx`` -- no throwaway training, no output-dir collision. After this, every worker
    reads both caches and NEVER loads the 3B teacher.

    The ``needs_logits``/``needs_features`` flags are the UNION across all configs (so
    whatever ANY config needs is warmed). No-ops entirely when no config needs the
    teacher (caller guards), exactly matching the serial path's per-config behavior.
    """
    from src.trainer.utils import precompute_teacher_logits
    from config.env import project_path

    resolved = [_apply_override(base_config, ov) for ov in config_overrides_list]
    needs_logits = any(cfg.distillation_config.weight_kl > 0 for cfg in resolved)
    needs_features = any(cfg.distillation_config.weight_mse > 0 for cfg in resolved)

    tcfg = base_config.trainer_config
    input_prefix = getattr(base_config.teacher_config, "input_prefix", "")
    add_special_tokens = getattr(base_config.teacher_config, "add_special_tokens", True)

    if needs_logits or needs_features:
        cache_base = getattr(tcfg, "cache_base_dir", None) or project_path
        teacher_bs = getattr(tcfg, "teacher_batch_size", None) or tcfg.batch_size
        print(
            f"[parallel] warming teacher logits/features cache "
            f"(needs_logits={needs_logits}, needs_features={needs_features})..."
        )
        precompute_teacher_logits(
            ctx.teacher_tokenizer,
            ctx.teacher_model,
            ctx.X_train,
            teacher_bs,
            tcfg.device,
            tcfg.max_len,
            needs_logits=needs_logits,
            needs_features=needs_features,
            input_prefix=input_prefix,
            add_special_tokens=add_special_tokens,
            project_path=cache_base,
            teacher_parent_dir=base_config.teacher_parent_dir,
            task_name=task_name,
            teacher_ckpt=ctx.teacher_ckpt,
            use_cache=True,
        )

    # Warm the teacher-eval cache (teacher_evaluation.json next to teacher_ckpt) too:
    # every worker's train_student calls evaluate_and_log_teacher BEFORE training, and
    # with the _NoOpTeacher stub it MUST hit that cache rather than forward the teacher.
    # evaluate_and_log_teacher logs to wandb on the compute path, so guard it with a
    # throwaway run + a throwaway run_dir (its per-run copy lands there, never colliding
    # with any real config's run_dir).
    print("[parallel] warming teacher-eval cache (teacher_evaluation.json)...")
    warm_run_dir = os.path.join(tcfg.output_dir, "_parallel_warm", f"{task_name}_teacher_eval")
    os.makedirs(warm_run_dir, exist_ok=True)
    try:
        wandb.init(
            project=getattr(tcfg, "wandb_project", None),
            name=f"_warm/{task_name}/teacher_eval",
            dir=project_output_path,
            mode="disabled",
            reinit=True,
        )
    except Exception:
        pass
    try:
        evaluate_and_log_teacher(
            ctx.teacher_model,
            ctx.teacher_tokenizer,
            ctx.X_test,
            ctx.y_test,
            task_name,
            base_config,
            warm_run_dir,
            ctx.teacher_ckpt,
        )
    finally:
        try:
            if wandb.run is not None:
                wandb.finish(exit_code=0)
        except Exception:
            pass


def _distill_task_batch_parallel(
    base_config: DistillationExperimentConfig,
    task_name: str,
    config_overrides_list: List[dict],
    parallel: int,
    prebuilt_ctx: Optional["TaskContext"] = None,
):
    """Config-parallel batch path (parallel > 1).

    Steps: (1) prepare_task ONCE; (2) WARM the HP-independent teacher caches ONCE with
    the LIVE teacher via a MINIMAL precompute + teacher-eval (NO full training of any
    config) -- skipped entirely when every config is pure-CE; (3) free the 3B teacher
    from GPU; (4) serialize the teacher-free ctx; (5) spawn ``parallel`` workers over
    ALL configs (0..n-1), each reloading its own ctx copy and running the SAME
    train_student. Per-config errors are isolated. Faithful to serial: same prepare,
    same seeds, same cached teacher values, same data -- and crucially EVERY config
    (including config-0) gets its FULL training in the pool, not a serial warm-up run.
    Only concurrency differs.

    ``prebuilt_ctx`` lets the fork path hand over the ctx it ALREADY built (when it fell back to
    spawn because the teacher could not be cache-skipped), so we don't redo prepare_task / reload
    the 3B teacher. None -> build it here as before (the normal spawn invocation is unchanged).
    """
    print(f"\n{'#' * 80}")
    print(
        f"### PARALLEL BATCH distillation for task={task_name}: "
        f"{len(config_overrides_list)} configs, parallel={parallel} ###"
    )
    print(f"{'#' * 80}")

    if prebuilt_ctx is not None:
        ctx = prebuilt_ctx
    else:
        prep_config = _seed_prep_config(base_config, config_overrides_list)
        # Exact skip-the-load gate (union across the sweep), same as the serial path.
        ctx = _call_prepare_task(prep_config, task_name, config_overrides_list)
    if ctx is None:
        print(f"[!] prepare_task returned None for {task_name}; nothing to run.")
        return

    n = len(config_overrides_list)
    results = {}  # index -> (ok, error_repr)

    # --- (2) MINIMAL teacher-cache warm with the LIVE teacher (no full training).
    # Only when at least one config actually needs the teacher (KL or MSE). Pure-CE
    # batches skip the warm entirely -- exactly the serial per-config behavior, where
    # precompute is skipped for weight_kl==0 AND weight_mse==0.
    any_needs_teacher = any(
        _config_needs_teacher(_apply_override(base_config, ov)) for ov in config_overrides_list
    )
    # If prepare_task already skipped the 3B load (caches valid), the teacher is the
    # _NoOpTeacher stub and every cache the warm would populate is ALREADY present — so
    # the warm is a no-op (it would only re-validate the same caches). Skip it; this is
    # also what keeps the stub from ever being forwarded.
    teacher_load_skipped = isinstance(ctx.teacher_model, _NoOpTeacher)
    if teacher_load_skipped:
        print("[parallel] teacher load was skipped (caches valid); no warm needed.")
    elif any_needs_teacher:
        _warm_teacher_cache(base_config, task_name, ctx, config_overrides_list)
    else:
        print("[parallel] no config needs the teacher (all pure-CE); skipping warm.")

    # --- (3) Free the 3B teacher from GPU before workers spawn.
    _free_teacher(ctx)

    # --- (4) Serialize the teacher-free ctx to disk for the workers.
    cache_path = os.path.join(
        base_config.trainer_config.output_dir,
        "_parallel_ctx_cache",
        f"{task_name}_taskctx.pt",
    )
    _serialize_task_ctx(ctx, cache_path)
    print(f"[parallel] cached teacher-free TaskContext -> {cache_path}")

    # --- (5) Run ALL configs (0..n-1) in a spawn-based worker pool. Every config --
    # including config-0 -- gets its FULL training here; nothing is run serially.
    work = [(i, base_config, task_name, config_overrides_list[i], cache_path) for i in range(n)]
    if work:
        pool = _spawn_pool(parallel)
        try:
            for index, ok_flag, err in pool.imap_unordered(_run_one_config_worker, work):
                results[index] = (ok_flag, err)
                if ok_flag:
                    print(f"[parallel] config {index + 1}/{n} OK")
                else:
                    print(f"[parallel] config {index + 1}/{n} FAILED:\n{err}")
        finally:
            pool.close()
            pool.join()

    n_ok = sum(1 for ok_flag, _ in results.values() if ok_flag)
    n_fail = sum(1 for ok_flag, _ in results.values() if not ok_flag)
    print(f"\n{'#' * 80}")
    print(
        f"### PARALLEL BATCH done for task={task_name}: "
        f"{n_ok} ok, {n_fail} failed (parallel={parallel}) ###"
    )
    print(f"{'#' * 80}")


# ---------------------------------------------------------------------------
# FORK-based config-parallel path (parallel > 1, parallel_mode="fork"). Shares the
# CPU data + teacher logits/features across all of a task's configs via Linux
# copy-on-write so memory stays ~1× per task. Additive; never touched when
# parallel == 1 or parallel_mode == "spawn".
# ---------------------------------------------------------------------------


def _cuda_context_exists() -> bool:
    """True iff a CUDA DRIVER context exists on any device in THIS process (never creates one).

    This is the reliable fork-poison detector. The trap we hit on the box: an import-time
    ``torch.cuda.is_available()`` (the ``DistillTrainerConfig.device`` field default) runs the CUDA
    Runtime API ``cudaGetDeviceCount`` -> ``cuInit``, which CREATES a primary driver context and
    poisons any later ``fork`` -- yet ``torch.cuda.is_initialized()`` STILL returns False, because
    that flag only tracks PyTorch's *Python-level* lazy init (``_lazy_init``), NOT the driver
    ``cuInit``. So a guard built on ``is_initialized()`` alone misses exactly this case (it did).

    ``torch._C._cuda_hasPrimaryContext(dev)`` queries the driver for an EXISTING primary context
    WITHOUT creating one -> it catches the ``cuInit`` poisoning that ``is_initialized()`` cannot.
    We enumerate devices via NVML (``device_count`` under ``PYTORCH_NVML_BASED_CUDA_CHECK=1``, set
    fork-safely at import) so the enumeration itself never calls ``cuInit``. Defense in depth:
    ``is_initialized()`` OR ``_is_in_bad_fork()`` OR any primary context => treat as initialized.

    Conservative by construction: ANY uncertainty (an exception while probing) returns True so the
    caller falls back to the always-safe spawn path rather than risk a poisoned fork. Returns False
    only when we can POSITIVELY confirm no context exists. Returns False if torch is unavailable.
    """
    try:
        import torch
    except Exception:
        return False  # no torch -> nothing could have created a CUDA context.
    try:
        if torch.cuda.is_initialized():
            return True  # PyTorch's own lazy init already ran.
    except Exception:
        return True  # cannot tell -> assume poisoned (fall back to spawn).
    try:
        if torch.cuda._is_in_bad_fork():
            return True
    except Exception:
        pass
    # Probe the driver for an existing PRIMARY context on each device WITHOUT creating one. We
    # need a device count that does NOT call cuInit; device_count() uses NVML when
    # PYTORCH_NVML_BASED_CUDA_CHECK=1 (set at import) and is fork-safe.
    has_primary = getattr(torch._C, "_cuda_hasPrimaryContext", None)
    if has_primary is None:
        # Old torch without the probe: we cannot positively confirm cleanliness -> be conservative.
        return False if not _nvml_cuda_present() else True
    try:
        n = torch.cuda.device_count()  # NVML-based (no cuInit) under the env var set at import.
    except Exception:
        return True  # cannot enumerate safely -> assume poisoned.
    for dev in range(n):
        try:
            if has_primary(dev):
                return True
        except Exception:
            return True  # probe failed -> assume poisoned.
    return False  # positively confirmed: no primary context on any device.


def _nvml_cuda_present() -> bool:
    """True iff NVML reports >=1 CUDA device, WITHOUT calling ``cuInit`` (fork-safe).

    Used only on torch builds lacking ``_cuda_hasPrimaryContext`` to decide the conservative
    default: if a GPU is present we cannot prove the context is clean, so we treat it as poisoned.
    """
    try:
        import torch

        os.environ.setdefault("PYTORCH_NVML_BASED_CUDA_CHECK", "1")
        return torch.cuda.device_count() > 0
    except Exception:
        return True  # uncertain -> conservative.


# Back-compat alias: the guard used to be named ``_cuda_is_initialized``. Kept so existing call
# sites / tests referencing the old name still resolve to the hardened detector.
_cuda_is_initialized = _cuda_context_exists


def _assert_no_cuda_before_fork():
    """Raise a clear error if a CUDA context exists right before we fork workers.

    A hard guard backing the fork-safety argument: the fork dispatcher only proceeds after the
    skip-load fast path (teacher cache valid -> 3B teacher never loaded -> no CUDA init). If that
    invariant is violated we must fail loudly here rather than fork a CUDA-poisoned process and
    corrupt every child silently. The dispatcher checks ``_cuda_is_initialized`` first and falls
    back to spawn; this assert is the defense-in-depth that turns any remaining violation into an
    immediate, named error instead of undefined CUDA behavior.
    """
    if _cuda_is_initialized():
        raise RuntimeError(
            "Refusing to fork: a CUDA context is already initialized in the parent. "
            "Fork-parallel requires the teacher to have been cache-skipped (no 3B load, no CUDA "
            "init) so children can initialize CUDA fresh post-fork. Use parallel_mode='spawn' "
            "instead, or warm the teacher caches so prepare_task can skip the load."
        )


def _preload_teacher_arrays_for_fork(
    base_config: DistillationExperimentConfig,
    task_name: str,
    ctx: "TaskContext",
    config_overrides_list: List[dict],
) -> None:
    """Load the teacher logits/features cache ONCE into ``ctx`` (CPU) for fork COW sharing.

    The fork children share these read-only arrays via copy-on-write instead of each re-reading
    the on-disk cache (which would copy the whole feature tensor per child -> N× memory). We load
    EXACTLY what the sweep needs (the union of logits/features across all configs, same gate the
    serial path uses) and ONLY when the cache is already valid on disk (it must be: fork is only
    chosen after the skip-load fast path, which itself validated these caches). On any miss we
    leave the fields None and the children fall back to the normal per-child cache read (correct,
    just not shared) -- never silently wrong. Stores numpy arrays (what ``SeqDataset`` accepts and
    copies into fresh tensors), so children never mutate the shared parent arrays.

    No CUDA is touched here (pure disk read of cached numpy) -> the no-CUDA-before-fork invariant
    is preserved.
    """
    import numpy as np  # noqa: F401  (kept for clarity; np.load below uses it)
    from src.trainer.utils import _get_cache_dir, _load_cache, _validate_cache
    from config.env import project_path

    resolved = [_apply_override(base_config, ov) for ov in config_overrides_list]
    needs_logits = any(cfg.distillation_config.weight_kl > 0 for cfg in resolved)
    needs_features = any(cfg.distillation_config.weight_mse > 0 for cfg in resolved)
    if not (needs_logits or needs_features):
        return  # pure-CE sweep: no teacher arrays needed at all.

    tcfg = base_config.trainer_config
    cache_base = getattr(tcfg, "cache_base_dir", None) or project_path
    cache_dir = _get_cache_dir(cache_base, base_config.teacher_parent_dir, task_name)
    logits, features, metadata = _load_cache(cache_dir, needs_features)
    if metadata is None or not _validate_cache(
        metadata, ctx.X_train, ctx.teacher_ckpt, tcfg.max_len
    ):
        # Cache not present/valid -> let each child precompute/read normally (fall through).
        print(
            "[fork] teacher cache not preloadable (missing/stale); children will read it per-config."
        )
        return
    ctx.teacher_logits = logits if needs_logits else None
    ctx.teacher_features = features if needs_features else None
    _log = []
    if ctx.teacher_logits is not None:
        _log.append(f"logits{list(ctx.teacher_logits.shape)}")
    if ctx.teacher_features is not None:
        _log.append(f"features{list(ctx.teacher_features.shape)}")
    print(f"[fork] preloaded teacher arrays for COW sharing: {', '.join(_log) or 'none'}")


def _run_one_config_fork_worker(args):
    """Fork-worker entry: train ONE config against the SHARED (COW) ctx, isolated.

    Unlike the spawn worker, this receives the LIVE in-memory ``ctx`` (inherited via fork COW), so
    it does NOT re-read the data/teacher cache from disk -- the big arrays are shared with the
    parent. To guarantee a child never corrupts the shared parent state, it trains on a SHALLOW
    COPY of the ctx (``dataclasses.replace`` with no changes): the big tensors/lists are shared by
    reference (read-only -- SeqDataset copies them into fresh tensors), while the only field
    ``train_student`` would ever rebind (none today; it ``replace``s the CONFIG, not the ctx) is
    insulated. Seeds via the SAME per-config path as serial (``set_seed`` inside ``train_student``).
    Returns ``(index, ok, error_repr)`` so the parent records successes/failures without one bad
    config aborting siblings.
    """
    import dataclasses

    index, base_config, task_name, override, ctx = args
    try:
        import torch
    except Exception:
        torch = None
    try:
        # Shallow per-child copy: shares the big read-only arrays, isolates any field rebinding.
        child_ctx = dataclasses.replace(ctx)
        cfg = _apply_override(base_config, override)
        train_student(cfg, task_name, child_ctx)
        return (index, True, None)
    except Exception:
        import traceback

        return (index, False, traceback.format_exc())
    finally:
        try:
            if wandb.run is not None:
                wandb.finish(exit_code=0)
        except Exception:
            pass
        import gc

        gc.collect()
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()


def _fork_pool(parallel: int):
    """Return a ``multiprocessing`` Pool over the FORK context.

    Fork (not spawn) is the whole point: the workers INHERIT the parent's already-loaded CPU data
    + teacher arrays via copy-on-write (shared, not re-loaded). This is ONLY safe because the
    caller guarantees NO CUDA context exists in the parent before the pool is built (see
    ``_assert_no_cuda_before_fork``). Isolated so tests can monkeypatch it with a synchronous
    executor (and so the fork context is created in exactly one place).
    """
    import multiprocessing as mp

    fork_ctx = mp.get_context("fork")
    return fork_ctx.Pool(processes=parallel)


def _distill_task_batch_fork(
    base_config: DistillationExperimentConfig,
    task_name: str,
    config_overrides_list: List[dict],
    parallel: int,
):
    """FORK config-parallel batch path (parallel > 1, parallel_mode='fork'): shared-memory.

    Steps: (1) prepare_task ONCE; (2) REQUIRE the teacher to have been cache-skipped (no 3B load
    -> no CUDA context) -- if not, fall back to the spawn path automatically (logged), since fork
    after CUDA init is unsafe; (3) preload the teacher logits/features cache ONCE into the ctx as
    shared CPU arrays; (4) assert no CUDA context exists, then fork ``parallel`` workers over ALL
    configs (0..n-1). Because the fork happens BEFORE any CUDA init and the big CPU arrays are only
    READ by children, Linux copy-on-write shares them -> memory ~1× (data + features) per task
    regardless of ``parallel``. Each child inits CUDA fresh post-fork and runs the SAME
    ``train_student`` on a shallow ctx copy, reading the shared arrays. Per-config errors are
    isolated. Faithful to serial: same prepare, same per-config seeds, same teacher values (same
    cache), same data, same batch order -- only concurrency differs.
    """
    print(f"\n{'#' * 80}")
    print(
        f"### FORK PARALLEL BATCH for task={task_name}: "
        f"{len(config_overrides_list)} configs, parallel={parallel} ###"
    )
    print(f"{'#' * 80}")

    prep_config = _seed_prep_config(base_config, config_overrides_list)
    # Exact skip-the-load gate (union across the sweep), same as serial/spawn.
    ctx = _call_prepare_task(prep_config, task_name, config_overrides_list)
    if ctx is None:
        print(f"[!] prepare_task returned None for {task_name}; nothing to run.")
        return

    # --- (2) Fork is ONLY safe when the teacher was cache-skipped (no 3B load -> no CUDA context).
    # If the 3B teacher WAS loaded (caches missing/stale), CUDA is now initialized and forking would
    # corrupt every child. Fall back to the audited spawn path automatically (which warms the cache,
    # frees the teacher, and uses fresh-Python spawn workers). Decision is logged.
    teacher_load_skipped = isinstance(ctx.teacher_model, _NoOpTeacher)
    if not teacher_load_skipped or _cuda_is_initialized():
        print(
            "[fork] WARNING: teacher was NOT cache-skipped (CUDA may be initialized in the parent); "
            "fork is unsafe -> falling back to parallel_mode='spawn'."
        )
        # The freshly-loaded teacher in ctx is reused by the spawn path's warm step.
        return _distill_task_batch_parallel(
            base_config, task_name, config_overrides_list, parallel, prebuilt_ctx=ctx
        )

    # --- (3) Preload the teacher logits/features cache ONCE into the ctx for COW sharing.
    _preload_teacher_arrays_for_fork(base_config, task_name, ctx, config_overrides_list)

    # --- (4) Hard guard: no CUDA context may exist before we fork (defense-in-depth).
    _assert_no_cuda_before_fork()

    n = len(config_overrides_list)
    results = {}  # index -> (ok, error_repr)
    # Workers inherit `ctx` via fork COW; we hand it directly (no disk serialization needed).
    work = [(i, base_config, task_name, config_overrides_list[i], ctx) for i in range(n)]
    if work:
        pool = _fork_pool(parallel)
        try:
            for index, ok_flag, err in pool.imap_unordered(_run_one_config_fork_worker, work):
                results[index] = (ok_flag, err)
                if ok_flag:
                    print(f"[fork] config {index + 1}/{n} OK")
                else:
                    print(f"[fork] config {index + 1}/{n} FAILED:\n{err}")
        finally:
            pool.close()
            pool.join()

    n_ok = sum(1 for ok_flag, _ in results.values() if ok_flag)
    n_fail = sum(1 for ok_flag, _ in results.values() if not ok_flag)
    print(f"\n{'#' * 80}")
    print(
        f"### FORK PARALLEL BATCH done for task={task_name}: "
        f"{n_ok} ok, {n_fail} failed (parallel={parallel}) ###"
    )
    print(f"{'#' * 80}")


def _apply_override(
    base_config: DistillationExperimentConfig,
    override: dict,
) -> DistillationExperimentConfig:
    """Apply one HP override dict to ``base_config`` via ``dataclasses.replace``.

    Any key whose value is a dict and that names a nested dataclass field on
    ``base_config`` (e.g. ``distillation_config``, ``trainer_config``) is replaced
    field-by-field onto that nested config; all other keys are treated as
    top-level DistillationExperimentConfig fields (e.g. ``random_state``). Returns
    a new config; ``base_config`` is NOT mutated, so the shared teacher/data
    context remains valid across configs.
    """
    override = dict(override)
    cfg = base_config
    # Nested-dataclass overrides: any key mapping to a dict that is an existing
    # config attribute is merged field-by-field (so we don't replace the whole
    # nested config, only the named fields). This generalizes beyond
    # distillation_config (e.g. trainer_config.early_stop_patience).
    for key in list(override.keys()):
        val = override[key]
        if isinstance(val, dict) and hasattr(cfg, key):
            nested = getattr(cfg, key)
            cfg = replace(cfg, **{key: replace(nested, **val)})
            override.pop(key)
    if override:
        cfg = replace(cfg, **override)
    return cfg


# Modes the selector understands. Documented here (and in the module ``MODES`` block) so a
# single source of truth drives both ``run_distillation`` and the ``--mode`` CLI choices.
MODES = ("slurm", "batch")


def run_distillation(
    mode: str,
    *,
    config: Optional[DistillationExperimentConfig] = None,
    base_config: Optional[DistillationExperimentConfig] = None,
    task_name: Optional[str] = None,
    config_overrides_list: Optional[List[dict]] = None,
    parallel: int = 1,
    parallel_mode: str = "fork",
):
    """Single documented selector that routes a ``mode`` to the matching DISPATCH path.

    This is the ONE place a caller switches execution strategy; the shared CORE
    (``prepare_task`` + ``train_student``) is identical regardless of ``mode``. It does
    NOT replace the legacy entry points -- ``python -m src.train.distill`` (slurm) and
    ``python -m src.train.distill_task`` (batch) still work byte-identically; this just
    gives both a common, self-documenting front door (see ``src.train.distill_run``).

    Args:
        mode: ``"slurm"`` (one HP config per process; scheduler-driven fan-out) or
            ``"batch"`` (single box loads the teacher ONCE and sweeps the config list).
        config: for ``mode="slurm"`` ONLY -- the fully-resolved single experiment config
            (with ``task_names`` set); routed verbatim through ``main`` so the SLURM
            per-config dispatch (``distill[slurm_config]``) is unchanged.
        base_config, task_name, config_overrides_list, parallel: for ``mode="batch"``
            ONLY -- the seed config, the task, its HP override list, and the concurrency
            knob (``1`` = serial loop, ``>1`` = config-parallel). Routed verbatim through
            ``distill_task_batch`` so both batch sub-modes are unchanged.

    Raises:
        ValueError: on an unknown ``mode`` or missing mode-required arguments.
    """
    if mode == "slurm":
        if config is None:
            raise ValueError("mode='slurm' requires `config` (a resolved experiment config).")
        return main(config)
    if mode == "batch":
        if base_config is None or task_name is None or config_overrides_list is None:
            raise ValueError(
                "mode='batch' requires `base_config`, `task_name`, and `config_overrides_list`."
            )
        return distill_task_batch(
            base_config,
            task_name,
            config_overrides_list,
            parallel=parallel,
            parallel_mode=parallel_mode,
        )
    raise ValueError(f"Unknown mode {mode!r}; expected one of {MODES}.")


def main(config: DistillationExperimentConfig):
    for task_name in config.task_names:
        # Update dataset config with current task
        distill_dataset_config = replace(config.dataset_config, task_name=task_name)
        distill_config = replace(config, dataset_config=distill_dataset_config)

        # Run distillation with SLURM if configured
        distill[distill_config.slurm_config](distill_config, task_name)


if __name__ == "__main__":
    config = tyro.extras.overridable_config_cli(configs, sort_subcommands=True)
    main(config)
