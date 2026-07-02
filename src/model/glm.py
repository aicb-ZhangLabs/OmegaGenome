import os
import csv
import json
import wandb
import torch
from dataclasses import dataclass, replace
from typing import Optional
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel

from torch.utils.data import DataLoader, Dataset
from ..data.dataset import (
    get_num_labels,
)
from ..trainer.utils import (
    evaluate_teacher_mcc,
    orig_get_best_checkpoint,
)
from datetime import datetime


@dataclass
class GLMConfig:
    model_name_or_path: str  # Can be either full model path or LoRA adapter path
    num_labels: int = 2
    ckpt_path: Optional[str] = None
    output_hidden_states: bool = True  # For feature extraction
    trust_remote_code: bool = True

    # Teacher-specific input formatting (e.g. Carbon needs the "<dna>" tag and
    # add_special_tokens=False). No-op defaults keep existing teachers unaffected.
    input_prefix: str = ""
    add_special_tokens: bool = True

    # Load dtype for the teacher ("bfloat16"/"float16"/None=fp32). Large teachers (Carbon-3B) need
    # bf16 to fit + it matches how they were fine-tuned. None keeps existing teachers at fp32.
    torch_dtype: Optional[str] = None

    # LoRA-specific fields
    base_model_path: Optional[str] = None  # Base model for LoRA adapters
    is_lora: Optional[bool] = None  # Auto-detect if None
    merge_lora: bool = True  # Merge adapter weights for faster inference

    def __post_init__(self):
        """Auto-detect if model is LoRA adapter if not specified."""
        if self.is_lora is None:
            import os

            adapter_config_path = os.path.join(self.model_name_or_path, "adapter_config.json")
            self.is_lora = os.path.exists(adapter_config_path)

            # If LoRA and base_model_path not provided, try to read from adapter_config
            if self.is_lora and self.base_model_path is None:
                try:
                    import json

                    with open(adapter_config_path, "r") as f:
                        adapter_config = json.load(f)
                    self.base_model_path = adapter_config.get("base_model_name_or_path")
                    if self.base_model_path:
                        print(f"Auto-detected base model: {self.base_model_path}")
                except Exception as e:
                    print(f"Warning: Could not read base model from adapter_config.json: {e}")


def build_glm(config: GLMConfig):
    """
    Build GLM model, handling both regular and LoRA fine-tuned models.
    """
    model_path = config.ckpt_path if config.ckpt_path else config.model_name_or_path

    _dtype = getattr(torch, config.torch_dtype) if config.torch_dtype else None  # None = fp32

    if config.is_lora:
        print(f"Loading LoRA adapter from: {model_path}")

        if not config.base_model_path:
            raise ValueError(
                "LoRA adapter detected but base_model_path is not set. "
                "Please specify base_model_path in GLMConfig."
            )

        print(f"Loading base model: {config.base_model_path}")

        # Load tokenizer from base model
        tokenizer = AutoTokenizer.from_pretrained(
            config.base_model_path, trust_remote_code=config.trust_remote_code
        )

        # Load base model
        base_model = AutoModelForSequenceClassification.from_pretrained(
            config.base_model_path,
            num_labels=config.num_labels,
            output_hidden_states=config.output_hidden_states,
            trust_remote_code=config.trust_remote_code,
            dtype=_dtype,
        )

        # Load LoRA adapter
        model = PeftModel.from_pretrained(base_model, model_path)

        # Merge adapter weights for faster inference
        if config.merge_lora:
            print("Merging LoRA weights into base model...")
            model = model.merge_and_unload()

    else:
        # Regular model loading (not LoRA)
        print(f"Loading regular model from: {model_path}")
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=config.trust_remote_code
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            num_labels=config.num_labels,
            output_hidden_states=config.output_hidden_states,
            trust_remote_code=config.trust_remote_code,
            dtype=_dtype,
        )
        # DNABERT-2's bundled flash-attn Triton kernel calls tl.dot(..., trans_b=True), which triton>=3
        # removed -> CompilationError at the teacher forward. bert_layers.py falls back to standard
        # (mathematically identical) attention when its module-global flash_attn_qkvpacked_func is None,
        # so null it out on the dynamically-loaded remote-code module. No-op for non-DNABERT-2 teachers.
        import sys as _sys
        for _mn, _mod in list(_sys.modules.items()):
            if _mn.endswith(".bert_layers") and getattr(_mod, "flash_attn_qkvpacked_func", None) is not None:
                _mod.flash_attn_qkvpacked_func = None
                print(f"[glm] disabled DNABERT-2 triton flash-attn in {_mn} -> standard-attn fallback")

    # Force the load dtype. trust_remote_code models (Carbon) can ignore from_pretrained's dtype and
    # stay fp32 (-> 12GB weights -> OOM), so cast explicitly. No-op when _dtype is None (other teachers).
    if _dtype is not None:
        model = model.to(_dtype)

    # Some autoregressive tokenizers (e.g. Carbon) define no pad token, which breaks batched
    # tokenization. Use eos as pad (masked by attention_mask -> predictions unchanged).
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    # Independently, the MODEL needs pad_token_id for batched sequence-classification pooling (it
    # locates the last non-pad token). Set it whenever the model lacks one but the tokenizer has it —
    # NOT gated on the tokenizer having just been patched (the tokenizer may already carry a pad token
    # while the model config's pad_token_id is None, which still crashes the forward at batch > 1).
    if getattr(model.config, "pad_token_id", None) is None and getattr(tokenizer, "pad_token_id", None) is not None:
        model.config.pad_token_id = tokenizer.pad_token_id

    return tokenizer, model


def get_best_checkpoint(parent_path: str, task_name: str, model_type: str = "default"):
    """Find best checkpoint for a task - supports Carbon-LoRA, GLM, and NT directory structures."""
    import re

    # Explicit per-teacher dispatch (clearer + each layout is isolated):
    #   NT   -> {parent}/finetuned_models/{task}_finetuned/model-best_mcc_score_X/
    #   GLM  -> Carbon-LoRA {parent}/{task}_finetuned/ (PEFT adapter), else {parent}/{task}/checkpoint-N/
    # (Enformer/Caduceus never reach here — find_teacher_checkpoint routes them to their own finders.)
    if "NT" in model_type or "nucleotide" in parent_path.lower():
        # NT checkpoint structure: finetuned_models/{task}_finetuned/model-best*mcc_score*
        task_dir = os.path.join(parent_path, "finetuned_models", f"{task_name}_finetuned")
        if not os.path.isdir(task_dir):
            return None, -1.0

        best_score = -1.0
        best_dir = None
        for d in os.listdir(task_dir):
            if d.startswith("model-best") and "mcc_score" in d:
                try:
                    match = re.search(r"mcc_score([\d.]+)", d)
                    if match:
                        score = float(match.group(1))
                        if score > best_score:
                            best_score, best_dir = score, os.path.join(task_dir, d)
                except (ValueError, AttributeError):
                    continue
        return best_dir, best_score
    else:
        # GLM family. Carbon-LoRA teachers save the PEFT adapter directly under {task}_finetuned/
        # (adapter_config.json present); the original GLM layout is {parent}/{task}/checkpoint-N/.
        lora_dir = os.path.join(parent_path, f"{task_name}_finetuned")
        if os.path.isfile(os.path.join(lora_dir, "adapter_config.json")):
            return lora_dir, -1.0
        # Original GLM checkpoint structure
        return orig_get_best_checkpoint(parent_path, task_name), -1.0


def get_teacher_model(config, task_name, teacher_ckpt):
    """
    Unified teacher model loading that supports GLM/DNABert2, NT, and Caduceus.

    This replaces the direct build_glm call to support multiple model types.

    Returns:
        tuple: (tokenizer, teacher_model, teacher_hidden)
    """
    model_type = getattr(config, "model_type", "glm")  # Default to 'glm' for backward compatibility
    if model_type == "enformer":
        # Import Enformer utilities only when needed
        from ..model.enformer import load_enformer_model, EnformerTokenizer

        num_labels = get_num_labels(task_name)

        # For Enformer, teacher_ckpt is the checkpoint file path
        # Extract enformer_dim and target_length from config if available
        enformer_dim = getattr(config.teacher_config, "enformer_dim", 1536)
        target_length = getattr(config.teacher_config, "target_length", 8)

        wrapped_model, _, base_model = load_enformer_model(
            teacher_ckpt,
            num_labels,
            config.trainer_config.device,
            enformer_dim=enformer_dim,
            target_length=target_length,
        )

        # Create dummy tokenizer for interface compatibility
        tokenizer = EnformerTokenizer()

        # Extract teacher hidden size
        teacher_hidden = wrapped_model.hidden_dim

        return tokenizer, wrapped_model, teacher_hidden

    elif model_type == "caduceus":
        # Import Caduceus utilities only when needed
        from ..trainer.utils import load_caduceus_model

        num_labels = get_num_labels(task_name)

        # For Caduceus, teacher_ckpt is the directory path
        # Check if there's a specific checkpoint file
        import re

        best_ckpt_file = None
        if os.path.isdir(teacher_ckpt):
            pattern = re.compile(r"epoch(\d+)_valmcc_(-?[0-9\.]+)\.pt")
            for filename in os.listdir(teacher_ckpt):
                if pattern.match(filename):
                    best_ckpt_file = os.path.join(teacher_ckpt, filename)
                    break

        wrapped_model, tokenizer, base_model = load_caduceus_model(
            teacher_ckpt, num_labels, config.trainer_config.device, best_ckpt_file
        )

        # # Extract teacher hidden size
        # if hasattr(wrapped_model, "hidden_dim") and wrapped_model.hidden_dim:
        #     teacher_hidden = wrapped_model.hidden_dim
        # else:
        #     teacher_hidden = 256  # Default for Caduceus
        # Extract teacher hidden size with fallback to dummy forward pass
        teacher_hidden = None
        if hasattr(wrapped_model, "hidden_dim") and wrapped_model.hidden_dim:
            teacher_hidden = wrapped_model.hidden_dim
            print(f"Got teacher_hidden={teacher_hidden} from wrapped_model.hidden_dim")

        # If still None, do a dummy forward pass to get the actual dimension
        if teacher_hidden is None:
            print(
                "Warning: hidden_dim not set, doing dummy forward pass to determine feature dimension"
            )
            dummy_input = torch.zeros(1, 10, dtype=torch.long, device=config.trainer_config.device)
            dummy_mask = torch.ones(1, 10, dtype=torch.long, device=config.trainer_config.device)

            with torch.no_grad():
                try:
                    _, dummy_features = wrapped_model(
                        dummy_input, attention_mask=dummy_mask, return_features=True
                    )
                    teacher_hidden = dummy_features.shape[-1]
                    print(f"Determined teacher_hidden={teacher_hidden} from dummy forward pass")
                except Exception as e:
                    print(f"Error during dummy forward pass: {e}")
                    teacher_hidden = 256  # Absolute fallback

        if teacher_hidden is None:
            teacher_hidden = 256  # Absolute fallback
            print(f"Using fallback teacher_hidden={teacher_hidden}")

        # Return in format compatible with existing code
        return tokenizer, wrapped_model, teacher_hidden

    else:  # Default GLM/NT path - unchanged
        teacher_config = replace(config.teacher_config, ckpt_path=teacher_ckpt)
        teacher_tokenizer, teacher_model = build_glm(teacher_config)

        # Extract teacher hidden size
        if hasattr(teacher_model, "config"):
            teacher_hidden = teacher_model.config.hidden_size
        else:
            teacher_hidden = 768  # Default fallback

        return teacher_tokenizer, teacher_model, teacher_hidden


def find_teacher_checkpoint(config, task_name):
    """
    Find the best teacher checkpoint based on model type.

    Returns: (checkpoint_path, score)
    """
    model_type = getattr(config, "model_type", "glm")
    if model_type == "enformer":
        # Import Enformer utilities only when needed
        from ..model.enformer import find_best_enformer_checkpoint

        checkpoint_path, score = find_best_enformer_checkpoint(task_name, config.teacher_parent_dir)
        return checkpoint_path, score

    elif model_type == "caduceus":
        # Import Caduceus utilities only when needed
        from ..trainer.utils import find_best_caduceus_checkpoint

        checkpoint_path, score, _ = find_best_caduceus_checkpoint(
            task_name, config.teacher_parent_dir
        )
        return checkpoint_path, score

    else:  # Default GLM/NT path
        # Determine if NT or GLM
        is_nt = "nucleotide" in config.teacher_config.model_name_or_path.lower()
        model_type_str = "NT" if is_nt else "GLM"

        if config.teacher_parent_dir:
            return get_best_checkpoint(
                config.teacher_parent_dir, task_name, model_type=model_type_str
            )
        else:
            teacher_ckpt = orig_get_best_checkpoint(config.trainer_config.output_dir, task_name)
            return teacher_ckpt, -1.0


def _teacher_eval_cache_path(config, teacher_ckpt) -> str:
    """Resolve the teacher_evaluation.json path EXACTLY as evaluate_and_log_teacher does.

    Single source of truth for the per-checkpoint teacher-eval cache location so the
    skip-the-load check (``teacher_eval_cache_is_valid``) and the writer
    (``evaluate_and_log_teacher``) never drift: Enformer ckpts are .pt files (cache in
    the parent dir); every other teacher uses the checkpoint directory itself.
    """
    model_type = getattr(config, "model_type", "glm")
    if model_type == "enformer":
        cache_dir = os.path.dirname(teacher_ckpt)
    else:
        cache_dir = teacher_ckpt
    if os.path.isfile(cache_dir):
        cache_dir = os.path.dirname(cache_dir)
    return os.path.join(cache_dir, "teacher_evaluation.json")


def teacher_eval_cache_is_valid(config, teacher_ckpt) -> bool:
    """True iff a cached teacher_evaluation.json would be a HIT in evaluate_and_log_teacher.

    Mirrors the cache-read guard inside ``evaluate_and_log_teacher`` byte-for-byte: the
    file must exist, parse, carry ``teacher_test_mcc``, and record the SAME
    ``teacher_checkpoint`` as ``teacher_ckpt``. A True result guarantees
    ``evaluate_and_log_teacher`` returns the cached MCC WITHOUT a teacher forward, so
    ``prepare_task`` can safely skip the 3B load when this (plus the logits/features
    cache) holds. Any read/parse/mismatch returns False (forces a real eval+load).
    """
    teacher_eval_file = _teacher_eval_cache_path(config, teacher_ckpt)
    if not os.path.exists(teacher_eval_file):
        return False
    try:
        with open(teacher_eval_file, "r") as f:
            data = json.load(f)
    except Exception:
        return False
    return data.get("teacher_checkpoint") == teacher_ckpt and "teacher_test_mcc" in data


def evaluate_and_log_teacher(
    teacher_model,
    teacher_tokenizer,
    X_test,
    y_test,
    task_name,
    config,
    run_dir,
    teacher_ckpt,
):
    """
    Evaluate teacher model on test set and log results.

    Returns:
        float: Teacher test MCC score
    """
    print(f"\n{'=' * 60}")
    print("Evaluating Teacher Model on Test Set")
    print(f"{'=' * 60}")
    # --- START: CACHE CHECK ---
    # --- START: DETERMINE CACHE DIRECTORY BASED ON MODEL TYPE ---
    # Resolve via the shared helper so the skip-the-3B-load check
    # (teacher_eval_cache_is_valid) and this reader/writer use the SAME path.
    teacher_eval_file = _teacher_eval_cache_path(config, teacher_ckpt)
    cache_dir = os.path.dirname(teacher_eval_file)
    # teacher_eval_file = os.path.join(teacher_ckpt, "teacher_evaluation.json")

    if os.path.exists(teacher_eval_file):
        try:
            with open(teacher_eval_file, "r") as f:
                teacher_eval_data = json.load(f)

            # Check if the cached checkpoint matches the current one
            if (
                teacher_eval_data.get("teacher_checkpoint") == teacher_ckpt
                and "teacher_test_mcc" in teacher_eval_data
            ):
                teacher_mcc = teacher_eval_data["teacher_test_mcc"]
                print(f"✓ Found cached teacher evaluation: {teacher_eval_file}")
                print(f"Cached Teacher Test MCC: {teacher_mcc:.4f}")
                print(f"{'=' * 60}\n")

                # Log to wandb (this is necessary as the original log is skipped)
                wandb.log(
                    {
                        "teacher/test_mcc": teacher_mcc,
                    }
                )
                return teacher_mcc
            else:
                print("Cached data is for a different checkpoint. Re-evaluating...")
        except Exception as e:
            print(
                f"Warning: Could not read cached teacher evaluation file. Re-evaluating. Error: {e}"
            )

    # --- END: CACHE CHECK ---
    # Create test dataset for teacher
    class SimpleTextDataset(Dataset):
        def __init__(self, texts, labels):
            self.texts = texts
            self.labels = labels

        def __len__(self):
            return len(self.texts)

        def __getitem__(self, idx):
            return {"text": self.texts[idx], "label": self.labels[idx]}

    # Create dataset and dataloader
    test_dataset = SimpleTextDataset(X_test, y_test)

    # Match the teacher's fine-tuning input format (e.g. Carbon's "<dna>" prefix + add_special_tokens
    # =False); defaults ("" / True) are a no-op for the other teachers. Also ensure a pad token exists
    # (autoregressive teachers ship none), else batched padding + classification pooling fail.
    _prefix = getattr(config.teacher_config, "input_prefix", "")
    _add_special = getattr(config.teacher_config, "add_special_tokens", True)
    if teacher_tokenizer.pad_token is None and teacher_tokenizer.eos_token is not None:
        teacher_tokenizer.pad_token = teacher_tokenizer.eos_token
    if getattr(teacher_model.config, "pad_token_id", None) is None and teacher_tokenizer.pad_token_id is not None:
        teacher_model.config.pad_token_id = teacher_tokenizer.pad_token_id

    # Collate function for teacher
    def collate_fn(batch):
        from ..trainer.utils import tokenize_teacher_inputs

        labels = [item["label"] for item in batch]
        # Same helper as precompute: applies prefix + add_special_tokens (HF tokenizers only), so a
        # custom-tokenizer teacher (Enformer) isn't passed an unsupported kwarg.
        encoded = tokenize_teacher_inputs(
            teacher_tokenizer,
            [item["text"] for item in batch],
            config.trainer_config.max_len,
            input_prefix=_prefix,
            add_special_tokens=_add_special,
        )

        # FIX: Handle missing attention_mask
        result = {
            "input_ids": encoded["input_ids"],
            "labels": torch.tensor(labels, dtype=torch.long),
        }

        # Only add attention_mask if it exists, otherwise create default
        if "attention_mask" in encoded:
            result["attention_mask"] = encoded["attention_mask"]
        else:
            result["attention_mask"] = torch.ones_like(encoded["input_ids"])

        return result

    # Robust against a minimal trainer_config wrapper that only carries `device`
    # (the config-driven DKD path builds such a wrapper): fall back to the JSON
    # teacher_batch_size, then batch_size, then a safe default of 4.
    _teacher_bs = (
        getattr(config.trainer_config, "teacher_batch_size", None)
        or getattr(config.trainer_config, "batch_size", None)
        or 4
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=_teacher_bs,
        shuffle=False,
        num_workers=getattr(config.trainer_config, "num_workers", 0),
        collate_fn=collate_fn,
    )

    # Evaluate
    teacher_mcc = evaluate_teacher_mcc(
        teacher_model,
        teacher_tokenizer,
        test_loader,
        config.trainer_config.device,
    )

    print(f"Teacher Test MCC: {teacher_mcc:.4f}")
    print(f"{'=' * 60}\n")

    # Save to file
    teacher_eval_file_run = os.path.join(run_dir, "teacher_evaluation.json")
    # teacher_eval_file = os.path.join(teacher_ckpt, "teacher_evaluation.json")
    teacher_eval_data = {
        "task": task_name,
        "teacher_checkpoint": teacher_ckpt,
        "teacher_test_mcc": float(teacher_mcc),
        "timestamp": datetime.now().isoformat(),
    }

    with open(teacher_eval_file, "w") as f:
        json.dump(teacher_eval_data, f, indent=2)
    with open(teacher_eval_file_run, "w") as f:
        json.dump(teacher_eval_data, f, indent=2)

    # Also append to a summary CSV for easy comparison across experiments
    summary_csv = os.path.join(config.trainer_config.output_dir, "teacher_scores_summary.csv")
    file_exists = os.path.exists(summary_csv)

    with open(summary_csv, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(
                [
                    "task",
                    "teacher_checkpoint",
                    "teacher_test_mcc",
                    "timestamp",
                    "run_dir",
                ]
            )
        writer.writerow(
            [
                task_name,
                teacher_ckpt,
                f"{teacher_mcc:.4f}",
                datetime.now().isoformat(),
                run_dir,
            ]
        )

    # Log to wandb
    wandb.log(
        {
            "teacher/test_mcc": teacher_mcc,
        }
    )

    return teacher_mcc
