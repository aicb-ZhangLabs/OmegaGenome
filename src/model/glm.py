from dataclasses import dataclass
from typing import Optional
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from dataclasses import dataclass
from typing import Optional


@dataclass
class GLMConfig:
    model_name_or_path: str  # Can be either full model path or LoRA adapter path
    num_labels: int = 2
    ckpt_path: Optional[str] = None
    output_hidden_states: bool = True  # For feature extraction
    trust_remote_code: bool = False  # For NT models

    # LoRA-specific fields
    base_model_path: Optional[str] = None  # Base model for LoRA adapters
    is_lora: Optional[bool] = None  # Auto-detect if None
    merge_lora: bool = True  # Merge adapter weights for faster inference

    def __post_init__(self):
        """Auto-detect if model is LoRA adapter if not specified."""
        if self.is_lora is None:
            import os

            adapter_config_path = os.path.join(
                self.model_name_or_path, "adapter_config.json"
            )
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
                    print(
                        f"Warning: Could not read base model from adapter_config.json: {e}"
                    )


# def build_glm(config: GLMConfig):
#     """Build GLM or NT model - unified interface for all transformer models"""
#     tokenizer = AutoTokenizer.from_pretrained(
#         config.model_name_or_path, trust_remote_code=config.trust_remote_code
#     )
#     model = AutoModelForSequenceClassification.from_pretrained(
#         config.ckpt_path if config.ckpt_path is not None else config.model_name_or_path,
#         num_labels=config.num_labels,
#         output_hidden_states=config.output_hidden_states,
#         trust_remote_code=config.trust_remote_code,
#     )
#     return tokenizer, model

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel
import os


# def build_glm(config):
#     """
#     Build GLM model, handling both regular and LoRA fine-tuned models.
#     """
#     model_path = config.get("model_path") or config.get("teacher_model_path")

#     # Check if this is a LoRA adapter directory
#     is_lora = os.path.exists(os.path.join(model_path, "adapter_config.json"))

#     if is_lora:
#         print(f"Loading LoRA adapter from: {model_path}")

#         # Load the adapter config to get the base model path
#         import json

#         with open(os.path.join(model_path, "adapter_config.json"), "r") as f:
#             adapter_config = json.load(f)

#         base_model_name = adapter_config.get("base_model_name_or_path")

#         if not base_model_name:
#             raise ValueError(
#                 "LoRA adapter_config.json doesn't contain base_model_name_or_path"
#             )

#         print(f"Loading base model: {base_model_name}")

#         # Load tokenizer from base model
#         tokenizer = AutoTokenizer.from_pretrained(
#             base_model_name, trust_remote_code=True
#         )

#         # Load base model
#         base_model = AutoModelForSequenceClassification.from_pretrained(
#             base_model_name,
#             num_labels=config.get("num_labels", 2),
#             trust_remote_code=True,
#         )

#         # Load LoRA adapter
#         model = PeftModel.from_pretrained(base_model, model_path)

#         # Optional: merge adapter weights into base model for faster inference
#         if config.get("merge_lora", True):
#             print("Merging LoRA weights into base model...")
#             model = model.merge_and_unload()

#     else:
#         # Regular model loading (not LoRA)
#         print(f"Loading regular model from: {model_path}")
#         tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
#         model = AutoModelForSequenceClassification.from_pretrained(
#             model_path, num_labels=config.get("num_labels", 2), trust_remote_code=True
#         )

#     return tokenizer, model
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel
import os


# def build_glm(config):
#     """
#     Build GLM model, handling both regular and LoRA fine-tuned models.
#     """
#     # Handle both dict and object configs
#     if hasattr(config, "model_path"):
#         model_path = config.model_path
#     elif hasattr(config, "teacher_model_path"):
#         model_path = config.teacher_model_path
#     else:
#         raise ValueError(
#             "Config must have 'model_path' or 'teacher_model_path' attribute"
#         )

#     # Get num_labels
#     num_labels = getattr(config, "num_labels", 2)

#     # Check if this is a LoRA adapter directory
#     is_lora = os.path.exists(os.path.join(model_path, "adapter_config.json"))

#     if is_lora:
#         print(f"Loading LoRA adapter from: {model_path}")

#         # Load the adapter config to get the base model path
#         import json

#         with open(os.path.join(model_path, "adapter_config.json"), "r") as f:
#             adapter_config = json.load(f)

#         # Try to get base model from config first, then from adapter_config
#         base_model_name = (
#             getattr(config, "base_model_path", None)
#             or getattr(config, "base_model_name", None)
#             or adapter_config.get("base_model_name_or_path")
#         )

#         if not base_model_name:
#             raise ValueError(
#                 "Cannot find base model path. Please specify 'base_model_path' in config"
#             )

#         print(f"Loading base model: {base_model_name}")

#         # Load tokenizer from base model
#         tokenizer = AutoTokenizer.from_pretrained(
#             base_model_name, trust_remote_code=True
#         )

#         # Load base model
#         base_model = AutoModelForSequenceClassification.from_pretrained(
#             base_model_name, num_labels=num_labels, trust_remote_code=True
#         )

#         # Load LoRA adapter
#         model = PeftModel.from_pretrained(base_model, model_path)

#         # Optional: merge adapter weights into base model for faster inference
#         merge_lora = getattr(config, "merge_lora", True)
#         if merge_lora:
#             print("Merging LoRA weights into base model...")
#             model = model.merge_and_unload()

#     else:
#         # Regular model loading (not LoRA)
#         print(f"Loading regular model from: {model_path}")
#         tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
#         model = AutoModelForSequenceClassification.from_pretrained(
#             model_path, num_labels=num_labels, trust_remote_code=True
#         )

#     return tokenizer, model


def build_glm(config: GLMConfig):
    """
    Build GLM model, handling both regular and LoRA fine-tuned models.
    """
    model_path = config.model_name_or_path

    if config.is_lora:
        print(f"Loading LoRA adapter from: {model_path}")

        if not config.base_model_path:
            raise ValueError(
                f"LoRA adapter detected but base_model_path is not set. "
                f"Please specify base_model_path in GLMConfig."
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
        )

    return tokenizer, model


def get_best_checkpoint(parent_path: str, task_name: str, model_type: str = "default"):
    """Find best checkpoint for a task - supports both GLM and NT directory structures"""
    import os
    import re

    if "NT" in model_type or "nucleotide" in parent_path.lower():
        # NT checkpoint structure: finetuned_models/{task}_finetuned/model-best*mcc_score*
        task_dir = os.path.join(
            parent_path, "finetuned_models", f"{task_name}_finetuned"
        )
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
        # Original GLM checkpoint structure
        from ..trainer.utils import get_best_checkpoint as orig_get_best_checkpoint

        return orig_get_best_checkpoint(parent_path, task_name), -1.0
