from dataclasses import dataclass
from typing import Optional
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel
import os


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
        # Original GLM checkpoint structure
        from ..trainer.utils import get_best_checkpoint as orig_get_best_checkpoint

        return orig_get_best_checkpoint(parent_path, task_name), -1.0
