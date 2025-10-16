from dataclasses import dataclass
from typing import Optional
from transformers import AutoTokenizer, AutoModelForSequenceClassification


@dataclass
class GLMConfig:
    model_name_or_path: str
    num_labels: int = 2
    ckpt_path: Optional[str] = None
    output_hidden_states: bool = True  # Added for feature extraction
    trust_remote_code: bool = False  # Added for NT models


def build_glm(config: GLMConfig):
    """Build GLM or NT model - unified interface for all transformer models"""
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name_or_path, trust_remote_code=config.trust_remote_code
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        config.ckpt_path if config.ckpt_path is not None else config.model_name_or_path,
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
