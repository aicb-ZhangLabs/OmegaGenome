from dataclasses import dataclass
from typing import Optional
from transformers import AutoTokenizer, AutoModelForSequenceClassification


@dataclass
class GLMConfig:
    model_name_or_path: str
    num_labels: int = 2
    ckpt_path: Optional[str] = None


def build_glm(config: GLMConfig):
    tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path, trust_remote_code=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        config.ckpt_path if config.ckpt_path is not None else config.model_name_or_path,
        num_labels=config.num_labels,
        output_hidden_states=True,
        trust_remote_code=True,
    )
    return tokenizer, model
