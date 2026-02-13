"""
Enformer model wrapper for distillation.
Provides compatibility between Enformer models and the OmegaGenome distillation framework.
"""

import os
import re
import torch
import torch.nn as nn
from typing import Optional, Tuple
from enformer_pytorch import from_pretrained


class EnformerFeatureExtractor(nn.Module):
    """Wraps Enformer model to extract features for distillation."""

    def __init__(self, model, classifier):
        super().__init__()
        self.model = model
        self.classifier = classifier
        self.hidden_dim = None

    def forward(
        self,
        input_ids,
        attention_mask=None,
        return_features=False,
        return_multi_features=False,
    ):
        # Enformer expects input_ids directly
        output = self.model(input_ids, return_embeddings=True)
        embeddings = output[1]  # Second element is embeddings

        # Pool embeddings
        feats = embeddings.permute(0, 2, 1)  # [B, D_hidden, L_target]
        pooled = feats.mean(dim=-1)  # [B, D_hidden]

        # Set hidden_dim dynamically if not set
        if self.hidden_dim is None:
            self.hidden_dim = pooled.shape[-1]

        logits = self.classifier(pooled)

        if return_multi_features:
            # For ReviewKD: return multiple feature levels
            multi_features = [embeddings, pooled]
            return logits, multi_features
        elif return_features:
            return logits, pooled

        return logits


def load_enformer_model(
    checkpoint_path: str,
    num_labels: int,
    device: str,
    enformer_dim: int = 1536,
    target_length: int = 8,
) -> Tuple[nn.Module, None, nn.Module]:
    """
    Load an Enformer model from checkpoint.

    Args:
        checkpoint_path: Path to the model checkpoint file (.pt)
        num_labels: Number of classification labels
        device: Device to load the model on
        enformer_dim: Enformer dimension (default: 1536)
        target_length: Target length (default: 8)

    Returns:
        tuple: (wrapped_model, None, base_enformer)
            - wrapped_model: EnformerFeatureExtractor for distillation
            - None: No tokenizer for Enformer (sequence-based)
            - base_enformer: The underlying Enformer model
    """
    print(f"Loading Enformer model from: {checkpoint_path}")

    # Build Enformer backbone
    base_enformer = from_pretrained(
        "EleutherAI/enformer-official-rough",
        target_length=target_length,
        use_tf_gamma=False,
    )

    # Build classifier head
    hidden_dim = enformer_dim * 2  # Enformer trunk output is dim * 2
    classifier = nn.Linear(hidden_dim, num_labels)

    # Load checkpoint
    state_dict = torch.load(checkpoint_path, map_location=device)

    # Load Enformer weights (if present in checkpoint)
    enformer_state = {
        k.replace("enformer.", ""): v for k, v in state_dict.items() if k.startswith("enformer.")
    }
    if enformer_state:
        base_enformer.load_state_dict(enformer_state, strict=False)

    # Load classifier weights
    classifier_state = {
        k.replace("classifier.", ""): v
        for k, v in state_dict.items()
        if k.startswith("classifier.")
    }
    if classifier_state:
        classifier.load_state_dict(classifier_state, strict=True)

    # Move to device
    base_enformer = base_enformer.to(device)
    classifier = classifier.to(device)

    # Wrap in feature extractor
    wrapped_model = EnformerFeatureExtractor(base_enformer, classifier)
    wrapped_model.hidden_dim = hidden_dim

    print(f"✓ Enformer model loaded (hidden_dim={hidden_dim})")

    return wrapped_model, None, base_enformer


def find_best_enformer_checkpoint(
    task_name: str, checkpoint_root: str
) -> Tuple[Optional[str], float]:
    """
    Find the best Enformer checkpoint for a given task.
    Expected directory structure: {checkpoint_root}/{task_name}_checkpoints/

    Args:
        task_name: Name of the task
        checkpoint_root: Root directory containing task checkpoints

    Returns:
        tuple: (checkpoint_path, best_score)
    """
    task_dir = os.path.join(checkpoint_root, f"{task_name}_checkpoints")

    if not os.path.isdir(task_dir):
        print(f"Warning: Checkpoint directory not found for task '{task_name}' at {task_dir}")
        return None, -1.0

    best_score = -1.0
    best_ckpt_path = None

    # Pattern: epoch{X}_mcc{Y}.pt
    pattern = re.compile(r"epoch(\d+)_mcc([0-9\.\-]+)\.pt")

    for filename in os.listdir(task_dir):
        match = pattern.match(filename)
        if match:
            try:
                mcc_score = float(match.group(2))
                if mcc_score > best_score:
                    best_score = mcc_score
                    best_ckpt_path = os.path.join(task_dir, filename)
            except (ValueError, IndexError):
                continue

    if best_ckpt_path:
        print(
            f"Found best Enformer checkpoint for '{task_name}': "
            f"{os.path.basename(best_ckpt_path)} (Val MCC: {best_score:.4f})"
        )
    else:
        print(f"Warning: No valid Enformer checkpoint found for task '{task_name}' in {task_dir}")

    return best_ckpt_path, best_score


class EnformerTokenizer:
    """
    Dummy tokenizer for Enformer to maintain interface compatibility.
    Enformer uses direct sequence encoding.
    """

    CHAR2IDX = {"A": 0, "C": 1, "G": 2, "T": 3, "N": 4}

    def __call__(
        self,
        sequences,
        padding="max_length",
        truncation=True,
        max_length=1024,
        return_tensors="pt",
    ):
        """
        Encode sequences for Enformer.

        Args:
            sequences: List of DNA sequences or single sequence
            padding: Padding strategy (only 'max_length' supported)
            truncation: Whether to truncate
            max_length: Maximum sequence length
            return_tensors: Return format ('pt' for PyTorch)

        Returns:
            Dict with 'input_ids' and 'attention_mask'
        """
        if isinstance(sequences, str):
            sequences = [sequences]

        encoded = []
        for seq in sequences:
            seq = seq.upper()[:max_length] if truncation else seq.upper()
            ids = [self.CHAR2IDX.get(ch, 4) for ch in seq]

            # Pad if needed
            if len(ids) < max_length:
                ids += [4] * (max_length - len(ids))

            encoded.append(ids)

        input_ids = torch.tensor(encoded, dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
