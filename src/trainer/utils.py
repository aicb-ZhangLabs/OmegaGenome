import os
import torch
import numpy as np

from tqdm import tqdm
from sklearn.metrics import matthews_corrcoef

import json
import glob
from typing import Set, Tuple, List, Dict
from datetime import datetime

import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig


def get_best_checkpoint(parent_dir, task_name):
    task_dir = os.path.join(parent_dir, task_name)
    if not os.path.isdir(task_dir):
        return None
    ckpts = [d for d in os.listdir(task_dir) if d.startswith("checkpoint-")]
    if not ckpts:
        return None
    best = max(
        ckpts,
        key=lambda d: int(d.split("-", 1)[1]) if d.split("-", 1)[1].isdigit() else -1,
    )
    return os.path.join(task_dir, best)


@torch.no_grad()
def precompute_teacher_logits(
    tokenizer, model, sequences, batch_size, device, max_length, needs_features=False
):
    model.eval()
    logits_list = []
    features_list = []

    # Debug flag to print structure once
    debug_printed = False

    for i in tqdm(
        range(0, len(sequences), batch_size), total=len(sequences) // batch_size
    ):
        batch = sequences[i : i + batch_size]
        tok = tokenizer(
            batch,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        input_ids = tok.input_ids.to(device)

        # FIX: Handle missing attention_mask
        if hasattr(tok, "attention_mask") and tok.attention_mask is not None:
            attention_mask = tok.attention_mask.to(device)
        else:
            attention_mask = torch.ones_like(input_ids)

        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attention_mask)

            # FIX: Handle both wrapped models (returns Tensor) and standard HF models
            if isinstance(out, torch.Tensor):
                logits = out
            else:
                logits = out.logits

            logits_list.append(logits.cpu())

            if needs_features:
                # Debug: print structure on first batch
                if not debug_printed:
                    print(f"Debug - Batch size: {len(batch)}")
                    print(f"Debug - Logits shape: {logits.shape}")
                    print(f"Debug - Output type: {type(out)}")
                    print(f"Debug - Has hidden_states: {hasattr(out, 'hidden_states')}")
                    if hasattr(out, "hidden_states") and out.hidden_states is not None:
                        hs = out.hidden_states
                        print(f"Debug - Hidden states type: {type(hs)}")
                        if isinstance(hs, (tuple, list)):
                            print(f"Debug - Num hidden layers: {len(hs)}")
                            print(f"Debug - Last hidden state shape: {hs[-1].shape}")
                        else:
                            print(
                                f"Debug - Hidden states shape (single tensor): {hs.shape}"
                            )
                    debug_printed = True

                # Extract features
                if hasattr(out, "hidden_states") and out.hidden_states is not None:
                    hs = out.hidden_states

                    # Check if it's a tuple/list of tensors (standard BERT format)
                    if isinstance(hs, (tuple, list)):
                        last_hidden = hs[-1]
                    else:
                        # Single tensor
                        last_hidden = hs

                    # Now handle the dimensionality
                    if last_hidden.dim() == 3:
                        # Standard format: [batch_size, seq_len, hidden_size]
                        hidden = last_hidden[:, 0, :]  # CLS token
                    elif last_hidden.dim() == 2:
                        # DNA-BERT2 format: [seq_len, hidden_size]
                        actual_batch_size = logits.shape[0]
                        if actual_batch_size == 1:
                            # Single sequence: pool across sequence dimension
                            hidden = last_hidden.mean(
                                dim=0, keepdim=True
                            )  # [1, hidden_size]
                        else:
                            # Multiple sequences but concatenated - need to split and pool
                            seq_len_per_sample = (
                                last_hidden.shape[0] // actual_batch_size
                            )
                            hidden_list = []
                            for b in range(actual_batch_size):
                                start_idx = b * seq_len_per_sample
                                end_idx = (b + 1) * seq_len_per_sample
                                seq_hidden = last_hidden[start_idx:end_idx]
                                # Pool this sequence
                                pooled = seq_hidden.mean(dim=0)  # [hidden_size]
                                hidden_list.append(pooled)
                            hidden = torch.stack(
                                hidden_list
                            )  # [batch_size, hidden_size]
                    else:
                        raise ValueError(
                            f"Unexpected hidden state shape: {last_hidden.shape}"
                        )
                else:
                    # Fallback: use pooler_output or logits
                    if hasattr(out, "pooler_output") and out.pooler_output is not None:
                        hidden = out.pooler_output
                    else:
                        print(
                            "Warning: Cannot extract hidden states, using logits as features"
                        )
                        hidden = logits

                features_list.append(hidden.cpu())

    logits = torch.cat(logits_list, dim=0).numpy()
    features = torch.cat(features_list, dim=0).numpy() if needs_features else None
    return logits, features


@torch.no_grad()
def evaluate_teacher_mcc(teacher_model, teacher_tokenizer, test_dataloader, device):
    """
    Evaluate teacher model on test set and return MCC score.

    Args:
        teacher_model: The teacher model to evaluate
        teacher_tokenizer: Tokenizer for the teacher model
        test_dataloader: DataLoader containing test data
        device: Device to run evaluation on

    Returns:
        float: Matthews Correlation Coefficient
    """
    teacher_model.eval()
    teacher_model.to(device)

    all_preds = []
    all_labels = []

    print("Evaluating teacher model...")
    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Teacher evaluation"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = teacher_model(input_ids=input_ids, attention_mask=attention_mask)

            # FIX: Handle both wrapped models (returns Tensor) and standard HF models (returns object with .logits)
            if isinstance(outputs, torch.Tensor):
                logits = outputs
            else:
                logits = outputs.logits

            preds = torch.argmax(logits, dim=-1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    mcc = matthews_corrcoef(all_labels, all_preds)
    return float(mcc)


"""
Experiment tracker for resuming interrupted hyperparameter searches.

This module tracks which experiments have completed by checking for marker files
in the output directory, allowing automatic resume of interrupted searches.
"""


class ExperimentTracker:
    """Tracks completed experiments to enable resume functionality."""

    def __init__(
        self,
        output_dir: str,
        completion_marker: str = "final_summary.json",
        start_timestamp: str = None,
    ):
        """
        Args:
            output_dir: Root output directory
            completion_marker: Filename that indicates completion
            start_timestamp: Only check experiments after this timestamp (format: "20251020_220000")
                            None = check all experiments
        """
        self.output_dir = output_dir
        self.completion_marker = completion_marker
        self.start_timestamp = start_timestamp

    def _is_after_start_timestamp(self, timestamp_str: str) -> bool:
        """Check if timestamp is after start_timestamp filter."""
        if self.start_timestamp is None:
            return True
        # Format: "20251020_144435"
        return timestamp_str >= self.start_timestamp

    def _get_hyperparam_str(self, config_dict: dict) -> str:
        """
        Create hyperparameter string from config dict.
        Should match create_run_hyperparams_str() from distill_trainer.py
        """
        sorted_keys = sorted(config_dict.keys())
        return "_".join([f"{k}{config_dict[k]}" for k in sorted_keys])

    def is_experiment_completed(self, task_name: str, hyperparam_config: dict) -> bool:
        """
        Check if an experiment (task + hyperparameters) has completed.

        Searches for completed runs by looking for the completion marker file:
        {output_dir}/{task_name}/*/*/*/{hyperparam_str}/{completion_marker}

        Args:
            task_name: Task name (e.g., "promoter_all")
            hyperparam_config: Dict with hyperparameter values

        Returns:
            True if at least one completed run exists
        """
        hyperparam_str = self._get_hyperparam_str(hyperparam_config)

        # Pattern: {output_dir}/{task_name}/*/*/*/{hyperparam_str}/{completion_marker}
        # Structure: {task}/{timestamp}/{uuid}/{hyperparam_str}/
        pattern = os.path.join(
            self.output_dir,
            task_name,
            "*",  # timestamp
            "*",  # uuid
            hyperparam_str,
            self.completion_marker,
        )

        matches = glob.glob(pattern)
        return len(matches) > 0

    def get_completed_experiments(self) -> Set[Tuple[str, str]]:
        """
        Scan output directory for all completed experiments.

        Returns:
            Set of (task_name, hyperparam_str) tuples for completed experiments
        """
        completed = set()

        if not os.path.exists(self.output_dir):
            return completed

        # Find all completion marker files
        pattern = os.path.join(
            self.output_dir,
            "*",  # task_name
            "*",  # timestamp
            "*",  # uuid
            "*",  # hyperparam_str
            self.completion_marker,
        )

        for marker_path in glob.glob(pattern):
            # Extract task_name and hyperparam_str from path
            # Path format: .../task_name/timestamp/uuid/hyperparam_str/final_summary.json
            parts = marker_path.split(os.sep)

            # Find indices
            try:
                output_dir_parts = self.output_dir.split(os.sep)
                output_dir_depth = len(output_dir_parts)

                # After output_dir: task_name/timestamp/uuid/hyperparam_str/marker
                task_name = parts[output_dir_depth]
                timestamp = parts[output_dir_depth + 1]  # Extract timestamp
                hyperparam_str = parts[output_dir_depth + 3]
                # Filter by timestamp
                if not self._is_after_start_timestamp(timestamp):
                    continue
                completed.add((task_name, hyperparam_str))
            except (IndexError, ValueError):
                continue

        return completed

    def generate_experiment_plan(
        self,
        all_experiments: List[Dict],
    ) -> Tuple[List[Dict], List[Dict], Dict]:
        """
        Generate execution plan by checking which experiments are completed.

        Args:
            all_experiments: List of experiment dicts, each containing:
                - task_name: str
                - hyperparam_config: dict (with weight_ce, weight_kl, etc.)
                - experiment_count: int

        Returns:
            Tuple of (incomplete_experiments, completed_experiments, summary_dict)
        """
        completed_set = self.get_completed_experiments()

        incomplete = []
        completed = []

        for exp in all_experiments:
            task_name = exp["task_name"]
            hyperparam_config = exp["hyperparam_config"]
            hyperparam_str = self._get_hyperparam_str(hyperparam_config)

            if (task_name, hyperparam_str) in completed_set:
                completed.append(exp)
            else:
                incomplete.append(exp)

        summary = {
            "total_experiments": len(all_experiments),
            "completed": len(completed),
            "incomplete": len(incomplete),
            "completion_rate": (
                f"{len(completed) / len(all_experiments) * 100:.1f}%"
                if all_experiments
                else "0%"
            ),
        }

        return incomplete, completed, summary

    def print_summary_report(
        self,
        incomplete: List[Dict],
        completed: List[Dict],
        summary: Dict,
        save_to_file: bool = True,
    ):
        """
        Print and optionally save a summary report of experiment status.

        Args:
            incomplete: List of incomplete experiments
            completed: List of completed experiments
            summary: Summary statistics dict
            save_to_file: Whether to save report to file
        """
        report_lines = []

        def add_line(line=""):
            report_lines.append(line)
            print(line)

        add_line("=" * 80)
        add_line("EXPERIMENT RESUME REPORT")
        add_line("=" * 80)
        add_line(f"Output directory: {self.output_dir}")
        add_line(f"Timestamp: {datetime.now().isoformat()}")
        add_line()
        add_line("SUMMARY:")
        add_line(f"  Total experiments: {summary['total_experiments']}")
        add_line(f"  Completed: {summary['completed']} ({summary['completion_rate']})")
        add_line(f"  Incomplete: {summary['incomplete']}")
        add_line("=" * 80)

        if completed:
            add_line()
            add_line(f"COMPLETED EXPERIMENTS ({len(completed)}):")
            add_line("-" * 80)

            # Group by task
            by_task = {}
            for exp in completed:
                task = exp["task_name"]
                if task not in by_task:
                    by_task[task] = []
                by_task[task].append(exp["hyperparam_config"])

            for task, configs in sorted(by_task.items()):
                add_line(f"  {task}: {len(configs)} hyperparameter combinations")

        if incomplete:
            add_line()
            add_line(f"INCOMPLETE EXPERIMENTS TO RUN ({len(incomplete)}):")
            add_line("-" * 80)

            # Group by task
            by_task = {}
            for exp in incomplete:
                task = exp["task_name"]
                if task not in by_task:
                    by_task[task] = []
                by_task[task].append(exp["hyperparam_config"])

            for task, configs in sorted(by_task.items()):
                add_line(f"  {task}: {len(configs)} hyperparameter combinations")
                for i, config in enumerate(configs[:3], 1):  # Show first 3
                    params = ", ".join([f"{k}={v}" for k, v in sorted(config.items())])
                    add_line(f"    {i}. {params}")
                if len(configs) > 3:
                    add_line(f"    ... and {len(configs) - 3} more")

        add_line("=" * 80)

        # Save to file
        if save_to_file:
            os.makedirs(self.output_dir, exist_ok=True)
            report_file = os.path.join(
                self.output_dir,
                f"resume_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            )
            with open(report_file, "w") as f:
                f.write("\n".join(report_lines))
            print(f"\nReport saved to: {report_file}")

            # Also save detailed JSON
            json_file = os.path.join(
                self.output_dir,
                f"resume_plan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            )
            with open(json_file, "w") as f:
                json.dump(
                    {
                        "summary": summary,
                        "completed": completed,
                        "incomplete": incomplete,
                    },
                    f,
                    indent=2,
                )
            print(f"Detailed plan saved to: {json_file}")

    def get_experiment_status(self, task_name: str, hyperparam_config: dict) -> dict:
        """
        Get detailed status: completed, partial (has checkpoints), or not started.

        Returns:
            {
                "status": "completed" | "partial" | "not_started",
                "latest_checkpoint": path or None,
                "latest_epoch": int or None
            }
        """
        hyperparam_str = self._get_hyperparam_str(hyperparam_config)

        # Check for completed
        if self.is_experiment_completed(task_name, hyperparam_config):
            return {
                "status": "completed",
                "latest_checkpoint": None,
                "latest_epoch": None,
            }

        # Check for partial progress
        pattern = os.path.join(
            self.output_dir, task_name, "*", "*", hyperparam_str, "epoch_*_valmcc_*"
        )
        checkpoints = glob.glob(pattern)

        if not checkpoints:
            return {
                "status": "not_started",
                "latest_checkpoint": None,
                "latest_epoch": None,
            }

        # Find latest checkpoint
        latest_ckpt = max(
            checkpoints, key=lambda p: int(p.split("epoch_")[1].split("_")[0])
        )
        latest_epoch = int(latest_ckpt.split("epoch_")[1].split("_")[0])

        return {
            "status": "partial",
            "latest_checkpoint": latest_ckpt,
            "latest_epoch": latest_epoch,
        }


"""
Caduceus model wrapper for distillation.
This module provides compatibility between Caduceus models and the OmegaGenome distillation framework.
"""


class CaduceusFeatureExtractor(nn.Module):
    """Wraps Caduceus model to extract features for distillation."""

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.hidden_dim = None  # Will be set dynamically

    def forward(
        self,
        input_ids,
        attention_mask=None,
        return_features=False,
        return_multi_features=False,
    ):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        logits = outputs.logits

        if return_multi_features:
            # Extract features from multiple layers for ReviewKD
            hidden_states = outputs.hidden_states
            last_hidden = hidden_states[-1]
            pooled = last_hidden.mean(dim=1)

            # Set hidden_dim dynamically if not set
            if self.hidden_dim is None:
                self.hidden_dim = pooled.shape[-1]

            # Take last few layers for multi-stage distillation
            multi_features = [
                pooled,  # Last layer pooled
                hidden_states[-2].mean(dim=1) if len(hidden_states) > 1 else pooled,
            ]
            return logits, multi_features

        elif return_features:
            # Extract final hidden state and pool
            last_hidden = outputs.hidden_states[-1]
            pooled = last_hidden.mean(dim=1)

            # Set hidden_dim dynamically if not set
            if self.hidden_dim is None:
                self.hidden_dim = pooled.shape[-1]

            return logits, pooled

        return logits


def load_caduceus_model(checkpoint_path, num_labels, device, best_ckpt_file=None):
    """
    Load a Caduceus model from checkpoint.

    Args:
        checkpoint_path: Path to the model checkpoint directory
        num_labels: Number of classification labels
        device: Device to load the model on
        best_ckpt_file: Optional path to a specific checkpoint file

    Returns:
        tuple: (wrapped_model, tokenizer, base_model)
    """
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, trust_remote_code=True)

    # Load model configuration
    config = AutoConfig.from_pretrained(checkpoint_path, trust_remote_code=True)

    # Create model from config
    base_model = AutoModelForSequenceClassification.from_config(
        config, trust_remote_code=True
    ).to(device)

    # Load weights
    if best_ckpt_file:
        # Use the specific best checkpoint file
        state_dict = torch.load(best_ckpt_file, map_location=device)
    else:
        # Fallback to default file if no specific checkpoint found
        state_dict = torch.load(
            os.path.join(checkpoint_path, "pytorch_model.bin"), map_location=device
        )

    base_model.load_state_dict(state_dict, strict=True)

    # Wrap in feature extractor for distillation
    wrapped_model = CaduceusFeatureExtractor(base_model)

    return wrapped_model, tokenizer, base_model


def find_best_caduceus_checkpoint(task_name, checkpoint_root):
    """
    Searches for the best Caduceus checkpoint for a given task.
    Expected directory structure: {checkpoint_root}/{task_name}_caduceus_finetuned/

    Args:
        task_name: Name of the task
        checkpoint_root: Root directory containing task checkpoints

    Returns:
        tuple: (checkpoint_dir, best_score, best_checkpoint_file)
    """
    import os
    import re

    task_dir = os.path.join(checkpoint_root, f"{task_name}_caduceus_finetuned")

    if not os.path.isdir(task_dir):
        print(
            f"Warning: Checkpoint directory not found for task '{task_name}' at {task_dir}"
        )
        return None, -1.0, None

    best_score = -1.0
    best_ckpt_path = None

    # Pattern for checkpoint files: epoch{X}_valmcc_{Y}.pt
    pattern = re.compile(r"epoch(\d+)_valmcc_(-?[0-9\.]+)\.pt")

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
            f"Found best Caduceus checkpoint for '{task_name}': {os.path.basename(best_ckpt_path)} (Val MCC: {best_score:.4f})"
        )
        return task_dir, best_score, best_ckpt_path
    else:
        # Check if there's a pre-trained model without specific checkpoint files
        if os.path.exists(os.path.join(task_dir, "config.json")):
            print(f"Found Caduceus model for '{task_name}' at {task_dir}")
            return task_dir, 0.0, None

        print(
            f"Warning: No valid Caduceus checkpoint found for task '{task_name}' in {task_dir}"
        )
        return None, -1.0, None
