#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LoRA-RANK ABLATION (OmegaGenome)
================================================
COPY of nt-finetune-list-detailed-10-17-lora-skip-r-fix-num-label.py.
Purpose: show NT-2.5B teacher's held-out ceiling does NOT materially rise as we
increase the LoRA rank r (48 baseline -> 96 -> 192) => the limit is task signal,
not adapter capacity.

What changed vs the original (everything else identical: 40 epochs, lr 1e-5,
bs 8, best-ckpt-by-MCC, target_modules=["query","value"]):
  * LoRA rank `r` is read from env var LORA_R (or argv[1]); default 96.
  * lora_alpha keeps the SAME r:alpha RATIO as the r=48 baseline (alpha=64):
        alpha = round((64/48) * r) = round(4/3 * r)
    so the LoRA scaling alpha/r ~= 1.333 is held CONSTANT across r.
        r=48  -> alpha=64   (baseline, not rerun here)
        r=96  -> alpha=128
        r=192 -> alpha=256
  * downstream_task_list defaults to ["H3K9me3","enhancers_types"]; a single
    task can be selected with env var TASK (one job == one (task,r) cell).
  * parent_path points to a fresh, r-stamped dir under
        ${OG_STORE}/NT/2b5-INCREASE-R-ablation-0629/
"""

# ----------------------------------------
# Imports
# ----------------------------------------
import os
import sys
import csv
import torch
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from datasets import load_dataset, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import matthews_corrcoef, f1_score

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    TrainerCallback,
)

# Import PEFT for LoRA
from peft import LoraConfig, get_peft_model, TaskType


# ----------------------------------------
# Ablation knobs (env-var driven so one job == one (task, r) cell)
# ----------------------------------------
def _get_lora_r():
    """Return the LoRA rank r from env LORA_R or argv[1] (default 96)."""
    if os.environ.get("LORA_R"):
        return int(os.environ["LORA_R"])
    if len(sys.argv) > 1:
        return int(sys.argv[1])
    return 96


LORA_R = _get_lora_r()
# Keep alpha/r ratio identical to the r=48 baseline (alpha=64 => 4/3 * r).
LORA_ALPHA = int(round((64.0 / 48.0) * LORA_R))
print(f"[ablation] LORA_R={LORA_R}  LORA_ALPHA={LORA_ALPHA}  (alpha/r={LORA_ALPHA / LORA_R:.4f})")

# ----------------------------------------
# Global path prefix for all file operations (r-stamped, fresh dir)
# ----------------------------------------
parent_path = f"{os.environ.get('OG_STORE', 'data')}/NT/2b5-INCREASE-R-ablation-0629/r{LORA_R}/"

# ----------------------------------------
# Downstream task list (the two lowest-ceiling / most "suspicious" tasks)
# ----------------------------------------
_default_tasks = ["H3K9me3", "enhancers_types"]
if os.environ.get("TASK"):
    downstream_task_list = [os.environ["TASK"]]
else:
    downstream_task_list = list(_default_tasks)
print("The following downstream tasks will be processed:", downstream_task_list)

# ----------------------------------------
# Device and base model
# ----------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_MODEL_NAME = "InstaDeepAI/nucleotide-transformer-2.5b-multi-species"
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)


# ----------------------------------------
# Helper functions
# ----------------------------------------
def compute_metrics_f1_score(eval_pred):
    """Computes F1 score for binary classification."""
    predictions = np.argmax(eval_pred.predictions, axis=-1)
    references = eval_pred.label_ids
    return {"f1_score": f1_score(references, predictions)}


def compute_metrics_mcc(eval_pred):
    """Computes Matthews correlation coefficient (MCC)."""
    predictions = np.argmax(eval_pred.predictions, axis=-1)
    references = eval_pred.label_ids
    return {"mcc_score": matthews_corrcoef(references, predictions)}


def tokenize_function(examples):
    return tokenizer(examples["data"])


def get_num_labels_and_metric(task_name):
    """Return (num_labels, metric_fn, metric_for_best_model) for each task."""
    if task_name in ["enhancers_types", "splice_sites_all"]:
        return 3, compute_metrics_mcc, "mcc_score"
    else:
        return 2, compute_metrics_mcc, "mcc_score"


def compute_test_metrics(predictions, references, num_labels):
    """Helper to compute both MCC and F1, handling binary vs multi-class F1."""
    mcc = matthews_corrcoef(references, predictions)
    if num_labels == 2:
        f1 = f1_score(references, predictions)
    else:
        f1 = f1_score(references, predictions, average="macro")
    return mcc, f1


# ----------------------------------------
# Custom callback for extra saving logic
# ----------------------------------------
class SaveMoreDetailsCallback(TrainerCallback):
    """
    Logs per-epoch val metric + test MCC/F1 to a CSV, saves a checkpoint every
    100 epochs and whenever a new best val metric is observed.
    """

    def __init__(self, test_dataset, num_labels, metric_for_best_model="f1_score"):
        super().__init__()
        self.test_dataset = test_dataset
        self.num_labels = num_labels
        self.metric_for_best_model = metric_for_best_model
        self.best_metric = float("-inf")

        csv_exists = os.path.exists("training_metrics.csv")
        mode = "a" if csv_exists else "w"
        self.csv_file = open("training_metrics.csv", mode, newline="")
        self.csv_writer = csv.writer(self.csv_file)
        if not csv_exists:
            self.csv_writer.writerow(["epoch", "metric_name", "metric_value"])

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        epoch = int(state.epoch)
        if not metrics:
            return

        if "eval_f1_score" in metrics:
            current_metric = metrics["eval_f1_score"]
            metric_name = "f1_score"
        elif "eval_mcc_score" in metrics:
            current_metric = metrics["eval_mcc_score"]
            metric_name = "mcc_score"
        else:
            return

        self.csv_writer.writerow([epoch, metric_name, current_metric])
        self.csv_file.flush()

        trainer = kwargs.get("trainer", None)
        if trainer is not None:
            test_output = trainer.predict(self.test_dataset)
            test_preds = np.argmax(test_output.predictions, axis=-1)
            test_labels = test_output.label_ids
            test_mcc, test_f1 = compute_test_metrics(test_preds, test_labels, self.num_labels)
            self.csv_writer.writerow([epoch, "test_mcc", test_mcc])
            self.csv_writer.writerow([epoch, "test_f1", test_f1])
            self.csv_file.flush()

        model = kwargs.get("model", None) or (trainer.model if trainer else None)
        if model:
            if epoch > 0 and (epoch % 100 == 0):
                output_dir = os.path.join(
                    args.output_dir,
                    f"model-epoch{epoch}-{metric_name}{current_metric:.4f}",
                )
                model.save_pretrained(output_dir)
                print(f"[Callback] Saved checkpoint at epoch {epoch} -> {output_dir}")

            if current_metric > self.best_metric:
                self.best_metric = current_metric
                best_dir = os.path.join(
                    args.output_dir,
                    f"model-best-epoch{epoch}-{metric_name}{current_metric:.4f}",
                )
                model.save_pretrained(best_dir)
                print(f"[Callback] New best metric ({current_metric:.4f}) -> {best_dir}")

    def on_train_end(self, args, state, control, **kwargs):
        self.csv_file.close()


# ------------------------------------------------
# Main iteration
# ------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)

    finetuned_models_dir = os.path.join(parent_path, "finetuned_models")
    finetuned_plots_dir = os.path.join(parent_path, "finetuned_plots")
    os.makedirs(finetuned_models_dir, exist_ok=True)
    os.makedirs(finetuned_plots_dir, exist_ok=True)

    for dataset_name in downstream_task_list:
        print(f"\n=== Fine-tuning on dataset: {dataset_name} (r={LORA_R}) ===")
        try:
            num_labels, metric_fn, metric_for_best_model = get_num_labels_and_metric(dataset_name)

            print("Loading base model and adding classification head...")
            base_model = AutoModelForSequenceClassification.from_pretrained(
                BASE_MODEL_NAME,
                num_labels=num_labels,
                trust_remote_code=True,
            )

            # Define LoRA config (ONLY r/alpha vary vs the baseline)
            lora_config = LoraConfig(
                task_type=TaskType.SEQ_CLS,
                r=LORA_R,
                lora_alpha=LORA_ALPHA,
                lora_dropout=0.1,
                target_modules=["query", "value"],
            )
            model = get_peft_model(base_model, lora_config)
            model.print_trainable_parameters()
            model.to(device)

            print("Loading dataset from HF Hub:")
            try:
                raw_ds = load_dataset(
                    "InstaDeepAI/nucleotide_transformer_downstream_tasks_revised",
                    split={"train": "train", "test": "test"},
                    trust_remote_code=True,
                )
                train_dataset = raw_ds["train"].filter(lambda ex: ex["task"] == dataset_name)
                test_dataset = raw_ds["test"].filter(lambda ex: ex["task"] == dataset_name)
            except Exception as e:
                print(f"Error loading dataset for {dataset_name}: {e}")
                print("Skipping this dataset.")
                continue

            train_sequences = train_dataset["sequence"]
            train_labels = train_dataset["label"]
            test_sequences = test_dataset["sequence"]
            test_labels = test_dataset["label"]

            train_sequences, val_sequences, train_labels, val_labels = train_test_split(
                train_sequences, train_labels, test_size=0.1, random_state=42
            )

            ds_train = Dataset.from_dict({"data": train_sequences, "labels": train_labels})
            ds_val = Dataset.from_dict({"data": val_sequences, "labels": val_labels})
            ds_test = Dataset.from_dict({"data": test_sequences, "labels": test_labels})

            print("Tokenizing data...")
            ds_train_tokenized = ds_train.map(
                tokenize_function, batched=True, remove_columns=["data"]
            )
            ds_val_tokenized = ds_val.map(tokenize_function, batched=True, remove_columns=["data"])
            ds_test_tokenized = ds_test.map(
                tokenize_function, batched=True, remove_columns=["data"]
            )

            output_dir = os.path.join(finetuned_models_dir, f"{dataset_name}_finetuned")
            print("Setting up TrainingArguments...")
            args = TrainingArguments(
                output_dir=output_dir,
                remove_unused_columns=False,
                eval_strategy="epoch",
                save_strategy="epoch",
                learning_rate=1e-5,
                per_device_train_batch_size=8,
                gradient_accumulation_steps=1,
                per_device_eval_batch_size=64,
                num_train_epochs=40,
                logging_steps=100,
                load_best_model_at_end=True,
                metric_for_best_model=metric_for_best_model,
                greater_is_better=True,
                label_names=["labels"],
                dataloader_drop_last=True,
            )

            print("Creating Trainer...")
            save_callback = SaveMoreDetailsCallback(
                test_dataset=ds_test_tokenized,
                num_labels=num_labels,
                metric_for_best_model=metric_for_best_model,
            )
            trainer = Trainer(
                model=model,
                args=args,
                train_dataset=ds_train_tokenized,
                eval_dataset=ds_val_tokenized,
                tokenizer=tokenizer,
                compute_metrics=metric_fn,
                callbacks=[save_callback],
            )

            print(f"Starting training on {dataset_name} (r={LORA_R}) ...")
            trainer.train()

            metric_key = "eval_" + (
                "f1_score" if metric_for_best_model == "f1_score" else "mcc_score"
            )
            metric_curve = [
                (x["step"], x[metric_key]) for x in trainer.state.log_history if metric_key in x
            ]
            if metric_curve:
                steps, metric_values = zip(*metric_curve)
                fig_path = os.path.join(finetuned_plots_dir, f"{dataset_name}_val_{metric_key}.png")
                plt.figure()
                plt.plot(steps, metric_values, label=f"Validation {metric_key}")
                plt.title(f"Validation {metric_key} - {dataset_name} (r={LORA_R})")
                plt.xlabel("Training steps")
                plt.ylabel(metric_key)
                plt.legend()
                plt.savefig(fig_path)
                print(f"Saved figure: {fig_path}")
                plt.close()
            else:
                print("No eval metric logs found to plot.")

            print("Evaluating on test set...")
            test_metrics = trainer.predict(ds_test_tokenized).metrics
            print(f"Test metrics for {dataset_name}:\n{test_metrics}")

            result_filename = os.path.join(finetuned_models_dir, f"{dataset_name}_test_metrics.txt")
            with open(result_filename, "w") as f:
                f.write(str(test_metrics))
            print(f"Saved test metrics to {result_filename}")

            print(f"[INFO] Completed dataset: {dataset_name} (r={LORA_R})")

        except Exception as e:
            print(f"Error processing dataset {dataset_name}: {e}")
            print("Skipping this dataset.\n")
            continue

    print("\nAll tasks completed.")
