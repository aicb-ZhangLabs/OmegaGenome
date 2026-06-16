"""Stage-1 teacher fine-tuning engine.

Teacher-agnostic: given any HF sequence-classification gLM (Carbon, AIDO.DNA,
GENERATOR, NT, DNABERT-2, ...) described by a ``GLMConfig`` plus a
``TeacherFinetuneConfig``, fine-tune it on one NT-benchmark task and report MCC.

Built on the HF ``Trainer`` so LoRA, bf16, and gradient checkpointing (needed for
billion-scale teachers) come for free. The only teacher-specific bits are read off
``GLMConfig``: ``input_prefix`` and ``add_special_tokens`` (e.g. Carbon's "<dna>" tag).
"""

import os
import csv
import json
from datetime import datetime

import numpy as np
import torch
from datasets import Dataset
from sklearn.metrics import matthews_corrcoef
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from src.model.glm import GLMConfig


def _compute_mcc(eval_pred):
    """HF Trainer ``compute_metrics``: argmax logits -> MCC."""
    logits, labels = eval_pred
    if isinstance(logits, tuple):  # some models return (logits, hidden_states, ...)
        logits = logits[0]
    preds = np.argmax(logits, axis=-1)
    return {"mcc": float(matthews_corrcoef(labels, preds))}


def build_teacher_for_finetune(teacher: GLMConfig, num_labels: int, ft):
    """Load an HF classification model + tokenizer and (optionally) attach LoRA.

    Returns (tokenizer, model). ``ft`` is a TeacherFinetuneConfig.
    """
    tokenizer = AutoTokenizer.from_pretrained(
        teacher.model_name_or_path, trust_remote_code=teacher.trust_remote_code
    )
    # Autoregressive teachers (Carbon, GENERATOR, ...) often ship no pad token.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if ft.bf16 else None
    model = AutoModelForSequenceClassification.from_pretrained(
        teacher.model_name_or_path,
        num_labels=num_labels,
        trust_remote_code=teacher.trust_remote_code,
        torch_dtype=dtype,
    )
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id

    if ft.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()  # required for grad-checkpointing + LoRA

    if ft.use_lora:
        from peft import LoraConfig, TaskType, get_peft_model

        peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=ft.lora_r,
            lora_alpha=ft.lora_alpha,
            lora_dropout=ft.lora_dropout,
            target_modules=ft.lora_target_modules,  # None lets peft pick by arch
            bias="none",
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    return tokenizer, model


def _tokenized_dataset(sequences, labels, tokenizer, teacher: GLMConfig, max_len: int):
    """Wrap raw DNA strings with the teacher's input_prefix and tokenize."""
    ds = Dataset.from_dict({"sequence": list(sequences), "label": list(labels)})

    def _tok(batch):
        texts = [teacher.input_prefix + s for s in batch["sequence"]]
        out = tokenizer(
            texts,
            truncation=True,
            max_length=max_len,
            add_special_tokens=teacher.add_special_tokens,
        )
        out["labels"] = batch["label"]
        return out

    return ds.map(_tok, batched=True, remove_columns=ds.column_names)


def finetune_teacher_task(
    ft,
    teacher: GLMConfig,
    task_name: str,
    num_labels: int,
    splits,
    run_dir: str,
):
    """Fine-tune ``teacher`` on one task and return its test MCC.

    ``splits`` = (X_train, y_train, X_val, y_val, X_test, y_test).
    The best checkpoint (by validation MCC) is saved to ``run_dir/{task}_finetuned``,
    which is exactly where the stage-2 distillation loader looks for teacher weights.
    """
    X_train, y_train, X_val, y_val, X_test, y_test = splits
    os.makedirs(run_dir, exist_ok=True)
    ckpt_dir = os.path.join(run_dir, f"{task_name}_finetuned")

    tokenizer, model = build_teacher_for_finetune(teacher, num_labels, ft)

    ds_train = _tokenized_dataset(X_train, y_train, tokenizer, teacher, ft.max_len)
    ds_val = _tokenized_dataset(X_val, y_val, tokenizer, teacher, ft.max_len)
    ds_test = _tokenized_dataset(X_test, y_test, tokenizer, teacher, ft.max_len)

    args = TrainingArguments(
        output_dir=os.path.join(run_dir, "hf_trainer", task_name),
        num_train_epochs=ft.epochs,
        per_device_train_batch_size=ft.batch_size,
        per_device_eval_batch_size=ft.eval_batch_size,
        gradient_accumulation_steps=ft.grad_accum,
        learning_rate=ft.lr,
        weight_decay=ft.weight_decay,
        warmup_ratio=ft.warmup_ratio,
        lr_scheduler_type=ft.lr_scheduler_type,
        bf16=ft.bf16,
        gradient_checkpointing=ft.gradient_checkpointing,
        logging_steps=ft.log_every,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="mcc",
        greater_is_better=True,
        report_to=["wandb"] if ft.use_wandb else [],
        run_name=f"{ft.teacher_name}/{task_name}",
        seed=ft.seed,
        dataloader_num_workers=ft.num_workers,
        remove_unused_columns=False,
    )

    callbacks = []
    if ft.early_stopping_patience > 0:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=ft.early_stopping_patience))

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds_train,
        eval_dataset=ds_val,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=_compute_mcc,
        callbacks=callbacks,
    )

    trainer.train()

    val_metrics = trainer.evaluate(ds_val, metric_key_prefix="val")
    test_metrics = trainer.evaluate(ds_test, metric_key_prefix="test")
    val_mcc = float(val_metrics.get("val_mcc", float("nan")))
    test_mcc = float(test_metrics.get("test_mcc", float("nan")))

    # Save the best teacher (LoRA adapter if use_lora, else full model) where stage-2 expects it.
    trainer.save_model(ckpt_dir)
    tokenizer.save_pretrained(ckpt_dir)
    _write_results(ft, teacher, task_name, val_mcc, test_mcc, run_dir, ckpt_dir)

    print(f"[{ft.teacher_name}] {task_name}: val_mcc={val_mcc:.4f}  test_mcc={test_mcc:.4f}")
    return test_mcc


def _write_results(ft, teacher, task_name, val_mcc, test_mcc, run_dir, ckpt_dir):
    """Persist per-task metrics (json per run + one shared summary CSV)."""
    record = {
        "teacher_name": ft.teacher_name,
        "teacher_model": teacher.model_name_or_path,
        "task": task_name,
        "val_mcc": val_mcc,
        "test_mcc": test_mcc,
        "use_lora": ft.use_lora,
        "checkpoint": ckpt_dir,
        "timestamp": datetime.now().isoformat(),
    }
    with open(os.path.join(ckpt_dir, "finetune_result.json"), "w") as f:
        json.dump(record, f, indent=2)

    summary_csv = os.path.join(ft.output_dir, "teacher_finetune_summary.csv")
    os.makedirs(ft.output_dir, exist_ok=True)
    exists = os.path.exists(summary_csv)
    with open(summary_csv, "a", newline="") as f:
        writer = csv.writer(f)
        if not exists:
            writer.writerow(list(record.keys()))
        writer.writerow([record[k] for k in record])
