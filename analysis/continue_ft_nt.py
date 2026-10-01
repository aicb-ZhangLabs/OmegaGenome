"""CONTINUE the FULL fine-tune (no-LoRA) of NT-2.5B on promoter_tata.

Question: does the no-LoRA teacher improve past the current eval MCC 0.8885 /
test MCC 0.8774 with MORE epochs, or is it saturated? The paper reports NT promoter_tata
teacher = 0.948.

What this does
--------------
1. Loads the already-trained full-FT checkpoint (the standard HF EsmForSequenceClassification
   saved by the original no-LoRA DeepSpeed run) -- either ``final_model`` (best = epoch-5
   ckpt-715, eval MCC 0.8885) or a numbered ``checkpoint-XXXX``.
2. Continues training for more epochs on the SAME data / tokenizer / eval pipeline used by
   the original LoRA finetune and by analysis/eval_fullft_nt.py (apples-to-apples):
   dataset InstaDeepAI/nucleotide_transformer_downstream_tasks_revised, task=promoter_tata,
   90/10 stratified train/val split (random_state=42), EsmTokenizer add_special_tokens=True,
   argmax -> matthews_corrcoef.
3. Evaluates eval(val) AND test MCC every epoch, keeps the best-val checkpoint, and writes
   the final best model to a NEW output dir (never overwrites the original).

Why single-GPU (no DeepSpeed) -- documented choice
--------------------------------------------------
The original run used DeepSpeed ZeRO-3 + CPU offload across 4 GPUs. A 2.5B ESM model in
fp16 with Adam fits on a single 49 GB (laniakea 6000Ada) or 80 GB (voyager H100) GPU with
gradient checkpointing (~30 GB), so we run single-GPU. This is functionally identical full
no-LoRA fine-tuning (same weights, data, eval), queues far faster on a congested cluster
(1 GPU vs 4), and avoids installing deepspeed into the shared /home venv. The output is the
same HF AutoModelForSequenceClassification, loadable by eval_fullft_nt.py unchanged.

Outputs go under a NEW dir on /extra (NOT /home, NOT overwriting the original).
"""

import os
import sys
import json
import argparse
from datetime import datetime

import numpy as np
import torch
from sklearn.metrics import matthews_corrcoef, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    TrainerCallback,
)
from torch.utils.data import Dataset

REPO = os.environ.get("OG_ROOT", os.getcwd())
sys.path.insert(0, REPO)
from src.data.dataset import (  # noqa: E402
    DatasetConfig,
    build_data_splits_from_huggingface,
    get_num_labels,
)

DATASET_NAME = "InstaDeepAI/nucleotide_transformer_downstream_tasks_revised"
MAX_LEN = 1000  # matches eval_fullft_nt.py / EsmTokenizer model_max_length

ORIG_PARENT = (
    os.environ.get("OG_STORE", "data") + "/NT/"
    "2b5-multi-species_nucleotide-transformer-finetune-results-NO-LORA-MULTI-GPU-"
    "epoch10-3-22-revised-deepspeed/finetuned_models"
)


class TokSeqDataset(Dataset):
    """Tokenize sequences once (EsmTokenizer, add_special_tokens=True) and serve HF-Trainer dicts."""

    def __init__(self, sequences, labels, tokenizer, max_len):
        enc = tokenizer(
            list(sequences),
            truncation=True,
            max_length=max_len,
            padding=False,
            add_special_tokens=True,
        )
        self.input_ids = enc["input_ids"]
        self.attention_mask = enc["attention_mask"]
        self.labels = [int(y) for y in labels]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return {
            "input_ids": self.input_ids[i],
            "attention_mask": self.attention_mask[i],
            "labels": self.labels[i],
        }


def compute_metrics(eval_pred):
    """Argmax logits -> MCC + F1 (single-label classification), matching the eval pipeline."""
    logits, labels = eval_pred
    if isinstance(logits, tuple):
        logits = logits[0]
    preds = np.argmax(logits, axis=-1)
    return {
        "mcc_score": float(matthews_corrcoef(labels, preds)),
        "f1_score": float(
            f1_score(labels, preds, average="binary" if len(set(labels)) <= 2 else "macro")
        ),
    }


class TestEvalCallback(TrainerCallback):
    """After each epoch's val eval, also score the held-out TEST split so we get a per-epoch
    test-MCC curve (the reported number). Records into a list of dicts."""

    def __init__(self, trainer, test_dataset, records):
        self.trainer = trainer
        self.test_dataset = test_dataset
        self.records = records

    def on_evaluate(self, args, state, control, **kwargs):
        out = self.trainer.predict(self.test_dataset, metric_key_prefix="test")
        m = out.metrics
        rec = {
            "epoch": float(state.epoch) if state.epoch is not None else None,
            "global_step": int(state.global_step),
            "val_mcc": None,  # filled from log_history below
            "test_mcc": float(m.get("test_mcc_score", float("nan"))),
            "test_f1": float(m.get("test_f1_score", float("nan"))),
        }
        # grab the most recent val mcc from the trainer's log history
        for e in reversed(state.log_history):
            if "eval_mcc_score" in e:
                rec["val_mcc"] = float(e["eval_mcc_score"])
                break
        self.records.append(rec)
        print(
            f"[epoch {rec['epoch']:.2f} step {rec['global_step']}] "
            f"val_mcc={rec['val_mcc']} test_mcc={rec['test_mcc']:.4f}",
            flush=True,
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="promoter_tata")
    ap.add_argument(
        "--resume-from",
        default=os.path.join(ORIG_PARENT, "promoter_tata_finetuned", "final_model"),
        help="HF model dir to load weights from (final_model = best epoch-5, or checkpoint-XXXX).",
    )
    ap.add_argument(
        "--output-dir",
        default=(
            os.environ.get("OG_STORE", "data") + "/NT/"
            "2b5-multi-species_nucleotide-transformer-finetune-results-NO-LORA-MULTI-GPU-"
            "epoch10-3-22-revised-deepspeed-promoter_tata-continued/finetuned_models/"
            "promoter_tata_finetuned"
        ),
    )
    ap.add_argument("--epochs", type=int, default=15, help="ADDITIONAL epochs to train.")
    ap.add_argument(
        "--lr", type=float, default=1e-6, help="Continue at the original tail LR (~1e-6)."
    )
    ap.add_argument("--train-bs", type=int, default=8)
    ap.add_argument("--eval-bs", type=int, default=32)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--warmup-ratio", type=float, default=0.0, help="No warmup when continuing.")
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--smoke", action="store_true", help="1-step resume-load smoke test.")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"[continue-ft] loading weights from {args.resume_from}", flush=True)
    if not os.path.isfile(os.path.join(args.resume_from, "config.json")):
        raise FileNotFoundError(f"no config.json under {args.resume_from}")
    tokenizer = AutoTokenizer.from_pretrained(args.resume_from, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.resume_from,
        num_labels=get_num_labels(args.task),
        torch_dtype=torch.float32,  # full FT trains in fp32 master weights; TrainingArguments fp16 casts compute
        local_files_only=True,
    )
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False
    try:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    except Exception as exc:
        print(f"[warn] grad checkpointing not enabled: {exc}", flush=True)

    cfg = DatasetConfig(task_name=args.task, data_path="", dataset_name=DATASET_NAME)
    X_train, y_train, X_val, y_val, X_test, y_test = build_data_splits_from_huggingface(cfg)
    print(
        f"[data] train={len(X_train)} val={len(X_val)} test={len(X_test)} "
        f"num_labels={get_num_labels(args.task)}",
        flush=True,
    )

    train_ds = TokSeqDataset(X_train, y_train, tokenizer, MAX_LEN)
    val_ds = TokSeqDataset(X_val, y_val, tokenizer, MAX_LEN)
    test_ds = TokSeqDataset(X_test, y_test, tokenizer, MAX_LEN)

    from transformers import DataCollatorWithPadding

    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    targs = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=1 if args.smoke else args.epochs,
        max_steps=1 if args.smoke else -1,
        per_device_train_batch_size=args.train_bs,
        per_device_eval_batch_size=args.eval_bs,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="linear",
        fp16=True,
        eval_strategy="epoch",
        save_strategy="no" if args.smoke else "epoch",
        save_total_limit=2,
        logging_steps=10,
        load_best_model_at_end=False if args.smoke else True,
        metric_for_best_model="eval_mcc_score",
        greater_is_better=True,
        report_to=[],
        dataloader_num_workers=2,
        remove_unused_columns=True,
        gradient_checkpointing=True,
    )

    per_epoch = []
    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        compute_metrics=compute_metrics,
    )
    trainer.add_callback(TestEvalCallback(trainer, test_ds, per_epoch))

    if args.smoke:
        print(
            "[smoke] running 1 training step to verify resume-load + forward/backward...",
            flush=True,
        )
        trainer.train()
        # also confirm a test-split eval works
        out = trainer.predict(test_ds, metric_key_prefix="test")
        print(
            f"[smoke] OK. 1-step done. test_mcc(pre-converge)={out.metrics.get('test_mcc_score'):.4f}",
            flush=True,
        )
        print("[smoke] resume-load + train step + test eval all succeeded.", flush=True)
        return

    print(f"[continue-ft] training {args.epochs} more epochs (lr={args.lr})...", flush=True)
    trainer.train()

    # Final: best-val model is loaded (load_best_model_at_end). Score test once more.
    final = trainer.predict(test_ds, metric_key_prefix="test")
    best_test_mcc = float(final.metrics.get("test_mcc_score", float("nan")))

    # Save the best model to the NEW dir as a plain HF model (loadable by eval_fullft_nt.py).
    best_dir = os.path.join(args.output_dir, "final_model")
    trainer.save_model(best_dir)
    tokenizer.save_pretrained(best_dir)

    # Build the per-epoch curve + summary
    val_curve = [
        {
            "epoch": e["epoch"],
            "step": e["step"],
            "val_mcc": e["eval_mcc_score"],
            "val_f1": e.get("eval_f1_score"),
        }
        for e in trainer.state.log_history
        if "eval_mcc_score" in e
    ]
    summary = {
        "task": args.task,
        "resume_from": args.resume_from,
        "added_epochs": args.epochs,
        "lr": args.lr,
        "baseline_val_mcc": 0.8885,
        "baseline_test_mcc": 0.8774,
        "paper_teacher_mcc": 0.948,
        "best_val_mcc_after_continue": (
            max((c["val_mcc"] for c in val_curve), default=None) if val_curve else None
        ),
        "final_best_model_test_mcc": best_test_mcc,
        "val_curve": val_curve,
        "per_epoch_test": per_epoch,
        "timestamp": datetime.now().isoformat(),
    }
    os.makedirs(os.path.join(args.output_dir, "_summary"), exist_ok=True)
    sp = os.path.join(args.output_dir, "_summary", "continue_summary.json")
    with open(sp, "w") as f:
        json.dump(summary, f, indent=2)
    print("\n=== CONTINUE-FT SUMMARY ===", flush=True)
    print(json.dumps(summary, indent=2), flush=True)
    print(f"summary -> {sp}", flush=True)
    print(f"best model -> {best_dir}", flush=True)


if __name__ == "__main__":
    main()
