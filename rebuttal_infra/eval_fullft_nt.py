"""R2.1.b rebuttal: evaluate already-trained FULL-fine-tune (no-LoRA) NT-2.5B teachers.

For each task in {H2AFZ, H3K27ac, promoter_tata} this loads the standard HF
``EsmForSequenceClassification`` saved at ``<parent>/<task>_finetuned/final_model/``
(NOT a LoRA adapter -> plain ``AutoModelForSequenceClassification.from_pretrained``),
tokenizes the TEST split of the SAME dataset the LoRA teacher used
(``InstaDeepAI/nucleotide_transformer_downstream_tasks_revised``, filtered to the task),
runs inference and reports test MCC (sklearn ``matthews_corrcoef``).

Mirrors the canonical pipeline in src/trainer/finetune_trainer.py (same tokenizer,
add_special_tokens=True, argmax->MCC) so numbers are apples-to-apples with the LoRA run.
Writes one JSON per task + a summary CSV under rebuttal_infra/results/ (NOT /home root, NOT /tmp).
"""

import os
import sys
import gc
import json
import csv
import glob
import argparse
from datetime import datetime

import numpy as np
import torch
from sklearn.metrics import matthews_corrcoef
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# repo root on sys.path so we can reuse src.data
REPO = "/home/pengchx3/text-dna/OmegaGenome_Revise_202606/code_carbon"
sys.path.insert(0, REPO)
from src.data.dataset import DatasetConfig, build_data_splits_from_huggingface, get_num_labels

PARENT = (
    "/extra/zhanglab0/INDV/pengchx3/NT/"
    "2b5-multi-species_nucleotide-transformer-finetune-results-NO-LORA-MULTI-GPU-"
    "epoch10-3-22-revised-deepspeed/finetuned_models"
)
# LoRA teacher parent (r=32 adapters; the distillation NT_PARENT_PATH). Per-task we pick the
# highest-val best-epoch ckpt (model-best-epochN-mcc_scoreX) and merge onto the NT-2.5b base.
LORA_PARENT = (
    "/extra/zhanglab0/INDV/pengchx3/NT/"
    "2b5-multi-species_nucleotide-transformer-finetune-results-lora-epoch20-10-17-"
    "revised-r32-fix-num-label/finetuned_models"
)
DATASET_NAME = "InstaDeepAI/nucleotide_transformer_downstream_tasks_revised"
RESULTS_DIR = os.path.join(REPO, "rebuttal_infra", "results")
MAX_LEN = 1000  # EsmTokenizer model_max_length; NT 6-mer tokens, test seqs << this


def _best_lora_ckpt(task):
    """Return the LoRA best-epoch adapter dir with the highest val mcc encoded in its name."""
    import glob, re

    cands = glob.glob(os.path.join(LORA_PARENT, f"{task}_finetuned", "model-best-epoch*"))
    if not cands:
        raise FileNotFoundError(f"no LoRA best-epoch ckpt for {task} under {LORA_PARENT}")

    def _mcc(p):
        m = re.search(r"mcc_score([0-9.]+)", os.path.basename(p))
        return float(m.group(1)) if m else -1.0

    return max(cands, key=_mcc)


@torch.no_grad()
def eval_task(task, batch_size=32, kind="full_ft", out_records=None):
    """Run test-split inference and return test MCC.

    kind="full_ft": load the standard HF EsmForSequenceClassification final_model directly.
    kind="lora":    load the NT-2.5b base + merge the best-val LoRA adapter (carries the
                    trained classifier head via modules_to_save), same eval loop.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if kind == "lora":
        adapter_dir = _best_lora_ckpt(task)
        import json as _json
        from peft import PeftModel

        base = _json.load(open(os.path.join(adapter_dir, "adapter_config.json")))[
            "base_model_name_or_path"
        ]
        print(f"[{task}|lora] base={base}  adapter={adapter_dir}", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(base, local_files_only=True)
        base_model = AutoModelForSequenceClassification.from_pretrained(
            base, num_labels=get_num_labels(task), torch_dtype=torch.float16,
            local_files_only=True,
        )
        model = PeftModel.from_pretrained(base_model, adapter_dir).merge_and_unload()
        model_dir = adapter_dir
    else:
        model_dir = os.path.join(PARENT, f"{task}_finetuned", "final_model")
        # Guard: a missing local dir makes HF silently treat the path as a repo id and crash.
        if not os.path.isfile(os.path.join(model_dir, "config.json")):
            raise FileNotFoundError(
                f"full-FT model not found for {task}: {model_dir} has no config.json "
                f"(this task's full-FT checkpoint was never saved)"
            )
        print(f"[{task}|full_ft] loading {model_dir}", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_dir, torch_dtype=torch.float16, local_files_only=True
        )
    model.to(device).eval()
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id

    cfg = DatasetConfig(task_name=task, data_path="", dataset_name=DATASET_NAME)
    _, _, _, _, X_test, y_test = build_data_splits_from_huggingface(cfg)
    y_test = list(y_test)
    print(f"[{task}] test N={len(X_test)}  num_labels={get_num_labels(task)}", flush=True)

    preds = []
    for i in range(0, len(X_test), batch_size):
        batch = list(X_test[i : i + batch_size])
        enc = tokenizer(
            batch,
            truncation=True,
            max_length=MAX_LEN,
            padding=True,
            add_special_tokens=True,
            return_tensors="pt",
        ).to(device)
        logits = model(**enc).logits.float()
        preds.extend(torch.argmax(logits, dim=-1).cpu().tolist())

    mcc = float(matthews_corrcoef(y_test, preds))
    print(f"[{task}|{kind}] test_mcc = {mcc:.4f}", flush=True)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    rec = {
        "task": task,
        "kind": kind,
        "model_dir": model_dir,
        "dataset": DATASET_NAME,
        "n_test": len(X_test),
        "test_mcc": mcc,
        "timestamp": datetime.now().isoformat(),
    }
    with open(os.path.join(RESULTS_DIR, f"{kind}_{task}.json"), "w") as f:
        json.dump(rec, f, indent=2)

    # Free GPU mem so only ONE 2.5B model is ever resident (24GB galaxy holds one at a time).
    del model
    if kind == "lora":
        del base_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return rec


def _assemble_summary():
    """Build eval_summary.csv from whatever per-task JSONs exist (robust to partial runs)."""
    rows = []
    for p in sorted(glob.glob(os.path.join(RESULTS_DIR, "*.json"))):
        try:
            r = json.load(open(p))
            if {"task", "kind", "test_mcc"} <= set(r):
                rows.append(r)
        except Exception:
            pass
    summary = os.path.join(RESULTS_DIR, "eval_summary.csv")
    with open(summary, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["task", "kind", "n_test", "test_mcc"])
        for r in sorted(rows, key=lambda r: (r["task"], r["kind"])):
            w.writerow([r["task"], r["kind"], r.get("n_test", ""), f"{r['test_mcc']:.4f}"])
    print("\n=== test MCC summary ===", flush=True)
    for r in sorted(rows, key=lambda r: (r["task"], r["kind"])):
        print(f"{r['task']:>16s}  {r['kind']:>8s}  {r['test_mcc']:.4f}  (N={r.get('n_test')})", flush=True)
    print(f"summary -> {summary}  ({len(rows)} records)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+", default=["H2AFZ", "H3K27ac", "promoter_tata"])
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument(
        "--kinds", nargs="+", default=["full_ft"], choices=["full_ft", "lora"],
        help="full_ft and/or lora; lora merges the best-val r32 adapter for the same test MCC.",
    )
    args = ap.parse_args()

    failures = []
    for kind in args.kinds:
        for t in args.tasks:
            try:
                eval_task(t, args.batch_size, kind=kind)
            except Exception as exc:  # one failure must not kill the rest
                print(f"[SKIP {t}|{kind}] {type(exc).__name__}: {exc}", flush=True)
                failures.append((t, kind, str(exc)))
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    _assemble_summary()  # built from JSONs on disk, so partial success is preserved
    if failures:
        print("\n=== SKIPPED ===", flush=True)
        for t, k, e in failures:
            print(f"  {t}|{k}: {e}", flush=True)


if __name__ == "__main__":
    main()
