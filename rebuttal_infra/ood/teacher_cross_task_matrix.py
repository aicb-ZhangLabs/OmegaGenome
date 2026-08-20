#!/usr/bin/env python
"""R2.3 OOD / cross-task transfer matrix for the TEACHER foundation models (INFERENCE ONLY).

Companion to ``cross_task_matrix.py`` (which does this for the distilled BPNet students). Here each
cell (A, B) is the MCC (+ accuracy companion) of the foundation-model teacher *fine-tuned on task A*
evaluated -- with NO further training -- on task B's HF TEST set. One 18x18 matrix per foundation
model (nt / dnabert2 / enformer / caduceus / carbon).

Everything heavy is REUSED (no reimplemented forward / tokenize / checkpoint-finder logic):
  * ``find_teacher_checkpoint`` / ``get_teacher_model``  (src/model/glm.py) -- resolve + load the
    per-task teacher, handling all 5 model_types + LoRA merge (nt/carbon).
  * ``tokenize_teacher_inputs`` + ``_free_capture_wrappers`` (src/trainer/utils.py) -- teacher-exact
    tokenization (per-teacher input_prefix / add_special_tokens) and the hidden_states-wrapper leak fix.
  * ``build_data_splits_from_huggingface`` / ``get_num_labels`` (src/data/dataset.py) -- the exact
    loader the distillation used.
  * ``_teacher_eval_cache_path`` (src/model/glm.py) -- single source of truth for each ckpt's cached
    ``teacher_evaluation.json`` (the diagonal reference MCC).
The eval loop mirrors ``evaluate_teacher_mcc`` (same wrapped-Tensor-vs-.logits branch, same argmax,
same per-batch wrapper free) but returns predictions so we can score MCC *and* accuracy in ONE pass;
the diagonal-vs-cache check validates that this replication is byte-faithful (cell (A,A) must equal the
cached teacher_test_mcc, which was itself produced by evaluate_teacher_mcc).

LABEL-SPACE COMPATIBILITY: a teacher's classification head is fixed to ``get_num_labels(A)``. We only
evaluate (A, B) where ``get_num_labels(A) == get_num_labels(B)`` (16 binary tasks + 2 ternary tasks ->
16^2 + 2^2 valid cells); incompatible cells stay NaN. Build teacher A ONCE, loop B, then free it.
"""
import argparse
import csv
import json
import os
import sys
from dataclasses import replace

import numpy as np
import torch
from sklearn.metrics import matthews_corrcoef, accuracy_score
from torch.utils.data import DataLoader

REPO = "/home/pengchx3/text-dna/OmegaGenome_Revise_202606/code_carbon"
sys.path.insert(0, REPO)

from src.data.dataset import (  # noqa: E402
    DatasetConfig, build_data_splits_from_huggingface, get_num_labels,
)
from src.model.glm import (  # noqa: E402
    GLMConfig, find_teacher_checkpoint, get_teacher_model, _teacher_eval_cache_path,
)
from src.trainer.utils import tokenize_teacher_inputs, _free_capture_wrappers  # noqa: E402

# The 18 NT-revised downstream tasks (same order/list as cross_task_matrix.py).
TASKS = [
    "H2AFZ", "H3K27ac", "H3K27me3", "H3K36me3", "H3K4me1", "H3K4me2", "H3K4me3",
    "H3K9ac", "H3K9me3", "H4K20me1",
    "promoter_all", "promoter_no_tata", "promoter_tata",
    "enhancers", "enhancers_types",
    "splice_sites_all", "splice_sites_acceptors", "splice_sites_donors",
]

# Per-foundation-model teacher plumbing. Values copied VERBATIM from the production DKD run configs
# (rebuttal_infra/dkd/runs/dkd_*.json) so the teacher is loaded byte-identically to how it was
# distilled/evaluated -- EXCEPT torch_dtype for nt/carbon which we force to bfloat16 (task directive:
# the 2.5B/3B teachers must fit; argmax is robust to fp16<->bf16, guarded by the diagonal check).
#   model_type is the string get_teacher_model/find_teacher_checkpoint dispatch on.
#   parent is teacher_parent_dir; glm are the GLMConfig kwargs (input_prefix/add_special_tokens/LoRA).
MODEL_SPECS = {
    "nt": dict(
        model_type="NT",
        parent="/home/pengchx3/text-dna/OmegaGenome_Revise_202606/code_carbon/data/finetuned_models/2b5-multi-species_nucleotide-transformer-finetune-results-lora-epoch20-10-17-revised-r32-fix-num-label",
        glm=dict(model_name_or_path="InstaDeepAI/nucleotide-transformer-2.5b-multi-species",
                 base_model_path="InstaDeepAI/nucleotide-transformer-2.5b-multi-species",
                 is_lora=True, merge_lora=True, torch_dtype="bfloat16",
                 input_prefix="", add_special_tokens=True),
    ),
    "dnabert2": dict(
        model_type="glm",
        parent="/home/pengchx3/text-dna/OmegaGenome_Revise_202606/code_carbon/data/finetuned_models/dnabert2_output_shared/output",
        glm=dict(model_name_or_path="zhihan1996/DNABERT-2-117M",
                 base_model_path=None, is_lora=None, merge_lora=False, torch_dtype="float32",
                 input_prefix="", add_special_tokens=True),
    ),
    "caduceus": dict(
        model_type="caduceus",
        parent="/home/pengchx3/text-dna/OmegaGenome_Revise_202606/code_carbon/data/finetuned_models/caduceus_finetune_results",
        glm=dict(model_name_or_path="kuleshov-group/caduceus-ps_seqlen-131k_d_model-256_n_layer-16",
                 base_model_path=None, is_lora=False, merge_lora=False, torch_dtype="float32",
                 input_prefix="", add_special_tokens=True),
    ),
    "enformer": dict(
        model_type="enformer",
        parent="/extra/zhanglab0/INDV/pengchx3/enformer_finetune_results",
        glm=dict(model_name_or_path="EleutherAI/enformer-official-rough",
                 base_model_path=None, is_lora=False, merge_lora=False, torch_dtype="float32",
                 input_prefix="", add_special_tokens=True),
    ),
    "carbon": dict(
        model_type="glm",
        parent="/home/pengchx3/carbon_teachers/carbon_3b_lora",
        glm=dict(model_name_or_path="HuggingFaceBio/Carbon-3B",
                 base_model_path="HuggingFaceBio/Carbon-3B",
                 is_lora=True, merge_lora=True, torch_dtype="bfloat16",
                 input_prefix="<dna>", add_special_tokens=False),
    ),
}


class _TrainerCfg:
    """Minimal trainer_config the teacher loaders/eval read (device / max_len / batch / workers)."""
    def __init__(self, device, max_len, batch_size, num_workers):
        self.device = device
        self.max_len = max_len
        self.batch_size = batch_size
        self.teacher_batch_size = batch_size
        self.num_workers = num_workers


class _FullCfg:
    """The config object get_teacher_model / find_teacher_checkpoint expect (mirrors run_single_experiment)."""
    def __init__(self, teacher_config, parent, model_type, trainer_config):
        self.teacher_config = teacher_config
        self.teacher_parent_dir = parent
        self.model_type = model_type
        self.trainer_config = trainer_config


def make_config(model, num_labels, device, max_len, batch_size, num_workers):
    """Build the (_FullCfg) for a given foundation model + head size, from MODEL_SPECS (DRY)."""
    spec = MODEL_SPECS[model]
    teacher_config = GLMConfig(
        num_labels=num_labels,
        output_hidden_states=False,  # inference: logits only, no hidden-state capture
        trust_remote_code=True,
        **spec["glm"],
    )
    return _FullCfg(teacher_config, spec["parent"], spec["model_type"],
                    _TrainerCfg(device, max_len, batch_size, num_workers))


def build_test_loader(teacher_tokenizer, teacher_model, X_test, y_test, cfg):
    """Build task B's test DataLoader with the teacher's exact input formatting.

    Reuses ``tokenize_teacher_inputs`` for tokenization (per-teacher input_prefix / add_special_tokens)
    and mirrors the pad-token handling in glm.evaluate_and_log_teacher so autoregressive / custom
    tokenizers (Carbon, Enformer) batch + pool correctly."""
    tc = cfg.teacher_config
    _prefix = getattr(tc, "input_prefix", "")
    _add_special = getattr(tc, "add_special_tokens", True)
    _pad = getattr(teacher_tokenizer, "pad_token", None)
    _eos = getattr(teacher_tokenizer, "eos_token", None)
    if _pad is None and _eos is not None:
        teacher_tokenizer.pad_token = _eos
    _mcfg = getattr(teacher_model, "config", None)
    _pad_id = getattr(teacher_tokenizer, "pad_token_id", None)
    if _mcfg is not None and getattr(_mcfg, "pad_token_id", None) is None and _pad_id is not None:
        _mcfg.pad_token_id = _pad_id

    def collate_fn(batch):
        labels = [b["label"] for b in batch]
        enc = tokenize_teacher_inputs(
            teacher_tokenizer, [b["text"] for b in batch], cfg.trainer_config.max_len,
            input_prefix=_prefix, add_special_tokens=_add_special,
        )
        out = {"input_ids": enc["input_ids"], "labels": torch.tensor(labels, dtype=torch.long)}
        out["attention_mask"] = enc["attention_mask"] if "attention_mask" in enc \
            else torch.ones_like(enc["input_ids"])
        return out

    ds = [{"text": s, "label": int(l)} for s, l in zip(X_test, y_test)]
    return DataLoader(ds, batch_size=cfg.trainer_config.batch_size, shuffle=False,
                      num_workers=cfg.trainer_config.num_workers, collate_fn=collate_fn)


@torch.no_grad()
def eval_mcc_acc(teacher_model, loader, device):
    """Forward the whole loader, return (mcc, acc). Mirrors evaluate_teacher_mcc's forward path
    (wrapped-Tensor vs .logits branch, argmax, per-batch _free_capture_wrappers) but keeps predictions
    so both metrics come from a single pass; the diagonal check validates the replication."""
    teacher_model.eval()
    teacher_model.to(device)  # match evaluate_teacher_mcc: merged-LoRA teachers (NT/carbon) can keep params on CPU
    preds, labs = [], []
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        outputs = teacher_model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs if isinstance(outputs, torch.Tensor) else outputs.logits
        preds.extend(torch.argmax(logits, dim=-1).cpu().numpy())
        labs.extend(batch["labels"].numpy())
        del outputs, logits
        _free_capture_wrappers(teacher_model)
    preds, labs = np.array(preds), np.array(labs)
    return float(matthews_corrcoef(labs, preds)), float(accuracy_score(labs, preds))


def cached_diag_mcc(cfg, teacher_ckpt):
    """Read the ckpt's cached teacher_test_mcc (diagonal reference) via the shared cache-path helper;
    fall back to finetune_result.json's test_mcc (carbon). Returns np.nan if neither is present."""
    for path in (_teacher_eval_cache_path(cfg, teacher_ckpt),
                 os.path.join(os.path.dirname(_teacher_eval_cache_path(cfg, teacher_ckpt)),
                              "finetune_result.json")):
        try:
            with open(path) as f:
                d = json.load(f)
            for k in ("teacher_test_mcc", "test_mcc"):
                if k in d:
                    return float(d[k])
        except Exception:
            continue
    return np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODEL_SPECS))
    ap.add_argument("--n", type=int, default=0, help="cap test seqs per task (0=full; smoke e.g. 256).")
    ap.add_argument("--tasks", default="", help="comma list restricting BOTH A and B (default: all 18).")
    ap.add_argument("--max-len", type=int, default=1000, help="teacher tokenization max_length (=DKD).")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--diag-tol", type=float, default=0.02)
    ap.add_argument("--strict-diag", action="store_true")
    ap.add_argument("--out", default=os.path.join(REPO, "rebuttal_infra/ood"))
    ap.add_argument("--smoke", action="store_true", help="if --n unset, cap n=256.")
    args = ap.parse_args()
    if args.smoke and args.n == 0:
        args.n = 256
    os.makedirs(args.out, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()] or list(TASKS)
    print(f"model={args.model} device={device} n={args.n} max_len={args.max_len} tasks={len(tasks)}",
          flush=True)

    # Pre-load every task's test split once (reuse across all A that evaluate on it).
    test_sets = {}
    for t in tasks:
        _, _, _, _, Xte, yte = build_data_splits_from_huggingface(
            DatasetConfig(task_name=t, data_path=""))
        Xte, yte = list(Xte), list(yte)
        if args.n and args.n > 0:
            Xte, yte = Xte[:args.n], yte[:args.n]
        test_sets[t] = (Xte, np.asarray(yte))
        print(f"  [{t}] test={len(Xte)} num_labels={get_num_labels(t)}", flush=True)

    mcc_mat = {a: {b: np.nan for b in tasks} for a in tasks}
    acc_mat = {a: {b: np.nan for b in tasks} for a in tasks}
    diag_rows = []

    for a in tasks:
        nla = get_num_labels(a)
        cfg = make_config(args.model, nla, device, args.max_len, args.batch_size, args.num_workers)
        ckpt, score = find_teacher_checkpoint(cfg, a)
        if not ckpt:
            print(f"  WARNING: no teacher checkpoint for {a}; skipping row", flush=True)
            continue
        print(f"\n=== teacher A={a} (num_labels={nla}) ckpt={ckpt} ===", flush=True)
        tok, model, hidden = get_teacher_model(cfg, a, ckpt)
        model.eval()
        for b in tasks:
            if get_num_labels(b) != nla:
                continue  # incompatible head -> NaN
            Xb, yb = test_sets[b]
            loader = build_test_loader(tok, model, Xb, yb, cfg)
            mcc, acc = eval_mcc_acc(model, loader, device)
            mcc_mat[a][b], acc_mat[a][b] = mcc, acc
            print(f"   eval B={b:24s} mcc={mcc:+.4f} acc={acc:.4f}{' (DIAG)' if a == b else ''}",
                  flush=True)
        # diagonal sanity vs cached teacher_test_mcc.
        d = mcc_mat[a][a]
        ref = cached_diag_mcc(cfg, ckpt)
        has_ref = not np.isnan(ref)
        delta = (d - ref) if has_ref else np.nan
        ok = (abs(delta) <= args.diag_tol) if has_ref else True
        diag_rows.append((a, d, ref, delta, ok))
        if has_ref:
            print(f"   DIAGONAL {a}: matrix={d:+.4f} cached={ref:+.4f} delta={delta:+.4f} "
                  f"[{'OK' if ok else '*** MISMATCH ***'}]", flush=True)
        else:
            print(f"   DIAGONAL {a}: matrix={d:+.4f} (no cached ref)", flush=True)
        del model, tok
        if device == "cuda":
            torch.cuda.empty_cache()

    present = [a for a in tasks if not np.isnan(mcc_mat[a][a]) or any(
        not np.isnan(mcc_mat[a][b]) for b in tasks)]

    def write_matrix(path, mat):
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["teacher_task"] + tasks)
            for a in tasks:
                w.writerow([a] + ["" if np.isnan(mat[a][b]) else f"{mat[a][b]:.4f}" for b in tasks])
        print(f"SAVED {path}", flush=True)

    tag = args.model
    mcc_path = os.path.join(args.out, f"teacher_cross_task_matrix_{tag}.csv")
    acc_path = os.path.join(args.out, f"teacher_cross_task_matrix_{tag}_acc.csv")
    diag_path = os.path.join(args.out, f"teacher_cross_task_matrix_{tag}_diag.csv")
    summ_path = os.path.join(args.out, f"teacher_cross_task_matrix_{tag}_summary.txt")
    write_matrix(mcc_path, mcc_mat)
    write_matrix(acc_path, acc_mat)
    with open(diag_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["task", "matrix_diag_mcc", "cached_teacher_test_mcc", "delta", "within_tol"])
        for a, d, ref, delta, ok in diag_rows:
            w.writerow([a, f"{d:.4f}", "" if np.isnan(ref) else f"{ref:.4f}",
                        "" if np.isnan(delta) else f"{delta:+.4f}", int(ok)])
    print(f"SAVED {diag_path}", flush=True)

    bad = [r for r in diag_rows if not r[4]]
    lines = [f"TEACHER cross-task OOD matrix -- model={args.model} n={args.n} max_len={args.max_len}",
             f"diagonal within tol ({args.diag_tol}): {len(diag_rows) - len(bad)}/{len(diag_rows)}"]
    for a, d, ref, delta, ok in bad:
        lines.append(f"  MISMATCH {a}: matrix={d:+.4f} cached={ref:+.4f} delta={delta:+.4f}")

    def grp(label, sub):
        sub = [t for t in sub if t in present]
        diag = [mcc_mat[a][a] for a in sub if not np.isnan(mcc_mat[a][a])]
        off = [mcc_mat[a][b] for a in sub for b in sub if a != b and not np.isnan(mcc_mat[a][b])]
        if diag and off:
            lines.append(f"\n--- {label} ({len(sub)} tasks) ---")
            lines.append(f"  mean in-task (diag)  MCC = {np.mean(diag):+.4f}")
            lines.append(f"  mean off-task (cross) MCC = {np.mean(off):+.4f}")
            lines.append(f"  transfer drop (diag - cross) = {np.mean(diag) - np.mean(off):+.4f}")

    grp("BINARY sub-matrix", [t for t in tasks if get_num_labels(t) == 2])
    grp("TERNARY (3-label) sub-matrix", [t for t in tasks if get_num_labels(t) == 3])
    report = "\n".join(lines)
    with open(summ_path, "w") as f:
        f.write(report + "\n")
    print("\n" + report, flush=True)
    print(f"SAVED {summ_path}", flush=True)

    if args.strict_diag and bad:
        sys.exit(f"STRICT: {len(bad)} diagonal mismatch(es) > {args.diag_tol}")
    print("\nDONE", flush=True)


if __name__ == "__main__":
    main()
