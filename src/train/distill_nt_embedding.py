#!/usr/bin/env python
"""R1.3 rebuttal: NT-embedding-input BPNet distillation (one job per task, best-HP).

Reviewer R1.3: "In the baseline, one-hot encoding is used. Wouldn't it be more reasonable
to use a sequence embedding from a model trained on lots of unlabeled data?"

This driver runs the SAME NT-2.5B -> BPNet distillation as the one-hot baseline, but feeds the
BPNet student PER-BP NT embeddings instead of one-hot (``input_mode='nt_embedding'``). Everything
else is held fixed for an apples-to-apples comparison:
  * teacher = the SAME finetuned NT-2.5B LoRA checkpoint (best val MCC) per task,
  * KD targets = the SAME teacher logits (+ optional features) the one-hot run used (reused from /
    written to the SAME ``data/cache`` via ``precompute_teacher_logits``),
  * losses / best-ckpt-by-val-MCC / eval = the SAME ``DistillationModel`` + trainer helpers,
  * hyperparameters = each task's BEST-HP row from ``rebuttal_infra/best_hp/best_hp_nt.yaml``.

The ONLY differences vs the one-hot path: (1) the student input is per-bp NT embeddings (cached by
``embedding_cache.precompute_perbp_embeddings``), and (2) the student has a tiny learned 1x1-conv
``input_adapter`` mapping embedding_dim -> 4 channels. Both are gated; the one-hot path is untouched.

Usage (one task):
    python -m src.train.distill_nt_embedding --task-name promoter_tata \
        --best-hp rebuttal_infra/best_hp/best_hp_nt.yaml \
        --results-csv <out.csv> [--epochs N] [--max-steps N] [--smoke]
"""

import os
os.environ.setdefault("PYTORCH_NVML_BASED_CUDA_CHECK", "1")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys
import csv
import json
import time
import argparse
import numpy as np
import torch
import wandb
import yaml
from datetime import datetime

# DEADLOCK DIAGNOSTIC: dump ALL thread stacks to stderr (slurm log) every 150s. Runs in-process (no
# ptrace/sudo needed), so a hang between "Student params" and epoch 1 prints the exact frozen line.
import faulthandler as _fh, sys as _sys
_fh.enable()  # dump a traceback on a *fatal* signal (segfault etc.) -- cheap, no background thread
# REMOVED dump_traceback_later(150, repeat=True): that periodic background stack-walker was a diagnostic
# for the (now-fixed) wandb deadlock. It walks every thread's frames every 150s while the main thread is
# deep in native CUDA/numpy, a plausible trigger for the remaining intermittent segfaults that hopped
# between _expand_to_perbp (numpy) and torch conv forward across different nodes. Not needed anymore.

from torch.utils.data import DataLoader

from src.model.bpnet_classifier import BPNetClassifier, BPNetClassifierConfig
from src.model.distillation import DistillationModel, DistillationModelConfig
from src.model.glm import GLMConfig, find_teacher_checkpoint, get_teacher_model, evaluate_and_log_teacher
from src.data.dataset import (
    get_num_labels,
    build_data_splits_from_huggingface,
    DatasetConfig,
    EmbeddingSeqDataset,
    SeqDataset,
)
from src.trainer.utils import precompute_teacher_logits
from src.trainer.embedding_cache import precompute_perbp_embeddings, embedding_cache_hit
from src.trainer.distill_trainer import (
    evaluate,
    save_checkpoint,
    _early_stop_step,
    create_run_directory,
)

# NT-2.5B teacher LoRA parent (epoch20, the dir whose model-best*mcc_score* ckpts the one-hot run used).
NT_PARENT_PATH_DEFAULT = (
    "/extra/zhanglab0/INDV/pengchx3/OmegaGenome_different_version/OmegaGenome_11_1_fix_ckpt/"
    "OmegaGenome/data/finetuned_models/"
    "2b5-multi-species_nucleotide-transformer-finetune-results-lora-epoch20-10-17-revised-r32-fix-num-label"
)
NT_EMBEDDING_DIM = 2560  # NT-2.5B hidden size (per-token / per-bp embedding width)

# DNABERT-2 embedding teacher (R1.3, fast variant): 117M BPE encoder, hidden size 768. Used ONLY as
# the per-bp EMBEDDING source for the student input; the KD targets (logits/features) still come from
# the SAME NT-2.5B teacher, so the comparison is purely "which embedding feeds the student input".
DNABERT2_MODEL = "zhihan1996/DNABERT-2-117M"
DNABERT2_EMBEDDING_DIM = 768


def _disable_dnabert2_flash_attn() -> int:
    """Force DNABERT-2's PyTorch attention fallback by nulling its Triton flash-attn fn.

    DNABERT-2 (trust_remote_code) lands its ``bert_layers`` module in sys.modules under a
    transformers_modules.* path. Setting ``flash_attn_qkvpacked_func`` to None there makes
    BertUnpadSelfAttention take the pure-PyTorch branch (numerically equivalent; only the kernel
    differs), avoiding the Triton>=3 ``trans_b`` incompat. Returns #modules patched. Mirrors the
    helper in benchmark_inference.py.
    """
    import sys
    patched = 0
    for name, mod in list(sys.modules.items()):
        if mod is None:
            continue
        # Match by the module suffix + the symbol's presence (robust to the local dir name, which may
        # not contain "DNABERT" -- e.g. a 'dnabert2_local' assembled snapshot dir).
        if name.endswith("bert_layers") and hasattr(mod, "flash_attn_qkvpacked_func"):
            mod.flash_attn_qkvpacked_func = None
            patched += 1
    return patched


def load_dnabert2_embedder(device: str, model_path: str = ""):
    """Load base DNABERT-2-117M as an embedding teacher (tokenizer, model, hidden_size=768).

    Uses ``AutoModel`` (the encoder, not a seq-classification head). The Triton flash-attn kernel is
    disabled in-process (pure-PyTorch attention fallback) for Triton>=3 compatibility. ``model_path``
    is a local assembled dir (offline); empty falls back to the HF id.
    """
    from transformers import AutoTokenizer, AutoModel
    src = model_path or DNABERT2_MODEL
    tok = AutoTokenizer.from_pretrained(src, trust_remote_code=True)
    model = AutoModel.from_pretrained(src, trust_remote_code=True)
    _patched = _disable_dnabert2_flash_attn()
    print(f"[dnabert2] Triton flash-attn disabled on {_patched} module(s) -> PyTorch attention fallback")
    model = model.to(device).eval()
    hidden = int(getattr(model.config, "hidden_size", DNABERT2_EMBEDDING_DIM))
    return tok, model, hidden


def make_dnabert2_hidden_state_fn():
    """Return a ``hidden_state_fn`` for precompute_perbp_embeddings that reads DNABERT-2 layer states.

    DNABERT-2's ``BertModel`` returns ``(encoded_layers, pooled)`` (no HF ``hidden_states`` attr), and
    its ``output_all_encoded_layers=True`` path emits UNPADDED ``[total_nnz, H]`` states (the pooler
    even crashes on it). To read a clean PADDED ``[B, T, H]`` layer-L output we instead TEMPORARILY
    slice ``encoder.layer`` to the first L blocks and call ``output_all_encoded_layers=False`` (returns
    the padded last-of-the-slice state). embedding_layer=None -> middle (n//2); -1 -> all 12 layers
    (full last layer); >=1 -> that many leading blocks; **0 -> the RAW BPE token embeddings** (the
    pre-transformer ``model.embeddings`` output, no context -- the "learned BPE/byte encoding" control).
    """
    import torch.nn as nn

    def _fn(model, input_ids, attention_mask, embedding_layer):
        n_layers = len(model.encoder.layer)
        if embedding_layer == 0:
            # Layer 0 = raw BPE token embeddings BEFORE any transformer block (no context). Padded
            # [B, T, 768] directly from the embedding module. token_type_ids default to zeros.
            tti = torch.zeros_like(input_ids)
            emb = model.embeddings(input_ids, tti)
            return emb, 0
        if embedding_layer is None:
            keep = n_layers // 2
        elif embedding_layer == -1 or embedding_layer >= n_layers:
            keep = n_layers
        else:
            keep = max(1, embedding_layer)
        orig = model.encoder.layer
        try:
            if keep != n_layers:
                model.encoder.layer = nn.ModuleList(list(orig)[:keep])
            seq_out, _ = model(
                input_ids=input_ids, attention_mask=attention_mask,
                output_all_encoded_layers=False,
            )  # padded [B, T, 768]
        finally:
            model.encoder.layer = orig
        return seq_out, keep

    return _fn


def load_best_hp(yaml_path: str, task_name: str) -> dict:
    """Return the best-HP dict for ``task_name`` from the rebuttal best_hp_nt.yaml.

    Maps the YAML field names (weight_ce/weight_kl/weight_mse) onto the trainer/loss names used
    here. Raises if the task is absent so a bad task name fails loud, not silent.
    """
    with open(yaml_path) as f:
        all_hp = yaml.safe_load(f)
    if task_name not in all_hp:
        raise KeyError(f"task '{task_name}' not in {yaml_path}")
    hp = all_hp[task_name]["hyperparameters"]
    return {
        "weight_ce": float(hp["weight_ce"]),
        "weight_kl": float(hp["weight_kl"]),
        "weight_mse": float(hp["weight_mse"]),
        "temperature": float(hp["temperature"]),
        "distill_method": hp.get("distill_method", "vanilla"),
        "lr": float(hp["lr"]),
        "batch_size": int(hp["batch_size"]),
        "epochs": int(hp["epochs"]),
        "model_size": all_hp[task_name]["metadata"].get("model_size", "original"),
    }


def count_params(model) -> dict:
    """Param-count breakdown for the embedding-input student (reported in the results table).

    ``front_end`` is the embedding-injection cost (adapter / stem_adapter+embed_proj+fuse_block for
    latefuse) computed as total - classifier - backbone-conv-stack, so it is correct across all three
    arms (replace4 / replaceK / latefuse) even though latefuse splits the backbone into _early/_late.
    """
    total = sum(p.numel() for p in model.parameters())
    classifier = sum(p.numel() for p in model.classifier.parameters())
    # Front-end modules across arms (any that exist).
    front_modules = []
    for attr in ("input_adapter", "stem_adapter", "embed_proj", "_fuse_block"):
        m = getattr(model, attr, None)
        if m is not None:
            front_modules.append(m)
    front_end = sum(p.numel() for m in front_modules for p in m.parameters())
    # Backbone = the conv stack. For latefuse it lives in _early/_late (plus the original backbone may
    # still hold the now-unused profile/total_count heads); approximate as total - classifier - front_end.
    backbone = total - classifier - front_end
    return {"total": total, "backbone": backbone, "classifier": classifier,
            "input_adapter": front_end}


def append_results_row(csv_path: str, row: dict):
    """Append one task's result to the shared results CSV (header written once)."""
    fieldnames = [
        "task", "input_mode", "embedding_source", "embedding_layer", "front_end",
        "student_best_val_mcc", "student_best_test_mcc",
        "teacher_test_mcc", "student_total_params", "student_backbone_params",
        "student_adapter_params", "best_epoch", "weight_ce", "weight_kl", "weight_mse",
        "temperature", "lr", "batch_size", "epochs_run", "embedding_dim", "timestamp",
    ]
    exists = os.path.isfile(csv_path)
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in fieldnames})


def main():
    p = argparse.ArgumentParser(description="NT-embedding-input BPNet distillation (R1.3)")
    p.add_argument("--task-name", required=True)
    p.add_argument("--best-hp", default="/home/pengchx3/text-dna/OmegaGenome_Revise_202606/rebuttal_infra/best_hp/best_hp_nt.yaml")
    p.add_argument("--nt-parent", default=NT_PARENT_PATH_DEFAULT)
    p.add_argument("--max-len", type=int, default=1000)
    p.add_argument("--teacher-batch-size", type=int, default=8, help="batch for the 2.5B teacher forward")
    p.add_argument("--epochs", type=int, default=0, help="override best-HP epochs (0 = use best-HP value)")
    p.add_argument("--max-steps", type=int, default=0, help="cap optimizer steps (0 = unlimited); for smoke tests")
    p.add_argument("--early-stop-patience", type=int, default=None,
                   help="stop training if val MCC has not improved for this many epochs (best-val ckpt is "
                        "still reloaded for the test eval, so the reported MCC is unchanged). None/0 = full schedule.")
    p.add_argument("--num-workers", type=int, default=8,
                   help="DataLoader worker processes (result-preserving: parallel fetch/collate only). "
                        "Matters for the 2560-dim embedding input; 0 = single-threaded (old behavior).")
    p.add_argument("--smoke", action="store_true", help="subset data + few steps to validate end-to-end")
    p.add_argument("--emb-cache", type=lambda s: s.lower() in ("1", "true", "yes"), default=None,
                   help="force embedding-cache use on/off (default: off in --smoke, on otherwise). Set to 1 "
                        "in a smoke to EXERCISE the cache HIT (in-RAM) read path on the sshfs node.")
    p.add_argument("--input-mode", choices=["nt_embedding", "onehot"], default="nt_embedding",
                   help="student input front-end. 'nt_embedding' (DEFAULT, the R1.3 experiment) feeds per-bp "
                        "NT embeddings; 'onehot' is a MATCHED CONTROL that runs the SAME teacher/KD-targets/"
                        "trainer/data-splits but with the one-hot BPNet student, for apples-to-apples diagnosis.")
    p.add_argument("--embedding-source", choices=["nt", "nt_base", "dnabert2"], default="nt",
                   help="WHICH pretrained model provides the student's per-bp INPUT embeddings. 'nt' "
                        "(default) reuses the NT-2.5B FINE-TUNED teacher; 'nt_base' uses the RAW base "
                        "NT-2.5B foundation model (NO LoRA, un-fine-tuned, hidden 2560); 'dnabert2' uses "
                        "base DNABERT-2-117M (much faster, hidden 768). The KD TARGETS (logits/features) "
                        "ALWAYS come from the NT-2.5B fine-tuned teacher either way, so this isolates the "
                        "input-representation effect.")
    p.add_argument("--embedding-layer", type=int, default=-999,
                   help="teacher hidden-state layer for the input embeddings. -999 (default) = MIDDLE "
                        "layer (n//2; more transferable than the task-specialized last layer); -1 = last "
                        "layer; >=0 = that exact index.")
    p.add_argument("--dnabert2-path", default="",
                   help="local dir with the assembled DNABERT-2 model+code (offline). If empty, falls back "
                        "to the HF id 'zhihan1996/DNABERT-2-117M'.")
    p.add_argument("--fusion", choices=["replace4", "replaceK", "latefuse_emb", "latefuse_onehot"],
                   default="replace4",
                   help="embedding FRONT-END (reuses cached embeddings; no re-precompute). replace4 "
                        "(default/control): adapter D->4 (capacity-matched bottleneck). replaceK: adapter "
                        "D->K + widened stem (no D->4 bottleneck). latefuse_emb: deep full-width embedding "
                        "concat on top of an embedding-derived 4-ch stem. latefuse_onehot: FAITHFUL -- "
                        "REAL one-hot stem + deep embedding concat (tests value-ADD over a real one-hot BPNet).")
    p.add_argument("--adapter-width", type=int, default=32, help="K for --fusion replaceK")
    p.add_argument("--adapter-mlp-hidden", type=int, default=0,
                   help="input_adapter projection depth for replace4/replaceK. 0 (default) = single 1x1 "
                        "conv (per-position LINEAR D->out). >0 = 2-layer MLP Conv1d(D->h)->ReLU->Conv1d(h->out) "
                        "(nonlinear per-position projection; first layer ~D*h params, D=2560).")
    p.add_argument("--fuse-width", type=int, default=32, help="embedding projection width for --fusion latefuse")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--model-size", default="",
                   help="OVERRIDE the best-HP student model_size (e.g. 'original' to force the 0.12M "
                        "deployable student even when best_hp lists 'medium'). Empty = use the best-HP "
                        "value. The best-HP LOSS weights/temperature/lr/epochs are unchanged; only the "
                        "BPNet student width changes. Use this for an apples-to-apples original-size "
                        "comparison (and to enable latefuse_onehot, which requires original).")
    p.add_argument("--cache-base", default="", help="base dir for embedding cache (default: nt-parent's grandparent /data root)")
    p.add_argument("--output-dir", default="")
    p.add_argument("--results-csv", default="")
    p.add_argument("--wandb-project", default="OmegaGenome-R13-NT-Embedding")
    p.add_argument("--wandb-mode", default=os.environ.get("WANDB_MODE", "online"))
    args = p.parse_args()

    os.environ["WANDB_MODE"] = args.wandb_mode
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    task_name = args.task_name
    num_labels = get_num_labels(task_name)

    # -999 sentinel -> middle layer (None passed to the cache, which resolves n//2 per teacher).
    embedding_layer = None if args.embedding_layer == -999 else args.embedding_layer
    # Per-bp input width depends on the embedding source (DNABERT-2=768, NT-2.5B=2560).
    emb_input_dim = DNABERT2_EMBEDDING_DIM if args.embedding_source == "dnabert2" else NT_EMBEDDING_DIM

    hp = load_best_hp(args.best_hp, task_name)
    if args.model_size:
        # Override the student width (loss HP unchanged). Lets us run the original-size deployable
        # student even when best_hp lists 'medium' (e.g. splice_sites_all), and enables latefuse_onehot.
        print(f"[model-size override] best-HP model_size '{hp['model_size']}' -> '{args.model_size}'")
        hp["model_size"] = args.model_size
    epochs = args.epochs or hp["epochs"]
    batch_size = hp["batch_size"]
    if args.smoke:
        epochs = min(epochs, 3)

    # From-scratch (no KD) uses NEITHER teacher logits/features NOR the teacher for anything the student
    # consumes -- the fine-tuned NT-2.5B teacher would be loaded ONLY to log the reference teacher MCC.
    from_scratch = hp["weight_kl"] == 0 and hp["weight_mse"] == 0
    # SKIP_TEACHER_EVAL=1 (env): on from-scratch arms, skip the fine-tuned NT-2.5B teacher load + eval
    # entirely (removes one huge sshfs model load from the stall path; teacher_test_mcc CSV col stays
    # empty -- we have it from Exp-1b). NEVER skip when the 'nt' embedding source REUSES the teacher as
    # the per-bp embedder (that path genuinely needs it); nt_base/dnabert2/onehot arms do not.
    skip_teacher = (
        os.environ.get("SKIP_TEACHER_EVAL") == "1"
        and from_scratch
        and not (args.input_mode == "nt_embedding" and args.embedding_source == "nt")
    )

    print(f"\n{'='*64}\nR1.3 NT-EMBEDDING DISTILL: {task_name}\n"
          f"best-HP: {hp}\nepochs={epochs} max_steps={args.max_steps} smoke={args.smoke}\n{'='*64}\n")

    # ---- teacher (NT-2.5B LoRA, best-val ckpt) ----
    NT_BASE = "InstaDeepAI/nucleotide-transformer-2.5b-multi-species"
    teacher_config = GLMConfig(
        model_name_or_path=NT_BASE,
        num_labels=num_labels, trust_remote_code=True, output_hidden_states=True,
        is_lora=True, base_model_path=NT_BASE, merge_lora=True,
    )

    class _Cfg:
        teacher_config = None
        teacher_parent_dir = args.nt_parent
        model_type = "nt"
        class trainer_config:  # noqa: N801
            device = None
    _Cfg.teacher_config = teacher_config
    _Cfg.trainer_config.device = device

    if skip_teacher:
        # SKIP_TEACHER_EVAL from-scratch: no fine-tuned teacher is needed (no KD targets, no reference
        # MCC). Leave teacher handles None; nothing downstream dereferences them on this path.
        print("[SKIP_TEACHER_EVAL] from-scratch + non-'nt' embedding -> skipping fine-tuned NT-2.5B "
              "teacher load & eval (teacher_test_mcc CSV col stays empty)")
        teacher_ckpt, score = None, None
        teacher_tokenizer, teacher_model, teacher_hidden = None, None, None
    else:
        teacher_ckpt, score = find_teacher_checkpoint(_Cfg, task_name)
        if teacher_ckpt is None:
            print(f"No teacher checkpoint for {task_name}; aborting")
            return 1
        print(f"Teacher ckpt: {teacher_ckpt} (val mcc {score})")

        teacher_tokenizer, teacher_model, teacher_hidden = get_teacher_model(_Cfg, task_name, teacher_ckpt)
        # build_glm (LoRA-merge path) returns the teacher on CPU; the one-hot baseline moves it to device
        # inside train_distill_task (distill_trainer.py:255). This driver calls precompute_teacher_logits /
        # precompute_perbp_embeddings DIRECTLY, which tokenize inputs onto `device`, so the teacher MUST be
        # on the same device or the first forward dies with a cpu/cuda index_select mismatch. Move it here.
        teacher_model = teacher_model.to(device)
        teacher_model.eval()
        assert teacher_hidden == NT_EMBEDDING_DIM, f"expected NT hidden {NT_EMBEDDING_DIM}, got {teacher_hidden}"

    # ---- data ----
    ds_cfg = DatasetConfig(task_name=task_name, data_path="", random_state=args.seed)
    X_train, y_train, X_val, y_val, X_test, y_test = build_data_splits_from_huggingface(ds_cfg)
    if args.smoke:
        n = batch_size * 4
        X_train, y_train = X_train[:n], y_train[:n]
        X_val, y_val = X_val[: batch_size * 2], y_val[: batch_size * 2]
        X_test, y_test = X_test[: batch_size * 2], y_test[: batch_size * 2]
        print(f"[SMOKE] subset -> train {len(X_train)} val {len(X_val)} test {len(X_test)}")

    # nt_parent = .../OmegaGenome/data/finetuned_models/<teacher>; the teacher logit/feature cache the
    # one-hot run wrote lives at .../OmegaGenome/data/cache/<teacher>/<task>, and _get_cache_dir appends
    # 'data/cache/...', so cache_base must be .../OmegaGenome (3 levels up) to HIT that existing cache.
    cache_base = args.cache_base or os.path.dirname(os.path.dirname(os.path.dirname(args.nt_parent)))
    needs_logits = hp["weight_kl"] > 0
    needs_features = hp["weight_mse"] > 0

    # ---- KD TARGETS: reuse the SAME pooled teacher logits/features cache the one-hot run used ----
    # From-scratch (weight_kl=weight_mse=0) uses NEITHER target, so skip the ~2.5 h teacher forward over
    # the train set entirely -- the loss multiplies them by 0 anyway, the dataset accepts None logits
    # (returns (x, y)), and prepare_batch sets tlog=None when weight_kl<=0. Result is identical. The
    # per-bp embedding INPUT is separate (cached below) and unaffected. With-KD arms keep the precompute.
    if needs_logits or needs_features:
        train_tlogits, train_tfeatures = precompute_teacher_logits(
            teacher_tokenizer, teacher_model, X_train, args.teacher_batch_size, device, args.max_len,
            needs_logits=needs_logits, needs_features=needs_features,
            project_path=cache_base, teacher_parent_dir=args.nt_parent,
            task_name=task_name, teacher_ckpt=teacher_ckpt, use_cache=True,
        )
    else:
        train_tlogits, train_tfeatures = None, None
        print("[from-scratch] weight_kl=weight_mse=0 -> skipping teacher-logit precompute (saves the teacher forward)")

    # ---- STUDENT INPUT ----
    # nt_embedding (default, the experiment): per-bp NT embeddings (cached).
    # onehot (matched control): no embeddings; the student one-hot-encodes char ids itself.
    emb_train = emb_val = emb_test = None
    dnabert2_model = None
    if args.input_mode == "nt_embedding":
        # Whether the per-bp embedding cache is consulted (off in smoke unless --emb-cache forces it).
        # Computed once here so the nt_base cache-hit skip and the precompute call stay consistent.
        use_emb_cache = args.emb_cache if args.emb_cache is not None else not args.smoke
        if args.embedding_source == "dnabert2":
            # Embedding source = base DNABERT-2-117M (fast). KD targets stay NT-2.5B (above). The cache
            # lands under a DNABERT-2-tagged teacher_parent_dir so it never collides with the NT cache.
            emb_tokenizer, dnabert2_model, dnabert2_hidden = load_dnabert2_embedder(device, args.dnabert2_path)
            assert dnabert2_hidden == emb_input_dim, f"DNABERT-2 hidden {dnabert2_hidden} != {emb_input_dim}"
            emb_model = dnabert2_model
            emb_teacher_parent = os.path.join(os.path.dirname(args.nt_parent), "dnabert2_emb")
            emb_ckpt = f"dnabert2-117m-base/{DNABERT2_MODEL}"
            emb_hsfn = make_dnabert2_hidden_state_fn()
        elif args.embedding_source == "nt_base":
            # Embedding source = the RAW base NT-2.5B foundation model (NO LoRA, un-fine-tuned). The KD
            # TARGETS (P2) and the reference teacher MCC (P1) still come from the FINE-TUNED NT-2.5B
            # teacher (loaded above); ONLY the student INPUT embedding changes. The per-bp cache lands
            # under a DISTINCT "nt_base_emb" teacher_parent_dir so it never collides with the fine-tuned
            # "nt_adapters" cache. hidden_states are the standard HF tuple (same class as the teacher),
            # so the SAME mid-layer extraction (emb_hsfn=None) applies; the random classification head is
            # unused (only hidden_states[mid] is read).
            emb_teacher_parent = os.path.join(os.path.dirname(args.nt_parent), "nt_base_emb")  # DISTINCT cache
            emb_ckpt = "nt-2.5b-base"
            emb_hsfn = None  # base NT has standard hidden_states -> same mid-layer extraction as the fine-tuned NT
            # LOAD-ONCE: the base NT-2.5B is a ~10GB sshfs load whose read intermittently STALLS. When the
            # per-bp cache for ALL three splits is already present+valid, precompute_perbp_embeddings reads
            # purely from disk and NEVER dereferences emb_model/emb_tokenizer, so skip the load entirely.
            # Only the FIRST embedding arm of a task (cache MISS -> replace4) actually loads base-NT to
            # BUILD the cache; replaceK/latefuse hit the cache and load nothing.
            cache_ready = use_emb_cache and all(
                embedding_cache_hit(cache_base, emb_teacher_parent, task_name, seqs, emb_ckpt,
                                    args.max_len, split=sp, embedding_layer=embedding_layer)
                for sp, seqs in (("train", X_train), ("val", X_val), ("test", X_test)))
            if cache_ready:
                print("[load-once] nt_base per-bp cache HIT for train/val/test -> skipping base NT-2.5B load")
                emb_tokenizer, emb_model = None, None
            else:
                from transformers import AutoTokenizer, AutoModelForSequenceClassification
                emb_tokenizer = AutoTokenizer.from_pretrained(NT_BASE, trust_remote_code=True)
                _base = AutoModelForSequenceClassification.from_pretrained(
                    NT_BASE, num_labels=num_labels, output_hidden_states=True, trust_remote_code=True)
                emb_model = _base.to(device).eval()
        else:  # 'nt': reuse the NT-2.5B teacher as the embedding source
            emb_tokenizer, emb_model = teacher_tokenizer, teacher_model
            emb_teacher_parent, emb_ckpt, emb_hsfn = args.nt_parent, teacher_ckpt, None
        # EMB_LOAD_IN_RAM=0 (default here): read the per-bp cache via mmap_mode='r' instead of loading
        # the full ~27GB fp16 array into RAM. The cache now lives in node-local /dev/shm (RAM), so mmap
        # reads are fast, and this avoids both the 27GB in-RAM copy AND the num_workers fork-COW blowup
        # that was deadlocking the DataLoader on the large embedding cache.
        emb_kw = dict(
            batch_size=args.teacher_batch_size, device=device, max_length=args.max_len,
            cache_base=cache_base, teacher_parent_dir=emb_teacher_parent, task_name=task_name,
            teacher_ckpt=emb_ckpt, use_cache=use_emb_cache,
            embedding_layer=embedding_layer, hidden_state_fn=emb_hsfn,
            load_in_ram=(os.environ.get("EMB_LOAD_IN_RAM", "1") == "1"),  # default True = working Exp-1b behavior
        )
        emb_train = precompute_perbp_embeddings(emb_tokenizer, emb_model, X_train, split="train", **emb_kw)
        emb_val = precompute_perbp_embeddings(emb_tokenizer, emb_model, X_val, split="val", **emb_kw)
        emb_test = precompute_perbp_embeddings(emb_tokenizer, emb_model, X_test, split="test", **emb_kw)

    # ---- teacher test MCC (cached eval; for the results table) ----
    teacher_mcc = None
    if not args.smoke and teacher_model is not None:
        run_dir_for_teacher = create_run_directory(
            args.output_dir or os.path.join(cache_base, "output", "r13_nt_embedding"),
            task_name, DistillationModelConfig(**{k: hp[k] for k in ("weight_ce", "weight_kl", "weight_mse", "temperature")}),
        )
        try:
            teacher_mcc = float(evaluate_and_log_teacher(
                teacher_model, teacher_tokenizer, X_test, y_test, task_name, _Cfg, run_dir_for_teacher, teacher_ckpt,
            ))
        except Exception as e:
            print(f"teacher eval skipped: {e}")

    # Free the 2.5B teacher (+ DNABERT-2 embedder if loaded); the student trains from cached arrays only.
    # teacher_model is None when SKIP_TEACHER_EVAL skipped the load (from-scratch); nothing to free then.
    if teacher_model is not None:
        del teacher_model
    if dnabert2_model is not None:
        del dnabert2_model
    if device == "cuda":
        torch.cuda.empty_cache()

    # ---- student (input_mode-gated: nt_embedding front-end vs one-hot control) ----
    # NOTE: input embedding_dim follows the EMBEDDING SOURCE (DNABERT-2=768 / NT-2.5B=2560); the
    # teacher_hidden_size (KD feature target) always follows the NT-2.5B teacher (=2560).
    student_cfg = BPNetClassifierConfig(
        num_labels=num_labels, model_type="bpnet", model_size=hp["model_size"],
        input_mode=args.input_mode,
        embedding_dim=(emb_input_dim if args.input_mode == "nt_embedding" else None),
        teacher_hidden_size=(NT_EMBEDDING_DIM if needs_features else None),
        front_end=args.fusion, adapter_width=args.adapter_width, fuse_width=args.fuse_width,
        adapter_mlp_hidden=args.adapter_mlp_hidden,
    )
    model = BPNetClassifier(student_cfg).to(device)
    params = count_params(model)
    print(f"Student params: {params}")

    distill_cfg = DistillationModelConfig(
        weight_ce=hp["weight_ce"], weight_kl=hp["weight_kl"], weight_mse=hp["weight_mse"],
        temperature=hp["temperature"], distill_method=hp["distill_method"],
    )
    distill_model = DistillationModel(distill_cfg, torch.nn.Identity(), model, device).to(device)

    # ---- datasets / loaders (identical KD-target contract across both input modes) ----
    if args.input_mode == "nt_embedding":
        # latefuse_onehot needs the REAL one-hot threaded alongside the cached embedding (packed into
        # element 0 as [L, D+4]); pass sequences+max_len so the dataset builds it. Other arms: embedding only.
        oh = args.fusion == "latefuse_onehot"
        train_ds = EmbeddingSeqDataset(emb_train, y_train, train_tlogits, train_tfeatures,
                                       sequences=(X_train if oh else None), max_len=args.max_len)
        val_ds = EmbeddingSeqDataset(emb_val, y_val, sequences=(X_val if oh else None), max_len=args.max_len)
        test_ds = EmbeddingSeqDataset(emb_test, y_test, sequences=(X_test if oh else None), max_len=args.max_len)
    else:  # onehot control: char-index ids, same teacher logits/features KD targets
        train_ds = SeqDataset(X_train, y_train, args.max_len, train_tlogits, train_tfeatures)
        val_ds = SeqDataset(X_val, y_val, args.max_len)
        test_ds = SeqDataset(X_test, y_test, args.max_len)
    pin = device == "cuda"
    # Data-pipeline parallelism (RESULT-PRESERVING): the sampler still decides batch composition, so
    # workers only fetch/collate faster -- identical batches, identical training. Critical here because
    # the per-bp NT embedding input is 2560-dim (vs 4 for one-hot), so single-threaded loading starves
    # the GPU. Workers share the RAM-loaded fp16 cache via fork COW (reads don't copy) -> no RAM blowup.
    nw = max(0, args.num_workers)
    # pin_memory only WITH workers. With num_workers=0 the main thread runs __getitem__ (numpy
    # _expand_to_perbp) while torch's pin-memory thread copies each batch concurrently; against a
    # multi-threaded numpy/BLAS backend this raced and segfaulted intermittently in _expand_to_perbp
    # (native crash, garbage frame line#). No workers -> no pin thread -> race gone. (Thread env vars
    # OMP/OPENBLAS/MKL=1 in the launcher also serialize numpy as a second guard.)
    lk = dict(pin_memory=(pin and nw > 0), num_workers=nw)
    if nw > 0:
        lk.update(persistent_workers=True, prefetch_factor=4)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, **lk)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, **lk)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, **lk)

    # ---- output / wandb ----
    out_dir = args.output_dir or os.path.join(cache_base, "output", "r13_nt_embedding")
    run_dir = create_run_directory(out_dir, task_name, distill_cfg)
    # FUNDAMENTAL DEADLOCK FIX (faulthandler-confirmed): the training step blocked forever on wandb's
    # mailbox IPC to its internal service process (wandb/sdk/mailbox), which hangs under this cluster's
    # runtime. mode="disabled" makes wandb.init a no-op (RunDisabled) -- no service, no mailbox, no
    # deadlock. Honour WANDB_MODE if it is offline/online; otherwise force disabled. Val curves are
    # still written to the results CSV, so nothing needed is lost.
    _wm = args.wandb_mode if args.wandb_mode in ("offline", "online") and os.environ.get("WANDB_FORCE_DISABLE") != "1" else "disabled"
    wandb.init(project=args.wandb_project, name=f"{task_name}/{args.input_mode}/{hp['model_size']}",
               dir=cache_base, config={**hp, **params, "input_mode": args.input_mode},
               tags=[task_name, "nt", args.input_mode, "r1.3"], mode=_wm)

    # ---- train (reuses DistillationModel + trainer helpers; best-ckpt by val MCC) ----
    optimizer = torch.optim.AdamW(model.parameters(), lr=hp["lr"])
    best_val_mcc, best_epoch, epochs_no_improve = -1.0, 0, 0
    global_step = 0
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            loss, metrics = distill_model(batch)
            loss.backward()
            optimizer.step()
            global_step += 1
            total_loss += loss.item()
            if args.max_steps and global_step >= args.max_steps:
                break
        val_metrics = evaluate(model, val_loader, device)
        is_best = val_metrics["mcc"] > best_val_mcc
        # Early-stop is opt-in (default None = full schedule, matching the R7 6-task runs). The BEST-VAL
        # checkpoint is saved below and reloaded for the test eval, so the reported MCC is identical to
        # what a full run would report -- patience only trims the flat post-best tail.
        epochs_no_improve, should_stop = _early_stop_step(is_best, epochs_no_improve, args.early_stop_patience)
        if is_best:
            best_val_mcc, best_epoch = val_metrics["mcc"], epoch
        save_checkpoint(model, epoch, val_metrics["mcc"], run_dir, is_best=is_best)
        wandb.log({"epoch": epoch, "train/epoch_loss": total_loss / max(1, len(train_loader)),
                   "val/mcc": val_metrics["mcc"], "best_val_mcc": best_val_mcc,
                   "time/elapsed_s": time.time() - t0}, step=global_step)
        print(f"[{task_name}] epoch {epoch}/{epochs} loss {total_loss/max(1,len(train_loader)):.4f} "
              f"val_mcc {val_metrics['mcc']:.4f}{' NEW BEST' if is_best else ''}")
        if args.max_steps and global_step >= args.max_steps:
            print("[max-steps reached]")
            break
        if should_stop:
            print(f"[early-stop] no val-MCC improvement for {args.early_stop_patience} epochs "
                  f"(best {best_val_mcc:.4f} @ epoch {best_epoch}); stopping at epoch {epoch}/{epochs}")
            break

    # ---- best-ckpt test eval ----
    best_dir = os.path.join(run_dir, "best_model")
    best_test_mcc = None
    if os.path.exists(best_dir):
        model.load_state_dict(torch.load(os.path.join(best_dir, "student.pt"), map_location=device))
        best_test_mcc = evaluate(model, test_loader, device)["mcc"]
    final_test = evaluate(model, test_loader, device)
    wandb.log({"best_test/mcc": best_test_mcc if best_test_mcc is not None else final_test["mcc"],
               "final_test/mcc": final_test["mcc"]})

    summary = {
        "task": task_name, "input_mode": args.input_mode,
        "best_val_mcc": best_val_mcc, "best_epoch": best_epoch,
        "best_test_mcc": best_test_mcc, "final_test_mcc": final_test["mcc"],
        "teacher_test_mcc": teacher_mcc, "params": params, "hp": hp,
    }
    with open(os.path.join(run_dir, "final_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nFINAL {task_name}: best_val_mcc {best_val_mcc:.4f} best_test_mcc "
          f"{best_test_mcc if best_test_mcc is None else round(best_test_mcc,4)} "
          f"teacher {teacher_mcc}")

    if args.results_csv:
        append_results_row(args.results_csv, {
            "task": task_name, "input_mode": args.input_mode,
            "embedding_source": (args.embedding_source if args.input_mode == "nt_embedding" else ""),
            "embedding_layer": ("mid" if embedding_layer is None else embedding_layer),
            "front_end": (args.fusion if args.input_mode == "nt_embedding" else ""),
            "student_best_val_mcc": round(best_val_mcc, 4),
            "student_best_test_mcc": (round(best_test_mcc, 4) if best_test_mcc is not None else ""),
            "teacher_test_mcc": (round(teacher_mcc, 4) if teacher_mcc is not None else ""),
            "student_total_params": params["total"], "student_backbone_params": params["backbone"],
            "student_adapter_params": params["input_adapter"], "best_epoch": best_epoch,
            "weight_ce": hp["weight_ce"], "weight_kl": hp["weight_kl"], "weight_mse": hp["weight_mse"],
            "temperature": hp["temperature"], "lr": hp["lr"], "batch_size": batch_size,
            "epochs_run": epoch, "embedding_dim": (emb_input_dim if args.input_mode == "nt_embedding" else ""),
            "timestamp": datetime.now().isoformat(),
        })
        print(f"Appended result -> {args.results_csv}")

    wandb.finish()
    return 0


if __name__ == "__main__":
    sys.exit(main())
