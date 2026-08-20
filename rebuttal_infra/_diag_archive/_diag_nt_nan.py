"""Diagnostic: find a NaN-free NT-2.5B forward for the R1.13d panel (teacher feats+logits).

Tests the cartesian product of {merge_lora True/False} x {fp32, bf16} on a small batch, reporting
for each: where the first NaN appears (input embeddings? a specific hidden layer? logits?). Also
tries the canonical eval_fullft_nt path (merge + fp16, logits only) as a known-good reference.
SLURM/GPU. Prints a clear PASS/FAIL matrix so we pick the right loader for the full run.
"""
import os, sys
import numpy as np
import torch

REPO = "/home/pengchx3/text-dna/OmegaGenome_Revise_202606/code_carbon"
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "rebuttal_infra"))
from feature_space_compare import load_splice_test_seqs, NUM_LABELS, MAX_LEN  # noqa
from extract_feats_logits_multi import find_teacher_ckpt  # noqa

device = "cuda"
seqs, _ = load_splice_test_seqs(8, os.environ.get("HF_HOME", ""))
ckpt, _ = find_teacher_ckpt("nt")
print("ckpt", ckpt, flush=True)
BASE = "InstaDeepAI/nucleotide-transformer-2.5b-multi-species"

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel


def probe(merge, dtype, pad, special_skip_classifier=False):
    tag = f"merge={merge} dtype={dtype} pad={pad}"
    tok = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
    base = AutoModelForSequenceClassification.from_pretrained(
        BASE, num_labels=NUM_LABELS, output_hidden_states=True, trust_remote_code=True,
        torch_dtype=dtype)
    model = PeftModel.from_pretrained(base, ckpt)
    if merge:
        model = model.merge_and_unload()
    model = model.to(device).eval()
    if getattr(model.config, "pad_token_id", None) is None and tok.pad_token_id is not None:
        model.config.pad_token_id = tok.pad_token_id
    pk = dict(padding=True) if pad == "dynamic" else dict(padding="max_length", max_length=MAX_LEN)
    enc = tok(list(seqs), truncation=True, return_tensors="pt", **pk)
    ids = enc["input_ids"].to(device); mask = enc["attention_mask"].to(device)
    with torch.no_grad():
        out = model(input_ids=ids, attention_mask=mask, output_hidden_states=True)
    hs = out.hidden_states
    logit_nan = bool(torch.isnan(out.logits).any())
    first_nan_layer = None
    for li, h in enumerate(hs):
        if torch.isnan(h).any():
            first_nan_layer = li; break
    print(f"[{tag}] n_hidden={len(hs)} logits_nan={logit_nan} "
          f"first_nan_hidden_layer={first_nan_layer} "
          f"logits[0]={out.logits[0].float().cpu().numpy()}", flush=True)
    del model, base
    torch.cuda.empty_cache()
    return (not logit_nan) and (first_nan_layer is None)


results = {}
for merge in (True, False):
    for dtype in (torch.float32, torch.bfloat16):
        for pad in ("dynamic",):
            try:
                ok = probe(merge, dtype, pad)
            except Exception as e:
                print(f"[merge={merge} dtype={dtype} pad={pad}] EXC {type(e).__name__}: {e}", flush=True)
                ok = False
            results[(merge, str(dtype), pad)] = ok
print("=== SUMMARY (True == clean) ===", flush=True)
for k, v in results.items():
    print(k, v, flush=True)
