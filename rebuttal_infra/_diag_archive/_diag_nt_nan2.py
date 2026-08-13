"""Diagnostic 2: NT-2.5B NaN at hidden layer 20 in ALL dtype/merge combos when output_hidden_states=True.

Hypotheses tested here:
  A) output_hidden_states=True is itself the trigger -> run WITHOUT it (logits only) and check logits.
  B) attention_mask handling -> compare attention_mask=ones (no mask) vs real mask.
  C) a specific layer's activation overflows even in fp32 -> dump per-layer max-abs to localize.
  D) capture the penultimate hidden via a forward HOOK (no output_hidden_states) -> clean feature path.
fp16 merged is the canonical eval_fullft_nt path; include it.
"""
import os, sys
import numpy as np
import torch

REPO = "/home/pengchx3/text-dna/OmegaGenome_Revise_202606/code_carbon"
sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, "rebuttal_infra"))
from feature_space_compare import load_splice_test_seqs, NUM_LABELS, MAX_LEN  # noqa
from extract_feats_logits_multi import find_teacher_ckpt  # noqa

device = "cuda"
seqs, _ = load_splice_test_seqs(8, os.environ.get("HF_HOME", ""))
ckpt, _ = find_teacher_ckpt("nt")
BASE = "InstaDeepAI/nucleotide-transformer-2.5b-multi-species"
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel

tok = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)


def build(merge, dtype):
    base = AutoModelForSequenceClassification.from_pretrained(
        BASE, num_labels=NUM_LABELS, trust_remote_code=True, torch_dtype=dtype)
    m = PeftModel.from_pretrained(base, ckpt)
    if merge:
        m = m.merge_and_unload()
    m = m.to(device).eval()
    if getattr(m.config, "pad_token_id", None) is None and tok.pad_token_id is not None:
        m.config.pad_token_id = tok.pad_token_id
    return m


def enc_batch():
    e = tok(list(seqs), truncation=True, padding=True, return_tensors="pt")
    return e["input_ids"].to(device), e["attention_mask"].to(device)


ids, mask = enc_batch()
ones = torch.ones_like(mask)

# A) logits only, no output_hidden_states (the eval_fullft path), merged fp16
for merge, dt, tag in [(True, torch.float16, "merge fp16"), (True, torch.float32, "merge fp32"),
                       (False, torch.float32, "unmerged fp32")]:
    m = build(merge, dt)
    with torch.no_grad():
        lo = m(input_ids=ids, attention_mask=mask).logits  # no output_hidden_states
        lo_ones = m(input_ids=ids, attention_mask=ones).logits
    print(f"[A {tag}] logits_only nan={bool(torch.isnan(lo).any())}  "
          f"with_mask=ones nan={bool(torch.isnan(lo_ones).any())}", flush=True)
    del m; torch.cuda.empty_cache()

# C+D) localize: register hooks on each EsmLayer to record output max-abs + nan, fp32 unmerged.
m = build(False, torch.float32)
esm = m.base_model.model.esm if hasattr(m, "base_model") else m.esm
layers = esm.encoder.layer
print(f"[C] n_esm_layers={len(layers)}", flush=True)
recs = []
def mk(i):
    def hook(mod, inp, out):
        o = out[0] if isinstance(out, tuple) else out
        recs.append((i, float(o.abs().max()), bool(torch.isnan(o).any())))
    return hook
hs = [l.register_forward_hook(mk(i)) for i, l in enumerate(layers)]
with torch.no_grad():
    _ = m(input_ids=ids, attention_mask=mask)
for h in hs: h.remove()
for i, mx, nan in recs[:32]:
    print(f"  layer {i:2d} maxabs={mx:.3e} nan={nan}", flush=True)

# D) penultimate via hook WITHOUT output_hidden_states -- is it clean for layers < first nan?
print("[D] done", flush=True)
print("=== DIAG2 DONE ===", flush=True)
