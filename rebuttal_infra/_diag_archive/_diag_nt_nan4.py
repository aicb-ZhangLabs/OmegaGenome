"""Diagnostic 4: pinpoint the exact op in EsmLayer 19 (absolute-PE NT-2.5B) that first goes non-finite,
by hooking the submodules of layer 19 (attention.self.query/key/value, attention.output, intermediate,
output, LayerNorms). fp32, merged. Reports per-submodule input/output max-abs + isfinite.
"""
import os, sys
import torch
REPO = "/home/pengchx3/text-dna/OmegaGenome_Revise_202606/code_carbon"
sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, "rebuttal_infra"))
from feature_space_compare import load_splice_test_seqs, NUM_LABELS  # noqa
from extract_feats_logits_multi import find_teacher_ckpt  # noqa
device = "cuda"
seqs, _ = load_splice_test_seqs(8, os.environ.get("HF_HOME", ""))
ckpt, _ = find_teacher_ckpt("nt")
BASE = "InstaDeepAI/nucleotide-transformer-2.5b-multi-species"
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel
tok = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
e = tok(list(seqs), truncation=True, padding=True, return_tensors="pt")
ids, mask = e["input_ids"].to(device), e["attention_mask"].to(device)
base = AutoModelForSequenceClassification.from_pretrained(
    BASE, num_labels=NUM_LABELS, trust_remote_code=True, torch_dtype=torch.float32)
m = PeftModel.from_pretrained(base, ckpt).merge_and_unload().to(device).eval()
if getattr(m.config, "pad_token_id", None) is None and tok.pad_token_id is not None:
    m.config.pad_token_id = tok.pad_token_id
esm = m.esm if hasattr(m, "esm") else m.base_model.model.esm
L = esm.encoder.layer[19]

def stat(t):
    if not torch.is_tensor(t):
        return "n/a"
    return f"maxabs={float(t.abs().max()):.3e} finite={bool(torch.isfinite(t).all())}"

hooks = []
def mk(name):
    def h(mod, inp, out):
        i = inp[0] if isinstance(inp, tuple) and len(inp) and torch.is_tensor(inp[0]) else None
        o = out[0] if isinstance(out, tuple) and len(out) and torch.is_tensor(out[0]) else (out if torch.is_tensor(out) else None)
        print(f"  [L19.{name}] in:{stat(i)}  out:{stat(o)}", flush=True)
    return h

for name, mod in [("attn.self.query", L.attention.self.query),
                  ("attn.self.key", L.attention.self.key),
                  ("attn.self.value", L.attention.self.value),
                  ("attn.self", L.attention.self),
                  ("attn.output.dense", L.attention.output.dense),
                  ("attn.LayerNorm", L.attention.LayerNorm if hasattr(L.attention,'LayerNorm') else L.LayerNorm),
                  ("intermediate", L.intermediate),
                  ("output.dense", L.output.dense),
                  ("output", L.output)]:
    try:
        hooks.append(mod.register_forward_hook(mk(name)))
    except Exception as ex:
        print("skip", name, ex)
# also the layer's own LayerNorm(s)
for nm, sub in L.named_modules():
    if sub.__class__.__name__ == "LayerNorm":
        hooks.append(sub.register_forward_hook(mk("LN:"+nm)))
with torch.no_grad():
    _ = m(input_ids=ids, attention_mask=mask)
print("=== DIAG4 DONE ===", flush=True)
