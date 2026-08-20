"""Diagnostic 3: NT-2.5B layer-19 NaN from clean (~2e3 maxabs) inputs. Localize WITHIN the layer and
test fixes:
  1) attn_implementation eager vs sdpa (HF EsmSelfAttention has both paths).
  2) instrument layer 19's attention: dump Q/K/V/scores/probs max-abs + nan to see the exact tensor.
  3) does the SAME single layer NaN when fed its own clean input in isolation (fp32)?
Pick the fix that yields nan-free hidden states + logits.
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
e = tok(list(seqs), truncation=True, padding=True, return_tensors="pt")
ids, mask = e["input_ids"].to(device), e["attention_mask"].to(device)


def build(dtype, attn=None):
    kw = dict(num_labels=NUM_LABELS, trust_remote_code=True, torch_dtype=dtype)
    if attn is not None:
        kw["attn_implementation"] = attn
    base = AutoModelForSequenceClassification.from_pretrained(BASE, **kw)
    m = PeftModel.from_pretrained(base, ckpt).merge_and_unload().to(device).eval()
    if getattr(m.config, "pad_token_id", None) is None and tok.pad_token_id is not None:
        m.config.pad_token_id = tok.pad_token_id
    return m


for attn in ("eager", "sdpa"):
    try:
        m = build(torch.float32, attn)
        with torch.no_grad():
            lo = m(input_ids=ids, attention_mask=mask).logits
        print(f"[attn={attn} fp32] logits nan={bool(torch.isnan(lo).any())} val0={lo[0].float().cpu().numpy()}", flush=True)
        del m; torch.cuda.empty_cache()
    except Exception as ex:
        print(f"[attn={attn}] EXC {type(ex).__name__}: {ex}", flush=True)
        torch.cuda.empty_cache()

# Instrument layer 19 internals (default attn) in fp32.
m = build(torch.float32)
esm = m.esm if hasattr(m, "esm") else m.base_model.model.esm
layer = esm.encoder.layer[19]
att = layer.attention.self  # EsmSelfAttention
orig_forward = att.forward
def traced(hidden_states, attention_mask=None, head_mask=None, encoder_hidden_states=None,
           encoder_attention_mask=None, past_key_value=None, output_attentions=False, **kw):
    q = att.transpose_for_scores(att.query(hidden_states))
    k = att.transpose_for_scores(att.key(hidden_states))
    v = att.transpose_for_scores(att.value(hidden_states))
    print(f"  [L19] in maxabs={float(hidden_states.abs().max()):.3e} "
          f"Q={float(q.abs().max()):.3e} K={float(k.abs().max()):.3e} V={float(v.abs().max()):.3e}", flush=True)
    if getattr(att, "rotary_embeddings", None) is not None:
        q, k = att.rotary_embeddings(q, k)
        print(f"  [L19] after-rotary Q={float(q.abs().max()):.3e} K={float(k.abs().max()):.3e}", flush=True)
    scores = torch.matmul(q, k.transpose(-1, -2))
    print(f"  [L19] scores maxabs={float(scores.abs().max()):.3e} nan={bool(torch.isnan(scores).any())} "
          f"inf={bool(torch.isinf(scores).any())}", flush=True)
    return orig_forward(hidden_states, attention_mask, head_mask, encoder_hidden_states,
                        encoder_attention_mask, past_key_value, output_attentions)
att.forward = traced
with torch.no_grad():
    _ = m(input_ids=ids, attention_mask=mask)
print("=== DIAG3 DONE ===", flush=True)
