"""Diagnostic 5: is the layer-19 NaN in the BASE NT-2.5B (no adapter) too, or only with LoRA?
Also: check layer-19 intermediate weight for NaN, dump the LayerNorm-output stats correctly (mean/var),
and test the fix candidate: cast EsmIntermediate + LayerNorm region compute via float64, OR simply
clamp the pre-FFN LayerNorm input. Decisive: run BASE-only forward + per-element inf scan.
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

def run(model, tag):
    esm = model.esm if hasattr(model, "esm") else model.base_model.model.esm
    L = esm.encoder.layer[19]
    inter = L.intermediate
    # weight nan check
    w_nan = any(torch.isnan(p).any().item() for p in inter.parameters())
    # capture LN output (pre-FFN). In pre-LN EsmLayer, output FFN does: LayerNorm(hidden) -> intermediate
    cap = {}
    def lnhook(mod, i, o):
        cap['ln_in'] = i[0].detach(); cap['ln_out'] = o.detach()
    def inhook(mod, i, o):
        cap['in_in'] = i[0].detach(); cap['in_out'] = o.detach()
    h1 = L.LayerNorm.register_forward_hook(lnhook)
    h2 = inter.register_forward_hook(inhook)
    with torch.no_grad():
        out = model(input_ids=ids, attention_mask=mask)
    h1.remove(); h2.remove()
    ln_in, ln_out = cap['ln_in'], cap['ln_out']
    print(f"[{tag}] inter.weight_nan={w_nan}", flush=True)
    print(f"[{tag}] L19.LayerNorm in: maxabs={float(ln_in.abs().max()):.3e} finite={bool(torch.isfinite(ln_in).all())} "
          f"min={float(ln_in.min()):.3e} max={float(ln_in.max()):.3e}", flush=True)
    print(f"[{tag}] L19.LayerNorm out: maxabs={float(ln_out.abs().max()):.3e} finite={bool(torch.isfinite(ln_out).all())} "
          f"nfinite={int(torch.isfinite(ln_out).sum())}/{ln_out.numel()}", flush=True)
    ii = cap['in_in']
    print(f"[{tag}] intermediate in: maxabs={float(ii.abs().max()):.3e} finite={bool(torch.isfinite(ii).all())}", flush=True)
    print(f"[{tag}] logits nan={bool(torch.isnan(out.logits).any())}", flush=True)
    # manual recompute of LN in float64 to see if it's a precision overflow
    ln = L.LayerNorm
    x = ln_in.double()
    mu = x.mean(-1, keepdim=True); var = x.var(-1, unbiased=False, keepdim=True)
    y = (x - mu) / torch.sqrt(var + ln.eps)
    print(f"[{tag}] fp64 manual LN out finite={bool(torch.isfinite(y).all())} maxabs={float(y.abs().max()):.3e} "
          f"min_var={float(var.min()):.3e}", flush=True)
    return out

# BASE only
base = AutoModelForSequenceClassification.from_pretrained(
    BASE, num_labels=NUM_LABELS, trust_remote_code=True, torch_dtype=torch.float32).to(device).eval()
if base.config.pad_token_id is None and tok.pad_token_id is not None:
    base.config.pad_token_id = tok.pad_token_id
run(base, "BASE-only")
del base; torch.cuda.empty_cache()

# BASE + merged adapter
base2 = AutoModelForSequenceClassification.from_pretrained(
    BASE, num_labels=NUM_LABELS, trust_remote_code=True, torch_dtype=torch.float32)
m = PeftModel.from_pretrained(base2, ckpt).merge_and_unload().to(device).eval()
if getattr(m.config, "pad_token_id", None) is None and tok.pad_token_id is not None:
    m.config.pad_token_id = tok.pad_token_id
run(m, "MERGED")
print("=== DIAG5 DONE ===", flush=True)
