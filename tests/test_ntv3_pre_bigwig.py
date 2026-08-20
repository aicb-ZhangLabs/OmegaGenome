"""Audit NTv3PreBigWigModel + build_bigwig_model — the 8M PRE backbone gains a track head and the
factory dispatches PRE (8M, AutoModelForMaskedLM) vs POST (100M/650M, AutoModel) correctly.

Loads the real NTv3-8M ckpt (small, ~31MB) on CPU. Skips gracefully if the weights aren't present.
Run: PYTHONPATH=. <venv>/python -m tests.test_ntv3_pre_bigwig
"""
import os
import torch

from transformers import AutoConfig
from src.model.ntv3_finetune import (
    NTV3_CROP_FRAC, NTv3PreBigWigModel, build_bigwig_model, _is_pretrained_ckpt,
)
from src.model.ntv3_teacher import prepare_local_snapshot

EIGHT_M = "/extra/zhanglab0/INDV/pengchx3/ntv3_local/8m_pre"
HUNDRED_M = "/extra/zhanglab0/INDV/pengchx3/ntv3_local/100m_post"
os.environ.setdefault("HF_HOME", "/home/pengchx3/text-dna/OmegaGenome_Revise_202606/code_carbon/.hf_cache")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

_n = 0
def ok(cond, msg):
    global _n
    assert cond, msg
    _n += 1


def _weights_present(d):
    w = os.path.join(d, "model.safetensors")
    return os.path.isfile(w) and os.path.exists(os.path.realpath(w))


def test_factory_dispatch_by_config():
    """_is_pretrained_ckpt: 8M PRE -> True, 100M POST -> False (config-only, cheap)."""
    if os.path.isdir(EIGHT_M):
        c8 = AutoConfig.from_pretrained(prepare_local_snapshot(EIGHT_M), trust_remote_code=True, local_files_only=True)
        ok(_is_pretrained_ckpt(c8) is True, "8M classified as PRE")
    if os.path.isdir(HUNDRED_M):
        c100 = AutoConfig.from_pretrained(prepare_local_snapshot(HUNDRED_M), trust_remote_code=True, local_files_only=True)
        ok(_is_pretrained_ckpt(c100) is False, "100M classified as POST")
    print("PASS test_factory_dispatch_by_config")


def test_pre_bigwig_8m_forward():
    """Build the 8M via the factory, forward, check the track-head output contract."""
    if not _weights_present(EIGHT_M):
        print("SKIP test_pre_bigwig_8m_forward (8M weights absent)")
        return
    T, B, L = 34, 2, 384
    m = build_bigwig_model(EIGHT_M, T, keep_target_center_fraction=NTV3_CROP_FRAC).eval()
    ok(isinstance(m, NTv3PreBigWigModel), "factory builds NTv3PreBigWigModel for the 8M PRE ckpt")
    ok(sum(p.numel() for p in m.parameters()) / 1e6 < 9.0, "8M-scale (<9M params)")
    with torch.no_grad():
        out = m(torch.randint(0, 4, (B, L)))
    y = out["bigwig_tracks_logits"]
    ok(tuple(y.shape) == (B, int(L * NTV3_CROP_FRAC), T), f"output [B, 0.375L, T]; got {tuple(y.shape)}")
    ok(torch.isfinite(y).all().item(), "finite outputs")
    ok((y >= 0).all().item(), "softplus head -> non-negative track signal")
    # gradients flow into both head and backbone (full fine-tune capable)
    m.train()
    loss = m(torch.randint(0, 4, (B, L)))["bigwig_tracks_logits"].mean()
    loss.backward()
    ok(m.bigwig_head.head.weight.grad is not None, "track head receives gradient")
    ok(any(p.grad is not None for p in m.backbone.parameters()), "backbone receives gradient (full-FT)")
    print("PASS test_pre_bigwig_8m_forward")


if __name__ == "__main__":
    test_factory_dispatch_by_config()
    test_pre_bigwig_8m_forward()
    print(f"\n{_n} assertions passed")
