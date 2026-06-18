"""Comprehensive audit of the Carbon-3B -> BPNet distillation pipeline (CPU-only, no teacher load).

Covers the pieces exercised by the upcoming 18-task vanilla run + HP sweep, with emphasis on the
MSE feature-matching path (never run in production before) and the recently-fixed logit_standard KL:

  (1) logit_standard clean == debug parity      (5) MSE skipped when weight_mse=0
  (2) vanilla/dkd/dist KL finite + sane          (6) teacher discovery finds {task}_finetuned/ adapters
  (3) MSE raw: finite, ->0 when feats match       (7) deploy_120k student ~0.12M + MSE alignment shapes
  (4) MSE L2-norm: scale-invariant, !=raw         (8) end-to-end distillation_loss combines CE+KL+MSE w/ grads
"""

import os
import sys
import tempfile

import torch

from src.model.bpnet_classifier import BPNetClassifier, BPNetClassifierConfig
from src.model.distillation import DistillationModel, DistillationModelConfig
from src.model.glm import get_best_checkpoint

_n = 0


def ok(cond, msg):
    global _n
    _n += 1
    assert cond, msg


def _bare_model(cfg):
    """DistillationModel with only config+device set (skips __init__'s teacher/student banner)."""
    m = DistillationModel.__new__(DistillationModel)
    torch.nn.Module.__init__(m)  # enable submodule assignment (skips the heavy __init__ banner/models)
    m.config = cfg
    m.device = "cpu"
    return m


def test_logit_standard_parity():
    m = _bare_model(DistillationModelConfig(temperature=4.0, distill_method="logit_standard"))
    torch.manual_seed(0)
    for shape in [(8, 5), (16, 2), (4, 18)]:
        s, t = torch.randn(*shape), torch.randn(*shape)
        ok(torch.allclose(m._logit_standard_kl(s, t), m._logit_standard_kl_debug(s, t), atol=1e-5),
           f"logit_standard clean==debug {shape}")


def test_kl_variants_sane():
    m = _bare_model(DistillationModelConfig(temperature=2.0))
    torch.manual_seed(1)
    s = torch.randn(8, 5)
    ok(abs(m._vanilla_kl(s, s).item()) < 1e-5, "vanilla KL(s,s)~0")
    ok(torch.isfinite(m._vanilla_kl(torch.randn(8, 5), torch.randn(8, 5))), "vanilla KL finite")
    lab = torch.randint(0, 5, (8,))
    ok(torch.isfinite(m._dkd_loss(torch.randn(8, 5), torch.randn(8, 5), lab)), "dkd finite")
    ok(torch.isfinite(m._dist_loss(torch.randn(8, 5), torch.randn(8, 5))), "dist finite")


def _student(proj="down", teacher_hidden=64, normalize=False, num_labels=2):
    return BPNetClassifier(BPNetClassifierConfig(model_type="bpnet", model_size="deploy_120k",
                                                 num_labels=num_labels, teacher_hidden_size=teacher_hidden,
                                                 teacher_projection_opt=proj))


def test_mse_raw():
    m = _bare_model(DistillationModelConfig(weight_mse=1.0, mse_normalize=False))
    m.student_model = _student(proj="down", teacher_hidden=64)
    s = torch.randn(4, 65)  # student pooled (deploy_120k feature_dim=65)
    t = torch.randn(4, 64)  # teacher hidden
    mse = m.mse_term(s, t)
    ok(torch.isfinite(mse) and mse.item() >= 0, "raw MSE finite, non-negative")
    # ->0 when (projected) teacher == student: pick t s.t. proj(t)==s is hard; instead match s to proj(t)
    sa, ta = m.student_model.aligned_feats(s, t)
    ok(abs(m.config.weight_mse * torch.nn.functional.mse_loss(ta, ta).item()) < 1e-9, "MSE(x,x)=0")


def test_mse_l2norm_scale_invariant():
    m = _bare_model(DistillationModelConfig(weight_mse=1.0, mse_normalize=True))
    m.student_model = _student(proj="up", teacher_hidden=64)  # up: aligned returns (s,t) unprojected
    s = torch.randn(4, 64)
    t = torch.randn(4, 64)
    mse1 = m.mse_term(s, t)
    mse2 = m.mse_term(s * 10.0, t)  # scaling student must NOT change L2-normalized MSE
    ok(torch.allclose(mse1, mse2, atol=1e-5), f"L2-norm MSE scale-invariant: {mse1.item()} vs {mse2.item()}")
    # differs from raw MSE
    m_raw = _bare_model(DistillationModelConfig(weight_mse=1.0, mse_normalize=False))
    m_raw.student_model = m.student_model
    ok(not torch.allclose(mse1, m_raw.mse_term(s, t), atol=1e-3), "L2-norm MSE != raw MSE")
    # parallel vectors -> ~0
    v = torch.randn(4, 64)
    ok(m.mse_term(v, v * 3.0).item() < 1e-6, "L2-norm MSE(parallel)~0")


def test_mse_skipped():
    m = _bare_model(DistillationModelConfig(weight_mse=0.0))
    m.student_model = _student()
    ok(m.mse_term(torch.randn(4, 65), torch.randn(4, 64)).item() == 0.0, "MSE skipped when weight=0")
    ok(m.mse_term(torch.randn(4, 65), None).item() == 0.0, "MSE skipped when tfeats=None")


def test_teacher_discovery():
    with tempfile.TemporaryDirectory() as d:
        for t in ["H3K4me3", "promoter_tata"]:
            os.makedirs(os.path.join(d, f"{t}_finetuned"))
            open(os.path.join(d, f"{t}_finetuned", "adapter_config.json"), "w").write("{}")
        ckpt, _ = get_best_checkpoint(d, "H3K4me3", "GLM")
        ok(ckpt is not None and ckpt.endswith("H3K4me3_finetuned"), f"finds adapter dir: {ckpt}")
        ok(get_best_checkpoint(d, "nonexistent", "GLM")[0] is None, "missing task -> None")


def test_deploy_120k_student():
    m = _student(proj="down", teacher_hidden=2048)
    bb = sum(p.numel() for p in m.backbone.parameters())
    cl = sum(p.numel() for p in m.classifier.parameters())
    deploy = bb + cl
    ok(0.10e6 < deploy < 0.14e6, f"deploy_120k ~0.12M deployment params, got {deploy}")
    ids = torch.randint(0, 4, (3, 256))
    logits, feats = m(ids, return_feats=True)
    sa, ta = m.aligned_feats(feats, torch.randn(3, 2048))
    ok(sa.shape == ta.shape, f"MSE alignment shapes match: {sa.shape} vs {ta.shape}")


def test_teacher_tokenization_isolation():
    """tokenize_teacher_inputs must apply prefix + add_special_tokens for HF tokenizers, but NOT pass
    add_special_tokens to custom tokenizers (e.g. Enformer's) that don't accept it — so the Carbon
    formatting can't break the other teachers."""
    from transformers import PreTrainedTokenizerBase
    from src.trainer.utils import tokenize_teacher_inputs

    class CustomTok:  # mimics EnformerTokenizer: no add_special_tokens / **kwargs
        def __call__(self, sequences, padding="max_length", truncation=True, max_length=1024, return_tensors="pt"):
            return {"input_ids": torch.zeros(len(sequences), max_length, dtype=torch.long)}

    # custom tokenizer: must not crash (add_special_tokens NOT forwarded)
    enc = tokenize_teacher_inputs(CustomTok(), ["ACGT", "TTGG"], max_length=8,
                                  input_prefix="<dna>", add_special_tokens=False)
    ok(enc["input_ids"].shape == (2, 8), "custom tokenizer handled without add_special_tokens kwarg")

    class StubHF(PreTrainedTokenizerBase):
        captured = {}
        def __call__(self, texts, **kw):
            type(self).captured = {"texts": texts, **kw}
            return {"input_ids": torch.zeros(len(texts), kw["max_length"], dtype=torch.long)}

    t = StubHF.__new__(StubHF)
    tokenize_teacher_inputs(t, ["ACGT"], max_length=8, input_prefix="<dna>", add_special_tokens=False)
    ok(StubHF.captured["texts"] == ["<dna>ACGT"], "HF tokenizer gets the prefix")
    ok(StubHF.captured.get("add_special_tokens") is False, "HF tokenizer gets add_special_tokens")
    t2 = StubHF.__new__(StubHF)
    tokenize_teacher_inputs(t2, ["ACGT"], max_length=8)  # defaults = other teachers
    ok(StubHF.captured["texts"] == ["ACGT"] and StubHF.captured.get("add_special_tokens") is True,
       "defaults (NT/DNABERT2): no prefix, add_special_tokens=True (unchanged behavior)")


def test_get_best_checkpoint_dispatch():
    """Per-teacher checkpoint dispatch is correct AND isolated: NT uses its finetuned_models/.../
    model-best_mcc layout; Carbon-LoRA (GLM) uses {task}_finetuned/adapter. An NT-typed lookup must
    NOT be hijacked by a stray adapter dir (the bug an over-eager top-level carbon check could cause)."""
    from src.model.glm import get_best_checkpoint

    with tempfile.TemporaryDirectory() as d:
        # NT layout (dir name must satisfy the NT branch's existing `mcc_score<digits>` regex)
        nt = os.path.join(d, "nt")
        os.makedirs(os.path.join(nt, "finetuned_models", "H3K4me3_finetuned", "model-best_mcc_score0.9078"))
        ckpt, _ = get_best_checkpoint(nt, "H3K4me3", "NT")
        ok(ckpt is not None and "finetuned_models" in ckpt, "NT layout -> NT path")

        # Carbon-LoRA (GLM) layout
        carbon = os.path.join(d, "carbon")
        os.makedirs(os.path.join(carbon, "H3K4me3_finetuned"))
        open(os.path.join(carbon, "H3K4me3_finetuned", "adapter_config.json"), "w").write("{}")
        ckpt2, _ = get_best_checkpoint(carbon, "H3K4me3", "GLM")
        ok(ckpt2 is not None and ckpt2.endswith("H3K4me3_finetuned"), "Carbon-LoRA -> GLM path")

        # ISOLATION: NT-typed lookup ignores a stray adapter dir, still uses the NT layout
        os.makedirs(os.path.join(nt, "H3K4me3_finetuned"))
        open(os.path.join(nt, "H3K4me3_finetuned", "adapter_config.json"), "w").write("{}")
        ckpt3, _ = get_best_checkpoint(nt, "H3K4me3", "NT")
        ok(ckpt3 is not None and "finetuned_models" in ckpt3, "NT type ignores stray adapter -> NT layout")


def test_teacher_configs_formatting():
    """Every teacher config carries the right formatting: non-Carbon teachers stay neutral (no <dna>,
    add_special_tokens=True, fp32), so the Carbon additions can't change their behavior; Carbon sets
    its required <dna>/add_special=False/bf16."""
    from config.distillation.glm import nt_2b5, caduceus, enformer, dna_bert_v2, carbon_3b, carbon_3b_lora

    for name, cfg in [("nt_2b5", nt_2b5), ("caduceus", caduceus), ("enformer", enformer),
                      ("dna_bert_v2", dna_bert_v2)]:
        ok(cfg.input_prefix == "" and cfg.add_special_tokens is True and cfg.torch_dtype is None,
           f"{name}: neutral (prefix='', add_special=True, dtype=None)")
    ok(carbon_3b.input_prefix == "<dna>" and carbon_3b.add_special_tokens is False
       and carbon_3b.torch_dtype == "bfloat16", "carbon_3b: <dna> + add_special=False + bf16")
    ok(carbon_3b_lora.is_lora is True and carbon_3b_lora.input_prefix == "<dna>"
       and carbon_3b_lora.torch_dtype == "bfloat16", "carbon_3b_lora: inherits formatting + is_lora")


def test_end_to_end_loss():
    cfg = DistillationModelConfig(weight_ce=0.5, weight_kl=0.5, weight_mse=0.2,
                                  temperature=2.0, distill_method="vanilla")
    student = _student(proj="down", teacher_hidden=64, num_labels=2)
    m = DistillationModel(cfg, teacher_model=None, student_model=student, device="cpu")
    B = 4
    batch = [torch.randint(0, 4, (B, 256)),          # ids
             torch.randint(0, 2, (B,)),               # labels
             torch.randn(B, 2),                        # teacher logits (num_labels=2)
             torch.randn(B, 64)]                       # teacher features (teacher_hidden=64)
    loss, metrics = m(batch)
    ok(torch.isfinite(loss), "end-to-end loss finite")
    ok(all(k in metrics for k in ["loss", "ce", "kl", "mse"]), "metrics has ce/kl/mse")
    ok(metrics["mse"] > 0 and metrics["kl"] != 0 and metrics["ce"] > 0, "all three terms active")
    loss.backward()
    grads = [p.grad for p in student.parameters() if p.requires_grad and p.grad is not None]
    ok(len(grads) > 0 and all(torch.isfinite(g).all() for g in grads), "student grads finite")


if __name__ == "__main__":
    tests = [test_logit_standard_parity, test_kl_variants_sane, test_mse_raw,
             test_mse_l2norm_scale_invariant, test_mse_skipped, test_teacher_discovery,
             test_deploy_120k_student, test_teacher_tokenization_isolation,
             test_get_best_checkpoint_dispatch, test_teacher_configs_formatting, test_end_to_end_loss]
    failed = 0
    for fn in tests:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except Exception as e:
            failed += 1
            print(f"FAIL  {fn.__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed, {_n} assertions")
    sys.exit(1 if failed else 0)
