"""Audit of NTv3FineTune (the benchmark fine-tuning head) — stub backbone, no heavy NTv3 load.

Angles: (1) output shape + central-crop correctness, (2) softplus non-negativity, (3) grad flow in
full fine-tune (head + backbone), (4) freeze_backbone (backbone frozen, head trainable + gets grads),
(5) loss/metric integration + numerical stability.
"""

import sys
import types

import numpy as np
import torch
import torch.nn as nn

from src.model.ntv3_finetune import NTV3_CROP_FRAC, NTv3FineTune, central_crop
from src.trainer.track_distill import track_distill_loss
from src.trainer.track_metrics import per_track_pearson

_n = 0


def ok(c, m):
    global _n
    assert c, "FAIL: " + m
    _n += 1


class _StubBackbone(nn.Module):
    """Returns an object with `.embedding` [B, L_full, embed_dim]; trainable so backbone grads test."""

    def __init__(self, embed_dim, l_full):
        super().__init__()
        self.l_full = l_full
        self.proj = nn.Embedding(8, embed_dim)

    def forward(self, input_ids, species_ids):
        return types.SimpleNamespace(embedding=self.proj(input_ids))  # [B, L_full, embed_dim]


def _model(embed_dim=16, l_full=800, num_tracks=5, freeze=False):
    return NTv3FineTune(_StubBackbone(embed_dim, l_full), embed_dim, num_tracks, freeze_backbone=freeze)


def test_shape_crop_nonneg():
    m = _model(l_full=800, num_tracks=5)
    out = m(torch.randint(0, 8, (2, 800)), torch.zeros(2, dtype=torch.long))
    l_out = round(800 * NTV3_CROP_FRAC)  # 300
    ok(tuple(out.shape) == (2, 5, l_out), f"shape [B,T,L_out] (got {tuple(out.shape)})")
    ok(bool((out >= 0).all()), "softplus -> non-negative")
    x = torch.arange(10).view(1, 10, 1).float()
    ok(central_crop(x, 4)[0, :, 0].tolist() == [3, 4, 5, 6], "central_crop centers correctly")


def test_grad_full_ft():
    m = _model(freeze=False)
    m(torch.randint(0, 8, (2, 800)), torch.zeros(2, dtype=torch.long)).pow(2).mean().backward()
    ok(all(p.grad is not None for p in m.head_parameters()), "head params get grads")
    ok(any(p.grad is not None and p.grad.abs().sum() > 0 for p in m.backbone.parameters()),
       "backbone gets grads in full fine-tune")


def test_freeze_backbone():
    m = _model(freeze=True)
    ok(all(not p.requires_grad for p in m.backbone.parameters()), "backbone frozen (no requires_grad)")
    ok(all(p.requires_grad for p in m.head_parameters()), "head trainable")
    m(torch.randint(0, 8, (2, 800)), torch.zeros(2, dtype=torch.long)).pow(2).mean().backward()
    ok(all(p.grad is not None for p in m.head_parameters()), "head still gets grads when frozen")
    ok(all(p.grad is None for p in m.backbone.parameters()), "frozen backbone gets NO grads")


def test_loss_metric_integration():
    m = _model(num_tracks=4)
    out = m(torch.randint(0, 8, (3, 800)), torch.zeros(3, dtype=torch.long))  # [3,4,L_out]
    l_out = out.shape[-1]
    target = torch.rand(3, l_out, 4) * 5  # [B, L_out, T] non-negative
    for kind in ("mse", "poisson", "pearson"):
        loss = track_distill_loss(out, target, kind)
        ok(torch.isfinite(loss) and loss.ndim == 0, f"loss '{kind}' finite scalar")
    mr, per = per_track_pearson(out.detach().numpy(), np.transpose(target.numpy(), (0, 2, 1)))
    ok(len(per) == 4 and np.isfinite(mr), "per_track_pearson over output finite")
    for mode in (True, False):
        m.train(mode)
        ok(torch.isfinite(m(torch.randint(0, 8, (1, 800)), torch.zeros(1, dtype=torch.long))).all(),
           f"output finite (train={mode})")


def test_apply_lora():
    from src.model.ntv3_finetune import NTV3_LORA_TARGETS, apply_lora

    ok(NTV3_LORA_TARGETS == ["linear", "mha_output", "fc1", "fc2"], "audited LoRA targets")

    class Stub(nn.Module):  # has an 'fc1' linear (a LoRA target) + a non-target 'other'
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(8, 8)
            self.other = nn.Linear(8, 8)

        def forward(self, x):
            return self.other(self.fc1(x))

    peft_m = apply_lora(Stub(), r=4, alpha=8)
    tr = [n for n, p in peft_m.named_parameters() if p.requires_grad]
    ok(tr and all("lora" in n.lower() for n in tr), "only LoRA adapters trainable")
    ok(any("fc1" in n for n in tr), "LoRA applied to the fc1 target")


if __name__ == "__main__":
    tests = [test_shape_crop_nonneg, test_grad_full_ft, test_freeze_backbone, test_loss_metric_integration,
             test_apply_lora]
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
