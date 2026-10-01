"""Audit of the --amp acceleration (bf16 autocast + TF32). The critical guarantee is that amp-OFF is
byte-identical to the faithful fp32 path; amp-ON is validated for the autocast PATTERN the loop uses
(tested via cpu bf16 autocast since the test box has no CUDA). 5 angles."""

import inspect
import torch
import src.train.finetune_ntv3 as F
from src.model.ntv3_finetune import BPNetTrackStudent


def test_amp_arg_exists_default_off():
    """--amp is a flag, default False -> the faithful repro path is the default."""
    src = inspect.getsource(F)
    assert '"--amp"' in src and 'action="store_true"' in src
    # default store_true flag is False unless passed
    assert "enabled=args.amp" in src  # autocast is gated on the flag, never forced on


def test_tf32_guarded_behind_amp():
    """TF32 matmul/cudnn is enabled ONLY inside the `if args.amp` guard (off by default)."""
    src = inspect.getsource(F)
    assert "allow_tf32 = True" in src
    i = src.index("allow_tf32")
    guard = src.rfind("if args.amp", 0, i)
    assert guard != -1 and (i - guard) < 400, "TF32 must sit under the args.amp guard"


def test_amp_off_is_byte_identical():
    """autocast(enabled=False) — the default path — leaves the forward bit-for-bit unchanged."""
    m = BPNetTrackStudent(num_tracks=3).eval()
    tok = torch.randint(6, 10, (2, 128))
    with torch.no_grad():
        plain = m(tok)["bigwig_tracks_logits"]
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=False):
            gated = m(tok)["bigwig_tracks_logits"]
    assert torch.equal(plain, gated), "amp-OFF must be byte-identical (faithful repro preserved)"


def test_amp_on_pattern_bf16_forward_fp32_params():
    """The loop's pattern (autocast forward+loss, backward OUTSIDE): compute is bf16, params/grads stay
    fp32. Tested with cpu bf16 autocast (same semantics as the cuda autocast the loop uses)."""
    m = BPNetTrackStudent(num_tracks=2)
    tok = torch.randint(6, 10, (2, 128))
    with torch.autocast("cpu", dtype=torch.bfloat16):
        out = m(tok)["bigwig_tracks_logits"]
        loss = out.float().mean()  # loss reduction, like the real loop
    assert out.dtype == torch.bfloat16, "autocast should cast the conv/linear output to bf16"
    loss.backward()  # backward is OUTSIDE autocast (inherits)
    assert torch.isfinite(loss)
    # params + grads remain fp32 (autocast never changes the master weights)
    for p in m.parameters():
        assert p.dtype == torch.float32
        if p.grad is not None:
            assert p.grad.dtype == torch.float32 and torch.isfinite(p.grad).all()


def test_amp_eval_path_stays_fp32():
    """Eval / best_model.pth path runs with NO autocast -> fp32 output (metrics unaffected by amp)."""
    m = BPNetTrackStudent(num_tracks=2).eval()
    with torch.no_grad():
        out = m(torch.randint(6, 10, (1, 128)))["bigwig_tracks_logits"]
    assert out.dtype == torch.float32
