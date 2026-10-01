"""BPNet track student: shape contract, one-hot correctness, crop, reuse of the carbon backbone,
KD-loop compatibility. The student REUSES code_carbon's tested BPNet tower (64-ch, dilation 2^i->512)."""

import torch
from src.model.ntv3_finetune import (
    BPNetTrackStudent,
    build_bigwig_model,
    NTV3_NUC_IDS,
    NTV3_CROP_FRAC,
)

C = 64  # the reused carbon `original` BPNet feature dim


def _student(num_tracks=1, **kw):
    return BPNetTrackStudent(num_tracks=num_tracks, **kw)


def test_forward_shape_and_crop():
    """Output is [B, L_out=0.375*L, T], non-negative; features share the cropped length (64-ch)."""
    m = _student(num_tracks=4)
    L = 256
    tokens = torch.randint(6, 10, (2, L))  # A/T/C/G ids (6..9)
    out = m(tokens)
    L_out = L - 2 * int(L * (1 - NTV3_CROP_FRAC) // 2)
    assert out["bigwig_tracks_logits"].shape == (2, L_out, 4), out["bigwig_tracks_logits"].shape
    assert out["features"].shape == (2, L_out, C)
    assert (out["bigwig_tracks_logits"] >= 0).all(), "softplus -> non-negative"


def test_onehot_mapping_correct():
    """The nuc LUT one-hots A/C/G/T into the right channels and zeros special/pad/N tokens."""
    m = _student()
    lut = m.nuc_lut
    for ch, base in enumerate("ACGT"):
        row = torch.zeros(4)
        row[ch] = 1.0
        assert torch.equal(lut[NTV3_NUC_IDS[base]], row), base
    for special in (0, 1, 2, 3, 4, 5, 10):  # unk/pad/mask/cls/eos/bos/N -> absent base
        assert lut[special].sum() == 0, special


def test_reuses_carbon_backbone_not_a_fork():
    """The conv tower is the carbon BPNet stem (full dilation 2^i->512), not a local reimplementation."""
    m = _student()
    dilations = [
        mod.dilation[0] for mod in m.backbone.modules() if isinstance(mod, torch.nn.Conv1d)
    ]
    assert 512 in dilations, f"expected full-RF dilation 512 from the carbon BPNet, got {dilations}"


def test_full_crop_keeps_length():
    """keep_target_center_fraction=1.0 keeps the whole length (no crop)."""
    m = _student(keep_target_center_fraction=1.0)
    out = m(torch.randint(6, 10, (1, 128)))
    assert out["bigwig_tracks_logits"].shape[1] == 128


def test_grads_flow():
    """A backward pass populates grads on every trainable parameter (no detached subgraph)."""
    m = _student(num_tracks=2)
    out = m(torch.randint(6, 10, (2, 128)))
    out["bigwig_tracks_logits"].sum().backward()
    assert all(p.grad is not None for p in m.parameters() if p.requires_grad)


def test_factory_dispatch_and_param_count():
    """build_bigwig_model(student_arch='bpnet') returns a BPNetTrackStudent; it is tiny (<2M params)."""
    m = build_bigwig_model("ignored", 1, student_arch="bpnet", nuc_ids=NTV3_NUC_IDS, vocab_size=11)
    assert isinstance(m, BPNetTrackStudent)
    n = sum(p.numel() for p in m.parameters())
    assert n < 2_000_000, f"{n} params — expected a tiny BPNet"
    assert m.config.embed_dim == C  # exposed like the NTv3 students (for KD feature alignment)


def test_variable_channels_and_depth():
    """channels/n_dilated knobs change width/depth and feature dim, keeping full-RF dilation (2^i)."""
    m = _student(num_tracks=2, channels=32, n_dilated=5)
    assert m.config.embed_dim == 32
    convs = [mod for mod in m.backbone.modules() if isinstance(mod, torch.nn.Conv1d)]
    # stem + 5 dilated blocks = 6 convs; widths are 32; deepest dilation = 2^5 = 32
    assert len(convs) == 6, len(convs)
    assert convs[-1].out_channels == 32
    assert max(c.dilation[0] for c in convs) == 32
    out = m(torch.randint(6, 10, (2, 256)))
    assert out["features"].shape[-1] == 32 and (out["bigwig_tracks_logits"] >= 0).all()


def test_kd_loss_compatibility():
    """Student output drops into the KD loss exactly like the NTv3 students (same contract)."""
    from src.trainer.track_distill import track_kd_loss, TrackKDConfig

    m = _student(num_tracks=3)
    tokens = torch.randint(6, 10, (2, 128))
    out = m(tokens)
    L_out = out["bigwig_tracks_logits"].shape[1]
    teacher = torch.rand(2, L_out, 3)
    targets = torch.rand(2, L_out, 3)
    cfg = TrackKDConfig(w_kl=0.5, w_ce=0.5, w_mse=0.0)
    loss, parts = track_kd_loss(
        out["bigwig_tracks_logits"], teacher, targets, cfg=cfg, student_layout="BLT"
    )
    assert torch.isfinite(loss) and loss.item() >= 0
