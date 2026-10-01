"""Audit of the specialist-teacher KD path (--teacher_specialist): the pure track-resolution helper
+ the KD contract for both generalist (index-select) and specialist (direct) teachers. 7 angles."""

import torch
from src.trainer.track_distill import resolve_kd_track_targets, track_kd_loss, TrackKDConfig
from src.model.ntv3_finetune import LinearHead


# --- resolve_kd_track_targets: the pure logic the training loop relies on (4 angles) ---
def test_resolve_no_subset_full_model():
    """No subset -> full-model student, teacher used whole, no index-select."""
    assert resolve_kd_track_targets(None, 34, False) == (34, None)
    assert resolve_kd_track_targets([], 34, True) == (34, None)  # empty subset == no subset


def test_resolve_subset_generalist_indexselects():
    """Subset + GENERALIST teacher: teacher stays 34-track, index-select the subset out of its output."""
    assert resolve_kd_track_targets([18], 34, False) == (34, [18])
    assert resolve_kd_track_targets([3, 7, 18], 34, False) == (34, [3, 7, 18])


def test_resolve_subset_specialist_direct():
    """Subset + SPECIALIST teacher: teacher has a |subset|-track head, used directly (no index-select)."""
    assert resolve_kd_track_targets([18], 34, True) == (1, None)
    assert resolve_kd_track_targets([3, 7, 18], 34, True) == (3, None)


def test_resolve_specialist_vs_generalist_differ_only_in_select():
    """Same subset: specialist loads a small head + no select; generalist loads native + selects."""
    g = resolve_kd_track_targets([18], 34, False)
    s = resolve_kd_track_targets([18], 34, True)
    assert g[0] == 34 and g[1] == [18]  # generalist: native head, select track 18
    assert s[0] == 1 and s[1] is None  # specialist: 1-track head, no select


# --- KD contract: both teacher kinds flow through the loss with a 1-track student (3 angles) ---
def _kd(student, teacher, gt):
    cfg = TrackKDConfig(w_kl=0.5, w_ce=0.5, w_mse=0.0, distill_loss="standardized_mse")
    loss, _ = track_kd_loss(student, teacher, gt, cfg=cfg, student_layout="BLT")
    return loss


def test_kd_generalist_indexselected_teacher():
    """34-track generalist teacher, index_select track 18 -> 1 track, matches 1-track student."""
    B, L = 2, 64
    student = torch.rand(B, L, 1)
    teacher34 = torch.rand(B, L, 34)
    kd_idx = torch.tensor([18])
    teacher1 = teacher34.index_select(-1, kd_idx)  # the loop's index-select step
    assert teacher1.shape == (B, L, 1)
    loss = _kd(student, teacher1, torch.rand(B, L, 1))
    assert torch.isfinite(loss) and loss.item() >= 0


def test_kd_specialist_teacher_direct():
    """1-track specialist teacher used directly (no index-select) with a 1-track student."""
    B, L = 2, 64
    loss = _kd(torch.rand(B, L, 1), torch.rand(B, L, 1), torch.rand(B, L, 1))
    assert torch.isfinite(loss) and loss.item() >= 0


def test_specialist_teacher_head_is_one_track():
    """A model built with num_tracks=1 (specialist teacher) has a 1-output head — the load contract."""
    head = LinearHead(embed_dim=16, num_labels=1)
    assert head.head.out_features == 1
    out = head(torch.randn(2, 8, 16))
    assert out.shape == (2, 8, 1) and (out >= 0).all()  # softplus -> non-negative 1-track signal
