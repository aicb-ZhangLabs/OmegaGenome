"""Comprehensive audit of the 3-term per-bp track KD loss (src/trainer/track_distill.track_kd_loss).

Covers: weighted-sum correctness, every gt_loss + distill_loss type, teacher/gt detachment, student
layout (BLT vs BTL), position alignment, feature term on/off, weight zeroing, teacher_bounded
semantics, and gradient flow. Pure synthetic tensors (CPU, fast). No model load.
Run: PYTHONPATH=. <venv>/python -m tests.test_track_kd_loss
"""
import torch

from src.trainer.track_distill import TrackKDConfig, track_kd_loss

_n = 0
def ok(cond, msg):
    global _n
    assert cond, msg
    _n += 1


def _data(B=2, L=48, T=5, req=False):
    # non-negative (student = softplus-like output; teacher/gt = non-neg signal)
    g = torch.Generator().manual_seed(0)
    student = torch.rand(B, L, T, generator=g, requires_grad=req)
    teacher = torch.rand(B, L, T, generator=g)
    gt = torch.rand(B, L, T, generator=g)
    return student, teacher, gt


def test_weighted_sum_and_components():
    s, t, g = _data()
    cfg = TrackKDConfig(w_ce=0.5, w_kl=0.5, w_mse=0.2, gt_loss="mse", distill_loss="mse")
    total, comp = track_kd_loss(s, t, g, cfg=cfg)
    ok(set(comp) == {"gt", "distill", "feat", "total"}, "components dict keys")
    # no feature tensors -> feat term is 0
    ok(float(comp["feat"]) == 0.0, "feat=0 when no feature tensors given")
    expect = 0.5 * float(comp["gt"]) + 0.5 * float(comp["distill"]) + 0.2 * float(comp["feat"])
    ok(abs(float(total) - expect) < 1e-5, f"total = weighted sum; {float(total)} vs {expect}")
    print("PASS weighted_sum_and_components")


def test_all_loss_types_run():
    s, t, g = _data()
    for gl in ("poisson_multinomial", "mse", "pearson"):
        for dl in ("poisson_multinomial", "mse", "pearson", "teacher_bounded"):
            total, comp = track_kd_loss(s, t, g, cfg=TrackKDConfig(gt_loss=gl, distill_loss=dl))
            ok(torch.isfinite(total).item(), f"finite total for gt={gl} distill={dl}")
    print("PASS all_loss_types_run")


def test_teacher_and_gt_detached():
    s = torch.rand(2, 48, 5, requires_grad=True)
    t = torch.rand(2, 48, 5, requires_grad=True)
    g = torch.rand(2, 48, 5, requires_grad=True)
    total, _ = track_kd_loss(s, t, g, cfg=TrackKDConfig(gt_loss="mse", distill_loss="mse"))
    total.backward()
    ok(s.grad is not None, "student receives gradient")
    ok(t.grad is None, "teacher detached (no grad)")
    ok(g.grad is None, "ground-truth detached (no grad)")
    print("PASS teacher_and_gt_detached")


def test_student_layout_btl():
    # BTL student [B,T,L] permuted to [B,L,T] should match an explicitly-permuted BLT call
    B, L, T = 2, 48, 5
    s_btl = torch.rand(B, T, L)
    t = torch.rand(B, L, T); g = torch.rand(B, L, T)
    cfg = TrackKDConfig(gt_loss="mse", distill_loss="mse")
    a, _ = track_kd_loss(s_btl, t, g, cfg=cfg, student_layout="BTL")
    b, _ = track_kd_loss(s_btl.permute(0, 2, 1).contiguous(), t, g, cfg=cfg, student_layout="BLT")
    ok(abs(float(a) - float(b)) < 1e-6, "BTL layout == manually-permuted BLT")
    print("PASS student_layout_btl")


def test_position_alignment():
    # student longer than teacher -> adaptive-pooled to teacher length (no crash, finite)
    s = torch.rand(2, 96, 5); t = torch.rand(2, 48, 5); g = torch.rand(2, 48, 5)
    total, _ = track_kd_loss(s, t, g, cfg=TrackKDConfig(gt_loss="mse", distill_loss="mse"))
    ok(torch.isfinite(total).item(), "aligns student positions to teacher/gt length")
    print("PASS position_alignment")


def test_feature_term():
    s, t, g = _data()
    sf = torch.rand(2, 16, requires_grad=True); tf = torch.rand(2, 16)
    cfg = TrackKDConfig(w_ce=0, w_kl=0, w_mse=1.0, gt_loss="mse", distill_loss="mse")
    total, comp = track_kd_loss(s, t, g, student_feat=sf, teacher_feat=tf, cfg=cfg)
    ok(float(comp["feat"]) > 0, "feature MSE active when feats provided + w_mse>0")
    ok(abs(float(total) - float(comp["feat"])) < 1e-5, "with w_ce=w_kl=0, total == feat term")
    print("PASS feature_term")


def test_weight_zeroing():
    s, t, g = _data()
    # distill weight 0 -> distill loss type shouldn't change the total
    base = TrackKDConfig(w_ce=1.0, w_kl=0.0, w_mse=0.0, gt_loss="mse", distill_loss="mse")
    a, _ = track_kd_loss(s, t, g, cfg=base)
    base2 = TrackKDConfig(w_ce=1.0, w_kl=0.0, w_mse=0.0, gt_loss="mse", distill_loss="pearson")
    b, _ = track_kd_loss(s, t, g, cfg=base2)
    ok(abs(float(a) - float(b)) < 1e-6, "w_kl=0 -> distill_loss choice doesn't affect total")
    print("PASS weight_zeroing")


def test_teacher_bounded_semantics():
    # student == teacher -> never 'worse' -> teacher_bounded distill = 0
    B, L, T = 2, 48, 5
    g = torch.rand(B, L, T)
    t = torch.rand(B, L, T)
    s_equal = t.clone()
    _, comp_eq = track_kd_loss(s_equal, t, g,
                               cfg=TrackKDConfig(w_ce=0, w_kl=1.0, w_mse=0, distill_loss="teacher_bounded"))
    ok(float(comp_eq["distill"]) == 0.0, "teacher_bounded=0 when student matches teacher")
    # student far from gt (worse than teacher) -> positive
    s_bad = g + 100.0
    _, comp_bad = track_kd_loss(s_bad, t, g,
                                cfg=TrackKDConfig(w_ce=0, w_kl=1.0, w_mse=0, distill_loss="teacher_bounded"))
    ok(float(comp_bad["distill"]) > 0.0, "teacher_bounded>0 when student worse than teacher")
    print("PASS teacher_bounded_semantics")


def test_pearson_zero_when_identical():
    t = torch.rand(2, 48, 5)
    g = torch.rand(2, 48, 5)
    # distill pearson with student==teacher -> corr 1 -> loss ~0
    _, comp = track_kd_loss(t.clone(), t, g, cfg=TrackKDConfig(w_ce=0, w_kl=1.0, w_mse=0, distill_loss="pearson"))
    ok(abs(float(comp["distill"])) < 1e-4, "pearson distill ~0 when student==teacher")
    print("PASS pearson_zero_when_identical")


if __name__ == "__main__":
    for fn in [test_weighted_sum_and_components, test_all_loss_types_run, test_teacher_and_gt_detached,
               test_student_layout_btl, test_position_alignment, test_feature_term, test_weight_zeroing,
               test_teacher_bounded_semantics, test_pearson_zero_when_identical]:
        fn()
    print(f"\n{_n} assertions passed")
