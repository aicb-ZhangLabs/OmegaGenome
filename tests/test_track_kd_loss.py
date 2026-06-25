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
    for gl in ("poisson_multinomial", "mse", "pearson", "dist", "standardized_mse"):
        for dl in ("poisson_multinomial", "mse", "pearson", "teacher_bounded", "dist", "standardized_mse"):
            total, comp = track_kd_loss(s, t, g, cfg=TrackKDConfig(gt_loss=gl, distill_loss=dl))
            ok(torch.isfinite(total).item(), f"finite total for gt={gl} distill={dl}")
    print("PASS all_loss_types_run")


def test_dist_and_standardized_mse():
    """Audit the new SOTA-grounded distill terms (DIST correlation-matching NeurIPS'22; per-track
    standardized MSE / logit-standardization CVPR'24), adapted to multitrack regression."""
    from src.trainer.track_distill import _dist_term, _standardized_mse_term
    torch.manual_seed(0)
    B, L, T = 3, 64, 5
    t = torch.rand(B, L, T)

    # identical student==teacher -> both terms ~0 (perfect correlation / zero standardized error)
    ok(float(_dist_term(t.clone(), t)) < 1e-4, "DIST = 0 when student matches teacher")
    ok(float(_standardized_mse_term(t.clone(), t)) < 1e-4, "standardized_mse = 0 when student matches teacher")
    # uncorrelated student -> DIST positive
    ok(float(_dist_term(torch.rand(B, L, T), t)) > 0, "DIST > 0 when student uncorrelated with teacher")

    # DIST is SCALE/SHIFT-invariant (matches structure, not magnitude): an affine map of the teacher
    # along positions has ~perfect correlation -> ~0 loss (the capacity-gap robustness property).
    aff = 7.0 * t + 3.0
    ok(float(_dist_term(aff, t)) < 1e-3, "DIST is scale+shift invariant (correlation-based)")
    # standardized MSE is also scale-invariant per track -> huge-magnitude student still bounded
    big = (t * 1000.0)
    ok(0 <= float(_standardized_mse_term(big, t)) < 1e-3, "standardized_mse is per-track scale-invariant")

    # gradient flows through both as a distill term, and DIST matches the eval (Pearson) direction
    s = torch.rand(B, L, T, requires_grad=True)
    for dl in ("dist", "standardized_mse"):
        tot, comp = track_kd_loss(s, t, torch.rand(B, L, T),
                                  cfg=TrackKDConfig(w_ce=0, w_kl=1.0, w_mse=0, distill_loss=dl))
        s.grad = None; tot.backward()
        ok(s.grad is not None and float(s.grad.abs().sum()) > 0, f"gradient flows for distill_loss={dl}")
    print("PASS dist_and_standardized_mse")


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


def test_feature_alignment_trainer_wiring():
    """End-to-end audit of the NEW trainer feature-alignment block (src/train/finetune_ntv3.py): the
    previous KD runs passed NO student_feat/teacher_feat, so the w_mse term was a silent no-op. This
    replicates the loop's block — student & teacher emit `{"bigwig_tracks_logits","features"}`, a lazily
    built student->teacher FitNets projector + its own optimizer — and verifies the alignment is real:
    feature term active, gradients reach BOTH student and projector, and the projector actually learns."""
    torch.manual_seed(0)
    B, L, T, S_DIM, T_DIM = 2, 48, 5, 16, 40  # student emb 16 -> teacher emb 40 (mismatched, like 256->1536)

    # mock student/teacher forwards matching the real wrappers' output contract (both return "features")
    student_logits_w = torch.nn.Linear(S_DIM, T, bias=False)
    def student_forward(x):
        feat = x  # [B, L, S_DIM] per-bp embedding (the cropped hidden state)
        return {"bigwig_tracks_logits": torch.relu(student_logits_w(feat)), "features": feat}
    def teacher_forward(x):  # frozen
        return {"bigwig_tracks_logits": torch.rand(B, L, T), "features": torch.rand(B, L, T_DIM)}

    x = torch.rand(B, L, S_DIM, requires_grad=True)
    gt = torch.rand(B, L, T)
    cfg = TrackKDConfig(w_ce=0.5, w_kl=0.5, w_mse=0.2, gt_loss="mse", distill_loss="mse")

    # --- replicate the trainer's per-step block ---
    feat_proj, feat_opt = None, None              # lazily built, as in the loop
    s_out = student_forward(x); t_out = teacher_forward(x)
    logits, s_feat = s_out["bigwig_tracks_logits"], s_out["features"]
    teacher_logits, t_feat = t_out["bigwig_tracks_logits"], t_out["features"]
    ok("features" in s_out and "features" in t_out, "both forwards expose `features` (the wired contract)")

    s_feat_proj = None
    if cfg.w_mse > 0:
        feat_proj = torch.nn.Linear(s_feat.shape[-1], t_feat.shape[-1])   # lazy build: 16 -> 40
        feat_opt = torch.optim.AdamW(feat_proj.parameters(), lr=1e-2)
        ok(feat_proj.in_features == S_DIM and feat_proj.out_features == T_DIM,
           "projector maps student emb -> teacher emb (handles dim mismatch)")
        s_feat_proj = feat_proj(s_feat)
    total, comp = track_kd_loss(logits, teacher_logits, gt, cfg=cfg, student_layout="BLT",
                                student_feat=s_feat_proj, teacher_feat=(t_feat if cfg.w_mse > 0 else None))
    ok(float(comp["feat"]) > 0, "feature term is ACTIVE now (was a no-op before the wiring)")

    w_before = feat_proj.weight.detach().clone()
    feat_opt.zero_grad()
    total.backward()
    ok(x.grad is not None and float(x.grad.abs().sum()) > 0, "gradient flows back into the STUDENT")
    ok(feat_proj.weight.grad is not None and float(feat_proj.weight.grad.abs().sum()) > 0,
       "gradient flows into the FitNets projector")
    feat_opt.step()
    ok(float((feat_proj.weight - w_before).abs().sum()) > 0, "projector's own optimizer actually updates it")

    # backward-compat: w_mse=0 -> no projector needed, feature term stays 0 (old behavior preserved)
    cfg0 = TrackKDConfig(w_ce=0.5, w_kl=0.5, w_mse=0.0, gt_loss="mse", distill_loss="mse")
    _, comp0 = track_kd_loss(logits.detach(), teacher_logits, gt, cfg=cfg0, student_layout="BLT")
    ok(float(comp0["feat"]) == 0.0, "w_mse=0 -> feature term off (backward compatible)")

    # robustness: unequal student/teacher emb SEQUENCE lengths must self-align (not crash)
    sf_short = torch.rand(B, L // 2, T_DIM, requires_grad=True)  # student emb half the teacher length
    tf_long = torch.rand(B, L, T_DIM)
    cfg_f = TrackKDConfig(w_ce=0, w_kl=0, w_mse=1.0, gt_loss="mse", distill_loss="mse")
    tot_f, comp_f = track_kd_loss(logits.detach(), teacher_logits, gt, cfg=cfg_f, student_layout="BLT",
                                  student_feat=sf_short, teacher_feat=tf_long)
    ok(float(comp_f["feat"]) > 0, "feature term self-aligns unequal emb lengths (no crash)")

    # scale control: features are L2-normalised before MSE, so the term is bounded (~[0,4]) regardless
    # of raw emb magnitude — it can't swamp the O(1) Poisson terms (the bug that motivated this).
    big_sf = torch.randn(B, L, T_DIM, requires_grad=True) * 1000.0  # huge-magnitude emb
    big_tf = torch.randn(B, L, T_DIM) * 1000.0
    _, comp_big = track_kd_loss(logits.detach(), teacher_logits, gt, cfg=cfg_f, student_layout="BLT",
                                student_feat=big_sf, teacher_feat=big_tf)
    ok(0 < float(comp_big["feat"]) <= 4.0 + 1e-4, "feature MSE is scale-invariant + bounded (L2-normalised)")
    print("PASS feature_alignment_trainer_wiring")


if __name__ == "__main__":
    for fn in [test_weighted_sum_and_components, test_all_loss_types_run, test_teacher_and_gt_detached,
               test_student_layout_btl, test_position_alignment, test_feature_term, test_weight_zeroing,
               test_teacher_bounded_semantics, test_pearson_zero_when_identical,
               test_feature_alignment_trainer_wiring, test_dist_and_standardized_mse]:
        fn()
    print(f"\n{_n} assertions passed")
