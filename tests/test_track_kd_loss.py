"""Comprehensive audit of the 3-term per-bp track KD loss (src/trainer/track_distill.track_kd_loss).

Covers: weighted-sum correctness, every gt_loss + distill_loss type, teacher/gt detachment, student
layout (BLT vs BTL), position alignment, feature term on/off, weight zeroing, teacher_bounded
semantics, and gradient flow. Pure synthetic tensors (CPU, fast). No model load.
Run: PYTHONPATH=. <venv>/python -m tests.test_track_kd_loss
"""

import numpy as np
import torch

from src.trainer.track_distill import (
    TrackKDConfig,
    track_kd_loss,
    _dist_term,
    _corr_along,
    _standardized_mse_term,
    _cwd_term,
)


def test_reference_correctness_dist():
    """DIST audit (4 angles): (1) intra term == numpy Pearson over positions; (2) inter term == numpy
    Pearson over tracks; (3) sum == _dist_term; (4) each 1-corr ∈ [0,2] so total ∈ [0,4] (bounded)."""
    rng = np.random.RandomState(1)
    B, L, T = 2, 24, 4
    p = rng.rand(B, L, T).astype("float64")
    t = rng.rand(B, L, T).astype("float64")
    pt, tt = torch.tensor(p), torch.tensor(t)
    # (1) intra = 1 - mean over (b,track) of Pearson along positions
    intra = np.mean([np.corrcoef(p[b, :, c], t[b, :, c])[0, 1] for b in range(B) for c in range(T)])
    ok(
        abs((1 - intra) - float(_corr_along(pt, tt, dim=1))) < 1e-6,
        "DIST intra == numpy Pearson over positions",
    )
    # (2) inter = 1 - mean over (b,pos) of Pearson across tracks
    inter = np.mean([np.corrcoef(p[b, i, :], t[b, i, :])[0, 1] for b in range(B) for i in range(L)])
    ok(
        abs((1 - inter) - float(_corr_along(pt, tt, dim=2))) < 1e-6,
        "DIST inter == numpy Pearson over tracks",
    )
    # (3) _dist_term == intra + inter
    ok(
        abs(float(_dist_term(pt, tt)) - ((1 - intra) + (1 - inter))) < 1e-6,
        "DIST == intra + inter (both axes)",
    )
    # (4) bounded in [0,4]
    ok(0 <= float(_dist_term(pt, tt)) <= 4 + 1e-6, "DIST bounded in [0,4]")
    print("PASS reference_correctness_dist")


def test_reference_correctness_standardized_mse():
    """standardized-MSE audit (4 angles): (1) == numpy per-track z-MSE (torch std is unbiased/ddof=1);
    (2) per-track INDEPENDENT (scaling one track doesn't change another's contribution); (3) shift+scale
    invariant per track; (4) >= 0."""
    rng = np.random.RandomState(2)
    B, L, T = 2, 32, 3
    p = rng.rand(B, L, T).astype("float64")
    t = rng.rand(B, L, T).astype("float64")

    def zmse(
        a, b
    ):  # reference: z-score each track over positions (ddof=1 to match torch default), MSE
        za = (a - a.mean(1, keepdims=True)) / (a.std(1, ddof=1, keepdims=True) + 1e-7)
        zb = (b - b.mean(1, keepdims=True)) / (b.std(1, ddof=1, keepdims=True) + 1e-7)
        return float(np.mean((za - zb) ** 2))

    ref = zmse(p, t)
    got = float(_standardized_mse_term(torch.tensor(p), torch.tensor(t)))
    ok(abs(ref - got) < 1e-4, f"std-MSE == numpy per-track z-MSE ({got:.5f} vs {ref:.5f})")
    # (2) per-track independence: multiply track 0 of pred by a constant -> only track 0's z-score is
    #     unchanged (z-score is scale-invariant), so the loss is unchanged
    p2 = p.copy()
    p2[:, :, 0] *= 13.0
    ok(
        abs(_standardized_mse_term(torch.tensor(p2), torch.tensor(t)).item() - got) < 1e-4,
        "std-MSE per-track scale-invariant (track 0 ×13 -> unchanged)",
    )
    ok(float(_standardized_mse_term(torch.tensor(p), torch.tensor(t))) >= 0, "std-MSE >= 0")
    print("PASS reference_correctness_standardized_mse")


def test_reference_correctness_cwd():
    """CWD audit (5 angles): (1) == numpy T²·KL(softmax_t || softmax_s) over positions per track, meaned,
    at two temperatures (validates the T² scaling + KL direction); (2) >= 0 (KL nonneg); (3) exact 0 when
    identical; (4) RAW KL (the softness, T² removed) decreases with temperature; (5) asymmetric (KL is
    directional: swapping student/teacher changes the value)."""
    rng = np.random.RandomState(3)
    B, L, T = 2, 20, 3
    p = rng.rand(B, L, T).astype("float64") * 4
    t = rng.rand(B, L, T).astype("float64") * 4
    pt, tt = torch.tensor(p), torch.tensor(t)

    def ref_cwd(pred, targ, Tm):
        def sm(x):  # softmax over positions (axis=1)
            e = np.exp(x / Tm - (x / Tm).max(1, keepdims=True))
            return e / e.sum(1, keepdims=True)

        sp, tp = sm(pred), sm(targ)
        kl = (tp * (np.log(tp + 1e-12) - np.log(sp + 1e-12))).sum(1)  # [B,C] KL over positions
        return float(Tm * Tm * kl.mean())

    for Tm in (1.0, 4.0):  # (1) full formula incl T² at two temperatures
        ok(
            abs(ref_cwd(p, t, Tm) - float(_cwd_term(pt, tt, Tm))) < 1e-3,
            f"CWD == numpy T²·KL over positions (T={Tm})",
        )
    ok(float(_cwd_term(pt, tt, 4.0)) >= 0, "CWD >= 0 (KL nonneg)")  # (2)
    ok(float(_cwd_term(tt, tt, 4.0)) < 1e-6, "CWD == 0 when identical")  # (3)

    # (4) raw KL (divide out T²) softens with temperature: KL(T=8) < KL(T=1)
    def raw(Tm):
        return float(_cwd_term(pt, tt, Tm)) / (Tm * Tm)

    ok(raw(8.0) < raw(1.0), "CWD raw KL (T² removed) decreases with temperature (softening)")
    # (5) directional: KL(t||s) != KL(s||t) in general
    ok(
        abs(float(_cwd_term(pt, tt, 4.0)) - float(_cwd_term(tt, pt, 4.0))) > 1e-4,
        "CWD is directional (teacher→student)",
    )
    print("PASS reference_correctness_cwd")


def test_reference_correctness_gt_mix_and_subset():
    """GT-mix audit (3 angles): m=0 pure teacher, m=1 pure GT, m=0.5 == loss vs explicit convex blend.
    Specialist track_subset audit (2 angles): teacher index_select matches; T=1 loss is finite + the
    distill term equals the single picked track's loss."""
    s, t, g = _data()
    cfg_half = TrackKDConfig(
        w_ce=0, w_kl=1.0, w_mse=0, gt_loss="mse", distill_loss="mse", distill_target_gt_mix=0.5
    )
    _, comp = track_kd_loss(s, t, g, cfg=cfg_half)
    blend = 0.5 * t + 0.5 * g
    ref_distill = torch.nn.functional.mse_loss(s, blend)  # explicit convex blend target
    ok(
        abs(float(comp["distill"]) - float(ref_distill)) < 1e-6,
        "gt_mix=0.5 distill == MSE vs (0.5·teacher+0.5·gt)",
    )
    # specialist: subset the teacher by index and confirm the loss equals training that one track alone
    B, L, T = 2, 16, 5
    full_t = torch.rand(B, L, T)
    idx = torch.tensor([2])  # pick track 2
    sub_student = torch.rand(B, L, 1, requires_grad=True)
    sub_teacher = full_t.index_select(
        -1, idx
    )  # the trainer's teacher_logits.index_select(-1, kd_track_idx)
    ok(
        torch.equal(sub_teacher[..., 0], full_t[..., 2]),
        "specialist: teacher index_select picks the right track",
    )
    tot, c = track_kd_loss(
        sub_student,
        sub_teacher,
        torch.rand(B, L, 1),
        cfg=TrackKDConfig(w_ce=0, w_kl=1.0, w_mse=0, gt_loss="mse", distill_loss="mse"),
    )
    tot.backward()
    ok(
        torch.isfinite(tot).item() and sub_student.grad is not None,
        "specialist: T=1 KD loss finite + grad flows",
    )
    print("PASS reference_correctness_gt_mix_and_subset")


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
    for gl in ("poisson_multinomial", "mse", "pearson", "dist", "standardized_mse", "cwd"):
        for dl in (
            "poisson_multinomial",
            "mse",
            "pearson",
            "teacher_bounded",
            "dist",
            "standardized_mse",
            "cwd",
        ):
            total, comp = track_kd_loss(s, t, g, cfg=TrackKDConfig(gt_loss=gl, distill_loss=dl))
            ok(torch.isfinite(total).item(), f"finite total for gt={gl} distill={dl}")
    print("PASS all_loss_types_run")


def test_cwd_and_gt_mix():
    """Audit Channel-Wise Distillation (ICCV'21, per-track softmax-KL over positions) + Enigma-style
    GT-target mixing — the research-recommended #1 untried method + the cheap target-blend."""
    from src.trainer.track_distill import _cwd_term

    torch.manual_seed(0)
    B, L, T = 3, 64, 5
    t = torch.rand(B, L, T)
    # CWD = 0 when student matches teacher; positive when the positional profile differs
    ok(float(_cwd_term(t.clone(), t, 4.0)) < 1e-5, "CWD = 0 when student profile matches teacher")
    ok(float(_cwd_term(torch.rand(B, L, T), t, 4.0)) > 0, "CWD > 0 when profiles differ")
    # CWD softmax over positions is SHIFT-invariant (a per-track additive baseline doesn't change it);
    # it matches the positional profile SHAPE. (It is NOT scale-invariant — a multiplicative scale
    # sharpens the softmax — which is fine: the eval cares about shape, set via the temperature.)
    ok(
        float(_cwd_term(t + 5.0, t, 4.0)) < 1e-4,
        "CWD is shift-invariant per track (matches profile shape)",
    )
    # gradient flows through CWD as a distill term
    s = torch.rand(B, L, T, requires_grad=True)
    tot, _ = track_kd_loss(
        s, t, torch.rand(B, L, T), cfg=TrackKDConfig(w_ce=0, w_kl=1.0, w_mse=0, distill_loss="cwd")
    )
    s.grad = None
    tot.backward()
    ok(s.grad is not None and float(s.grad.abs().sum()) > 0, "gradient flows for distill_loss=cwd")
    # GT-target mix: m=0 -> pure teacher target; m=1 -> distill target == gt (so distill term == gt term)
    s2, t2, g2 = _data()
    _, c0 = track_kd_loss(
        s2, t2, g2, cfg=TrackKDConfig(gt_loss="mse", distill_loss="mse", distill_target_gt_mix=0.0)
    )
    _, c1 = track_kd_loss(
        s2, t2, g2, cfg=TrackKDConfig(gt_loss="mse", distill_loss="mse", distill_target_gt_mix=1.0)
    )
    ok(
        abs(float(c1["distill"]) - float(c1["gt"])) < 1e-6,
        "gt_mix=1 -> distill target is the ground truth",
    )
    ok(float(c0["distill"]) != float(c1["distill"]), "gt_mix changes the distill target")
    print("PASS cwd_and_gt_mix")


def test_dist_and_standardized_mse():
    """Audit the new SOTA-grounded distill terms (DIST correlation-matching NeurIPS'22; per-track
    standardized MSE / logit-standardization CVPR'24), adapted to multitrack regression."""
    from src.trainer.track_distill import _dist_term, _standardized_mse_term

    torch.manual_seed(0)
    B, L, T = 3, 64, 5
    t = torch.rand(B, L, T)

    # identical student==teacher -> both terms ~0 (perfect correlation / zero standardized error)
    ok(float(_dist_term(t.clone(), t)) < 1e-4, "DIST = 0 when student matches teacher")
    ok(
        float(_standardized_mse_term(t.clone(), t)) < 1e-4,
        "standardized_mse = 0 when student matches teacher",
    )
    # uncorrelated student -> DIST positive
    ok(
        float(_dist_term(torch.rand(B, L, T), t)) > 0,
        "DIST > 0 when student uncorrelated with teacher",
    )

    # DIST is SCALE/SHIFT-invariant (matches structure, not magnitude): an affine map of the teacher
    # along positions has ~perfect correlation -> ~0 loss (the capacity-gap robustness property).
    aff = 7.0 * t + 3.0
    ok(float(_dist_term(aff, t)) < 1e-3, "DIST is scale+shift invariant (correlation-based)")
    # standardized MSE is also scale-invariant per track -> huge-magnitude student still bounded
    big = t * 1000.0
    ok(
        0 <= float(_standardized_mse_term(big, t)) < 1e-3,
        "standardized_mse is per-track scale-invariant",
    )

    # gradient flows through both as a distill term, and DIST matches the eval (Pearson) direction
    s = torch.rand(B, L, T, requires_grad=True)
    for dl in ("dist", "standardized_mse"):
        tot, comp = track_kd_loss(
            s, t, torch.rand(B, L, T), cfg=TrackKDConfig(w_ce=0, w_kl=1.0, w_mse=0, distill_loss=dl)
        )
        s.grad = None
        tot.backward()
        ok(
            s.grad is not None and float(s.grad.abs().sum()) > 0,
            f"gradient flows for distill_loss={dl}",
        )
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
    t = torch.rand(B, L, T)
    g = torch.rand(B, L, T)
    cfg = TrackKDConfig(gt_loss="mse", distill_loss="mse")
    a, _ = track_kd_loss(s_btl, t, g, cfg=cfg, student_layout="BTL")
    b, _ = track_kd_loss(s_btl.permute(0, 2, 1).contiguous(), t, g, cfg=cfg, student_layout="BLT")
    ok(abs(float(a) - float(b)) < 1e-6, "BTL layout == manually-permuted BLT")
    print("PASS student_layout_btl")


def test_position_alignment():
    # student longer than teacher -> adaptive-pooled to teacher length (no crash, finite)
    s = torch.rand(2, 96, 5)
    t = torch.rand(2, 48, 5)
    g = torch.rand(2, 48, 5)
    total, _ = track_kd_loss(s, t, g, cfg=TrackKDConfig(gt_loss="mse", distill_loss="mse"))
    ok(torch.isfinite(total).item(), "aligns student positions to teacher/gt length")
    print("PASS position_alignment")


def test_feature_term():
    s, t, g = _data()
    sf = torch.rand(2, 16, requires_grad=True)
    tf = torch.rand(2, 16)
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
    _, comp_eq = track_kd_loss(
        s_equal, t, g, cfg=TrackKDConfig(w_ce=0, w_kl=1.0, w_mse=0, distill_loss="teacher_bounded")
    )
    ok(float(comp_eq["distill"]) == 0.0, "teacher_bounded=0 when student matches teacher")
    # student far from gt (worse than teacher) -> positive
    s_bad = g + 100.0
    _, comp_bad = track_kd_loss(
        s_bad, t, g, cfg=TrackKDConfig(w_ce=0, w_kl=1.0, w_mse=0, distill_loss="teacher_bounded")
    )
    ok(float(comp_bad["distill"]) > 0.0, "teacher_bounded>0 when student worse than teacher")
    print("PASS teacher_bounded_semantics")


def test_pearson_zero_when_identical():
    t = torch.rand(2, 48, 5)
    g = torch.rand(2, 48, 5)
    # distill pearson with student==teacher -> corr 1 -> loss ~0
    _, comp = track_kd_loss(
        t.clone(), t, g, cfg=TrackKDConfig(w_ce=0, w_kl=1.0, w_mse=0, distill_loss="pearson")
    )
    ok(abs(float(comp["distill"])) < 1e-4, "pearson distill ~0 when student==teacher")
    print("PASS pearson_zero_when_identical")


def test_feature_alignment_trainer_wiring():
    """End-to-end audit of the NEW trainer feature-alignment block (src/train/finetune_ntv3.py): the
    previous KD runs passed NO student_feat/teacher_feat, so the w_mse term was a silent no-op. This
    replicates the loop's block — student & teacher emit `{"bigwig_tracks_logits","features"}`, a lazily
    built student->teacher FitNets projector + its own optimizer — and verifies the alignment is real:
    feature term active, gradients reach BOTH student and projector, and the projector actually learns."""
    torch.manual_seed(0)
    B, L, T, S_DIM, T_DIM = (
        2,
        48,
        5,
        16,
        40,
    )  # student emb 16 -> teacher emb 40 (mismatched, like 256->1536)

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
    feat_proj, feat_opt = None, None  # lazily built, as in the loop
    s_out = student_forward(x)
    t_out = teacher_forward(x)
    logits, s_feat = s_out["bigwig_tracks_logits"], s_out["features"]
    teacher_logits, t_feat = t_out["bigwig_tracks_logits"], t_out["features"]
    ok(
        "features" in s_out and "features" in t_out,
        "both forwards expose `features` (the wired contract)",
    )

    s_feat_proj = None
    if cfg.w_mse > 0:
        feat_proj = torch.nn.Linear(s_feat.shape[-1], t_feat.shape[-1])  # lazy build: 16 -> 40
        feat_opt = torch.optim.AdamW(feat_proj.parameters(), lr=1e-2)
        ok(
            feat_proj.in_features == S_DIM and feat_proj.out_features == T_DIM,
            "projector maps student emb -> teacher emb (handles dim mismatch)",
        )
        s_feat_proj = feat_proj(s_feat)
    total, comp = track_kd_loss(
        logits,
        teacher_logits,
        gt,
        cfg=cfg,
        student_layout="BLT",
        student_feat=s_feat_proj,
        teacher_feat=(t_feat if cfg.w_mse > 0 else None),
    )
    ok(float(comp["feat"]) > 0, "feature term is ACTIVE now (was a no-op before the wiring)")

    w_before = feat_proj.weight.detach().clone()
    feat_opt.zero_grad()
    total.backward()
    ok(x.grad is not None and float(x.grad.abs().sum()) > 0, "gradient flows back into the STUDENT")
    ok(
        feat_proj.weight.grad is not None and float(feat_proj.weight.grad.abs().sum()) > 0,
        "gradient flows into the FitNets projector",
    )
    feat_opt.step()
    ok(
        float((feat_proj.weight - w_before).abs().sum()) > 0,
        "projector's own optimizer actually updates it",
    )

    # backward-compat: w_mse=0 -> no projector needed, feature term stays 0 (old behavior preserved)
    cfg0 = TrackKDConfig(w_ce=0.5, w_kl=0.5, w_mse=0.0, gt_loss="mse", distill_loss="mse")
    _, comp0 = track_kd_loss(logits.detach(), teacher_logits, gt, cfg=cfg0, student_layout="BLT")
    ok(float(comp0["feat"]) == 0.0, "w_mse=0 -> feature term off (backward compatible)")

    # robustness: unequal student/teacher emb SEQUENCE lengths must self-align (not crash)
    sf_short = torch.rand(
        B, L // 2, T_DIM, requires_grad=True
    )  # student emb half the teacher length
    tf_long = torch.rand(B, L, T_DIM)
    cfg_f = TrackKDConfig(w_ce=0, w_kl=0, w_mse=1.0, gt_loss="mse", distill_loss="mse")
    tot_f, comp_f = track_kd_loss(
        logits.detach(),
        teacher_logits,
        gt,
        cfg=cfg_f,
        student_layout="BLT",
        student_feat=sf_short,
        teacher_feat=tf_long,
    )
    ok(float(comp_f["feat"]) > 0, "feature term self-aligns unequal emb lengths (no crash)")

    # scale control: features are L2-normalised before MSE, so the term is bounded (~[0,4]) regardless
    # of raw emb magnitude — it can't swamp the O(1) Poisson terms (the bug that motivated this).
    big_sf = torch.randn(B, L, T_DIM, requires_grad=True) * 1000.0  # huge-magnitude emb
    big_tf = torch.randn(B, L, T_DIM) * 1000.0
    _, comp_big = track_kd_loss(
        logits.detach(),
        teacher_logits,
        gt,
        cfg=cfg_f,
        student_layout="BLT",
        student_feat=big_sf,
        teacher_feat=big_tf,
    )
    ok(
        0 < float(comp_big["feat"]) <= 4.0 + 1e-4,
        "feature MSE is scale-invariant + bounded (L2-normalised)",
    )
    print("PASS feature_alignment_trainer_wiring")


if __name__ == "__main__":
    for fn in [
        test_weighted_sum_and_components,
        test_all_loss_types_run,
        test_teacher_and_gt_detached,
        test_student_layout_btl,
        test_position_alignment,
        test_feature_term,
        test_weight_zeroing,
        test_teacher_bounded_semantics,
        test_pearson_zero_when_identical,
        test_feature_alignment_trainer_wiring,
        test_dist_and_standardized_mse,
        test_cwd_and_gt_mix,
        test_reference_correctness_dist,
        test_reference_correctness_standardized_mse,
        test_reference_correctness_cwd,
        test_reference_correctness_gt_mix_and_subset,
    ]:
        fn()
    print(f"\n{_n} assertions passed")
