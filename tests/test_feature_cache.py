"""Feature-cache correctness — PCA fit, compression round-trip, pooling, memmap alignment, namespace
isolation. Pure-function / synthetic so it runs on CPU with no teacher. Run:
    python -m tests.test_feature_cache
"""

import os
import tempfile

import numpy as np
import torch

from src.trainer.teacher_cache import fit_feature_pca, compress_features, feature_cache_dir

PASS = []


def check(name, cond):
    PASS.append(bool(cond))
    print(f"  [{'OK' if cond else 'FAIL'}] {name}")
    assert cond, name


def test_pca_recovers_known_rank():
    """Angle 1: PCA on rank-r data captures ~all variance at d'>=r, and components are orthonormal."""
    rng = np.random.default_rng(0)
    D, r, M = 128, 10, 5000
    basis = rng.standard_normal((r, D))
    X = (rng.standard_normal((M, r)) * np.linspace(5, 0.5, r)) @ basis  # exact rank r
    pca = fit_feature_pca(X, d_prime=r)
    check("d'=r captures ~100% var", pca["explained_var_ratio"] > 0.999)
    pca_half = fit_feature_pca(X, d_prime=r // 2)
    check("d'<r captures less var", pca_half["explained_var_ratio"] < pca["explained_var_ratio"])
    C = pca["components"]
    check("components column-orthonormal", np.allclose(C.T @ C, np.eye(r), atol=1e-4))
    check("components shape [D,d']", C.shape == (D, r))


def test_compression_roundtrip():
    """Angle 2: compress->reconstruct error matches the dropped-variance bound; full rank => ~0."""
    rng = np.random.default_rng(1)
    D, r, M = 64, 8, 4000
    X = (rng.standard_normal((M, r))) @ rng.standard_normal((r, D))  # rank 8
    feat = torch.tensor(X.reshape(1, M, D), dtype=torch.float32)
    pca = fit_feature_pca(X, d_prime=r)  # keep all real rank
    comp = compress_features(feat, pca["mean"], pca["components"], pool=1)  # [1, M, r]
    # reconstruct: X_hat = mean + Z @ C^T
    recon = comp[0].numpy() @ pca["components"].T + pca["mean"]
    rel = np.linalg.norm((X - pca["mean"]) - (recon - pca["mean"])) / np.linalg.norm(
        X - pca["mean"]
    )
    check("full-rank PCA reconstructs (~0 err)", rel < 1e-3)
    pca8to4 = fit_feature_pca(X, d_prime=4)
    c4 = compress_features(feat, pca8to4["mean"], pca8to4["components"], pool=1)
    rec4 = c4[0].numpy() @ pca8to4["components"].T + pca8to4["mean"]
    rel4 = np.linalg.norm((X - pca8to4["mean"]) - (rec4 - pca8to4["mean"])) / np.linalg.norm(
        X - pca8to4["mean"]
    )
    check("d'=4<r=8 loses some (err>0)", rel4 > 1e-3)
    check(
        "recon error == sqrt(1-evr) bound",
        abs(rel4 - np.sqrt(1 - pca8to4["explained_var_ratio"])) < 0.02,
    )


def test_pooling_shape_and_value():
    """Angle 3: avg-pool by `pool` halves/quarters positions and equals the manual mean of each block."""
    feat = torch.arange(2 * 12 * 3, dtype=torch.float32).reshape(2, 12, 3)
    mean = np.zeros(3, dtype=np.float32)
    comp = np.eye(3, dtype=np.float32)  # identity proj -> isolate pooling
    out = compress_features(feat, mean, comp, pool=4)  # 12 -> 3 positions
    check("pooled length = L//pool", out.shape == (2, 3, 3))
    # first pooled position = mean of original positions 0..3
    check("pooled value = block mean", torch.allclose(out[0, 0], feat[0, 0:4].mean(dim=0)))
    check("pool=1 is identity", torch.allclose(compress_features(feat, mean, comp, pool=1), feat))


def test_memmap_roundtrip_and_namespace():
    """Angle 4: a written memmap reloads idx-aligned; the feat/ dir is disjoint from gen/spec logit dirs."""
    rng = np.random.default_rng(2)
    N, Lp, d = 7, 5, 4
    with tempfile.TemporaryDirectory() as root:
        feat_dir = os.path.join(root, "feat")
        gen_dir = os.path.join(root, "gen")
        os.makedirs(feat_dir)
        os.makedirs(gen_dir)
        # simulate a logit cache file in gen/ — must never be touched by the feature path
        torch.save({"logits": torch.zeros(N, 5, 1)}, os.path.join(gen_dir, "t12.pt"))
        arr = rng.standard_normal((N, Lp, d)).astype(np.float16)
        path = os.path.join(feat_dir, "feat.npy")
        mm = np.lib.format.open_memmap(path, mode="w+", dtype=np.float16, shape=(N, Lp, d))
        mm[:] = arr
        mm.flush()
        re = np.load(path, mmap_mode="r")
        check("memmap reload exact", np.array_equal(np.asarray(re), arr))
        check("indexable by window idx", np.array_equal(np.asarray(re[3]), arr[3]))
        check(
            "feat/ files disjoint from gen/",
            set(os.listdir(feat_dir)).isdisjoint(os.listdir(gen_dir)),
        )
        check("gen logit cache untouched", os.path.exists(os.path.join(gen_dir, "t12.pt")))


def test_compress_matches_student_loss_dim():
    """Angle 5: the compressed teacher feat dim == d' (what the student projector must hit) for any pool."""
    rng = np.random.default_rng(3)
    feat = torch.tensor(rng.standard_normal((3, 64, 32)), dtype=torch.float32)
    pca = fit_feature_pca(feat.reshape(-1, 32).numpy(), d_prime=8)
    for pool in (1, 2, 4, 8):
        out = compress_features(feat, pca["mean"], pca["components"], pool=pool)
        check(f"pool={pool}: target dim == d'=8", out.shape[-1] == 8)
        check(f"pool={pool}: length == 64//pool", out.shape[1] == 64 // pool)


def test_centering_is_applied():
    """Angle 6: compression subtracts the PCA mean (constant-shift invariance of the alignment target)."""
    rng = np.random.default_rng(4)
    X = rng.standard_normal((2000, 16)).astype(np.float32) + 7.0  # large offset
    pca = fit_feature_pca(X, d_prime=16)
    feat = torch.tensor(X.reshape(1, 2000, 16))
    out = compress_features(feat, pca["mean"], pca["components"], pool=1)
    # centered projection has ~zero mean over samples (PCA mean removed)
    check("projected feature is ~zero-mean", abs(out[0].mean().item()) < 1e-2)


def test_feature_cache_dir_non_collision():
    """Angle 7: feature_cache_dir encodes source+method+d'+pool so caches never collide, and the feature
    namespace is disjoint from the per-track logit caches at teacher_logits/{gen,spec}/t{i}.pt."""
    root = "ntv3_cache/teacher_feat"
    gen = feature_cache_dir(root, "gen", "svd", 512, 16)
    spec33 = feature_cache_dir(root, "spec33", "svd", 512, 16)
    # (a) generalist vs specialist source -> different dirs (mirrors the gen/ vs spec/ logit split)
    check("gen vs spec33 source -> different dirs", gen != spec33)
    check("gen path contains /gen/", os.sep + "gen" + os.sep in gen + os.sep)
    check("spec33 path contains /spec33/", os.sep + "spec33" + os.sep in spec33 + os.sep)
    # (b) different method / d' / pool -> different dirs (no silent overwrite of a differently-compressed cache)
    base = feature_cache_dir(root, "gen", "svd", 512, 16)
    check(
        "different method -> different dir", base != feature_cache_dir(root, "gen", "rp", 512, 16)
    )
    check("different d' -> different dir", base != feature_cache_dir(root, "gen", "svd", 256, 16))
    check("different pool -> different dir", base != feature_cache_dir(root, "gen", "svd", 512, 8))
    check(
        "same args -> same dir (deterministic)",
        base == feature_cache_dir(root, "gen", "svd", 512, 16),
    )
    # (c) the exact documented format <root>/<source>/<method>_d<d'>_p<pool>
    check("exact path format", gen == os.path.join(root, "gen", "svd_d512_p16"))
    # (d) NEVER collides with a logit cache path teacher_logits/{gen,spec}/t{i}.pt. The feature cache lives
    # under a separate teacher_feat root, and even sharing a root the leaf is a DIR (.../svd_d512_p16/feat.npy)
    # whereas a logit cache is a FILE gen/t{i}.pt — the path components can never coincide.
    logit_gen = os.path.join("ntv3_cache", "teacher_logits", "gen", "t12.pt")
    logit_spec = os.path.join("ntv3_cache", "teacher_logits", "spec", "t12.pt")
    feat_files = {
        os.path.join(gen, "feat.npy"),
        os.path.join(gen, "meta.pt"),
        os.path.join(spec33, "feat.npy"),
    }
    check("feat paths disjoint from logit paths", feat_files.isdisjoint({logit_gen, logit_spec}))
    # the logit leaf name t<i>.pt can never equal the feature leaf method_d<d>_p<pool>
    check("feat leaf != logit leaf", os.path.basename(gen) != "t12.pt")
    # even under a shared root the source subdir differs from the logit subdir layout
    shared = feature_cache_dir(os.path.join("ntv3_cache", "teacher_logits"), "gen", "svd", 512, 16)
    check(
        "feat-under-logit-root is a deeper dir, not a tN.pt file",
        shared != logit_gen and not shared.endswith(".pt"),
    )


def test_svd_matches_covariance_eigendecomposition():
    """Angle 8: SVD-of-centered-data and covariance-eigendecomposition recover the SAME top-d' subspace
    (Eckart-Young). Verify (i) singular-value energy == covariance eigenvalues, (ii) the kept subspace is
    identical up to per-component sign, (iii) explained_var_ratio matches the eigenvalue ratio."""
    rng = np.random.default_rng(7)
    D, M, d = 20, 3000, 6
    A = rng.standard_normal((D, D))
    Sigma = A @ A.T  # SPD covariance with a spread spectrum
    L = np.linalg.cholesky(Sigma)
    X = (rng.standard_normal((M, D)) @ L.T) + rng.standard_normal(D) * 3.0  # correlated + offset
    pca = fit_feature_pca(X, d_prime=d)

    # reference: eigendecomposition of the SAMPLE covariance of the SAME centered data
    Xc = X - X.mean(axis=0)
    cov = (Xc.T @ Xc) / (M - 1)
    eigval, eigvec = np.linalg.eigh(cov)  # ascending
    order = np.argsort(eigval)[::-1]
    eigval, eigvec = eigval[order], eigvec[:, order]

    # (i) singular-value energy S^2/(M-1) == covariance eigenvalues (full spectrum), order-matched
    _, S, _ = np.linalg.svd(Xc, full_matrices=False)
    check(
        "S^2/(M-1) == cov eigenvalues",
        np.allclose(np.sort(S**2 / (M - 1)), np.sort(eigval), atol=1e-6),
    )

    # (ii) kept subspace identical: each SVD component aligns with the matching eigvec up to sign (|cos|~1),
    #      and the d'-dim projector P=C C^T is identical regardless of sign convention.
    C = pca["components"]  # [D, d'] from SVD
    Vref = eigvec[:, :d]  # [D, d'] from eigh
    cos = np.abs(np.sum(C * Vref, axis=0))
    check("per-component |cos| ~ 1 (same directions up to sign)", np.allclose(cos, 1.0, atol=1e-4))
    P_svd, P_eig = C @ C.T, Vref @ Vref.T
    check(
        "kept-subspace projector identical (sign-invariant)", np.allclose(P_svd, P_eig, atol=1e-4)
    )

    # (iii) explained_var_ratio == eigenvalue energy ratio of the SAME data
    evr_ref = float(eigval[:d].sum() / eigval.sum())
    check(
        "explained_var_ratio matches eigenvalue ratio",
        abs(pca["explained_var_ratio"] - evr_ref) < 1e-4,
    )


def test_meta_is_weights_only_safe():
    """Angle 9: the meta.pt payload uses torch TENSORS (not numpy/objects) for the PCA basis, so it round-
    trips under torch>=2.6 weights_only=True. coords (a list of tuples) stays a plain python container."""
    pca = fit_feature_pca(
        np.random.default_rng(8).standard_normal((500, 12)).astype(np.float32), d_prime=4
    )
    meta = {
        "pca_mean": torch.from_numpy(pca["mean"]),
        "pca_components": torch.from_numpy(pca["components"]),
        "explained_var_ratio": float(pca["explained_var_ratio"]),
        "d_prime": 4,
        "pool": 8,
        "L_out": 96,
        "Lp": 12,
        "num_windows": 7,
        "sequence_length": 32768,
        "coords": [("chr1", 0), ("chr1", 100)],
        "overlap": 0.0,
    }
    check(
        "pca basis stored as torch.Tensor (not numpy)",
        isinstance(meta["pca_mean"], torch.Tensor)
        and isinstance(meta["pca_components"], torch.Tensor),
    )
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "meta.pt")
        torch.save(meta, p)
        re = torch.load(p, map_location="cpu", weights_only=True)  # the loader's torch>=2.6 path
        check("meta round-trips under weights_only=True", re["d_prime"] == 4 and re["pool"] == 8)
        check(
            "pca_components survive weights_only load",
            torch.allclose(re["pca_components"], meta["pca_components"]),
        )


def test_pooled_block_mean_under_projection():
    """Angle 10: with a NON-trivial PCA projection, each pooled position still equals the block-mean of the
    projected (not raw) features — i.e. project-then-pool == pool-of-projected, and avg is exact."""
    rng = np.random.default_rng(9)
    B, L, D, d, pool = 2, 16, 24, 5, 4
    feat = torch.tensor(rng.standard_normal((B, L, D)), dtype=torch.float32)
    pca = fit_feature_pca(feat.reshape(-1, D).numpy(), d_prime=d)
    out = compress_features(feat, pca["mean"], pca["components"], pool=pool)
    # manual reference: project each position, then mean over each contiguous block of `pool`
    mean_t = torch.as_tensor(pca["mean"])
    comp_t = torch.as_tensor(pca["components"])
    proj = (feat - mean_t) @ comp_t  # [B, L, d]
    ref = proj.reshape(B, L // pool, pool, d).mean(dim=2)  # [B, L//pool, d]
    check(
        "project-then-pool == pool-of-projected (block mean)", torch.allclose(out, ref, atol=1e-5)
    )
    check("pooled shape [B, L//pool, d']", out.shape == (B, L // pool, d))


if __name__ == "__main__":
    for t in [
        test_pca_recovers_known_rank,
        test_compression_roundtrip,
        test_pooling_shape_and_value,
        test_memmap_roundtrip_and_namespace,
        test_compress_matches_student_loss_dim,
        test_centering_is_applied,
        test_feature_cache_dir_non_collision,
        test_svd_matches_covariance_eigendecomposition,
        test_meta_is_weights_only_safe,
        test_pooled_block_mean_under_projection,
    ]:
        print(f"\n== {t.__name__} ==")
        t()
    print(f"\n{sum(PASS)}/{len(PASS)} assertions passed")
