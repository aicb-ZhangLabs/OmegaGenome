"""Shared teacher-logit caching — ONE implementation used by both the standalone precompute
(`precompute_teacher_logits.py`) and the cache-along-training hook (`finetune_ntv3 --cache_logits`), so
the two can never diverge.

A cache stores a frozen teacher's per-bp track logits over a FIXED non-overlapping train window set
(overlap=0, limit=N). Because windows are deterministic given (regions, seq_len, limit, seed), the cache —
indexed by window idx — aligns 1:1 with a cached-KD run that rebuilds the same set. Files are per track:
`{gen,spec}/t{idx}.pt`, each `{logits [N,L_out,1], track_subset=[idx], coords, limit_num_samples, overlap,
sequence_length, teacher, teacher_specialist, track_ids}`. Logit-only (features are ~750GB, infeasible).
"""
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.data.ntv3_ft_data import GenomeBigWigDataset
from src.model.ntv3_finetune import NTV3_CROP_FRAC, load_finetuned_bigwig_teacher
from src.trainer.track_distill import resolve_kd_track_targets


# =============================================================================================
# FEATURE caching (the generalist teacher's BACKBONE hidden state, shared across all 34 tracks).
# Separate `feat/` namespace -> never conflicts with the per-track logit caches at gen/spec/t{i}.pt.
# Compression (per the rigorous feasibility analysis): the 1536-dim feature is low effective-rank, so we
# PCA-project channels to d' (near-lossless), and avg-pool positions by `pool` to keep the cache RAM-loadable.
# =============================================================================================

def feature_cache_dir(root: str, source: str, method: str, d_prime: int, pool: int) -> str:
    """Path for a feature cache, ENCODING both the teacher source AND the compression so nothing ever
    collides: ``<root>/<source>/<method>_d<d'>_p<pool>`` (e.g. ``teacher_feat/gen/svd_d512_p16``).
    - ``source``: ``gen`` (34-track generalist teacher) vs ``spec<i>`` (a single-track specialist teacher) —
      generalist and specialist features are DIFFERENT tensors, so they get different dirs (mirrors the
      logit caches' gen/ vs spec/ split).
    - ``method``/``d_prime``/``pool``: SVD vs random-projection, or different rank/pooling, each distinct.
    Features are shared across TRACKS within one source (one backbone state feeds all heads), so there is no
    per-track axis below ``source``."""
    return os.path.join(root, source, f"{method}_d{int(d_prime)}_p{int(pool)}")


def fit_feature_pca(sample_md: np.ndarray, d_prime: int) -> dict:
    """PCA basis (mean + top-``d_prime`` principal components) of a ``[M, D]`` teacher-feature sample, via the
    **SVD of the centered data** (NOT the covariance eigendecomposition). PCA and SVD return the SAME optimal
    rank-d' subspace (Eckart-Young), but SVD on the centered matrix is numerically more stable — it never forms
    ``XᵀX``, so it avoids squaring the condition number (which corrupts small singular values). At D=1536 the
    economy SVD is a few seconds. Pure numpy, unit-testable. Returns ``{'mean':[D], 'components':[D, d'],
    'explained_var_ratio': float}``; ``components`` columns are orthonormal (the top-d' right singular vectors)."""
    X = np.asarray(sample_md, dtype=np.float64)
    assert 0 < d_prime <= X.shape[1], f"d_prime {d_prime} must be in (0, D={X.shape[1]}]"
    mean = X.mean(axis=0)
    Xc = X - mean
    # economy SVD: Xc = U·diag(S)·Vᵀ; principal directions = rows of Vt (= columns of V); var ∝ S².
    _, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    comps = Vt[:d_prime].T                               # [D, d'] top-d' principal directions
    var = S ** 2                                          # ∝ per-component variance (constant 1/(M-1) cancels)
    evr = float(var[:d_prime].sum() / (var.sum() + 1e-12))
    return {"mean": mean.astype(np.float32), "components": comps.astype(np.float32),
            "explained_var_ratio": evr}


def compress_features(feat_bld: torch.Tensor, mean, components, pool: int) -> torch.Tensor:
    """Center -> project to d' (PCA) -> avg-pool positions by ``pool``. ``feat`` ``[B, L, D]`` ->
    ``[B, L//pool, d']``. This compressed tensor IS the cached feature-alignment target (the student later
    projects+pools its own features into this same reduced space and matches direction)."""
    mean_t = torch.as_tensor(mean, device=feat_bld.device, dtype=feat_bld.dtype)
    comp_t = torch.as_tensor(components, device=feat_bld.device, dtype=feat_bld.dtype)
    proj = (feat_bld - mean_t) @ comp_t                  # [B, L, d']
    if pool > 1:
        proj = F.avg_pool1d(proj.transpose(1, 2), pool).transpose(1, 2)  # [B, L//pool, d']
    return proj.contiguous()


def cache_teacher_features(model, ds, out_dir, d_prime, pool, n_fit_windows, seq_len, coords,
                           device, amp, batch_size=4, num_workers=8, limit_num_samples=None):
    """Build the generalist FEATURE cache into ``out_dir`` (the separate ``feat/`` namespace):
      1) forward the teacher over the first ``n_fit_windows`` -> fit PCA(``d_prime``) on those features;
      2) forward ALL windows -> compress (center->project d'->avg-pool ``pool``) -> write a memmap
         ``out_dir/feat.npy`` of shape ``[N, L_out//pool, d']`` fp16, idx-aligned to the logit cache;
      3) write ``out_dir/meta.pt`` (pca basis, coords, pool, d', seq_len, window count, evr).
    Idempotent: returns early if both files already exist. Features are SHARED across all 34 tracks, so this
    ONE cache serves every track's feature-alignment KD (no per-track feature files)."""
    os.makedirs(out_dir, exist_ok=True)
    feat_path, meta_path = os.path.join(out_dir, "feat.npy"), os.path.join(out_dir, "meta.pt")
    if os.path.exists(feat_path) and os.path.exists(meta_path):
        print(f"cache_features: {feat_path} exists -> skip (idempotent)", flush=True)
        return feat_path
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    model.eval()

    # --- pass 1: fit PCA on the first n_fit_windows ---
    sample, seen = [], 0
    for batch in loader:
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            f = model(batch["tokens"].to(device))["features"]      # [b, L, D]
        sample.append(f.float().reshape(-1, f.shape[-1]).cpu().numpy())
        seen += f.shape[0]
        if seen >= n_fit_windows:
            break
    sample = np.concatenate(sample, axis=0)
    # subsample positions so the covariance fit is bounded regardless of n_fit_windows*L
    if sample.shape[0] > 120_000:
        rng = np.random.default_rng(0)
        sample = sample[rng.choice(sample.shape[0], 120_000, replace=False)]
    pca = fit_feature_pca(sample, d_prime)
    L_out = int(np.array(_probe_len(model, ds, device, amp)))
    Lp = L_out // pool
    print(f"cache_features: PCA d'={d_prime} explains {pca['explained_var_ratio']:.4f} of var; "
          f"L_out={L_out} pool={pool} -> Lp={Lp}; writing [{len(ds)}, {Lp}, {d_prime}] fp16", flush=True)

    # --- pass 2: compress ALL windows into the memmap, idx-aligned ---
    mm = np.lib.format.open_memmap(feat_path, mode="w+", dtype=np.float16, shape=(len(ds), Lp, d_prime))
    for batch in loader:
        ix = batch["idx"].numpy()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            f = model(batch["tokens"].to(device))["features"].float()
        c = compress_features(f, pca["mean"], pca["components"], pool)    # [b, Lp, d']
        mm[ix] = c.cpu().numpy().astype(np.float16)
    mm.flush()
    # store the PCA basis as torch TENSORS (not numpy) so meta.pt loads under torch>=2.6 weights_only=True.
    torch.save({"pca_mean": torch.from_numpy(pca["mean"]),
                "pca_components": torch.from_numpy(pca["components"]),
                "explained_var_ratio": float(pca["explained_var_ratio"]), "d_prime": int(d_prime),
                "pool": int(pool), "L_out": int(L_out), "Lp": int(Lp), "num_windows": int(len(ds)),
                # ORIGINAL build limit (n_windows): the loader MUST pin train by this (not num_windows), else
                # sample_regions_for_total_length resamples different regions -> off-by-N window mismatch.
                "limit_num_samples": int(limit_num_samples if limit_num_samples is not None else len(ds)),
                "sequence_length": int(seq_len), "coords": coords, "overlap": 0.0}, meta_path)
    print(f"cache_features: wrote {feat_path} ({mm.nbytes/1e9:.1f} GB) + {meta_path}", flush=True)
    return feat_path


def _probe_len(model, ds, device, amp):
    """Teacher feature length L_out for one window (so the memmap is sized before the main pass)."""
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
        f = model(ds[0]["tokens"].unsqueeze(0).to(device))["features"]
    return f.shape[1]


def build_cache_dataset(fasta, bw_paths, train_regions, seq_len, tokenizer, n_windows):
    """The FIXED non-overlapping window set a cache and its cached-KD run BOTH construct (overlap=0,
    limit=n_windows). Only `bw_paths[:1]` is read (the teacher needs tokens, not targets)."""
    return GenomeBigWigDataset(fasta, bw_paths[:1], train_regions, seq_len, tokenizer,
                               transform_fn=lambda x: x, overlap=0.0,
                               keep_target_center_fraction=NTV3_CROP_FRAC, limit_num_samples=n_windows)


def window_coords(ds):
    """(chrom, start) per window i — the deterministic alignment fingerprint stored in every cache."""
    coords = []
    for i in range(len(ds)):
        ci = max(0, np.searchsorted(ds._cumulative_starts, i, side="right") - 1)
        ri = ds.region_info[ci]
        coords.append((ri["chr_name"],
                       int(ri["region_start_offset"] + (i - ds._cumulative_starts[ci]) * ds.stride)))
    return coords


def compute_logits(model, ds, kd_track_idx, device, amp, batch_size=8, num_workers=8):
    """Run `model` over `ds` IN ORDER -> per-bp logits ``[N, L_out, C]`` (idx-aligned). ``kd_track_idx``
    (a tensor, for a generalist teacher) index-selects the wanted tracks out of the teacher's full output;
    None means the model already emits exactly the tracks to cache."""
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    model.eval()
    cache = None
    for batch in loader:
        tok = batch["tokens"].to(device)
        ix = batch["idx"].numpy()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            t = model(tok)["bigwig_tracks_logits"]
        if kd_track_idx is not None:
            t = t.index_select(-1, kd_track_idx)
        t = t.float().cpu().numpy()
        if cache is None:
            cache = np.zeros((len(ds), t.shape[1], t.shape[2]), dtype=np.float32)
        cache[ix] = t
    return cache


# =============================================================================================
# JOINT (all-track) MEMMAP logit cache: the per-track `.pt` caches above are loaded fully into RAM by the
# trainer, so the joint 34-track cache (~107GB fp32 / ~53GB fp16) can't use them. This variant streams the
# teacher's per-bp logits to a disk memmap (fp16) that the trainer reads a batch-slice at a time — one 650M
# pass makes every downstream student run teacher-free WITHOUT the RAM blow-up. Same window set / coords /
# idx-alignment as the .pt caches, so the trainer's alignment asserts and track-subset check are unchanged.
# =============================================================================================

def _probe_logits_shape(model, ds, kd_track_idx, device, amp):
    """(L_out, C) of the teacher's per-bp logits for one window — so the memmap is sized before the pass.
    Applies ``kd_track_idx`` (generalist index-select) exactly as the main loop does, so C is the CACHED
    channel count (== the student's track count), not the teacher's full head width."""
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
        t = model(ds[0]["tokens"].unsqueeze(0).to(device))["bigwig_tracks_logits"]
    if kd_track_idx is not None:
        t = t.index_select(-1, kd_track_idx)
    return int(t.shape[1]), int(t.shape[2])


def compute_logits_memmap(model, ds, out_dir, kd_track_idx, device, amp, batch_size=8, num_workers=8,
                          track_subset=None, track_ids=None, seq_len=None, limit_num_samples=None,
                          teacher_tag=None, specialist=False):
    """Stream ``model``'s per-bp logits over ``ds`` IN ORDER to a disk memmap (fp16), for the JOINT
    all-track cache that is too large (~53GB) to hold in RAM. Writes:
      - ``out_dir/logits.npy`` : ``[N, L_out, C]`` fp16, idx-aligned (mirrors the feature-cache memmap);
      - ``out_dir/meta.pt``    : coords + the SAME keys the .pt cache carries (``track_subset``,
        ``limit_num_samples``, ``overlap``, ``sequence_length``, ``num_windows``, ...), so the trainer's
        coord-alignment assert and ``track_subset != idx`` check work identically on either layout.
    fp16 matches the AMP KD run's teacher-forward precision; the trainer upcasts to fp32 on read (the
    cached==live faithfulness the audit checks holds within fp16 rounding). ``kd_track_idx`` (generalist)
    index-selects the wanted tracks; None means the model already emits exactly the tracks to cache.
    ``track_subset`` is stored verbatim (pass None for a joint/full-native cache so it matches a joint
    student whose ``idx`` is None). Idempotent: returns early if both files already exist."""
    os.makedirs(out_dir, exist_ok=True)
    logit_path, meta_path = os.path.join(out_dir, "logits.npy"), os.path.join(out_dir, "meta.pt")
    if os.path.exists(logit_path) and os.path.exists(meta_path):
        print(f"compute_logits_memmap: {logit_path} exists -> skip (idempotent)", flush=True)
        return logit_path
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    model.eval()
    L_out, C = _probe_logits_shape(model, ds, kd_track_idx, device, amp)
    mm = np.lib.format.open_memmap(logit_path, mode="w+", dtype=np.float16, shape=(len(ds), L_out, C))
    for batch in loader:
        ix = batch["idx"].numpy()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            t = model(batch["tokens"].to(device))["bigwig_tracks_logits"]
        if kd_track_idx is not None:
            t = t.index_select(-1, kd_track_idx)
        mm[ix] = t.float().cpu().numpy().astype(np.float16)
    mm.flush()
    torch.save({"track_subset": list(track_subset) if track_subset else track_subset,
                "num_windows": int(len(ds)),
                "limit_num_samples": int(limit_num_samples if limit_num_samples is not None else len(ds)),
                "overlap": 0.0, "sequence_length": int(seq_len) if seq_len is not None else None,
                "teacher": teacher_tag, "teacher_specialist": bool(specialist),
                "track_ids": list(track_ids) if track_ids is not None else None,
                "coords": window_coords(ds), "dtype": "float16", "L_out": L_out, "C": C}, meta_path)
    print(f"compute_logits_memmap: wrote {logit_path} [{len(ds)},{L_out},{C}] fp16 "
          f"({mm.nbytes/1e9:.1f} GB) + {meta_path}", flush=True)
    return logit_path


class MemmapLogitCache:
    """Per-batch reader over a disk-streamed joint teacher-logit cache (``logits.npy`` fp16), a DROP-IN
    for the in-RAM ``.pt`` tensor in the KD loop: ``.shape`` and ``cache[idx_tensor]`` mirror the tensor
    API, so the trainer reads it identically without loading ~53GB into RAM. ``__getitem__`` fancy-indexes
    the memmap with the batch's window indices (a small copy) and upcasts fp16 -> fp32, so the loss math
    matches the fp32 ``.pt`` path within fp16 rounding."""

    def __init__(self, mm):
        self._mm = mm  # np.memmap [N, L_out, C] fp16, mmap_mode='r'

    @property
    def shape(self):
        return self._mm.shape

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.cpu().numpy()
        return torch.from_numpy(np.ascontiguousarray(self._mm[idx])).float()  # [b, L_out, C] fp32 CPU


def open_memmap_logit_cache(out_dir, in_ram=False):
    """Open a memmap logit cache (``logits.npy`` fp16 + ``meta.pt``).
    Returns ``(MemmapLogitCache, meta_dict)`` — the reader drops into the KD loop where the ``.pt``
    tensor does; ``meta`` carries coords / track_subset / limit_num_samples / overlap (same keys as .pt).

    ``in_ram``: when False (default) the logits are ``mmap_mode='r'`` and read a batch-slice at a time
    (per-batch RANDOM reads). Over an sshfs mount those random memmap reads can hang uninterruptibly on a
    network blip (finding 'sshfs memmap hang'). When True, the WHOLE ``logits.npy`` (fp16) is pulled into
    RAM with ONE sequential read (``np.load`` w/o ``mmap_mode``) and the reader then fancy-indexes an
    in-memory ndarray — no per-batch disk I/O, so no sshfs hang and no node-local /tmp staging needed. The
    array is byte-identical to the on-disk fp16 (same values, upcast to fp32 per-batch on read), so KD is
    numerically unchanged. ~53GB for the joint-34 cache — fits a big-RAM node (e.g. laniakea 1TB)."""
    logit_path = os.path.join(out_dir, "logits.npy")
    if in_ram:
        print(f"open_memmap_logit_cache: loading {logit_path} fully into RAM (one sequential read) ...",
              flush=True)
        mm = np.ascontiguousarray(np.load(logit_path))  # full fp16 array resident in RAM
        print(f"open_memmap_logit_cache: cache RAM-resident, shape={mm.shape} dtype={mm.dtype} "
              f"({mm.nbytes/1e9:.1f} GB)", flush=True)
    else:
        mm = np.load(logit_path, mmap_mode="r")
    meta = torch.load(os.path.join(out_dir, "meta.pt"), map_location="cpu")
    return MemmapLogitCache(mm), meta


def cache_record(logits_NL1, coords, track_idx, track_id, n_windows, seq_len, teacher_tag, specialist):
    """The on-disk dict for one single-track cache file (the exact format the cached-KD loader expects)."""
    return {"logits": torch.from_numpy(np.ascontiguousarray(logits_NL1)), "track_subset": [track_idx],
            "num_windows": int(logits_NL1.shape[0]), "limit_num_samples": n_windows, "overlap": 0.0,
            "sequence_length": seq_len, "teacher": teacher_tag, "teacher_specialist": bool(specialist),
            "track_ids": [track_id], "coords": coords}


def save_per_track(cache, coords, out_idx, track_ids, out_dir, n_windows, seq_len, teacher_tag,
                   specialist, skip_existing=True):
    """Save ``cache[:, :, c]`` -> ``out_dir/t{out_idx[c]}.pt`` for each output channel c. Idempotent
    (skips existing when ``skip_existing``). Returns the list of paths written."""
    os.makedirs(out_dir, exist_ok=True)
    assert cache.shape[2] == len(out_idx) == len(track_ids), \
        f"channel/idx/id length mismatch: {cache.shape[2]}/{len(out_idx)}/{len(track_ids)}"
    written = []
    for c, i in enumerate(out_idx):
        path = os.path.join(out_dir, f"t{i}.pt")
        if skip_existing and os.path.exists(path):
            continue
        torch.save(cache_record(cache[:, :, c:c + 1], coords, i, track_ids[c], n_windows, seq_len,
                                 teacher_tag, specialist), path)
        written.append(path)
    return written


def build_cache_from_teacher(out_path, teacher_ckpt, teacher_base, teacher_specialist, idx, native_n,
                             fasta, bw_paths, bw_ids, train_regions, tokenizer, seq_len, n_windows,
                             device, amp, batch_size=8):
    """Cache-along-distillation: build the (single combined) cache for ``idx`` from a teacher CKPT — one
    teacher pass over the fixed window set — when a cached-KD run finds its cache missing. Far cheaper
    than the per-step live forward, and leaves the cache for next time. Loads + frees the teacher here.

    NOTE: ``bw_paths``/``bw_ids`` are the SUBSET (already sliced to ``idx`` by the caller, length==len(idx));
    ``idx``/``native_n`` are the original dataset indices, used only for the generalist index-select +
    cache labeling. (The teacher needs only tokens, so any one of the subset paths suffices.)"""
    assert len(bw_ids) == len(idx) == len(bw_paths), \
        f"expected subset bw_paths/bw_ids of length {len(idx)}, got {len(bw_paths)}/{len(bw_ids)}"
    tnt, kd_idx = resolve_kd_track_targets(idx, native_n, teacher_specialist)
    kd_track_idx = torch.tensor(kd_idx, device=device) if kd_idx is not None else None
    teacher = load_finetuned_bigwig_teacher(teacher_ckpt, teacher_base, tnt, device=device)
    ds = build_cache_dataset(fasta, bw_paths, train_regions, seq_len, tokenizer, n_windows)
    cache = compute_logits(teacher, ds, kd_track_idx, device, amp, batch_size)
    coords = window_coords(ds)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    torch.save({"logits": torch.from_numpy(np.ascontiguousarray(cache)), "track_subset": idx,
                "num_windows": int(cache.shape[0]), "limit_num_samples": n_windows, "overlap": 0.0,
                "sequence_length": seq_len, "teacher": teacher_ckpt, "teacher_specialist": teacher_specialist,
                "track_ids": list(bw_ids), "coords": coords}, out_path)
    del teacher
    if device == "cuda":
        torch.cuda.empty_cache()
    print(f"cache-along-distillation: built {out_path} (N={cache.shape[0]}) from {teacher_ckpt}", flush=True)


def cache_teacher_logits(model, tokenizer, fasta, bw_paths, train_regions, bw_ids, track_subset_idx,
                         cache_dir, n_windows, seq_len, device, amp, teacher_tag="best-ckpt",
                         batch_size=8):
    """Cache the per-track logits of a trained teacher's BEST checkpoint to ``{gen,spec}/t{idx}.pt``. The
    caller must pass the BEST-VAL model (not an intermediate state) — only the converged teacher gives a
    meaningful cache. Generalist (``track_subset_idx`` falsy) caches all its tracks (channel c == dataset
    track c); specialist caches its subset. Idempotent: skips entirely if all target files already exist."""
    subdir = "spec" if track_subset_idx else "gen"
    out_dir = os.path.normpath(os.path.join(cache_dir, subdir))
    out_idx = list(track_subset_idx) if track_subset_idx else list(range(len(bw_ids)))
    if all(os.path.exists(os.path.join(out_dir, f"t{i}.pt")) for i in out_idx):
        print(f"cache_logits: all {len(out_idx)} {subdir} caches exist -> skip (not first training)", flush=True)
        return []
    ds = build_cache_dataset(fasta, bw_paths, train_regions, seq_len, tokenizer, n_windows)
    cache = compute_logits(model, ds, kd_track_idx=None, device=device, amp=amp, batch_size=batch_size)
    written = save_per_track(cache, window_coords(ds), out_idx, list(bw_ids), out_dir, n_windows, seq_len,
                             teacher_tag, bool(track_subset_idx))
    print(f"cache_logits: wrote {len(written)} {subdir} caches (N={cache.shape[0]}) -> {out_dir}", flush=True)
    return written
