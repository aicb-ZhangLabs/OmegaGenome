"""Cached-KD correctness: the train windows MUST be reconstructable identically to the precompute's, or
cache[idx] misaligns. Guards the window-construction determinism (the bug the dry-run caught). 4 angles."""

import numpy as np
from src.data.ntv3_ft_data import GenomeBigWigDataset

REGIONS = [("chr1", 0, 32768 * 300), ("chr2", 100, 100 + 32768 * 200)]


def _ds(overlap=0.0, limit=256, seq=32768):
    return GenomeBigWigDataset(
        "/fake.fa",
        ["/fake.bw"],
        REGIONS,
        seq,
        tokenizer=None,
        transform_fn=lambda x: x,
        overlap=overlap,
        limit_num_samples=limit,
    )


def _coords(ds):
    out = []
    for j in range(len(ds)):
        ci = max(0, np.searchsorted(ds._cumulative_starts, j, side="right") - 1)
        ri = ds.region_info[ci]
        out.append(
            (
                ri["chr_name"],
                ri["region_start_offset"] + (j - ds._cumulative_starts[ci]) * ds.stride,
            )
        )
    return out


def test_same_params_same_windows():
    """Identical (regions, seq_len, limit, overlap) -> identical window count AND coords (cache aligns)."""
    a, b = _ds(), _ds()
    assert len(a) == len(b) and _coords(a) == _coords(b)


def test_overlap_changes_windows():
    """Different overlap -> different windows: this is exactly why cached mode must pin overlap=0."""
    assert len(_ds(overlap=0.0)) != len(_ds(overlap=0.9)), "overlap must change the tiling"


def test_limit_caps_window_count():
    """limit_num_samples budget bounds the window count (overlap=0 -> ~limit windows)."""
    assert len(_ds(limit=256, overlap=0.0)) <= 256
    assert len(_ds(limit=50, overlap=0.0)) < len(_ds(limit=256, overlap=0.0))


def test_coords_are_stride_spaced():
    """Within a region, consecutive windows are exactly `stride` apart (the coord-assert relies on this)."""
    ds = _ds(overlap=0.0)
    c = _coords(ds)
    same_region = [c[i] for i in range(len(c)) if c[i][0] == "chr1"]
    diffs = {same_region[i + 1][1] - same_region[i][1] for i in range(len(same_region) - 1)}
    assert diffs == {ds.stride}, f"expected stride {ds.stride}, got {diffs}"
