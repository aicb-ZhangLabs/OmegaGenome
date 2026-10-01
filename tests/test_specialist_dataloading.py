"""Audit: single-track (specialist) training reads ONLY the target bigWig, not all 34 (idea-2 speedup).
Mocks the bigWig handles + fasta so it runs with no data files."""

import numpy as np
import torch
import src.data.ntv3_ft_data as D
from src.data.ntv3_ft_data import GenomeBigWigDataset


class _FakeHandle:
    def __init__(self, path):
        self.path = path

    def values(self, chrom, start, end, numpy=True):
        return np.ones(end - start, dtype=np.float32)


def _build(monkey_paths, seq_len=16):
    reads = []
    # record every bigWig path actually opened
    D_orig = D._get_bigwig_handle

    def fake_handle(p):
        reads.append(p)
        return _FakeHandle(p)

    D._get_bigwig_handle = fake_handle

    class _FakeTok:
        def __call__(self, seq, **kw):
            return {"input_ids": torch.zeros(1, len(seq), dtype=torch.long)}

    ds = GenomeBigWigDataset(
        "/fake.fa",
        monkey_paths,
        [("chr1", 0, 4 * seq_len)],
        seq_len,
        _FakeTok(),
        transform_fn=lambda x: x,
        overlap=0.0,
        keep_target_center_fraction=1.0,
    )
    ds._get_seq = lambda chrom, start, end: "A" * (end - start)
    return ds, reads, D_orig


def test_single_track_reads_only_target():
    """A specialist dataset built on 1 of 34 bigWig paths reads exactly that 1 path per window."""
    all34 = [f"/bw/track_{i}.bw" for i in range(34)]
    target = [all34[18]]  # specialist subset = track 18 only
    ds, reads, orig = _build(target)
    try:
        item = ds[0]
        assert reads == [all34[18]], f"read {reads}, expected only track 18"
        assert item["bigwig_targets"].shape[-1] == 1, item["bigwig_targets"].shape
        # NONE of the other 33 tracks were ever opened
        assert not any(p in reads for p in all34 if p != all34[18])
    finally:
        D._get_bigwig_handle = orig


def test_joint_reads_all_for_contrast():
    """Sanity contrast: a 34-track dataset reads all 34 — confirming the speedup is real (34x fewer)."""
    all34 = [f"/bw/track_{i}.bw" for i in range(34)]
    ds, reads, orig = _build(all34)
    try:
        item = ds[0]
        assert len(reads) == 34 and item["bigwig_targets"].shape[-1] == 34
    finally:
        D._get_bigwig_handle = orig
