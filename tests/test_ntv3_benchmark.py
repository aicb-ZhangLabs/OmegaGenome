"""Comprehensive test of the NTv3 Benchmark data loader on synthetic genome/bigWig/splits/metadata.

Angles: splits parsing, window tiling (default/stride/cap/too-big), metadata parsing, FASTA sequence
read, bigWig targets + log1p, and the missing-bigWig guard. No download / no network.
"""

import os
import sys
import tempfile

import numpy as np

from src.data.ntv3_benchmark import BenchmarkData, load_splits, load_track_meta, sample_windows

_n = 0


def ok(c, m):
    global _n
    assert c, "FAIL: " + m
    _n += 1


def _fixture(d):
    open(f"{d}/genome.fasta", "w").write(">chr1\n" + "ACGT" * 500 + "\n")  # 2000 bp
    os.makedirs(f"{d}/tracks", exist_ok=True)
    import pyBigWig

    bw = pyBigWig.open(f"{d}/tracks/T1.bigwig", "w")
    bw.addHeader([("chr1", 2000)])
    bw.addEntries(["chr1", "chr1"], [0, 1000], ends=[1000, 2000], values=[3.0, 7.0])
    bw.close()
    open(f"{d}/meta.tsv", "w").write(
        "file_id\tspecies_common_name\tspecies_name\tfile_assembly\tmean\tstd\tassay\n"
        "T1\thuman\thuman\thg38\t1.0\t0.5\tATAC-seq\n"
        "T2\tmouse\tmouse\tmm10\t2.0\t0.3\tDNase-seq\n"  # different species -> filtered out
    )
    open(f"{d}/splits.bed", "w").write("chr1\t0\t1000\ttrain\nchr1\t1000\t2000\ttest\n")


def test_splits_and_windows():
    with tempfile.TemporaryDirectory() as d:
        _fixture(d)
        sp = load_splits(f"{d}/splits.bed")
        ok(sp["train"] == [("chr1", 0, 1000)] and sp["test"] == [("chr1", 1000, 2000)], "splits parsed")
        ok(sample_windows(sp["train"], 500) == [("chr1", 0, 500), ("chr1", 500, 1000)], "tile default")
        ok(len(sample_windows(sp["train"], 300, stride=100)) == 8, "tile stride")
        ok(len(sample_windows(sp["train"], 300, n=2)) == 2, "tile cap n")
        ok(sample_windows(sp["train"], 5000) == [], "window > interval -> empty")


def test_meta_and_data():
    with tempfile.TemporaryDirectory() as d:
        _fixture(d)
        meta = load_track_meta(f"{d}/meta.tsv", species="human")
        ok(len(meta) == 1 and meta[0].assay == "ATAC-seq" and meta[0].mean == 1.0, "meta human-only + fields")
        bd = BenchmarkData(f"{d}/genome.fasta", f"{d}/tracks", meta)
        ok(bd.labels == ["T1"], "labels")
        ok(bd.sequences([("chr1", 0, 8)]) == ["ACGTACGT"], "fasta sequence read")
        tgt = bd.targets([("chr1", 0, 2000)], nbins=2, log1p=True)
        ok(tgt.shape == (1, 2, 1), "targets shape")
        ok(abs(tgt[0, 0, 0] - np.log1p(3.0)) < 1e-3 and abs(tgt[0, 1, 0] - np.log1p(7.0)) < 1e-3,
           "targets log1p of [3,7]")
        raw = bd.targets([("chr1", 0, 2000)], nbins=2, log1p=False)
        ok(abs(raw[0, 0, 0] - 3.0) < 1e-3, "targets raw (no log1p)")


def test_missing_bigwig_guard():
    with tempfile.TemporaryDirectory() as d:
        _fixture(d)
        meta = load_track_meta(f"{d}/meta.tsv", species="human")
        meta[0].file_id = "DOES_NOT_EXIST"
        try:
            BenchmarkData(f"{d}/genome.fasta", f"{d}/tracks", meta)
            ok(False, "missing bigWig must raise")
        except FileNotFoundError:
            ok(True, "missing bigWig raises FileNotFoundError")


if __name__ == "__main__":
    tests = [test_splits_and_windows, test_meta_and_data, test_missing_bigwig_guard]
    failed = 0
    for fn in tests:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except Exception as e:
            failed += 1
            print(f"FAIL  {fn.__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed, {_n} assertions")
    sys.exit(1 if failed else 0)
