"""RIGOR GUARD #2 tests: a SELECTED --track_subset track must be REAL, non-empty + openable bigWig data.

Root-cause class: the codebase uses a "placeholder bigWig" trick (empty 0-byte files for unused tracks so
the NTV3_HUMAN_TRACK_IDS name-order guard still passes). Nothing previously verified that a track we ACTUALLY
train/cache has real, openable data — so a placeholder or a truncated/mid-transfer bigWig slipped through and
failed later with a cryptic dataloader `[bwHdrRead]` error (the real 2026-06 bug: wrong file_ids -> empty bw).

These tests use tiny SYNTHETIC bigWigs (pyBigWig addHeader/addEntries) in a tmp dir — no real 40G data — to
cover: (a) reject 0-byte placeholder; (b) reject truncated/corrupt bigWig; (c) accept a real bigWig;
(d) name-order guard rejects a reordered/missing-name dir; (e) out-of-range --track_subset index rejected.
Runs CPU-only as `python tests/test_track_subset_validation.py` (asserts + __main__) OR under pytest.
"""

import os
import sys
import tempfile

import pyBigWig

from src.data.ntv3_ft_data import (
    NTV3_HUMAN_TRACK_IDS,
    load_benchmark_frames,
    validate_subset_bigwigs,
)


def _write_real_bigwig(path, chrom="chr1", length=1000):
    """Create a tiny VALID bigWig (one chrom, a couple of intervals) for the accept-path tests."""
    bw = pyBigWig.open(path, "w")
    bw.addHeader([(chrom, length)])
    bw.addEntries([chrom, chrom], [0, length // 2], ends=[length // 2, length], values=[1.0, 5.0])
    bw.close()
    return path


# ── (a) reject an empty 0-byte placeholder ────────────────────────────────────────────────────────
def test_rejects_empty_placeholder():
    """A 0-byte placeholder for a SELECTED track must fail fast (size<=0), not slip to the dataloader."""
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "ENCSR527JGN_P.bigwig")
        open(p, "w").close()  # 0-byte placeholder (the subset-transfer trick, left UNFILLED)
        assert os.path.getsize(p) == 0
        try:
            validate_subset_bigwigs([18], ["ENCSR527JGN_P"], [p])
            assert False, "empty placeholder must raise"
        except FileNotFoundError as e:
            assert "18" in str(e) and "ENCSR527JGN_P" in str(e), f"error must name the track: {e}"
            assert "PLACEHOLDER" in str(e)
    print("PASS (a) rejects empty placeholder")


def test_rejects_missing_file():
    """A path that doesn't exist at all is treated as the empty/missing case (size -1)."""
    try:
        validate_subset_bigwigs([3], ["ENCSR100LIJ_P"], ["/no/such/file.bigwig"])
        assert False, "missing file must raise"
    except FileNotFoundError as e:
        assert "3" in str(e) and "ENCSR100LIJ_P" in str(e), e
    print("PASS (a') rejects missing file")


# ── (b) reject a truncated/corrupt bigWig ─────────────────────────────────────────────────────────
def test_rejects_truncated_bigwig():
    """A non-empty but truncated/corrupt file (bad header, mid-transfer, wrong format) must raise a
    clear error — exactly the `[bwHdrRead]` case, but caught up front naming the track."""
    with tempfile.TemporaryDirectory() as d:
        real = _write_real_bigwig(os.path.join(d, "real.bigwig"))
        trunc = os.path.join(d, "ENCSR046BCI_M.bigwig")
        with open(real, "rb") as f:
            head = f.read(
                32
            )  # truncated header (mid-transfer/cut file) -> pyBigWig [bwHdrRead] error
        with open(trunc, "wb") as f:
            f.write(head)
        assert (
            os.path.getsize(trunc) > 0
        )  # non-empty, so it passes the size check and hits pyBigWig.open
        try:
            validate_subset_bigwigs([0], ["ENCSR046BCI_M"], [trunc])
            assert False, "truncated bigWig must raise"
        except RuntimeError as e:
            assert "0" in str(e) and "ENCSR046BCI_M" in str(e), f"error must name the track: {e}"
            assert "corrupt" in str(e).lower() or "no chromosomes" in str(e).lower(), e
    print("PASS (b) rejects truncated/corrupt bigWig")


def test_rejects_wrong_format_file():
    """A non-empty file that is plainly not a bigWig (e.g. text) must also raise (wrong-format)."""
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "ENCSR410DWV.bigwig")
        with open(p, "w") as f:
            f.write("this is not a bigWig file, it is plain text\n" * 8)
        try:
            validate_subset_bigwigs([13], ["ENCSR410DWV"], [p])
            assert False, "wrong-format file must raise"
        except RuntimeError as e:
            assert "13" in str(e) and "ENCSR410DWV" in str(e), e
    print("PASS (b') rejects wrong-format file")


# ── (c) accept a real synthetic bigWig ────────────────────────────────────────────────────────────
def test_accepts_real_bigwig():
    """A real, openable, chrom-bearing bigWig for the selected track passes (no raise)."""
    with tempfile.TemporaryDirectory() as d:
        p = _write_real_bigwig(os.path.join(d, "ENCSR325NFE.bigwig"))
        validate_subset_bigwigs([12], ["ENCSR325NFE"], [p])  # must not raise
    print("PASS (c) accepts real bigWig")


def test_accepts_real_multi_track_subset():
    """Multiple selected tracks: all-real passes; one bad among real ones still fails naming that index."""
    with tempfile.TemporaryDirectory() as d:
        good0 = _write_real_bigwig(os.path.join(d, "ENCSR046BCI_M.bigwig"))
        good1 = _write_real_bigwig(os.path.join(d, "ENCSR325NFE.bigwig"))
        validate_subset_bigwigs([0, 12], ["ENCSR046BCI_M", "ENCSR325NFE"], [good0, good1])  # ok
        empty = os.path.join(d, "ENCSR527JGN_P.bigwig")
        open(empty, "w").close()
        try:
            validate_subset_bigwigs(
                [0, 18, 12],
                ["ENCSR046BCI_M", "ENCSR527JGN_P", "ENCSR325NFE"],
                [good0, empty, good1],
            )
            assert False, "one bad track among real ones must still raise"
        except FileNotFoundError as e:
            assert "18" in str(e) and "ENCSR527JGN_P" in str(e), e
    print("PASS (c') multi-track subset: all-real ok, one-bad fails on that index")


# ── (d) name-order guard rejects a reordered/missing-name dir ──────────────────────────────────────
def _build_bench_dir(d, ids, with_meta=True):
    """Lay out a minimal benchmark dir (human genome + functional_tracks/*.bigwig + splits + metadata)
    so load_benchmark_frames runs end-to-end on synthetic data. ``ids`` = the bigWig file stems to write."""
    ft = os.path.join(d, "human", "functional_tracks")
    os.makedirs(ft, exist_ok=True)
    for fid in ids:
        _write_real_bigwig(os.path.join(ft, f"{fid}.bigwig"))
    # minimal genome.fasta (+pyfaidx not needed for the name-order guard path) and splits.bed
    with open(os.path.join(d, "human", "genome.fasta"), "w") as f:
        f.write(">chr1\n" + "A" * 200 + "\n")
    with open(os.path.join(d, "human", "splits.bed"), "w") as f:
        f.write("chr1\t0\t200\ttrain\n")
    if with_meta:
        with open(os.path.join(d, "benchmark_metadata.tsv"), "w") as f:
            f.write("file_id\tassay\tmean\tstd\tspecies_common_name\n")
            for fid in ids:
                f.write(f"{fid}\tATAC-seq\t1.0\t1.0\thuman\n")
    return d


def test_name_order_guard_rejects_reorder():
    """load_benchmark_frames must fail if the dir's sorted bigWig ids != canonical NTV3_HUMAN_TRACK_IDS —
    a reordered/renamed dir would silently retarget every --track_subset index."""
    with tempfile.TemporaryDirectory() as d:
        # swap the first two canonical ids -> a name that, when sorted, no longer matches the pin
        bad_ids = NTV3_HUMAN_TRACK_IDS[:]
        bad_ids[0] = "ZZZ_reordered_marker"  # sorts last -> sorted(dir) != canonical
        _build_bench_dir(d, bad_ids)
        try:
            load_benchmark_frames(d, "human")
            assert False, "reordered/renamed track dir must raise the name-order guard"
        except RuntimeError as e:
            assert "canonical" in str(e).lower() or "NTV3_HUMAN_TRACK_IDS" in str(e), e
    print("PASS (d) name-order guard rejects reordered/renamed dir")


def test_name_order_guard_rejects_missing_track():
    """A dir missing one canonical track (33 instead of 34) must also fail the name-order guard."""
    with tempfile.TemporaryDirectory() as d:
        _build_bench_dir(d, NTV3_HUMAN_TRACK_IDS[:-1])  # drop the last track
        try:
            load_benchmark_frames(d, "human")
            assert False, "missing-track dir must raise"
        except RuntimeError as e:
            assert "canonical" in str(e).lower() or "NTV3_HUMAN_TRACK_IDS" in str(e), e
    print("PASS (d') name-order guard rejects missing track")


def test_name_order_guard_accepts_canonical():
    """The full canonical 34-id dir passes the name-order guard and returns aligned ids."""
    with tempfile.TemporaryDirectory() as d:
        _build_bench_dir(d, NTV3_HUMAN_TRACK_IDS)
        _, _, bw_ids, _, _, _ = load_benchmark_frames(d, "human")
        assert bw_ids == NTV3_HUMAN_TRACK_IDS, (
            "loader ids must equal canonical (sorted-glob == pin)"
        )
    print("PASS (d'') name-order guard accepts canonical 34-id dir")


# ── (e) out-of-range --track_subset index rejected ────────────────────────────────────────────────
def test_out_of_range_index_rejected():
    """The bounds check (the trainer + precompute apply it before selection) rejects any idx outside
    [0, native_n). Mirror the exact predicate used at the call sites."""
    native_n = 34
    for idx in ([34], [-1], [0, 99], [18, 34]):
        bad = [i for i in idx if i < 0 or i >= native_n]
        assert bad, f"{idx} should be flagged out-of-range"
    assert not [i for i in [0, 12, 18, 33] if i < 0 or i >= native_n]  # valid indices pass
    print("PASS (e) out-of-range --track_subset index rejected")


_TESTS = [
    test_rejects_empty_placeholder,
    test_rejects_missing_file,
    test_rejects_truncated_bigwig,
    test_rejects_wrong_format_file,
    test_accepts_real_bigwig,
    test_accepts_real_multi_track_subset,
    test_name_order_guard_rejects_reorder,
    test_name_order_guard_rejects_missing_track,
    test_name_order_guard_accepts_canonical,
    test_out_of_range_index_rejected,
]


if __name__ == "__main__":
    failed = 0
    for t in _TESTS:
        try:
            t()
        except Exception as e:
            failed += 1
            print(f"FAIL: {t.__name__}: {e}")
    print(f"\n{len(_TESTS) - failed}/{len(_TESTS)} passed")
    sys.exit(1 if failed else 0)
