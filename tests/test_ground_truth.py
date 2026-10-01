"""Test the ground-truth bigWig reader on a synthetic bigWig (no network, no large download)."""

import os
import sys
import tempfile


from src.data.ground_truth import ground_truth_targets, prepare_bigwigs


def test_atomic_download(monkeypatch=None):
    """prepare_bigwigs must write atomically: a present dst is complete, never a half-written file.

    Simulate a download that fails midway (writes a truncated .part). The final dst must NOT exist
    (so no reader can open a truncated bigWig), and a clean retry must produce the full file.
    """
    import src.data.ground_truth as gt

    with tempfile.TemporaryDirectory() as d:
        manifest = {"tracks": [{"label": "trk", "encode_acc": "ENCSRX"}]}
        gt.resolve_bigwig = lambda acc, assembly="GRCh38": ("ENCFFX", "http://fake/url")
        dst = os.path.join(d, "trk__ENCFFX.bigWig")

        def fail_midway(url, part):  # writes a truncated temp then dies before replace
            with open(part, "wb") as f:
                f.write(b"BIGWIGHEADER_TRUNCATED")
            raise RuntimeError("connection reset")

        gt.urllib.request.urlretrieve = fail_midway
        try:
            prepare_bigwigs(manifest, d)
        except RuntimeError:
            pass
        assert not os.path.exists(dst), "failed download left a file at dst (not atomic!)"

        gt.urllib.request.urlretrieve = lambda url, part: open(part, "wb").write(b"FULL_CONTENT")
        paths = prepare_bigwigs(manifest, d)
        assert paths == [dst] and os.path.exists(dst), "retry did not produce the file"
        assert open(dst, "rb").read() == b"FULL_CONTENT", "wrong content after atomic replace"
        print("PASS atomic_download: no truncated dst on failure; clean retry succeeds")


def test_binning():
    import pyBigWig

    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "t.bigWig")
        bw = pyBigWig.open(p, "w")
        bw.addHeader([("chr1", 1000)])
        # first 500 bp = value 1.0, next 500 bp = value 5.0 (explicit-start intervals, span=500)
        bw.addEntries(["chr1", "chr1"], [0, 500], ends=[500, 1000], values=[1.0, 5.0])
        bw.close()

        out = ground_truth_targets([p], [("chr1", 0, 1000)], nbins=2)  # [1, 2, 1]
        assert out.shape == (1, 2, 1), f"shape {out.shape}"
        assert abs(out[0, 0, 0] - 1.0) < 1e-3 and abs(out[0, 1, 0] - 5.0) < 1e-3, (
            f"bins {out[0, :, 0]}"
        )
        # chr-name fallback (chr1 vs 1) shouldn't crash; here both present check tolerant
        print("PASS binning:", out[0, :, 0])


if __name__ == "__main__":
    try:
        test_atomic_download()
        test_binning()
        print("\n2/2 passed")
    except Exception as e:
        print(f"FAIL: {e}")
        sys.exit(1)
