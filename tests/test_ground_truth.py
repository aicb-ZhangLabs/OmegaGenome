"""Test the ground-truth bigWig reader on a synthetic bigWig (no network, no large download)."""

import os
import sys
import tempfile

import numpy as np

from src.data.ground_truth import ground_truth_targets


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
        assert abs(out[0, 0, 0] - 1.0) < 1e-3 and abs(out[0, 1, 0] - 5.0) < 1e-3, f"bins {out[0,:,0]}"
        # chr-name fallback (chr1 vs 1) shouldn't crash; here both present check tolerant
        print("PASS binning:", out[0, :, 0])


if __name__ == "__main__":
    try:
        test_binning()
        print("\n1/1 passed")
    except Exception as e:
        print(f"FAIL: {e}")
        sys.exit(1)
