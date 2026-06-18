"""Ground-truth ENCODE bigWig signal for the NTv3 tracks — to evaluate the teacher vs measured signal.

ENCODE serves signed/expiring S3 URLs that pyBigWig cannot open remotely, so each track's bigWig is
DOWNLOADED once to a local cache (SSD), then read locally. Resolves each track's experiment accession
to a released GRCh38 bigWig (signal p-value for ChIP; DNase has no p-value track, so falls back to a
read-depth-normalized track), and reads per-bp signal binned to the teacher's output length.

Reuses the v1 track manifest (`config/distillation/ntv3_v1_tracks.json`) — no new metadata.
"""

import json
import os
import urllib.request
from typing import List, Tuple

import numpy as np

_EXP = "https://www.encodeproject.org/experiments/{acc}/?format=json"
# preferred bigWig output types, in order: ChIP has "signal p-value"; DNase does not.
_PREF = ["signal p-value", "read-depth normalized signal", "fold change over control", "signal of unique reads"]


def resolve_bigwig(acc: str, assembly: str = "GRCh38") -> Tuple[str, str]:
    """Experiment accession -> (bigWig file accession, download URL) for a released ``assembly`` bigWig."""
    req = urllib.request.Request(_EXP.format(acc=acc), headers={"Accept": "application/json"})
    files = json.load(urllib.request.urlopen(req, timeout=30)).get("files", [])
    # accept released OR archived (older DNase signal tracks are archived but still the measured signal)
    bws = [f for f in files if f.get("file_format") == "bigWig" and f.get("status") in ("released", "archived")
           and f.get("assembly") == assembly]
    if not bws:
        raise ValueError(f"no released {assembly} bigWig for {acc}")
    for ot in _PREF:
        m = [f for f in bws if f.get("output_type") == ot]
        if m:
            return m[0]["accession"], "https://www.encodeproject.org" + m[0]["href"]
    return bws[0]["accession"], "https://www.encodeproject.org" + bws[0]["href"]


def prepare_bigwigs(manifest: dict, cache_dir: str) -> List[str]:
    """Download each track's bigWig to ``cache_dir`` (skip if already present). Returns local paths."""
    os.makedirs(cache_dir, exist_ok=True)
    paths, missing = [], []
    for t in manifest["tracks"]:
        try:
            facc, url = resolve_bigwig(t["encode_acc"])
        except Exception as e:  # one bad track shouldn't sink the others
            print(f"  SKIP {t['label']} ({t['encode_acc']}): {e}", flush=True)
            paths.append(None)
            missing.append(t["label"])
            continue
        dst = os.path.join(cache_dir, f"{t['label']}__{facc}.bigWig")
        if not (os.path.exists(dst) and os.path.getsize(dst) > 0):
            print(f"  downloading {t['label']} ({facc}) ...", flush=True)
            urllib.request.urlretrieve(url, dst)
        paths.append(dst)
    if missing:
        print(f"  WARNING: no bigWig for {missing} -> those tracks score NaN", flush=True)
    return paths


def ground_truth_targets(bigwig_paths: List[str], coords, nbins: int) -> np.ndarray:
    """[N, nbins, T] binned-mean measured signal over the windows for each track's bigWig."""
    import pyBigWig  # lazy: module imports without pyBigWig installed

    bws = [pyBigWig.open(p) if p else None for p in bigwig_paths]  # None = missing track -> zeros
    out = np.zeros((len(coords), nbins, len(bws)), dtype=np.float32)
    try:
        for i, (chrom, start, end) in enumerate(coords):
            for j, bw in enumerate(bws):
                if bw is None:
                    continue
                c = chrom if chrom in bw.chroms() else chrom.replace("chr", "")  # chr1 vs 1
                vals = bw.stats(c, start, end, type="mean", nBins=nbins)
                out[i, :, j] = np.nan_to_num([v if v is not None else 0.0 for v in vals])
    finally:
        for bw in bws:
            if bw is not None:
                bw.close()
    return out
