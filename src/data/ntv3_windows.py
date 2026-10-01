"""Fetch genomic windows (hg38 etc.) for NTv3 multi-track distillation via the UCSC REST API.

No local genome / no /extra needed — sequences come from https://api.genome.ucsc.edu. Used to
build the 32 kb input windows whose NTv3 per-bp tracks become the student's distillation targets.
"""

import json
import time
import urllib.request
from typing import List, Optional, Tuple

_UCSC = "https://api.genome.ucsc.edu/getData/sequence?genome={g};chrom={c};start={s};end={e}"


def fetch_sequence(chrom: str, start: int, end: int, assembly: str = "hg38") -> str:
    """Fetch a 0-based [start, end) DNA window from UCSC; non-ACGT -> N."""
    url = _UCSC.format(g=assembly, c=chrom, s=start, e=end)
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    seq = json.load(urllib.request.urlopen(req, timeout=60))["dna"].upper()
    return "".join(ch if ch in "ACGT" else "N" for ch in seq)


def tile_windows(
    chrom: str,
    start: int,
    end: int,
    window: int,
    stride: Optional[int] = None,
    n: Optional[int] = None,
) -> List[Tuple[str, int, int]]:
    """Tile [start, end) into ``window``-bp windows (default non-overlapping). Optional cap ``n``."""
    stride = stride or window
    coords, pos = [], start
    while pos + window <= end:
        coords.append((chrom, pos, pos + window))
        pos += stride
        if n is not None and len(coords) >= n:
            break
    return coords


def fetch_windows(
    coords: List[Tuple[str, int, int]], assembly: str = "hg38", sleep: float = 0.1
) -> List[str]:
    """Fetch sequences for a list of (chrom, start, end) windows (polite sleep between calls)."""
    seqs = []
    for chrom, s, e in coords:
        seqs.append(fetch_sequence(chrom, s, e, assembly))
        time.sleep(sleep)
    return seqs
