"""Loader for the official NTv3 Benchmark dataset (InstaDeepAI/NTv3_benchmark_dataset).

Faithful-reproduction data layer: train/val/test windows from THEIR ``splits.bed``, sequences from
THEIR ``genome.fasta`` (pyfaidx), and per-bp targets from THEIR ``functional_tracks/*.bigwig`` with
the paper's processing (``log(1+x)``; per-track mean/std available in ``benchmark_metadata.tsv``).

Reuses ``ground_truth.ground_truth_targets`` for the bigWig binning — this module only adds the
splits parsing, the FASTA sequence reader, and the log1p transform.
"""

import csv
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from src.data.ground_truth import ground_truth_targets

Coord = Tuple[str, int, int]


def load_splits(path: str) -> Dict[str, List[Coord]]:
    """Parse splits.bed -> {split_name: [(chrom, start, end), ...]} (split = 4th column)."""
    out: Dict[str, List[Coord]] = {}
    with open(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) < 4:
                continue
            chrom, start, end, split = parts[0], int(parts[1]), int(parts[2]), parts[3]
            out.setdefault(split, []).append((chrom, start, end))
    return out


def sample_windows(intervals: List[Coord], window: int, stride: int = None, n: int = None) -> List[Coord]:
    """Tile ``intervals`` into ``window``-bp windows (default non-overlapping). Optional cap ``n``."""
    stride = stride or window
    coords: List[Coord] = []
    for chrom, start, end in intervals:
        pos = start
        while pos + window <= end:
            coords.append((chrom, pos, pos + window))
            if n is not None and len(coords) >= n:
                return coords
            pos += stride
    return coords


@dataclass
class TrackMeta:
    file_id: str
    assay: str
    mean: float
    std: float


def load_track_meta(meta_path: str, species: str = "human") -> List[TrackMeta]:
    """benchmark_metadata.tsv -> ordered TrackMeta list for one species' functional tracks."""
    with open(meta_path) as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    return [
        TrackMeta(r["file_id"], r["assay"], float(r["mean"]), float(r["std"]))
        for r in rows
        if r["species_common_name"] == species
    ]


class BenchmarkData:
    """Sequences (genome.fasta via pyfaidx) + per-bp targets (their bigWigs) over windows."""

    def __init__(self, genome_fasta: str, track_dir: str, track_meta: List[TrackMeta]):
        import pyfaidx

        self.fa = pyfaidx.Fasta(genome_fasta, sequence_always_upper=True)
        self.track_meta = track_meta
        self.bw_paths = [os.path.join(track_dir, f"{m.file_id}.bigwig") for m in track_meta]
        missing = [p for p in self.bw_paths if not os.path.exists(p)]
        if missing:
            raise FileNotFoundError(f"{len(missing)} benchmark bigWigs missing, e.g. {missing[:2]}")

    @property
    def labels(self) -> List[str]:
        return [m.file_id for m in self.track_meta]

    def sequences(self, coords: List[Coord]) -> List[str]:
        return [str(self.fa[c][s:e]) for c, s, e in coords]

    def targets(self, coords: List[Coord], nbins: int, log1p: bool = True) -> np.ndarray:
        """[N, nbins, T] binned signal; log(1+x) per the paper's eval procedure when ``log1p``."""
        out = ground_truth_targets(self.bw_paths, coords, nbins)  # reuse the bigWig binning
        return np.log1p(np.clip(out, 0.0, None)) if log1p else out
