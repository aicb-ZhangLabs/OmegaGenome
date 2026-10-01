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


def sample_windows(
    intervals: List[Coord], window: int, stride: int = None, n: int = None
) -> List[Coord]:
    """Tile ``intervals`` into ``window``-bp windows (default non-overlapping). Optional cap ``n``.

    NOTE: this is the zero-shot teacher-eval data path. The benchmark *fine-tuning* path uses the
    faithful ``GenomeBigWigDataset`` (dense overlap across all regions) instead — see ``ntv3_ft_data``.
    """
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


def stage_to_local(data_dir: str, stage_dir: str, species: str = "human") -> str:
    """Copy a species' genome(+.fai) + functional_tracks bigWigs + splits + metadata to fast local
    storage (e.g. node-local /tmp), so per-window bigWig reads aren't sshfs-latency-bound.

    Returns ``stage_dir`` (use it as the data_dir). Idempotent — skips files already present with a
    matching size, so a re-run or a shared stage dir doesn't recopy.
    """
    import shutil

    rels = [
        f"{species}/genome.fasta",
        f"{species}/genome.fasta.fai",
        f"{species}/splits.bed",
        "benchmark_metadata.tsv",
    ]
    ft = os.path.join(data_dir, species, "functional_tracks")
    if os.path.isdir(ft):
        rels += [
            f"{species}/functional_tracks/{f}"
            for f in sorted(os.listdir(ft))
            if f.endswith(".bigwig")
        ]
    for rel in rels:
        src, dst = os.path.join(data_dir, rel), os.path.join(stage_dir, rel)
        if not os.path.exists(src):
            continue
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if not (os.path.exists(dst) and os.path.getsize(dst) == os.path.getsize(src)):
            # Atomic copy (copy to a pid-scoped temp then os.replace) so concurrent stagers — e.g.
            # several seeds sharing one node-local dir — can't read or finish a half-written file.
            tmp = f"{dst}.tmp.{os.getpid()}"
            shutil.copy2(src, tmp)
            os.replace(tmp, dst)
    return stage_dir


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
