"""Faithful data layer for the official NTv3 benchmark fine-tuning recipe.

Ports the data pipeline from InstaDeepAI's official notebook
``notebooks_tutorials/03_fine_tuning_posttrained_model_biwig.ipynb`` (the "reproduce paper
results" setup) into our module layout, unchanged in behaviour:

  * ``GenomeBigWigDataset`` — dense overlapping windows tiled across ALL split regions
    (randomised by the DataLoader's ``shuffle``), with process-local FASTA/bigWig handles for
    multi-worker safety and on-the-fly per-window bigWig reads.
  * ``make_target_scaling_fn`` — the official target transform: divide by the per-track mean,
    then smooth-clip values above 10 (``2*sqrt(10x) - 10``). NOT log1p.
  * ``crop_center`` — keep the central ``keep_target_center_fraction`` of the length axis (0.375),
    matching the model's track-output crop.
  * ``sample_regions_for_total_length`` — accumulate whole/partial regions up to a token budget
    (used to bound the validation set), seeded.

Separate from ``ntv3_benchmark.py`` (the zero-shot teacher-eval data layer, which bins to fixed
output length + log1p); this module is the fine-tuning-faithful path.
"""

import bisect
import os
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

Region = Tuple[str, int, int]


def crop_center(x, keep_target_center_fraction: float = 0.375):
    """Keep the central ``keep_target_center_fraction`` along the length axis (..., seq_len, tracks).

    Works for numpy arrays and torch tensors (same slicing). Matches the model's track-output crop,
    so prediction and target cover the identical central region.
    """
    seq_len = x.shape[-2]
    target_offset = int(seq_len * (1 - keep_target_center_fraction) // 2)
    target_length = seq_len - 2 * target_offset
    return x[..., target_offset:target_offset + target_length, :]


def sample_regions_for_total_length(regions: List[Region], total_length_needed: int,
                                    seed: int = 0) -> List[Region]:
    """Accumulate regions (whole, or a random sub-window of the last one) until ``total_length_needed``
    base pairs are covered. Used to cap the validation set to a fixed token budget. Seeded."""
    sampled: List[Region] = []
    rng = np.random.RandomState(seed)
    accumulated = 0
    for chr_name, start, end in regions:
        region_length = end - start
        remaining = total_length_needed - accumulated
        if region_length >= remaining:
            max_start = region_length - remaining
            offset = rng.randint(0, max_start + 1) if max_start > 0 else 0
            sampled.append((chr_name, start + offset, start + offset + remaining))
            accumulated += remaining
            break
        sampled.append((chr_name, start, end))
        accumulated += region_length
        if accumulated >= total_length_needed:
            break
    return sampled


def make_target_scaling_fn(track_means: np.ndarray) -> Callable[[torch.Tensor], torch.Tensor]:
    """Official target transform: ``x / track_mean`` then smooth-clip ( >10 -> 2*sqrt(10x)-10 ).

    ``track_means`` is the per-track ``mean`` column from ``benchmark_metadata.tsv`` (track order).
    """
    means_t = torch.tensor(np.asarray(track_means), dtype=torch.float32)

    def transform_fn(x: torch.Tensor) -> torch.Tensor:
        scaled = x / means_t.to(x.device)
        return torch.where(scaled > 10.0, 2.0 * torch.sqrt(scaled * 10.0) - 10.0, scaled)

    return transform_fn


# --- process-local file-handle caches (one set of handles per DataLoader worker process) ---
_fasta_cache: Dict[Tuple[int, str], Any] = {}
_bigwig_cache: Dict[Tuple[int, str], Any] = {}

# sshfs-resilience knobs for bigWig I/O: the tracks live on an sshfs mount that can transiently raise
# ("Received an error during file opening!" on open, or an error mid-read) during a brief mount stall.
# A bounded exponential backoff survives these blips instead of crashing a multi-hour run. Widened to
# 8 attempts / 32s cap after a 2026-07-20 collision (two jobs' cache-load->read transitions storming
# sshfs at once) outlasted the original 5-attempt / 8s window: total retry window is now ~63s.
_BW_RETRY_ATTEMPTS = 8
_BW_RETRY_BASE_DELAY = 0.5  # seconds; backoff 0.5,1,2,4,8,16,32,32 (capped at _BW_RETRY_MAX_DELAY)
_BW_RETRY_MAX_DELAY = 32.0


def _retry_sshfs(fn: Callable[[], Any], what: str, path: str,
                 on_retry: Callable[[], None] = None):
    """Run an sshfs-backed bigWig I/O op ``fn`` with bounded exponential-backoff retries.

    Retries ``fn`` up to ``_BW_RETRY_ATTEMPTS`` times with exponential backoff (0.5s doubling, capped at
    ``_BW_RETRY_MAX_DELAY``), logging a warning per retry, and only re-raises (as a RuntimeError chaining
    the last error) after all attempts fail (~63s total window with the defaults). This
    directly survives the TRANSIENT sshfs errors seen on the bigWig ``open``/``.values`` reads. Optional
    ``on_retry`` runs before each backoff sleep — used to drop a stale cached handle so the next attempt
    REOPENS it (a stale sshfs handle cannot just be re-read). ``what``/``path`` are for messages only.
    """
    delay = _BW_RETRY_BASE_DELAY
    for attempt in range(1, _BW_RETRY_ATTEMPTS + 1):
        try:
            return fn()
        except Exception as e:  # transient sshfs open/read error (or, on last try, a real failure)
            if attempt == _BW_RETRY_ATTEMPTS:
                raise RuntimeError(
                    f"bigWig {what} failed after {_BW_RETRY_ATTEMPTS} attempts: {path}") from e
            print(f"[ntv3_ft_data] WARN transient bigWig {what} failed "
                  f"(attempt {attempt}/{_BW_RETRY_ATTEMPTS}): {e!r} :: {path}; retry in {delay:.1f}s",
                  flush=True)
            if on_retry is not None:
                on_retry()
            time.sleep(delay)
            delay = min(delay * 2.0, _BW_RETRY_MAX_DELAY)


def _get_fasta_handle(fasta_path: str):
    import pyfaidx

    key = (os.getpid(), str(Path(fasta_path).resolve()))
    if key not in _fasta_cache:
        _fasta_cache[key] = pyfaidx.Fasta(fasta_path, as_raw=True, sequence_always_upper=True)
    return _fasta_cache[key]


def _get_bigwig_handle(bigwig_path: str):
    """Return a process-local cached pyBigWig handle, opening (with sshfs retry) on first use."""
    import pyBigWig

    abs_path = str(Path(bigwig_path).resolve())
    key = (os.getpid(), abs_path)
    if key not in _bigwig_cache:
        if not Path(abs_path).exists():
            raise FileNotFoundError(f"BigWig not found: {abs_path}")
        _bigwig_cache[key] = _retry_sshfs(lambda: pyBigWig.open(abs_path), "open", abs_path)
    return _bigwig_cache[key]


def _read_bigwig_window(bigwig_path: str, chrom: str, start: int, end: int):
    """Read one window's ``values(chrom,start,end)`` from a bigWig, retrying transient sshfs errors.

    On failure the cached handle may be stale, so ``on_retry`` drops it and the next attempt reopens it
    via ``_get_bigwig_handle``. Returns the numpy value array (length ``end-start``).
    """
    abs_path = str(Path(bigwig_path).resolve())
    key = (os.getpid(), abs_path)
    return _retry_sshfs(
        lambda: _get_bigwig_handle(abs_path).values(chrom, start, end, numpy=True),
        "read", abs_path, on_retry=lambda: _bigwig_cache.pop(key, None))


class GenomeBigWigDataset(Dataset):
    """Reference genome + bigWig tracks over dense overlapping windows across a split's regions.

    Each region is tiled into ``sequence_length`` windows at ``stride = (1-overlap)*sequence_length``;
    a global index maps (via binary search) onto the per-region windows. Multi-worker safe (lazy,
    process-local handles). ``__getitem__`` returns tokenized DNA + center-cropped, scaled targets.
    Faithful port of the official notebook's dataset.
    """

    def __init__(self, fasta_path: str, bigwig_path_list: List[str], regions: List[Region],
                 sequence_length: int, tokenizer, transform_fn: Callable[[torch.Tensor], torch.Tensor],
                 overlap: float = 0.0, keep_target_center_fraction: float = 1.0,
                 limit_num_samples: int = None):
        super().__init__()
        self.fasta_path = fasta_path
        self.bigwig_path_list = bigwig_path_list
        self.sequence_length = sequence_length
        self.tokenizer = tokenizer
        self.transform_fn = transform_fn
        self.keep_target_center_fraction = keep_target_center_fraction
        self.stride = max(1, int((1 - overlap) * sequence_length))

        if limit_num_samples is not None:
            regions = sample_regions_for_total_length(regions, limit_num_samples * sequence_length)
        self.region_info, self._cumulative_starts, self.num_samples = self._process_regions(regions)

    def _process_regions(self, regions: List[Region]):
        """Index how many ``sequence_length`` windows each region yields (for global-idx lookup)."""
        region_info, cumulative_starts, total = [], [], 0
        for chr_name, region_s, region_e in regions:
            region_length = region_e - region_s
            n = (region_length - self.sequence_length) // self.stride + 1 if region_length >= self.sequence_length else 0
            if n > 0:
                region_info.append({"chr_name": chr_name, "region_start_offset": region_s, "num_samples": n})
                cumulative_starts.append(total)
                total += n
        return region_info, cumulative_starts, int(total)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        ci = bisect.bisect_right(self._cumulative_starts, idx) - 1
        chrom = self.region_info[ci]["chr_name"]
        region_start = self.region_info[ci]["region_start_offset"]
        start = region_start + (idx - self._cumulative_starts[ci]) * self.stride
        end = start + self.sequence_length

        seq = self._get_seq(chrom, start, end)
        tokens = self.tokenizer(seq, padding="max_length", truncation=True,
                                max_length=self.sequence_length, return_tensors="pt")["input_ids"][0]

        targets = np.array([_read_bigwig_window(p, chrom, start, end)
                            for p in self.bigwig_path_list]).T  # (seq_len, num_tracks)
        targets = torch.nan_to_num(torch.tensor(targets, dtype=torch.float32), nan=0.0)
        if self.keep_target_center_fraction < 1.0:
            targets = crop_center(targets, self.keep_target_center_fraction)
        targets = self.transform_fn(targets)
        # ``idx`` lets a cached-teacher KD run look up precomputed teacher logits for THIS exact window
        # (windows are deterministic given regions+seq_len+limit+seed, so idx aligns 1:1 with the cache).
        return {"tokens": tokens, "bigwig_targets": targets, "idx": idx}

    def _get_seq(self, chrom, start, end):
        return _get_fasta_handle(self.fasta_path)[chrom][start:end]


# ── CANONICAL track order (rigor guard against the index/assay confusion) ──────────────────────────
# The dataset order is ``sorted(glob("*.bigwig"))`` = alphabetical by file id, and ``--track_subset i``
# selects the i-th entry HERE. This is NOT the assay-sorted order of results/ntv3_650m_per_track_3seed.csv
# (so a CSV "row 18" is NOT ``--track_subset 18``). This list pins the human order so any data reshuffle
# fails loudly instead of silently retargeting every specialist experiment. (idx -> (file_id, assay).)
NTV3_HUMAN_TRACKS = [
    ("ENCSR046BCI_M", "PRO-cap"), ("ENCSR046BCI_P", "PRO-cap"),
    ("ENCSR100LIJ_M", "PRO-cap"), ("ENCSR100LIJ_P", "PRO-cap"),
    ("ENCSR114HGS_M", "PRO-cap"), ("ENCSR114HGS_P", "PRO-cap"),
    ("ENCSR154HRN_M", "eCLIP"), ("ENCSR154HRN_P", "eCLIP"),
    ("ENCSR249ROI_M", "eCLIP"), ("ENCSR249ROI_P", "eCLIP"),
    ("ENCSR321PWZ_M", "eCLIP"), ("ENCSR321PWZ_P", "eCLIP"),
    ("ENCSR325NFE", "ATAC-seq"), ("ENCSR410DWV", "ATAC-seq"),
    ("ENCSR484LTQ_M", "eCLIP"), ("ENCSR484LTQ_P", "eCLIP"),
    ("ENCSR487QSB", "ATAC-seq"),
    ("ENCSR527JGN_M", "polyA plus RNA-seq"), ("ENCSR527JGN_P", "polyA plus RNA-seq"),
    ("ENCSR619DQO_M", "total RNA-seq"), ("ENCSR619DQO_P", "total RNA-seq"),
    ("ENCSR628PLS", "ATAC-seq"), ("ENCSR682BFG", "Histone ChIP-seq"),
    ("ENCSR701YIC", "total RNA-seq"), ("ENCSR754DRC", "Histone ChIP-seq"),
    ("ENCSR799DGV_M", "PRO-cap"), ("ENCSR799DGV_P", "PRO-cap"),
    ("ENCSR814RGG", "ATAC-seq"),
    ("ENCSR862QCH_M", "eCLIP"), ("ENCSR862QCH_P", "eCLIP"),
    ("ENCSR863PSM", "Histone ChIP-seq"),
    ("ENCSR935RNW_M", "PRO-cap"), ("ENCSR935RNW_P", "PRO-cap"),
    ("ENCSR962OTG", "Histone ChIP-seq"),
]
NTV3_HUMAN_TRACK_IDS = [t[0] for t in NTV3_HUMAN_TRACKS]


def load_benchmark_frames(data_dir: str, species: str = "human"):
    """Resolve a local (already-downloaded/staged) benchmark dir to the inputs the FT pipeline needs.

    Returns ``(fasta_path, bigwig_paths, bigwig_ids, regions_by_split, track_means, track_assays)``
    where ``regions_by_split[split]`` is a list of (chrom, start, end) and ``track_means`` /
    ``track_assays`` are aligned to ``bigwig_ids``. Mirrors the notebook's ``prepare_genomics_inputs``
    ordering (metadata reindexed to the bigWig file ids), but reads from local files rather than
    ``snapshot_download``.
    """
    import pandas as pd

    root = Path(data_dir)
    fasta_path = str(root / species / "genome.fasta")
    bw_dir = root / species / "functional_tracks"
    bw_files = sorted(bw_dir.glob("*.bigwig"))
    bigwig_paths = [str(p) for p in bw_files]
    bigwig_ids = [p.stem for p in bw_files]
    # RIGOR GUARD: pin the human track order so --track_subset i always means the same track. If the data
    # dir ever yields a different/reordered set, fail loudly here rather than silently mis-target experiments.
    if species == "human" and bigwig_ids != NTV3_HUMAN_TRACK_IDS:
        raise RuntimeError(
            f"human bigWig order does not match the canonical NTV3_HUMAN_TRACK_IDS "
            f"({len(bigwig_ids)} ids); --track_subset indices would be wrong. "
            f"first mismatch: idx {next((i for i,(a,b) in enumerate(zip(bigwig_ids, NTV3_HUMAN_TRACK_IDS)) if a!=b), 'len-differs')}")

    splits_df = pd.read_csv(root / species / "splits.bed", sep="\t", header=None,
                            names=["chr_name", "start", "end", "split"],
                            dtype={"chr_name": str, "start": int, "end": int, "split": str})
    regions_by_split = {
        s: [(r.chr_name, int(r.start), int(r.end)) for r in g.itertuples()]
        for s, g in splits_df.groupby("split")
    }

    meta = pd.read_csv(root / "benchmark_metadata.tsv", sep="\t")
    meta = meta[meta["species_common_name"] == species].set_index("file_id").loc[bigwig_ids].reset_index()
    track_means = meta["mean"].to_numpy()
    track_assays = meta["assay"].tolist()
    return fasta_path, bigwig_paths, bigwig_ids, regions_by_split, track_means, track_assays


def validate_subset_bigwigs(idx: List[int], bw_ids: List[str], bw_paths: List[str]) -> None:
    """Fail fast if any SELECTED track's bigWig is not REAL, non-empty + openable data.

    The name-order guard in ``load_benchmark_frames`` only checks the file *names* exist + match the
    canonical order — it does NOT verify a track we actually train/cache has real data. So a 0-byte
    placeholder (the subset-transfer trick: empty files for unused tracks to pass the name guard) or a
    truncated/corrupt/mid-transfer bigWig would slip through and fail later with a cryptic dataloader
    ``[bwHdrRead]`` error. This checks each selected file is size>0 AND pyBigWig-openable AND has
    chromosomes, raising a clear error naming the exact track index + file_id. Reusable by the trainer
    and the cache builders (so a placeholder is impossible to train on or cache from silently).

    ``idx`` are the ORIGINAL dataset indices; ``bw_ids``/``bw_paths`` are the SUBSET lists (already sliced
    to ``idx``, len == len(idx)) — i.e. ``zip(idx, bw_ids, bw_paths)`` is the intended call shape.
    """
    import pyBigWig as _pbw

    for _i, _bid, _p in zip(idx, bw_ids, bw_paths):
        _sz = os.path.getsize(_p) if os.path.exists(_p) else -1
        if _sz <= 0:
            raise FileNotFoundError(
                f"--track_subset {_i} ({_bid}): bigWig is empty/missing ({_sz} bytes): {_p}\n"
                f"  -> this is an UNFILLED PLACEHOLDER. Transfer/stage the REAL bigWig for track {_i}.")
        try:
            _h = _pbw.open(_p); _chroms = _h.chroms(); _h.close()
        except Exception as _e:  # truncated/corrupt header, mid-transfer file, wrong format, ...
            raise RuntimeError(
                f"--track_subset {_i} ({_bid}): bigWig unreadable/corrupt ({_sz} bytes): {_p}\n"
                f"  pyBigWig: {_e}. Re-stage/re-transfer this track's bigWig (clear /tmp stage dir).") from _e
        if not _chroms:
            raise RuntimeError(f"--track_subset {_i} ({_bid}): bigWig has no chromosomes: {_p}")
