"""
Inference LATENCY + PEAK MEMORY benchmark for the NTv3 multi-track regression models
(teacher 650M + distilled students 8M / 100M / BPNet-0.12M), for the paper's regression
latency and memory table.

For each (model, task) it loads the model ONCE, builds the FULL NTv3 test loader (overlap=0,
all test windows = 10531 @ seqlen 32768), times inference, and records:
  - per-batch latency (mean / median / p90, ms)
  - whole-test-set total time (s)
  - throughput (windows/s)
  - peak memory (GPU: torch.cuda.max_memory_allocated; CPU: peak RSS via resource)
  - params (total)
  - device, gpu/cpu name, batch size, seqlen, n_test (windows + batches)

Heavy models on CPU (the 650M teacher) are timed over a small `--cpu-batches` budget and the
whole-test-set total is EXTRAPOLATED (linear) and labelled `extrapolated=True`.

The forward contract is identical for every model: model(tokens)["bigwig_tracks_logits"] -> [B,L_out,T].
Re-uses the exact builders/loader the fine-tune entrypoint uses, so the timed graph is the real one.

Output: one JSON record per (model, task) appended to --out-csv's sibling .jsonl, plus a CSV row.
"""

import argparse
import gc
import json
import os
import platform
import resource
import time

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.ntv3_benchmark import stage_to_local
from src.data.ntv3_ft_data import GenomeBigWigDataset, load_benchmark_frames, make_target_scaling_fn
from src.model.ntv3_finetune import (
    NTV3_CROP_FRAC,
    build_bigwig_model,
    load_finetuned_bigwig_teacher,
)


def _peak_rss_mb():
    """Peak resident set size of this process in MB (CPU-side peak memory)."""
    # ru_maxrss is KB on Linux.
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def _gpu_name():
    return torch.cuda.get_device_name(0) if torch.cuda.is_available() else None


def _cpu_name():
    """Best-effort human CPU model string from /proc/cpuinfo, falling back to platform."""
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return platform.processor() or platform.machine()


def _build_one_model(spec, num_tracks, tokenizer, device):
    """Build a single model from a spec dict and load its fine-tuned weights (if any).

    spec keys: kind ('ntv3'|'bpnet'|'teacher'), model (base/backbone path or HF id),
    ckpt (best_model.pth or None), channels, n_dilated.
    Returns (model, n_params_total).
    """
    kind = spec["kind"]
    if kind == "teacher":
        # 650M post teacher: build arch on the base and overwrite with the fine-tuned state_dict.
        model = load_finetuned_bigwig_teacher(
            spec["ckpt"], spec["model"], num_tracks, device=device
        )
    else:
        nuc_ids = {t: tokenizer.convert_tokens_to_ids(t) for t in "ACGT"}
        model = build_bigwig_model(
            spec["model"],
            num_tracks,
            keep_target_center_fraction=NTV3_CROP_FRAC,
            student_arch=("bpnet" if kind == "bpnet" else "ntv3"),
            channels=spec.get("channels"),
            n_dilated=spec.get("n_dilated"),
            nuc_ids=nuc_ids,
            vocab_size=tokenizer.vocab_size,
        ).to(device)
        if spec.get("ckpt") and os.path.exists(spec["ckpt"]):
            sd = torch.load(spec["ckpt"], map_location="cpu")
            sd = sd["model"] if isinstance(sd, dict) and "model" in sd else sd
            sd = {
                k[len("_orig_mod.") :] if k.startswith("_orig_mod.") else k: v
                for k, v in sd.items()
            }
            model.load_state_dict(
                sd, strict=False
            )  # rotary buffers tolerated; arch already correct
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    n_params = sum(p.numel() for p in model.parameters())
    return model, n_params


@torch.no_grad()
def _time_inference(model, loader, device, amp, max_batches=None):
    """Time inference over the loader. Returns dict with per-batch latencies (ms) and totals.

    Synchronizes around each batch on CUDA. If max_batches is set, only that many batches are timed
    (the caller marks the whole-set total as extrapolated). Data-loading time is EXCLUDED from the
    per-batch compute latency (we measure pure model forward), but the whole-set total is reported as
    the compute-only sum (data loading is host-side and overlaps in production).
    """
    is_cuda = device.startswith("cuda")
    lat_ms = []
    n_windows = 0
    n_batches = 0
    for batch in loader:
        tokens = batch["tokens"].to(device, non_blocking=True)
        if is_cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        if amp and is_cuda:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                _ = model(tokens)["bigwig_tracks_logits"]
        else:
            _ = model(tokens)["bigwig_tracks_logits"]
        if is_cuda:
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        lat_ms.append(dt * 1000.0)
        n_windows += tokens.shape[0]
        n_batches += 1
        if max_batches is not None and n_batches >= max_batches:
            break
    lat = np.asarray(lat_ms)
    return {
        "per_batch_ms_mean": float(lat.mean()),
        "per_batch_ms_median": float(np.median(lat)),
        "per_batch_ms_p90": float(np.percentile(lat, 90)),
        "per_batch_ms_min": float(lat.min()),
        "compute_total_s": float(lat.sum() / 1000.0),
        "n_batches_timed": n_batches,
        "n_windows_timed": n_windows,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--stage_dir", default=None)
    ap.add_argument("--species", default="human")
    ap.add_argument("--sequence_length", type=int, default=32768)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--device", default=None, help="cuda|cpu (default: cuda if available)")
    ap.add_argument(
        "--amp", action="store_true", help="bf16 autocast on CUDA (matches training default)"
    )
    ap.add_argument("--warmup-batches", type=int, default=3)
    ap.add_argument(
        "--cpu-batches",
        type=int,
        default=4,
        help="for heavy models on CPU: time this many batches, extrapolate whole-set total",
    )
    ap.add_argument(
        "--max-test-windows",
        type=int,
        default=None,
        help="cap test windows (None = full test set, the default)",
    )
    ap.add_argument("--specs", required=True, help="path to a JSON list of model specs")
    ap.add_argument(
        "--out-prefix", required=True, help="output path prefix (.jsonl + .csv written)"
    )
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if device.startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    data_dir = args.data_dir
    if args.stage_dir:
        print(f"staging genome+bigWigs -> {args.stage_dir}", flush=True)
        data_dir = stage_to_local(args.data_dir, args.stage_dir, args.species)

    fasta, bw_paths_all, bw_ids_all, regions_by_split, track_means_all, track_assays_all = (
        load_benchmark_frames(data_dir, args.species)
    )
    native_n = len(bw_ids_all)
    print(
        f"native tracks={native_n} | test regions span -> building full test windows (overlap=0)",
        flush=True,
    )

    from transformers import AutoTokenizer

    # tokenizer is identical across NTv3 sizes; load from the first ntv3 spec's model or a default.
    specs = json.load(open(args.specs))

    gpu = _gpu_name()
    cpu = _cpu_name()
    print(
        f"device={device} gpu={gpu} cpu={cpu} batch={args.batch_size} seqlen={args.sequence_length}",
        flush=True,
    )

    jsonl_path = args.out_prefix + ".jsonl"
    csv_path = args.out_prefix + ".csv"
    csv_cols = [
        "model",
        "task",
        "kind",
        "n_tracks",
        "params_M",
        "device",
        "gpu",
        "cpu",
        "batch_size",
        "seqlen",
        "n_test_windows",
        "n_test_batches",
        "per_batch_ms_mean",
        "per_batch_ms_median",
        "per_batch_ms_p90",
        "whole_test_total_s",
        "extrapolated",
        "throughput_win_per_s",
        "peak_mem_MB",
        "peak_mem_kind",
        "amp",
    ]
    write_header = not os.path.exists(csv_path)
    csv_f = open(csv_path, "a")
    if write_header:
        csv_f.write(",".join(csv_cols) + "\n")

    # Cache test loaders by (track_subset tuple) so single-track tasks reuse one build per index,
    # and the joint-34 build is shared.
    tok_cache = {}

    for spec in specs:
        name = spec["name"]
        task = spec["task"]
        kind = spec["kind"]
        track_subset = spec.get("track_subset")  # list[int] or None (joint-34)
        tok_model = spec["model"]
        if tok_model not in tok_cache:
            tok_cache[tok_model] = AutoTokenizer.from_pretrained(
                tok_model, trust_remote_code=True, local_files_only=os.path.isdir(tok_model)
            )
        tokenizer = tok_cache[tok_model]

        if track_subset:
            idx = list(track_subset)
            bw_paths = [bw_paths_all[i] for i in idx]
            track_means = (
                track_means_all[idx]
                if hasattr(track_means_all, "__getitem__") and not isinstance(track_means_all, list)
                else [track_means_all[i] for i in idx]
            )
        else:
            idx = None
            bw_paths = bw_paths_all
            track_means = track_means_all
        T = len(bw_paths)
        transform_fn = make_target_scaling_fn(track_means)

        test_ds = GenomeBigWigDataset(
            fasta,
            bw_paths,
            regions_by_split["test"],
            args.sequence_length,
            tokenizer,
            transform_fn,
            overlap=0.0,
            keep_target_center_fraction=NTV3_CROP_FRAC,
            limit_num_samples=args.max_test_windows,
        )
        test_loader = DataLoader(
            test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
        )
        n_test = len(test_ds)
        n_batches_full = (n_test + args.batch_size - 1) // args.batch_size
        print(
            f"\n=== {name} [{task}] kind={kind} T={T} | test windows={n_test} batches={n_batches_full} ===",
            flush=True,
        )

        model, n_params = _build_one_model(spec, T, tokenizer, device)
        print(f"params={n_params / 1e6:.3f}M", flush=True)

        # warmup
        if device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            nb = 0
            for batch in test_loader:
                tokens = batch["tokens"].to(device)
                if args.amp and device.startswith("cuda"):
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        _ = model(tokens)["bigwig_tracks_logits"]
                else:
                    _ = model(tokens)["bigwig_tracks_logits"]
                nb += 1
                if nb >= args.warmup_batches:
                    break
        if device.startswith("cuda"):
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()

        # heavy on CPU -> time a few batches, extrapolate
        is_heavy_cpu = (device == "cpu") and (kind == "teacher")
        max_b = args.cpu_batches if is_heavy_cpu else None
        timing = _time_inference(model, test_loader, device, args.amp, max_batches=max_b)

        if is_heavy_cpu:
            per_win = timing["compute_total_s"] / max(1, timing["n_windows_timed"])
            whole_total = per_win * n_test
            extrapolated = True
        else:
            whole_total = timing["compute_total_s"]
            extrapolated = False
        throughput = n_test / whole_total if whole_total > 0 else float("nan")

        if device.startswith("cuda"):
            peak_mem = torch.cuda.max_memory_allocated() / (1024**2)
            peak_kind = "cuda_max_allocated"
        else:
            peak_mem = _peak_rss_mb()
            peak_kind = "cpu_peak_rss"

        rec = {
            "model": name,
            "task": task,
            "kind": kind,
            "n_tracks": T,
            "params_M": n_params / 1e6,
            "device": device,
            "gpu": gpu,
            "cpu": cpu,
            "batch_size": args.batch_size,
            "seqlen": args.sequence_length,
            "n_test_windows": n_test,
            "n_test_batches": n_batches_full,
            "per_batch_ms_mean": timing["per_batch_ms_mean"],
            "per_batch_ms_median": timing["per_batch_ms_median"],
            "per_batch_ms_p90": timing["per_batch_ms_p90"],
            "whole_test_total_s": whole_total,
            "extrapolated": extrapolated,
            "throughput_win_per_s": throughput,
            "peak_mem_MB": peak_mem,
            "peak_mem_kind": peak_kind,
            "amp": bool(args.amp),
            "n_batches_timed": timing["n_batches_timed"],
            "n_windows_timed": timing["n_windows_timed"],
        }
        with open(jsonl_path, "a") as jf:
            jf.write(json.dumps(rec) + "\n")
        csv_f.write(",".join(str(rec[c]) for c in csv_cols) + "\n")
        csv_f.flush()
        print(
            f"RESULT {name}: per_batch={rec['per_batch_ms_mean']:.1f}ms "
            f"whole_test={whole_total:.1f}s (extrap={extrapolated}) "
            f"peak_mem={peak_mem:.0f}MB params={n_params / 1e6:.2f}M",
            flush=True,
        )

        del model
        gc.collect()
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    csv_f.close()
    print(f"\n=== DONE. wrote {jsonl_path} and {csv_path} ===", flush=True)


if __name__ == "__main__":
    main()
