"""
Utilities for benchmarking inference latency and memory usage.
"""

import os
import gc
import csv
import time
import torch
import numpy as np
import contextlib
from dataclasses import dataclass, asdict
from typing import Optional, List, Dict, Any, Union
from datetime import datetime
from tqdm import tqdm


def _fp16_autocast_ctx(device: str, dtype: torch.dtype):
    """Return an fp16 autocast context for the timed forward, or a no-op for fp32/CPU.

    Why: some teachers (Enformer, DNABERT-2) keep a few internal tensors (bias/LayerNorm,
    positional/embedding buffers) in float32 even after ``model.half()``, which raises a
    "expected scalar type Half but found Float" mismatch. ``torch.autocast`` runs the
    matmul/conv ops in fp16 while transparently promoting those mixed bias/LN ops, so EVERY
    teacher runs under the SAME fp16 path (single-precision latency/memory table, R1.11).
    Returns a nullcontext when dtype is fp32 or the device is CPU (CUDA-only fp16 autocast).
    """
    if dtype == torch.float16 and torch.cuda.is_available() and "cuda" in device:
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return contextlib.nullcontext()


@dataclass
class BenchmarkResult:
    """Container for benchmark results."""

    model_name: str
    model_type: str  # "teacher" or "student"
    task_name: str
    batch_size: int
    seq_length: int
    num_samples: int
    # Latency metrics (in milliseconds)
    mean_latency_ms: float
    std_latency_ms: float
    min_latency_ms: float
    max_latency_ms: float
    median_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    # Throughput
    throughput_samples_per_sec: float
    # Memory metrics (in MB)
    peak_memory_mb: float
    allocated_memory_mb: float
    reserved_memory_mb: float
    # Model info
    num_parameters: int
    # Whole-dataset wall-clock inference time (seconds). If total_time_extrapolated is True,
    # this is mean_per_batch_latency * n_total_batches (timed only a subset; the big teachers
    # are too slow to run the full test set), else it is the measured sum over all batches (R2.2).
    total_time_s: float = 0.0
    total_time_extrapolated: bool = False
    # Exact test-set size for this task (n samples). num_samples = samples actually timed.
    n_test: int = 0
    # Native (pre-padding) median sequence length in bp for this task; seq_length = padded length fed to model.
    native_seq_len: int = 0
    # GPU hardware the benchmark ran on (same-hardware comparison for the rebuttal table).
    gpu_name: str = ""
    # CPU model name (recorded for the CPU-latency sub-table, R1.11 deployment claim).
    cpu_name: str = ""
    # Compute device actually used for this row ("cuda" or "cpu").
    device: str = ""
    # Numeric precision the forward ran at ("fp16" or "fp32"). With the precision-unification
    # fix every (teacher AND student) row in a sweep shares one value -> the latency/memory
    # table is single-precision (R1.11; no apples-to-oranges fp16-vs-fp32 split).
    precision: str = ""
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()
        if not self.gpu_name:
            self.gpu_name = (
                torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
            )
        if not self.cpu_name:
            self.cpu_name = _get_cpu_name()

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _get_cpu_name() -> str:
    """Human-readable CPU model from /proc/cpuinfo (Linux); falls back to platform."""
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except Exception:
        pass
    import platform

    return platform.processor() or platform.machine() or "unknown-cpu"


def count_parameters(model: torch.nn.Module) -> int:
    """Count total number of parameters in a model."""
    return sum(p.numel() for p in model.parameters())


def full_memory_cleanup(device: str = "cuda"):
    """
    Aggressively clean up ALL GPU memory to get accurate per-model measurements.
    Call this BEFORE loading each model.
    """
    if torch.cuda.is_available() and "cuda" in device:
        # Run garbage collection first to free Python objects
        gc.collect()

        # Empty CUDA cache
        torch.cuda.empty_cache()

        # Reset all memory stats
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.reset_accumulated_memory_stats()

        # Synchronize to ensure all operations complete
        torch.cuda.synchronize()

        # Second round of cleanup
        gc.collect()
        torch.cuda.empty_cache()

        # Small sleep to let GPU fully release memory
        time.sleep(0.5)


def reset_memory_stats(device: str = "cuda"):
    """Reset CUDA memory statistics before timing."""
    if torch.cuda.is_available() and "cuda" in device:
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()


def _process_rss_mb() -> float:
    """Resident-set-size (RSS) of this process in MB, via psutil (CPU-memory proxy)."""
    try:
        import psutil

        return psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
    except Exception:
        return 0.0


def get_memory_stats(device: str = "cuda") -> Dict[str, float]:
    """Get current memory statistics in MB.

    On CUDA: GPU allocator stats. On CPU: process RSS (psutil) as peak/allocated,
    so CPU runs still report a meaningful memory footprint for the rebuttal table.
    """
    if torch.cuda.is_available() and "cuda" in device:
        torch.cuda.synchronize()
        return {
            "peak_memory_mb": torch.cuda.max_memory_allocated() / 1024 / 1024,
            "allocated_memory_mb": torch.cuda.memory_allocated() / 1024 / 1024,
            "reserved_memory_mb": torch.cuda.memory_reserved() / 1024 / 1024,
        }
    rss = _process_rss_mb()
    return {
        "peak_memory_mb": rss,
        "allocated_memory_mb": rss,
        "reserved_memory_mb": 0.0,
    }


def get_model_size_mb(model: torch.nn.Module) -> float:
    """Calculate model size in MB based on parameters."""
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    return (param_size + buffer_size) / 1024 / 1024


@torch.no_grad()
def benchmark_model_inference(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: str,
    model_name: str,
    model_type: str,
    task_name: str,
    warmup_batches: int = 10,
    num_batches: Optional[int] = None,
    dtype: torch.dtype = torch.float32,
) -> BenchmarkResult:
    """
    Benchmark model inference latency and memory usage.

    Memory measurement is isolated - we measure only this model's footprint.

    Args:
        dtype: forward precision. fp16 runs the forward under a CUDA fp16 autocast so the
            BPNet student matches the teachers' precision column (R1.11 single-precision
            table). The student is NOT ``model.half()``-cast because its forward one-hot
            encodes the long ``input_ids`` into float32 internally (half weights would break
            the index op); autocast gives fp16 compute without that. fp32 = no autocast.
    """
    autocast_ctx = _fp16_autocast_ctx(device, dtype)
    # ============================================================
    # STEP 1: Full cleanup BEFORE loading model to GPU
    # ============================================================
    full_memory_cleanup(device)

    # Record baseline memory (should be ~0 after cleanup)
    baseline_memory = (
        torch.cuda.memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0
    )
    print(f"  Baseline memory after cleanup: {baseline_memory:.2f} MB")

    # ============================================================
    # STEP 2: Load model to device and measure model memory
    # ============================================================
    model.eval()
    model.to(device)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    model_memory = (
        torch.cuda.memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0
    )
    model_size_calc = get_model_size_mb(model)
    print(
        f"  Model memory on GPU: {model_memory:.2f} MB (calculated size: {model_size_calc:.2f} MB)"
    )

    # Get batch info from first batch
    sample_batch = next(iter(dataloader))
    if isinstance(sample_batch, (list, tuple)):
        batch_size = sample_batch[0].shape[0]
        seq_length = sample_batch[0].shape[1] if len(sample_batch[0].shape) > 1 else 0
    else:
        batch_size = sample_batch.shape[0]
        seq_length = sample_batch.shape[1] if len(sample_batch.shape) > 1 else 0

    # ============================================================
    # STEP 3: Reset memory stats, then do warmup
    # ============================================================
    reset_memory_stats(device)

    print(f"  Warming up with {warmup_batches} batches...")
    warmup_iter = iter(dataloader)
    for _ in range(min(warmup_batches, len(dataloader))):
        try:
            batch = next(warmup_iter)
        except StopIteration:
            warmup_iter = iter(dataloader)
            batch = next(warmup_iter)

        if isinstance(batch, (list, tuple)):
            inputs = batch[0].to(device)
        else:
            inputs = batch.to(device)

        with autocast_ctx:
            _ = model(inputs)

        # Clear intermediate tensors
        del inputs

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # ============================================================
    # STEP 4: Reset stats again, then benchmark
    # ============================================================
    reset_memory_stats(device)

    latencies = []
    total_samples = 0
    max_batches = num_batches if num_batches else len(dataloader)

    print(f"  Benchmarking {model_name} on {task_name}...")
    for i, batch in enumerate(tqdm(dataloader, total=max_batches, desc="Benchmarking")):
        if i >= max_batches:
            break

        if isinstance(batch, (list, tuple)):
            inputs = batch[0].to(device)
            current_batch_size = inputs.shape[0]
        else:
            inputs = batch.to(device)
            current_batch_size = inputs.shape[0]

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        start_time = time.perf_counter()
        with autocast_ctx:
            output = model(inputs)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        end_time = time.perf_counter()

        latency_ms = (end_time - start_time) * 1000
        latencies.append(latency_ms)
        total_samples += current_batch_size

        # Clean up batch tensors to avoid memory accumulation
        del inputs, output

    # ============================================================
    # STEP 5: Collect final memory stats
    # ============================================================
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    memory_stats = get_memory_stats(device)

    # Compute statistics
    latencies = np.array(latencies)
    measured_total_s = latencies.sum() / 1000
    throughput = total_samples / measured_total_s if measured_total_s > 0 else 0

    n_total_batches = len(dataloader)
    is_extrapolated = max_batches < n_total_batches
    if is_extrapolated:
        whole_dataset_s = float(np.mean(latencies)) / 1000.0 * n_total_batches
    else:
        whole_dataset_s = float(measured_total_s)
    # Exact test-set size = batches * batch_size (last batch may be short; this is an upper bound).
    n_test_total = n_total_batches * batch_size

    result = BenchmarkResult(
        model_name=model_name,
        model_type=model_type,
        task_name=task_name,
        batch_size=batch_size,
        seq_length=seq_length,
        num_samples=total_samples,
        mean_latency_ms=float(np.mean(latencies)),
        std_latency_ms=float(np.std(latencies)),
        min_latency_ms=float(np.min(latencies)),
        max_latency_ms=float(np.max(latencies)),
        median_latency_ms=float(np.median(latencies)),
        p95_latency_ms=float(np.percentile(latencies, 95)),
        p99_latency_ms=float(np.percentile(latencies, 99)),
        throughput_samples_per_sec=throughput,
        peak_memory_mb=memory_stats["peak_memory_mb"],
        allocated_memory_mb=memory_stats["allocated_memory_mb"],
        reserved_memory_mb=memory_stats["reserved_memory_mb"],
        num_parameters=count_parameters(model),
        total_time_s=whole_dataset_s,
        total_time_extrapolated=is_extrapolated,
        n_test=n_test_total,
        native_seq_len=int(seq_length),
        device="cuda" if (torch.cuda.is_available() and "cuda" in device) else "cpu",
        precision="fp16" if dtype == torch.float16 else "fp32",
    )

    # ============================================================
    # STEP 6: Cleanup after benchmark
    # ============================================================
    model.cpu()  # Move model back to CPU
    del model
    full_memory_cleanup(device)

    return result


@torch.no_grad()
def benchmark_teacher_inference(
    model: torch.nn.Module,
    tokenizer,
    sequences: List[str],
    device: str,
    model_name: str,
    task_name: str,
    batch_size: int = 16,
    max_length: int = 1000,
    warmup_batches: int = 10,
    num_batches: Optional[int] = None,
    dtype: torch.dtype = torch.float16,
    input_prefix: str = "",
    add_special_tokens: bool = True,
) -> BenchmarkResult:
    """
    Benchmark teacher model inference with tokenization.

    Memory measurement is isolated - we measure only this model's footprint.

    Args:
        dtype: weight precision to benchmark at (torch.float16 = fp16, torch.float32 = fp32).
            Cast is applied after moving the model to ``device`` so peak-memory reflects it.
        input_prefix: teacher input formatting prefix (e.g. Carbon-3B needs "<dna>"); "" = no-op.
        add_special_tokens: passed to the tokenizer (Carbon-3B uses False to match its fine-tune).
    """
    # Apply the teacher's input prefix once (e.g. Carbon "<dna>"); no-op for others.
    if input_prefix:
        sequences = [f"{input_prefix}{s}" for s in sequences]
    # ============================================================
    # STEP 1: Full cleanup BEFORE loading model to GPU
    # ============================================================
    full_memory_cleanup(device)

    baseline_memory = (
        torch.cuda.memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0
    )
    print(f"  Baseline memory after cleanup: {baseline_memory:.2f} MB")

    # ============================================================
    # STEP 2: Load model to device
    # ============================================================
    model.eval()
    model.to(device)
    if dtype == torch.float16:
        # Half-precision inference (fp16). Cast float weights/buffers to half so the
        # peak-memory column genuinely reflects fp16 storage. Some teachers (Enformer,
        # DNABERT-2) leave a few internal float32 tensors behind, so we ALSO wrap the
        # forward in an fp16 autocast (see _fp16_autocast_ctx) which promotes those mixed
        # bias/LayerNorm ops automatically -> every teacher runs under the same fp16 path.
        model.half()
    autocast_ctx = _fp16_autocast_ctx(device, dtype)
    print(f"  Benchmarking at precision: {dtype}")

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    model_memory = (
        torch.cuda.memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0
    )
    model_size_calc = get_model_size_mb(model)
    print(
        f"  Model memory on GPU: {model_memory:.2f} MB (calculated size: {model_size_calc:.2f} MB)"
    )

    # Create batches
    num_sequences = len(sequences)
    batches = [
        sequences[i : i + batch_size] for i in range(0, num_sequences, batch_size)
    ]
    max_batches = num_batches if num_batches else len(batches)

    # ============================================================
    # STEP 3: Warmup
    # ============================================================
    reset_memory_stats(device)

    print(f"  Warming up with {warmup_batches} batches...")
    for i in range(min(warmup_batches, len(batches))):
        batch_seqs = batches[i % len(batches)]
        tok = tokenizer(
            batch_seqs,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
            add_special_tokens=add_special_tokens,
        )

        if isinstance(tok, dict):
            input_ids = tok["input_ids"].to(device)
            attention_mask = tok.get("attention_mask", torch.ones_like(input_ids)).to(
                device
            )
        else:
            input_ids = tok.input_ids.to(device)
            attention_mask = (
                tok.attention_mask.to(device)
                if hasattr(tok, "attention_mask")
                else torch.ones_like(input_ids)
            )

        with autocast_ctx:
            _ = model(input_ids=input_ids, attention_mask=attention_mask)

        # Clear intermediate tensors
        del input_ids, attention_mask, tok

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # ============================================================
    # STEP 4: Reset and benchmark
    # ============================================================
    reset_memory_stats(device)

    latencies = []
    total_samples = 0

    print(f"  Benchmarking {model_name} on {task_name}...")
    for i, batch_seqs in enumerate(tqdm(batches[:max_batches], desc="Benchmarking")):
        tok = tokenizer(
            batch_seqs,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
            add_special_tokens=add_special_tokens,
        )

        if isinstance(tok, dict):
            input_ids = tok["input_ids"].to(device)
            attention_mask = tok.get("attention_mask", torch.ones_like(input_ids)).to(
                device
            )
        else:
            input_ids = tok.input_ids.to(device)
            attention_mask = (
                tok.attention_mask.to(device)
                if hasattr(tok, "attention_mask")
                else torch.ones_like(input_ids)
            )

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        start_time = time.perf_counter()
        with autocast_ctx:
            output = model(input_ids=input_ids, attention_mask=attention_mask)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        end_time = time.perf_counter()

        latency_ms = (end_time - start_time) * 1000
        latencies.append(latency_ms)
        total_samples += len(batch_seqs)

        # Clear intermediate tensors
        del input_ids, attention_mask, tok, output

    # ============================================================
    # STEP 5: Collect memory stats
    # ============================================================
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    memory_stats = get_memory_stats(device)

    # Statistics
    latencies = np.array(latencies)
    measured_total_s = latencies.sum() / 1000
    throughput = total_samples / measured_total_s if measured_total_s > 0 else 0

    # Whole-dataset time. If we only timed a subset of batches (num_batches < all), report
    # the EXTRAPOLATED full-test-set time = mean_per_batch_latency * n_total_batches, flagged.
    n_total_batches = len(batches)
    is_extrapolated = max_batches < n_total_batches
    if is_extrapolated:
        whole_dataset_s = float(np.mean(latencies)) / 1000.0 * n_total_batches
    else:
        whole_dataset_s = float(measured_total_s)

    # Native (pre-padding) median sequence length in bp for this task.
    native_len = int(np.median([len(s) for s in sequences])) if sequences else 0

    result = BenchmarkResult(
        model_name=model_name,
        model_type="teacher",
        task_name=task_name,
        batch_size=batch_size,
        seq_length=max_length,
        num_samples=total_samples,
        mean_latency_ms=float(np.mean(latencies)),
        std_latency_ms=float(np.std(latencies)),
        min_latency_ms=float(np.min(latencies)),
        max_latency_ms=float(np.max(latencies)),
        median_latency_ms=float(np.median(latencies)),
        p95_latency_ms=float(np.percentile(latencies, 95)),
        p99_latency_ms=float(np.percentile(latencies, 99)),
        throughput_samples_per_sec=throughput,
        peak_memory_mb=memory_stats["peak_memory_mb"],
        allocated_memory_mb=memory_stats["allocated_memory_mb"],
        reserved_memory_mb=memory_stats["reserved_memory_mb"],
        num_parameters=count_parameters(model),
        total_time_s=whole_dataset_s,
        total_time_extrapolated=is_extrapolated,
        n_test=num_sequences,
        native_seq_len=native_len,
        device="cuda" if (torch.cuda.is_available() and "cuda" in device) else "cpu",
        precision="fp16" if dtype == torch.float16 else "fp32",
    )

    # ============================================================
    # STEP 6: Cleanup after benchmark
    # ============================================================
    model.cpu()
    del model
    full_memory_cleanup(device)

    return result


def save_benchmark_results_csv(
    results: List[BenchmarkResult],
    output_path: str,
):
    """Save benchmark results to CSV file."""
    if not results:
        print("No results to save.")
        return

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fieldnames = list(asdict(results[0]).keys())

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(result.to_dict())

    print(f"✓ Saved benchmark results to: {output_path}")


def log_benchmark_to_wandb(result: BenchmarkResult, prefix: str = "benchmark"):
    """Log benchmark result to wandb."""
    import wandb

    metrics = {
        f"{prefix}/{result.task_name}/{result.model_type}/{result.model_name}/mean_latency_ms": result.mean_latency_ms,
        f"{prefix}/{result.task_name}/{result.model_type}/{result.model_name}/std_latency_ms": result.std_latency_ms,
        f"{prefix}/{result.task_name}/{result.model_type}/{result.model_name}/median_latency_ms": result.median_latency_ms,
        f"{prefix}/{result.task_name}/{result.model_type}/{result.model_name}/p95_latency_ms": result.p95_latency_ms,
        f"{prefix}/{result.task_name}/{result.model_type}/{result.model_name}/p99_latency_ms": result.p99_latency_ms,
        f"{prefix}/{result.task_name}/{result.model_type}/{result.model_name}/throughput": result.throughput_samples_per_sec,
        f"{prefix}/{result.task_name}/{result.model_type}/{result.model_name}/peak_memory_mb": result.peak_memory_mb,
        f"{prefix}/{result.task_name}/{result.model_type}/{result.model_name}/num_parameters": result.num_parameters,
    }

    wandb.log(metrics)
    return metrics


def normalize_task_names(task_names: Union[str, List[str]]) -> List[str]:
    """
    Normalize task_names input to always return a list.

    Args:
        task_names: Either a single task name string or a list of task names

    Returns:
        List of task names
    """
    if isinstance(task_names, str):
        return [task_names]
    return list(task_names)
