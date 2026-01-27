"""
Utilities for benchmarking inference latency and memory usage.
"""

import os
import gc
import csv
import time
import torch
import numpy as np
from dataclasses import dataclass, asdict
from typing import Optional, List, Dict, Any, Union
from datetime import datetime
from tqdm import tqdm


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
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


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


def get_memory_stats(device: str = "cuda") -> Dict[str, float]:
    """Get current CUDA memory statistics in MB."""
    if torch.cuda.is_available() and "cuda" in device:
        torch.cuda.synchronize()
        return {
            "peak_memory_mb": torch.cuda.max_memory_allocated() / 1024 / 1024,
            "allocated_memory_mb": torch.cuda.memory_allocated() / 1024 / 1024,
            "reserved_memory_mb": torch.cuda.memory_reserved() / 1024 / 1024,
        }
    return {
        "peak_memory_mb": 0.0,
        "allocated_memory_mb": 0.0,
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
) -> BenchmarkResult:
    """
    Benchmark model inference latency and memory usage.

    Memory measurement is isolated - we measure only this model's footprint.
    """
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
    total_time_sec = latencies.sum() / 1000
    throughput = total_samples / total_time_sec if total_time_sec > 0 else 0

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
) -> BenchmarkResult:
    """
    Benchmark teacher model inference with tokenization.

    Memory measurement is isolated - we measure only this model's footprint.
    """
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
    total_time_sec = latencies.sum() / 1000
    throughput = total_samples / total_time_sec if total_time_sec > 0 else 0

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
