"""
CPU inference-latency benchmark for the FOUR teacher models (Enformer, DNABERT-2,
Caduceus, Carbon-3B) for the OmegaGenome rebuttal CPU-latency table (T18 / R1.11).

Reuses the EXACT GPU-measurement path (get_teacher_model + benchmark_teacher_inference,
the same code that produced the T15 GPU numbers) so CPU numbers are apples-to-apples:
same batch_size=16, same max_length=1000, same forward. On CPU we force fp32 (no autocast;
_fp16_autocast_ctx is a no-op off CUDA) and pin torch threads to the box/node core count.

Per-teacher we time `--num-batches` batches after `--warmup-batches` warmup, and report
mean/median ms/batch + peak RSS. Whole-task (N=3000) and all-18-task totals are computed
by the caller from mean_ms (ceil(3000/16)=188 and ceil(38822/16)=2427 batches).

Usage (single teacher, few batches, on the CPU node/login shell):
    python scripts/cpu_teacher_latency.py --teacher enformer --num-batches 5 --warmup 2
"""
import argparse
import gc
import json
import math
import os
import time

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("WANDB_MODE", "offline")
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch
from dataclasses import dataclass, replace


def _cpu_name():
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return "unknown-cpu"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", required=True,
                    choices=["enformer", "dnabert2", "caduceus", "carbon3b", "nt", "bpnet"],
                    help="teacher model, or 'bpnet' for the BPNet student (re-measured in the same job)")
    ap.add_argument("--task", default="splice_sites_all")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-length", type=int, default=1000)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--num-batches", type=int, default=5)
    ap.add_argument("--threads", type=int, default=None,
                    help="torch.set_num_threads (default: os.cpu_count())")
    ap.add_argument("--out-jsonl", default=None)
    args = ap.parse_args()

    threads = args.threads or os.cpu_count()
    # Pin BOTH torch and OpenMP thread pools to the SAME N (no oversubscription; N must be <=
    # physical cores). OMP_NUM_THREADS must be set before torch spins up its OpenMP pool, but
    # torch.set_num_threads is authoritative for the intra-op pool regardless; we set both so the
    # recorded thread count is unambiguous and every model shares identical parallelism.
    os.environ["OMP_NUM_THREADS"] = str(threads)
    os.environ.setdefault("MKL_NUM_THREADS", str(threads))
    torch.set_num_threads(threads)

    from config.distillation.data import nucletide_transformer_revised_benchmark
    from src.data.dataset import (
        get_num_labels, build_data_splits_from_huggingface, SeqDataset,
    )
    from src.model.glm import find_teacher_checkpoint, get_teacher_model
    from src.trainer.benchmark_utils import (
        benchmark_teacher_inference, benchmark_model_inference,
    )
    from src.train.benchmark_inference import (
        get_teacher_config_and_path, _disable_dnabert2_flash_attn,
    )

    device = "cpu"
    task_name = args.task

    print(f"[cfg] model={args.teacher} task={task_name} bs={args.batch_size} "
          f"maxlen={args.max_length} warmup={args.warmup} num_batches={args.num_batches} "
          f"threads={threads} omp={os.environ['OMP_NUM_THREADS']} cpu={_cpu_name()}", flush=True)

    # --- data ---
    dataset_config = replace(nucletide_transformer_revised_benchmark, task_name=task_name)
    _, _, _, _, X_test, y_test = build_data_splits_from_huggingface(dataset_config)
    print(f"[data] test samples={len(X_test)}", flush=True)

    n_task = 3000
    n_all = 38822
    bpt_task = math.ceil(n_task / args.batch_size)   # 188
    bpt_all = math.ceil(n_all / args.batch_size)     # 2427

    # --- BPNet student: re-measured in the SAME job (SeqDataset + DataLoader + benchmark_model_inference,
    # the exact student path from benchmark_inference.py) so its row shares hardware/threads with the teachers. ---
    if args.teacher == "bpnet":
        from torch.utils.data import DataLoader
        from src.model.bpnet_classifier import BPNetClassifier, BPNetClassifierConfig
        num_labels = get_num_labels(task_name)
        test_ds = SeqDataset(X_test, y_test, args.max_length)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
        model = BPNetClassifier(BPNetClassifierConfig(
            num_labels=num_labels, model_type="bpnet", model_size="original"))
        result = benchmark_model_inference(
            model=model, dataloader=test_loader, device=device,
            model_name="bpnet_original", model_type="student", task_name=task_name,
            warmup_batches=args.warmup, num_batches=args.num_batches, dtype=torch.float32,
        )
        whole_task_s = result.mean_latency_ms / 1000.0 * bpt_task
        all18_s = result.mean_latency_ms / 1000.0 * bpt_all
        rec = {
            "teacher": "bpnet", "task": task_name, "device": "cpu", "precision": "fp32",
            "threads": threads, "omp_threads": int(os.environ["OMP_NUM_THREADS"]), "cpu": _cpu_name(),
            "batch_size": args.batch_size, "max_length": args.max_length,
            "n_batches_timed": result.num_samples // args.batch_size, "warmup": args.warmup,
            "mean_ms_per_batch": result.mean_latency_ms, "median_ms_per_batch": result.median_latency_ms,
            "std_ms": result.std_latency_ms, "min_ms": result.min_latency_ms, "max_ms": result.max_latency_ms,
            "peak_rss_mb": result.peak_memory_mb, "num_parameters": result.num_parameters,
            "whole_task_s_N3000": whole_task_s, "batches_per_task": bpt_task,
            "all18task_s": all18_s, "batches_all18": bpt_all,
        }
        print("[RESULT] " + json.dumps(rec), flush=True)
        if args.out_jsonl:
            os.makedirs(os.path.dirname(args.out_jsonl), exist_ok=True)
            with open(args.out_jsonl, "a") as f:
                f.write(json.dumps(rec) + "\n")
        return

    # --- teacher config + checkpoint (same as GPU path) ---
    teacher_config, teacher_parent_dir, model_type = get_teacher_config_and_path(args.teacher)

    @dataclass
    class MinimalConfig:
        teacher_config: any
        teacher_parent_dir: str
        model_type: str

    mini = MinimalConfig(teacher_config, teacher_parent_dir, model_type)
    teacher_ckpt, score = find_teacher_checkpoint(mini, task_name)
    print(f"[ckpt] {teacher_ckpt} (score={score})", flush=True)
    assert teacher_ckpt is not None, "no checkpoint found"

    @dataclass
    class TrainerConfig:
        device: str

    @dataclass
    class FullConfig:
        teacher_config: any
        teacher_parent_dir: str
        model_type: str
        trainer_config: TrainerConfig

    num_labels = get_num_labels(task_name)
    full = FullConfig(
        replace(teacher_config, num_labels=num_labels),
        teacher_parent_dir, model_type, TrainerConfig(device=device),
    )

    if args.teacher in ("dnabert2", "caduceus"):
        # Both models pull in modules with a MODULE-LEVEL `@triton.autotune(...)` decorator that
        # eagerly calls driver.active (a GPU driver) at import and raises "0 active drivers" on a
        # CPU-only box -- before any model code runs. (DNABERT-2: flash_attn_triton.py; Caduceus:
        # mamba_ssm/ops/triton/layer_norm.py.) These Triton kernels are never used on CPU (DNABERT-2
        # takes the pure-PyTorch attention fallback; mamba_ssm falls back to its reference
        # selective_scan). Neutralise the decorator to a pass-through so the import succeeds
        # without touching a GPU; the forward runs on the CPU reference path.
        import triton
        triton.autotune = lambda *a, **k: (lambda fn: fn)  # no-op decorator (kernel unused on CPU)

    if args.teacher == "caduceus":
        # Even after the import is unblocked, Caduceus's RMSNorm routes through mamba_ssm's
        # FUSED Triton layer_norm (_layer_norm_fwd -> torch.cuda.device(...)), which errors on
        # CPU ("invalid argument to exchangeDevice"). Replace mamba_ssm's rms_norm_fn / layer_norm_fn
        # with a pure-PyTorch RMSNorm/LayerNorm that is numerically equivalent, so the SAME forward
        # graph (the Mamba matmuls/convs that dominate latency) runs on CPU. This mirrors the fused
        # kernel's math (optional residual add, then RMS/LayerNorm) -- only the norm kernel differs.
        import torch.nn.functional as F
        from mamba_ssm.ops.triton import layer_norm as _mln

        def _torch_norm(x, weight, bias, residual=None, eps=1e-6,
                        prenorm=False, residual_in_fp32=False, is_rms_norm=False,
                        **kwargs):
            in_dtype = x.dtype
            x = x.float()
            if residual is not None:
                x = x + (residual.float() if residual_in_fp32 else residual.to(x.dtype).float())
            residual_out = x
            if is_rms_norm:
                var = x.pow(2).mean(-1, keepdim=True)
                out = x * torch.rsqrt(var + eps)
            else:
                out = F.layer_norm(x, (x.shape[-1],), eps=eps)
            out = out * weight.float()
            if bias is not None:
                out = out + bias.float()
            out = out.to(in_dtype)
            if prenorm:
                return out, residual_out.to(in_dtype)
            return out

        def _rms_norm_fn(x, weight, bias, residual=None, prenorm=False,
                         residual_in_fp32=False, eps=1e-6, **kwargs):
            return _torch_norm(x, weight, bias, residual=residual, eps=eps,
                               prenorm=prenorm, residual_in_fp32=residual_in_fp32,
                               is_rms_norm=True)

        def _layer_norm_fn(x, weight, bias, residual=None, prenorm=False,
                           residual_in_fp32=False, eps=1e-6, is_rms_norm=False, **kwargs):
            return _torch_norm(x, weight, bias, residual=residual, eps=eps,
                               prenorm=prenorm, residual_in_fp32=residual_in_fp32,
                               is_rms_norm=is_rms_norm)

        _mln.rms_norm_fn = _rms_norm_fn
        _mln.layer_norm_fn = _layer_norm_fn

        def _rmsnorm_forward(self, x, residual=None, prenorm=False, residual_in_fp32=False):
            return _rms_norm_fn(x, self.weight, getattr(self, "bias", None),
                                residual=residual, prenorm=prenorm,
                                residual_in_fp32=residual_in_fp32, eps=self.eps)
        _mln.RMSNorm.forward = _rmsnorm_forward

        # Force mamba_simple onto its CPU reference path: the fused CUDA kernels
        # (mamba_inner_fn / causal_conv1d_cuda / selective_scan_cuda) hard-require x.is_cuda.
        # Null causal_conv1d_fn so the mixer skips the fully-fused fast path AND uses the plain
        # nn.Conv1d branch; swap selective_scan_fn -> selective_scan_ref (pure-PyTorch SSM scan).
        # Same math, CPU-runnable; the SSM scan/convs still dominate the timed latency.
        from mamba_ssm.modules import mamba_simple as _ms
        from mamba_ssm.ops.selective_scan_interface import selective_scan_ref as _ssr
        _ms.causal_conv1d_fn = None
        _ms.causal_conv1d_update = None
        _ms.selective_scan_fn = _ssr
        _ms.mamba_inner_fn = None

    t_load = time.perf_counter()
    tokenizer, teacher_model, _ = get_teacher_model(full, task_name, teacher_ckpt)
    print(f"[load] model loaded in {time.perf_counter()-t_load:.1f}s", flush=True)

    if args.teacher == "dnabert2":
        n = _disable_dnabert2_flash_attn()
        print(f"[dnabert2] disabled Triton flash-attn on {n} module(s)", flush=True)

    result = benchmark_teacher_inference(
        model=teacher_model,
        tokenizer=tokenizer,
        sequences=X_test,
        device=device,
        model_name=args.teacher,
        task_name=task_name,
        batch_size=args.batch_size,
        max_length=args.max_length,
        warmup_batches=args.warmup,
        num_batches=args.num_batches,
        dtype=torch.float32,
        input_prefix=getattr(teacher_config, "input_prefix", ""),
        add_special_tokens=getattr(teacher_config, "add_special_tokens", True),
    )

    whole_task_s = result.mean_latency_ms / 1000.0 * bpt_task
    all18_s = result.mean_latency_ms / 1000.0 * bpt_all

    rec = {
        "teacher": args.teacher, "task": task_name, "device": "cpu", "precision": "fp32",
        "threads": threads, "omp_threads": int(os.environ["OMP_NUM_THREADS"]), "cpu": _cpu_name(),
        "batch_size": args.batch_size, "max_length": args.max_length,
        "n_batches_timed": args.num_batches, "warmup": args.warmup,
        "mean_ms_per_batch": result.mean_latency_ms,
        "median_ms_per_batch": result.median_latency_ms,
        "std_ms": result.std_latency_ms,
        "min_ms": result.min_latency_ms, "max_ms": result.max_latency_ms,
        "peak_rss_mb": result.peak_memory_mb,
        "num_parameters": result.num_parameters,
        "whole_task_s_N3000": whole_task_s, "batches_per_task": bpt_task,
        "all18task_s": all18_s, "batches_all18": bpt_all,
    }
    print("[RESULT] " + json.dumps(rec), flush=True)
    if args.out_jsonl:
        os.makedirs(os.path.dirname(args.out_jsonl), exist_ok=True)
        with open(args.out_jsonl, "a") as f:
            f.write(json.dumps(rec) + "\n")

    del teacher_model
    gc.collect()


if __name__ == "__main__":
    main()
