"""Aggregate NTv3 latency benchmark JSONL records into a clean per-(model,task) markdown table
plus teacher-vs-student fold-speedups, for the paper's latency section.

Usage: python scripts/aggregate_latency.py <results_gpu.jsonl> [<results_cpu.jsonl> ...]
Prints markdown to stdout.
"""

import json
import sys


def load(paths):
    recs = []
    for p in paths:
        try:
            for line in open(p):
                line = line.strip()
                if line:
                    recs.append(json.loads(line))
        except FileNotFoundError:
            pass
    return recs


def main():
    recs = load(sys.argv[1:])
    if not recs:
        print("no records")
        return
    # de-dup by (model,task,device) keeping the last
    seen = {}
    for r in recs:
        seen[(r["model"], r["task"], r["device"])] = r
    recs = list(seen.values())

    # device meta
    devs = {}
    for r in recs:
        devs[r["device"]] = (
            r.get("gpu"),
            r.get("cpu"),
            r.get("batch_size"),
            r.get("seqlen"),
            r.get("amp"),
            r.get("n_test_windows"),
        )
    print("## Hardware / settings")
    for d, (gpu, cpu, bs, sl, amp, nt) in devs.items():
        hw = gpu if d.startswith("cuda") else cpu
        print(f"- **{d}**: {hw} | batch={bs} | seqlen={sl} | amp={amp} | n_test_windows={nt}")
    print()

    order = ["t0", "t3", "t11", "t12", "t14", "t18", "t20", "t30", "t33", "joint34"]
    models = ["NTv3-650M-teacher", "NTv3-100M", "NTv3-8M", "BPNet-0.12M"]

    for dev in sorted(devs):
        print(f"## {dev}")
        print(
            "| model | task | params(M) | n_tracks | per-batch (ms) | whole-test (s) | "
            "throughput (win/s) | peak mem (MB) | extrapolated |"
        )
        print("|---|---|--:|--:|--:|--:|--:|--:|:--:|")
        dr = [r for r in recs if r["device"] == dev]
        dr.sort(
            key=lambda r: (
                models.index(r["model"]) if r["model"] in models else 9,
                order.index(r["task"]) if r["task"] in order else 99,
            )
        )
        for r in dr:
            print(
                f"| {r['model']} | {r['task']} | {r['params_M']:.3f} | {r['n_tracks']} | "
                f"{r['per_batch_ms_mean']:.1f} | {r['whole_test_total_s']:.1f} | "
                f"{r['throughput_win_per_s']:.1f} | {r['peak_mem_MB']:.0f} | "
                f"{'yes' if r['extrapolated'] else 'no'} |"
            )
        print()

        # fold speedup vs teacher (joint-34, same device) using per-batch latency
        teach = next(
            (r for r in dr if r["model"] == "NTv3-650M-teacher" and r["task"] == "joint34"), None
        )
        if teach:
            tb = teach["per_batch_ms_mean"]
            print(
                f"### Teacher-vs-student fold-speedup ({dev}, per-batch latency; teacher 650M joint-34 = {tb:.1f} ms)"
            )
            print("| student | task | per-batch (ms) | speedup vs teacher |")
            print("|---|---|--:|--:|")
            for r in dr:
                if r["model"] == "NTv3-650M-teacher":
                    continue
                print(
                    f"| {r['model']} | {r['task']} | {r['per_batch_ms_mean']:.1f} | "
                    f"{tb / r['per_batch_ms_mean']:.1f}x |"
                )
            print()


if __name__ == "__main__":
    main()
