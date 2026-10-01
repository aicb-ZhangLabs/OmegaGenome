#!/usr/bin/env python3
"""Faithfully resume-extend (or clone) an NTv3 single-track finetune run to more steps.

Reconstructs the EXACT `src.train.finetune_ntv3` invocation from a finished run's saved
`config` (in its `ntv3_finetune_result.json`), changing ONLY `--num_steps_training` (and the
output RUNTAG). Because every flag — track_subset, cached_teacher_logits, model, kd weights —
is read back from the run's own config, there is no manual flag-mapping and the track→cache
mapping cannot be mis-set. The job auto-resumes from `latest_state.pth` in the output dir.

Usage (dry-run prints the sbatch command; --submit actually submits):
    python slurm/resume_extend.py --run-dir <SSD>/ntv3_targets/ntv3_bpnet_t30_cachedkd \
        --new-steps 240000 --node galaxy
    python slurm/resume_extend.py --run-dir ... --new-steps 40000 --node voyager --submit

--runtag lets you write to a NEW dir (clone, e.g. a fresh 100M run seeded from an 8M config);
default reuses the source dir name so it resumes in place.
"""

import os
import argparse
import json
import subprocess
import sys

# Flags the sbatch wrapper (slurm/ntv3_finetune.sbatch) already supplies itself; never re-emit.
WRAPPER_HANDLED = {
    "data_dir",
    "model",
    "out",
    "stage_dir",
    "wandb",
    "compile",
    "dry_run",
    "smoke",
    "amp",
}
# store_true config fields -> emit the bare flag only when True.
BOOL_FLAGS = {"use_lora", "cache_logits", "teacher_specialist"}


def build_passthrough(cfg, new_steps):
    """Return the list of `--flag value` pass-through args reconstructed from a saved config."""
    args = []
    for k, v in sorted(cfg.items()):
        if k in WRAPPER_HANDLED or v is None:
            continue
        if k == "num_steps_training":
            v = new_steps  # the one override
        if k in BOOL_FLAGS:
            if v:
                args.append(f"--{k}")
            continue
        args += [f"--{k}", str(v)]
    return args


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--run-dir", required=True, help="source run dir with ntv3_finetune_result.json"
    )
    ap.add_argument("--new-steps", type=int, required=True)
    # Single node only: the sbatch script is --nodes=1, so a comma nodelist makes SLURM compute
    # -N 2-1 and reject the job. Pick one node (cap-aware: voyager<=3, laniakea<=6).
    ap.add_argument("--node", required=True, choices=["galaxy", "voyager", "laniakea"])
    ap.add_argument(
        "--runtag",
        default=None,
        help="output dir name; default = source dir basename (resume in place)",
    )
    ap.add_argument(
        "--model",
        default=None,
        help="override MODEL env (e.g. a 100m_pre dir for a capacity clone)",
    )
    ap.add_argument(
        "--track",
        default=None,
        help="clone the source config onto a NEW track index: overrides "
        "track_subset and (if the source used a cached teacher) swaps cached_teacher_logits to "
        "the same dir's t<track>.pt. Requires --runtag (must not resume into the source dir).",
    )
    ap.add_argument(
        "--cpus",
        type=int,
        default=None,
        help="override --cpus-per-task (low value fits a CPU-saturated node's idle cores)",
    )
    ap.add_argument(
        "--submit", action="store_true", help="actually sbatch (default: dry-run print)"
    )
    a = ap.parse_args()

    cfg = json.load(open(os.path.join(a.run_dir, "ntv3_finetune_result.json")))["config"]
    if a.track is not None:
        assert a.runtag, (
            "--track requires --runtag (a NEW output dir; never clone a track into the source dir)"
        )
        cfg["track_subset"] = a.track
        # swap the cached teacher logits to the matching track in the same cache dir (KD runs only)
        if cfg.get("cached_teacher_logits"):
            d = os.path.dirname(cfg["cached_teacher_logits"])
            cfg["cached_teacher_logits"] = os.path.join(d, f"t{a.track}.pt")
    runtag = a.runtag or os.path.basename(a.run_dir.rstrip("/"))
    model = a.model or cfg["model"]
    passthrough = build_passthrough(cfg, a.new_steps)

    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sbatch = os.path.join(repo, "slurm", "ntv3_finetune.sbatch")
    env = {"RUNTAG": runtag, "MODEL": model}
    if a.node == "galaxy":
        env["NO_COMPILE"] = "1"  # galaxy GLIBC lacks triton's libstdc++ -> eager
    sbatch_flags = ["sbatch", f"--nodelist={a.node}", f"--job-name=ntv3-{runtag}"]
    if a.cpus:
        sbatch_flags += [f"--cpus-per-task={a.cpus}"]
    cmd = sbatch_flags + [sbatch] + passthrough

    env_str = " ".join(f"{k}={v}" for k, v in env.items())
    print(
        f"# track_subset={cfg.get('track_subset')!r}  cached_logits={cfg.get('cached_teacher_logits')!r}"
    )
    print(
        f"# {cfg.get('num_steps_training')} -> {a.new_steps} steps   model={model}   out=ntv3_targets/{runtag}"
    )
    print(env_str + " " + " ".join(cmd))
    if a.submit:
        r = subprocess.run(cmd, env={**os.environ, **env}, capture_output=True, text=True)
        print(r.stdout.strip(), r.stderr.strip(), file=sys.stderr)
        sys.exit(r.returncode)


if __name__ == "__main__":
    main()
