#!/usr/bin/env python3
"""Generate 3-seed job specs for the BEST hyperparameters per task (from best_hyperparams.json), for
slurm/auto_submit_specs.sh. One line per (task, seed) with the task's best-on-val raw hyperparameters.
Usage: python slurm/gen_3seed_best_specs.py [--best best_hyperparams.json] [--seeds 0 1 2] [--out best_3seed_specs.txt]
"""

import argparse
import json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--best", default="best_hyperparams.json")
    ap.add_argument("--variant", default="raw", choices=["raw", "l2norm"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--out", default="best_3seed_specs.txt")
    args = ap.parse_args()

    best = json.load(open(args.best)).get(args.variant, {})
    cfg = "carbon-raw" if args.variant == "raw" else "carbon-l2norm"
    lines = []
    for task in sorted(best):
        hp = best[task]["hyperparameters"]
        for s in args.seeds:
            lines.append(
                f"{cfg} --task-names {task} "
                f"--distillation-config.weight-ce {hp['weight_ce']} --distillation-config.weight-kl {hp['weight_kl']} "
                f"--distillation-config.weight-mse {hp['weight_mse']} --distillation-config.temperature {hp['temperature']} "
                f"--trainer-config.early-stop-patience 100 --random-state {s} --slurm-config.mode run"
            )
    with open(args.out, "w") as f:
        f.write("\n".join(lines) + ("\n" if lines else ""))
    print(
        f"{len(best)} tasks x {len(args.seeds)} seeds = {len(lines)} specs ({args.variant}) -> {args.out}"
    )


if __name__ == "__main__":
    main()
