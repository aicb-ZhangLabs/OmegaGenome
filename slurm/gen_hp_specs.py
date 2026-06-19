#!/usr/bin/env python3
"""Generate carbon-raw HP-search job specs (one carbon_distill.sbatch arg-line per grid combo x task),
for slurm/auto_submit_specs.sh. The grid is read from carbon_hp_raw_focused_config (single source of
truth). RESUME-AWARE: skips any (task, weight_ce, weight_kl, weight_mse, temperature) combo that already
has a matching raw final_summary.json. Usage: python slurm/gen_hp_specs.py [--out hp_specs.txt]
"""
import argparse, glob, json, os, sys
sys.path.insert(0, ".")
from itertools import product
# Use the BASE config's full grid for the HP search (kl[0,.25,.5,1] x mse[0,1,2,5] x T[.5,1,1.5,2,4]
# = 80 combos x 18 tasks = 1440). carbon_hp_raw_focused_config remains available as the focused option.
from config.distillation.experiments.carbon import carbon_base_hyperparam_raw_config as C

BASE = "/tmp/galaxy_srv_disk00/pengchx3/carbon_distillation"


def done_combos():
    """Set of (task, ce, kl, mse, temp) already completed (raw variant) from final_summary.json."""
    done = set()
    for f in glob.glob(os.path.join(BASE, "**", "final_summary.json"), recursive=True):
        if "mse_normalizeTrue" in f:  # l2norm runs don't count for the raw search
            continue
        try:
            s = json.load(open(f)); hp = s["hyperparameters"]
            done.add((s["task"], float(hp["weight_ce"]), float(hp["weight_kl"]),
                      float(hp["weight_mse"]), float(hp["temperature"])))
        except Exception:
            continue
    return done


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="hp_specs.txt")
    args = ap.parse_args()
    done = done_combos()
    lines, skipped = [], 0
    for task, ce, kl, mse, temp in product(C.task_names, C.weight_ces, C.weight_kls, C.weight_mses, C.temperatures):
        if (task, float(ce), float(kl), float(mse), float(temp)) in done:
            skipped += 1
            continue
        lines.append(
            f"carbon-raw --task-names {task} "
            f"--distillation-config.weight-ce {ce} --distillation-config.weight-kl {kl} "
            f"--distillation-config.weight-mse {mse} --distillation-config.temperature {temp} "
            f"--slurm-config.mode run"
        )
    with open(args.out, "w") as f:
        f.write("\n".join(lines) + ("\n" if lines else ""))
    total = len(C.task_names) * len(C.weight_ces) * len(C.weight_kls) * len(C.weight_mses) * len(C.temperatures)
    print(f"grid: {len(C.weight_kls)}kl x {len(C.weight_mses)}mse x {len(C.temperatures)}T x "
          f"{len(C.task_names)}tasks = {total} combos; {skipped} already done; wrote {len(lines)} specs -> {args.out}")


if __name__ == "__main__":
    main()
