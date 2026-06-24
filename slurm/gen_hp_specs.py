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


def done_combos(scope_leaf=""):
    """Set of (task, ce, kl, mse, temp) already completed (raw variant) from final_summary.json.

    ``scope_leaf`` restricts the glob to ``BASE/<scope_leaf>/**`` — REQUIRED when re-searching with a
    different student (e.g. ``carbon-raw-original`` writes under ``BASE/original``): final_summary.json
    does NOT record model_size, so an unscoped glob would mix the new grid with the old deploy_120k
    results and wrongly skip everything. Empty scope (default) globs all of BASE (legacy behavior).
    """
    root = os.path.join(BASE, scope_leaf) if scope_leaf else BASE
    done = set()
    for f in glob.glob(os.path.join(root, "**", "final_summary.json"), recursive=True):
        if "mse_normalizeTrue" in f:  # l2norm runs don't count for the raw search
            continue
        try:
            s = json.load(open(f)); hp = s["hyperparameters"]
            done.add((s["task"], float(hp["weight_ce"]), float(hp["weight_kl"]),
                      float(hp["weight_mse"]), float(hp["temperature"])))
        except Exception:
            continue
    return done


def _temps_for_kl(kl, temperatures):
    """Temperatures to sweep for a given kl weight. Temperature ONLY enters the loss via the KL term,
    and ``kl_term`` returns 0 *before* any softmax/T when ``weight_kl <= 0`` (src/model/distillation.py).
    So for kl==0 every temperature gives the identical loss — collapse to a single canonical T (the
    first) to avoid running ~N_T redundant copies. For kl>0, sweep all temperatures."""
    return [temperatures[0]] if float(kl) == 0.0 else list(temperatures)


def grid_combos(C):
    """Canonical HP-grid combos (task, ce, kl, mse, temp), with the kl=0 × T redundancy removed.
    To stage the search by mse weight, edit the config's ``weight_mses`` and regenerate (the grid is
    read straight from the config — no override flag)."""
    for task, ce, kl, mse in product(C.task_names, C.weight_ces, C.weight_kls, C.weight_mses):
        for temp in _temps_for_kl(kl, C.temperatures):
            yield (task, ce, kl, mse, temp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="hp_specs.txt")
    # --cfg picks the run config emitted per line; its scope leaf (the output_dir's last path component)
    # is auto-derived so resume can't mix a different-student re-search (e.g. carbon-raw-original, which
    # writes under .../original) with the old deploy_120k grid. Stage the mse search by editing the
    # config's weight_mses and rerunning (the 3 stages: [0,1] -> [0.25] -> [2,5]).
    ap.add_argument("--cfg", default="carbon-raw",
                    help="Experiment config name emitted per line (carbon-raw | carbon-raw-original).")
    args = ap.parse_args()
    from config.distillation.experiments.carbon import experiment_configs
    scope_leaf = os.path.basename(experiment_configs[args.cfg][1].trainer_config.output_dir)
    done = done_combos(scope_leaf)
    lines, skipped, total = [], 0, 0
    for task, ce, kl, mse, temp in grid_combos(C):
        total += 1
        if (task, float(ce), float(kl), float(mse), float(temp)) in done:
            skipped += 1
            continue
        lines.append(
            f"{args.cfg} --task-names {task} "
            f"--distillation-config.weight-ce {ce} --distillation-config.weight-kl {kl} "
            f"--distillation-config.weight-mse {mse} --distillation-config.temperature {temp} "
            f"--trainer-config.early-stop-patience 100 "  # search: best-val ckpt -> same MCC, ~2x faster
            f"--slurm-config.mode run"
        )
    with open(args.out, "w") as f:
        f.write("\n".join(lines) + ("\n" if lines else ""))
    full = len(C.task_names) * len(C.weight_ces) * len(C.weight_kls) * len(C.weight_mses) * len(C.temperatures)
    print(f"cfg={args.cfg} scope={scope_leaf} mse={C.weight_mses}\n"
          f"grid: {len(C.weight_kls)}kl x {len(C.weight_mses)}mse x {len(C.temperatures)}T x "
          f"{len(C.task_names)}tasks = {full} naive; {total} canonical (kl=0 collapses T, saves {full-total}); "
          f"{skipped} already done; wrote {len(lines)} specs -> {args.out}")


if __name__ == "__main__":
    main()
