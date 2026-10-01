"""Generate the two HP-search stage tables from results/carbon_grid_results.csv:
  (1) carbon_hp_stage1_3.csv   — base MSE config = stages 1+3 (weight_mse in {0,1,2,5})
  (2) carbon_hp_stage1_2_3.csv — full = stages 1+2+3 (adds stage-2 weight_mse=0.2)
plus a best-on-VAL-per-task summary for each. HP-search grid = the single-seed sweep
(random_state not in {0,1,2}; the {0,1,2} rows are the separate 3-seed reruns, excluded here).

Stage map (by weight_mse): stage1={0,1}, stage2={0.2}, stage3={2,5}. Read-only over the grid CSV;
emits new files (does not touch existing outputs). Run:
    .venv_carbon_portable/bin/python slurm/gen_stage_csvs.py
"""

import argparse
import csv
import os

SEED_3 = {"0", "1", "2"}  # the 3-seed reruns (excluded from the HP-search tables)
STAGE = {
    "0.0": 1,
    "0": 1,
    "1.0": 1,
    "1": 1,  # weight_mse -> stage
    "0.2": 2,
    "0.25": 2,
    "2.0": 3,
    "2": 3,
    "5.0": 3,
    "5": 3,
}
COLS = [
    "task",
    "variant",
    "stage",
    "weight_ce",
    "weight_kl",
    "weight_mse",
    "temperature",
    "lr",
    "batch_size",
    "random_state",
    "best_val_mcc",
    "best_test_mcc",
    "final_test_mcc",
    "best_epoch",
]


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--grid", default="results/carbon_grid_results.csv", help="input grid CSV (collate_runs)"
    )
    ap.add_argument(
        "--tag", default="", help="output filename tag, e.g. 'original' or 'deploy120k'"
    )
    args = ap.parse_args()
    tag = f"_{args.tag}" if args.tag else ""
    rows = list(csv.DictReader(open(args.grid)))
    grid = [r for r in rows if r["random_state"] not in SEED_3]  # single-seed HP-search sweep only
    print(
        f"MODEL TAG: {args.tag or '(default/deploy_120k)'} | grid rows {len(rows)} -> sweep {len(grid)}"
    )
    for r in grid:
        r["stage"] = STAGE.get(r["weight_mse"], "?")

    def emit(name, mse_keep):
        sel = [r for r in grid if r["weight_mse"] in mse_keep]
        # sort by task, then best_val_mcc desc (so the winner per task is the first row of its block)
        sel.sort(key=lambda r: (r["task"], -(_f(r["best_val_mcc"]) or -1)))
        path = os.path.join("results", name)
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=COLS, extrasaction="ignore")
            w.writeheader()
            w.writerows(sel)
        # best-on-VAL per task (never pick on test)
        best = {}
        for r in sel:
            v = _f(r["best_val_mcc"])
            if v is not None and (r["task"] not in best or v > _f(best[r["task"]]["best_val_mcc"])):
                best[r["task"]] = r
        bpath = path.replace(".csv", "_best.csv")
        with open(bpath, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=COLS, extrasaction="ignore")
            w.writeheader()
            w.writerows([best[t] for t in sorted(best)])
        print(f"  {name}: {len(sel)} rows, {len(best)} tasks -> {path}  (best-per-task -> {bpath})")
        return best

    print("=== stage 1+3 (base MSE config: weight_mse in {0,1,2,5}) ===")
    base = emit(f"carbon_hp_stage1_3{tag}.csv", {"0.0", "1.0", "2.0", "5.0"})
    print("=== stage 1+2+3 (full: adds weight_mse=0.2) ===")
    full = emit(f"carbon_hp_stage1_2_3{tag}.csv", {"0.0", "1.0", "2.0", "5.0", "0.2"})

    # Where did stage 2 (0.2) CHANGE the best HP for a task? -> these are the stage-2-contributed winners
    # that also deserve a 3-seed run.
    print("=== tasks where stage 2 (0.2) gives the full-grid best-on-val (vs base config) ===")
    changed = []
    for t in sorted(full):
        if t in base and full[t]["weight_mse"] == "0.2":
            changed.append(t)
            print(
                f"  {t}: full-best weight_mse=0.2 val={full[t]['best_val_mcc']} "
                f"(base-config best val={base[t]['best_val_mcc']}, mse={base[t]['weight_mse']})"
            )
    if not changed:
        print("  (none — stage 2 never produced the grid's best-on-val for any task)")
    print(f"\nstage-2-contributed tasks needing 3-seed: {changed or 'none'}")


if __name__ == "__main__":
    main()
