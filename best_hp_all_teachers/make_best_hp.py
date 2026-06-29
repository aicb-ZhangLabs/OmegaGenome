#!/usr/bin/env python3
"""Extract the best distillation hyperparameters per NT-task from a teacher's HP-search CSV.

Mirrors the carbon-3B ``deploy_120k`` best-HP organization
(see code_carbon/slurm/extract_best_hyperparams.py + best_hyperparams.json):
for every task we pick the run with the highest **validation** MCC (``summary.best_val_mcc``),
never test, so the selected hyperparameters do not leak the test set; we then record the
test MCC *of that same best-val row* (``summary.best_test/mcc``).

Input is a flat wandb-export CSV (one row per run). Teachers differ in column naming:
  * nt / caduceus / enformer:  task in ``config.dataset_config.task_name``;
        HP knobs in ``config.distillation_config.*``; lr/bs/epochs in ``config.trainer_config.*``.
  * dnabert2:  task NOT stored in a column (parse from ``name``); HP knobs in flat
        ``config.weight_ce|weight_kl|weight_mse|temperature``; lr/bs/epochs in ``config.lr|batch_size|epochs``.
A per-teacher COLUMN_MAP handles these differences; the core selection logic is shared.

Outputs (per teacher) under ``--out-dir``:
  * ``best_hp_<teacher>.yaml``  -- one block per task (nested {task: {hyperparameters, best_val_mcc, ...}})
  * ``best_hp_<teacher>.csv``   -- tidy 18-row table
  * ``best_hp_<teacher>.md``    -- markdown rendering of the same table

Usage:
  python make_best_hp.py --csv <path> --teacher <nt|dnabert2|caduceus|enformer> --out-dir <dir>
"""
import argparse
import csv
import os

# Canonical NT-classification task list (18). Ordered specific-prefix-first so that
# name-prefix matching disambiguates e.g. enhancers_types before enhancers.
KNOWN_TASKS = [
    "H2AFZ", "H3K27ac", "H3K27me3", "H3K36me3", "H3K4me1", "H3K4me2", "H3K4me3",
    "H3K9ac", "H3K9me3", "H4K20me1",
    "enhancers_types", "enhancers",
    "promoter_no_tata", "promoter_tata", "promoter_all",
    "splice_sites_acceptors", "splice_sites_donors", "splice_sites_all",
]
_TASKS_BY_LEN = sorted(KNOWN_TASKS, key=len, reverse=True)  # longest prefix first

# Per-teacher column maps. None for "task_col" means parse the task from the run ``name``.
# Each *_col tuple is tried in order; the first column present & non-empty in the row wins.
COLUMN_MAP = {
    "nt": {
        "task_col": "config.dataset_config.task_name",
        "weight_ce": ("config.distillation_config.weight_ce",),
        "weight_kl": ("config.distillation_config.weight_kl",),
        "weight_mse": ("config.distillation_config.weight_mse",),
        "temperature": ("config.distillation_config.temperature",),
        "distill_method": ("config.distillation_config.distill_method",),
        "kl_method": ("config.distillation_config.kl_method",),
        "dkd_alpha": ("config.distillation_config.dkd_alpha",),
        "dkd_beta": ("config.distillation_config.dkd_beta",),
        "lr": ("config.trainer_config.lr",),
        "batch_size": ("config.trainer_config.batch_size",),
        "epochs": ("config.trainer_config.epochs",),
        "seed": ("config.random_state", "config.dataset_config.random_state"),
        "model_size": ("config.student_config.model_size",),
        "hidden_dim": ("config.student_config.hidden_dim",),
    },
    "caduceus": "LIKE_NT",
    "enformer": "LIKE_NT",
    "dnabert2": {
        "task_col": None,  # parse from name
        "weight_ce": ("config.weight_ce", "config.distillation_config.weight_ce"),
        "weight_kl": ("config.weight_kl", "config.distillation_config.weight_kl"),
        "weight_mse": ("config.weight_mse", "config.distillation_config.weight_mse"),
        "temperature": ("config.temperature", "config.distillation_config.temperature"),
        "distill_method": ("config.distillation_config.distill_method",),
        "kl_method": ("config.distillation_config.kl_method",),
        "dkd_alpha": ("config.distillation_config.dkd_alpha",),
        "dkd_beta": ("config.distillation_config.dkd_beta",),
        "lr": ("config.lr", "config.trainer_config.lr"),
        "batch_size": ("config.batch_size", "config.trainer_config.batch_size"),
        "epochs": ("config.epochs", "config.trainer_config.epochs"),
        "seed": ("config.random_state", "config.dataset_config.random_state"),
        "model_size": ("config.student_config.model_size",),
        "hidden_dim": ("config.student_config.hidden_dim",),
    },
}
# shared columns across all teachers
VAL_COL = "summary.best_val_mcc"
TEST_COL = "summary.best_test/mcc"
EPOCH_COL = "summary.epoch"

# Hyperparameter fields (in display order) emitted into the YAML "hyperparameters" block.
HP_FIELDS = ["weight_ce", "weight_kl", "weight_mse", "temperature",
             "distill_method", "kl_method", "dkd_alpha", "dkd_beta",
             "lr", "batch_size", "epochs"]
# Extra metadata fields emitted alongside hyperparameters.
META_FIELDS = ["seed", "model_size", "hidden_dim"]


def resolve_map(teacher):
    """Return the concrete COLUMN_MAP for a teacher, expanding the 'LIKE_NT' alias."""
    m = COLUMN_MAP[teacher]
    return COLUMN_MAP["nt"] if m == "LIKE_NT" else m


def task_from_name(name):
    """Infer the NT task from a run name by longest-known-prefix match; None if unknown."""
    for t in _TASKS_BY_LEN:
        if name.startswith(t):
            return t
    return None


def first_present(row, cols):
    """Return the first non-empty value among the candidate columns, else '' ."""
    for c in cols:
        v = row.get(c, "")
        if v is not None and str(v).strip() != "":
            return str(v).strip()
    return ""


def to_num(s):
    """Best-effort numeric coercion: int when whole, float otherwise, else the raw string."""
    if s == "":
        return None
    try:
        f = float(s)
        return int(f) if f.is_integer() else f
    except (ValueError, TypeError):
        return s


def get_task(row, cmap):
    """Resolve a row's task from its task column, falling back to name parsing."""
    tcol = cmap["task_col"]
    if tcol:
        v = row.get(tcol, "")
        if v and str(v).strip():
            return str(v).strip()
    return task_from_name(row.get("name", ""))


def select_best(csv_path, teacher):
    """Group CSV rows by task and keep the row with the highest validation MCC per task.

    Returns (best, counts): best maps task -> chosen row's extracted record dict;
    counts maps task -> number of candidate rows (with a parseable val-MCC) considered.
    Rows with an empty ``summary.best_val_mcc`` are skipped (crashed/failed runs).
    """
    cmap = resolve_map(teacher)
    best = {}
    counts = {}
    with open(csv_path, newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            v = row.get(VAL_COL, "")
            if v is None or str(v).strip() == "":
                continue
            try:
                val = float(v)
            except ValueError:
                continue
            task = get_task(row, cmap)
            if task is None:
                continue
            counts[task] = counts.get(task, 0) + 1
            if task not in best or val > best[task]["best_val_mcc"]:
                rec = {"task": task, "best_val_mcc": val}
                tm = row.get(TEST_COL, "")
                rec["best_test_mcc"] = float(tm) if tm and str(tm).strip() else None
                ep = row.get(EPOCH_COL, "")
                rec["best_epoch"] = to_num(str(ep).strip()) if ep else None
                for k in HP_FIELDS:
                    rec[k] = to_num(first_present(row, cmap[k]))
                for k in META_FIELDS:
                    rec[k] = to_num(first_present(row, cmap[k]))
                best[task] = rec
    return best, counts


def round_or_none(x, n=4):
    """Round a float to n places; pass through None/non-floats unchanged."""
    return round(x, n) if isinstance(x, float) else x


def write_yaml(best, counts, out_path):
    """Write a nested {task: {hyperparameters, metadata, best_val/test_mcc}} YAML, no PyYAML dep."""
    def fmt(v):
        if v is None:
            return "null"
        if isinstance(v, str):
            return v
        return repr(v)

    lines = []
    for task in sorted(best):
        r = best[task]
        lines.append(f"{task}:")
        lines.append("  hyperparameters:")
        for k in HP_FIELDS:
            lines.append(f"    {k}: {fmt(r[k])}")
        lines.append("  metadata:")
        for k in META_FIELDS:
            lines.append(f"    {k}: {fmt(r[k])}")
        lines.append(f"  best_val_mcc: {fmt(round_or_none(r['best_val_mcc']))}")
        lines.append(f"  best_test_mcc: {fmt(round_or_none(r['best_test_mcc']))}")
        lines.append(f"  best_epoch: {fmt(r['best_epoch'])}")
        lines.append(f"  n_candidates: {counts.get(task, 1)}")
        lines.append("")
    with open(out_path, "w") as f:
        f.write("\n".join(lines))


def write_table(best, counts, csv_path, md_path):
    """Write a tidy 18-row CSV plus a markdown rendering of the same per-task best-HP table."""
    cols = (["task"] + HP_FIELDS + META_FIELDS
            + ["best_val_mcc", "best_test_mcc", "best_epoch", "n_candidates"])
    rows = []
    for task in sorted(best):
        r = best[task]
        out = {"task": task}
        for k in HP_FIELDS + META_FIELDS:
            out[k] = r[k]
        out["best_val_mcc"] = round_or_none(r["best_val_mcc"])
        out["best_test_mcc"] = round_or_none(r["best_test_mcc"])
        out["best_epoch"] = r["best_epoch"]
        out["n_candidates"] = counts.get(task, 1)
        rows.append(out)
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    with open(md_path, "w") as f:
        f.write("| " + " | ".join(cols) + " |\n")
        f.write("|" + "|".join(["---"] * len(cols)) + "|\n")
        for out in rows:
            f.write("| " + " | ".join("" if out[c] is None else str(out[c]) for c in cols) + " |\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", required=True, help="path to teacher HP-search flat CSV")
    ap.add_argument("--teacher", required=True, choices=list(COLUMN_MAP.keys()))
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    best, counts = select_best(args.csv, args.teacher)

    yaml_path = os.path.join(args.out_dir, f"best_hp_{args.teacher}.yaml")
    csv_out = os.path.join(args.out_dir, f"best_hp_{args.teacher}.csv")
    md_out = os.path.join(args.out_dir, f"best_hp_{args.teacher}.md")
    write_yaml(best, counts, yaml_path)
    write_table(best, counts, csv_out, md_out)

    print(f"[{args.teacher}] {len(best)}/18 tasks; wrote:\n  {yaml_path}\n  {csv_out}\n  {md_out}")
    print(f"{'task':<24}{'val':>8}{'test':>8}{'  n':>5}  hp(ce/kl/mse/T)")
    for task in sorted(best):
        r = best[task]
        hp = f"{r['weight_ce']}/{r['weight_kl']}/{r['weight_mse']}/{r['temperature']}"
        tv = "NA" if r["best_test_mcc"] is None else f"{r['best_test_mcc']:.4f}"
        print(f"{task:<24}{r['best_val_mcc']:>8.4f}{tv:>8}{counts.get(task,1):>5}  {hp}")


if __name__ == "__main__":
    main()
