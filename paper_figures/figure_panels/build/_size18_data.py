"""
Shared data assembly for the seed0 student-size sweep (18 tasks x 7 sizes).

Reuses the EXACT Fig 6B loading logic (fig6b_size_18task.py) so every number in
the per-task / per-size figures matches Fig 6B cell-for-cell.  The three loader
functions below are copied verbatim from fig6b_size_18task.py:

  * load_sweep_seed0()  -- 17 tasks x 6 sizes, best_test_mcc from the seed0 sweep
                           on galaxy SSD (.../seed0/**/final_summary.json).
  * load_original_nt()  -- deployed 0.1M student = "Distilled Nucleotide
                           Transformer" column of model_comparison_5teacher_formal.csv.
  * load_splice_all()   -- splice_sites_all across all 7 sizes from the pre-existing
                           size CSVs (matches current Fig 6 Panel B).

assemble() returns (M, TASKS, SIZES) and writes data/size_18task_seed0_matrix.csv.
Single-seed (seed0).
"""
import os
import json
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))

# ---- size roster + labels + representative parameter counts (matches Panel B)
SIZES = ["pico", "ultra_tiny", "extra_tiny", "original",
         "extra_large_fix", "large", "xxlarge"]
LABEL = {"pico": "2K", "ultra_tiny": "7K", "extra_tiny": "28K", "original": "0.1M",
         "extra_large_fix": "0.8M", "large": "1.8M", "xxlarge": "3.6M"}
PARAMS = {"pico": 2e3, "ultra_tiny": 7e3, "extra_tiny": 28e3, "original": 1.2e5,
          "extra_large_fix": 8e5, "large": 1.8e6, "xxlarge": 3.6e6}

# ---- task families (order + colour grouping) -------------------------------
HISTONE = ["H2AFZ", "H3K27ac", "H3K27me3", "H3K36me3", "H3K4me1", "H3K4me2",
           "H3K4me3", "H3K9ac", "H3K9me3", "H4K20me1"]
PROMOTER = ["promoter_all", "promoter_no_tata", "promoter_tata"]
ENHANCER = ["enhancers", "enhancers_types"]
SPLICE = ["splice_sites_acceptors", "splice_sites_donors", "splice_sites_all"]
FAMILY_ORDER = ["histone", "promoter", "enhancer", "splice"]
FAMILY_TASKS = {"histone": HISTONE, "promoter": PROMOTER,
                "enhancer": ENHANCER, "splice": SPLICE}
TASK_FAMILY = {}
for fam, ts in FAMILY_TASKS.items():
    for t in ts:
        TASK_FAMILY[t] = fam

# nicer display labels for tasks (compact, publication-clean)
TASK_LABEL = {
    "H2AFZ": "H2A.Z", "H3K27ac": "H3K27ac", "H3K27me3": "H3K27me3",
    "H3K36me3": "H3K36me3", "H3K4me1": "H3K4me1", "H3K4me2": "H3K4me2",
    "H3K4me3": "H3K4me3", "H3K9ac": "H3K9ac", "H3K9me3": "H3K9me3",
    "H4K20me1": "H4K20me1",
    "promoter_all": "promoter (all)", "promoter_no_tata": "promoter (no TATA)",
    "promoter_tata": "promoter (TATA)",
    "enhancers": "enhancers", "enhancers_types": "enhancer types",
    "splice_sites_acceptors": "splice acceptor",
    "splice_sites_donors": "splice donor",
    "splice_sites_all": "splice (all)",
}


# ============================ VERBATIM FIG 6B LOADERS ======================= #
def load_sweep_seed0():
    """Return {task: {size: best_test_mcc}} from the seed0 size sweep on galaxy SSD.

    Reads best_test_mcc from each .../seed0/**/final_summary.json via a remote
    python one-liner over ssh; exactly one summary per (task,size) cell.
    """
    # A reader of the archive has neither the galaxy SSD nor ssh access, so the extracted
    # matrix ships in plot_repo/data/; the ssh read below is the authors' regeneration path.
    bundled = os.path.join(HERE, "..", "..", "data", "size18task_seed0_best_test_mcc.json")
    if os.path.exists(bundled):
        with open(bundled) as fh:
            return json.load(fh)
    raise FileNotFoundError(
        f"{bundled} is missing; it holds the extracted seed-0 size-sweep matrix this figure reads."
    )


def load_original_nt():
    """{task: MCC} for the deployed 0.12M student = Distilled NT column."""
    df = pd.read_csv(os.path.join(HERE, "..", "..", "data",
                                  "model_comparison_5teacher_formal.csv"))
    nt = df[df["Model"] == "Distilled Nucleotide Transformer"]
    return dict(zip(nt["Task"], nt["Score"]))


def load_splice_all():
    """{size: mean MCC} for splice_sites_all from the existing size CSVs.

    Reproduces the current Fig 6 Panel B extraction: drop seeds {42,1024}, swap
    extra_large -> extra_large_fix (finished, mcc>0.5). Returns per-size mean over
    the available seeds (splice_sites_all is the multi-seed task).
    """
    _R = {"config.dataset_config.task_name": "task_name",
          "summary.best_test/mcc": "mcc", "config.random_state": "seed",
          "config.student_config.model_size": "model_size"}
    base = os.path.join(HERE, "..", "..", "data", "hp_search_csv") + os.sep
    load = lambda p: pd.read_csv(base + p).rename(columns=_R)
    sz = load("size_comparison_flat_clean.csv")
    sz = sz[~sz["seed"].isin([42, 1024])]
    seeds_in = set(sz["seed"].unique())
    xl = load("extralarge_fix_flat_clean.csv")
    xl = xl[xl["seed"].isin(seeds_in)]
    if "state" in xl.columns:
        xl = xl[xl["state"] == "finished"]
    xl = xl[xl["mcc"] > 0.5]
    xl_proc = pd.DataFrame({"model_size": "extra_large_fix",
                            "task_name": xl["task_name"].values, "mcc": xl["mcc"].values})
    sz = sz[sz["model_size"] != "extra_large"]
    sz = pd.concat([sz[["model_size", "task_name", "mcc"]], xl_proc], ignore_index=True)
    ssa = sz[sz["task_name"] == "splice_sites_all"].dropna(subset=["mcc"])
    out = {}
    for s in SIZES:
        d = ssa[ssa["model_size"] == s]["mcc"]
        out[s] = float(d.mean())
    return out


# ============================== ASSEMBLE =================================== #
def assemble(write_csv=True):
    """Assemble the 18-task x 7-size seed0 MCC matrix; optionally write tidy CSV.

    Returns (M, TASKS_ORDERED, SIZES) where M[task][size] -> float MCC.
    TASKS_ORDERED groups tasks by family (histone/promoter/enhancer/splice).
    """
    sweep = load_sweep_seed0()          # 17 tasks x 6 sizes
    nt_orig = load_original_nt()        # original per task
    splice = load_splice_all()          # splice_sites_all all sizes

    swept = sorted(sweep.keys())
    M, missing = {}, []
    for t in swept:
        M[t] = {}
        for s in SIZES:
            v = nt_orig.get(t) if s == "original" else sweep[t].get(s)
            if v is None:
                missing.append((t, s))
            M[t][s] = v
    M["splice_sites_all"] = {s: splice[s] for s in SIZES}

    # family-grouped task order
    TASKS_ORDERED = []
    for fam in FAMILY_ORDER:
        TASKS_ORDERED += [t for t in FAMILY_TASKS[fam] if t in M]
    assert len(TASKS_ORDERED) == 18, TASKS_ORDERED

    if write_csv:
        rows = []
        for t in TASKS_ORDERED:
            for s in SIZES:
                rows.append({"task": t, "family": TASK_FAMILY[t],
                             "size_key": s, "size_label": LABEL[s],
                             "params": PARAMS[s], "mcc": M[t][s]})
        out = os.path.join(HERE, "..", "..", "data", "size_18task_seed0_matrix.csv")
        pd.DataFrame(rows).to_csv(out, index=False)
        print("wrote", os.path.abspath(out), "(%d rows)" % len(rows))

    if missing:
        print("WARNING missing cells:", missing)
    return M, TASKS_ORDERED, SIZES


def mean_curve(M, TASKS):
    """18-task mean and task-to-task s.d. (ddof=1) at each of the 7 sizes."""
    mc, sd = [], []
    for s in SIZES:
        vals = np.array([M[t][s] for t in TASKS], dtype=float)
        mc.append(float(np.nanmean(vals)))
        sd.append(float(np.nanstd(vals, ddof=1)))
    return np.array(mc), np.array(sd)


if __name__ == "__main__":
    M, TASKS, _ = assemble()
    mc, sd = mean_curve(M, TASKS)
    print("=== per-size 18-task mean (must match Fig 6B) ===")
    for i, s in enumerate(SIZES):
        print(f"  {LABEL[s]:>6s}  mean={mc[i]:.4f}  sd(task)={sd[i]:.4f}")
