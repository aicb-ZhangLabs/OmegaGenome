"""Merge the DIST/LS kl=0.5 HP-fix 3-seed values into method_comparison_18task.csv.

Replaces the DIST/LS mcc_mean/mcc_std of the 9 re-run cells with the fair
method-appropriate kl=0.5 config values, adds an `hp_note` provenance column,
and prints OLD vs NEW per-method 18-task means + win counts. Two cells that did
NOT recover to the sane band even at kl=0.5 (DIST/LS H3K9me3) are tagged
'kl0.5_NONRECOVERED_pending_kl0.25' and flagged for a later kl=0.25 escalation.
"""

import numpy as np
import pandas as pd
import os

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(HERE, "method_comparison_18task.csv")
runs = {
    ("DIST", "H3K27ac"): [0.4567130134335276, 0.43619207916182373, 0.4456868563459183],
    ("DIST", "H3K27me3"): [0.551894231285542, 0.5564607612513951, 0.5536141998976913],
    ("DIST", "H3K4me1"): [0.4466523510047447, 0.4622104590288521, 0.4517910223567496],
    ("DIST", "H3K9me3"): [0.3194264116631796, 0.17045729691814082, 0.21520264493646468],
    ("DIST", "enhancers"): [0.4890784816652044, 0.4810139874804782, 0.5023139621942607],
    ("DIST", "splice_sites_acceptors"): [
        0.9126776282221157,
        0.9182015481382734,
        0.9306474020561267,
    ],
    ("LS", "H3K9me3"): [0.4256960627020052, 0.09596548933807657, 0.3846998748185089],
    ("LS", "splice_sites_acceptors"): [0.7340803928946976, 0.7096340447902604, 0.7949835000246069],
    ("LS", "splice_sites_donors"): [0.8499748513345635, 0.7452816438777944, 0.8746857478133722],
}
NONREC = {("DIST", "H3K9me3"), ("LS", "H3K9me3")}  # <0.4, still degenerate at kl=0.5
METHODS = ["OmegaGenome", "DKD", "DIST", "LS"]
df = pd.read_csv(CSV)
if "hp_note" not in df.columns:
    df["hp_note"] = ""
df["hp_note"] = df["hp_note"].fillna("")


def method_mean(d):
    w = d.pivot(index="task", columns="method", values="mcc_mean")
    return {m: (w[m].mean(), w[m].std(ddof=1)) for m in METHODS}, w


def wins(w):
    win = w.idxmax(axis=1)
    return {m: int((win == m).sum()) for m in METHODS}


old_stats, old_w = method_mean(df)
old_wins = wins(old_w)
for (m, t), vals in runs.items():
    mean = float(np.mean(vals))
    std = float(np.std(vals, ddof=1))
    mask = (df["method"] == m) & (df["task"] == t)
    assert mask.sum() == 1, (m, t, mask.sum())
    df.loc[mask, "mcc_mean"] = mean
    df.loc[mask, "mcc_std"] = std
    df.loc[mask, "n_seeds"] = 3
    df.loc[mask, "hp_note"] = (
        "kl0.5_NONRECOVERED_pending_kl0.25" if (m, t) in NONREC else "kl0.5_fix"
    )
df.to_csv(CSV, index=False)
new_stats, new_w = method_mean(df)
new_wins = wins(new_w)
print("\n=== per-method 18-task mean (mean +/- across-task s.d.) ===")
print(f"{'method':12} {'OLD':>16} {'NEW':>16}   OLD_wins NEW_wins")
for m in METHODS:
    om, osd = old_stats[m]
    nm, nsd = new_stats[m]
    print(
        f"{m:12} {om:.4f}+/-{osd:.4f}  {nm:.4f}+/-{nsd:.4f}    {old_wins[m]:>5}   {new_wins[m]:>5}"
    )
print("total wins OLD", sum(old_wins.values()), "NEW", sum(new_wins.values()))
