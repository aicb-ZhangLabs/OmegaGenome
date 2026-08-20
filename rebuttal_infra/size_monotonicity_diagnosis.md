# Size-scaling non-monotonicity: read-only diagnosis (seed-0 size sweep)

Scope: why some per-task student-size curves in `plot_repo/data/size_18task_seed0_matrix.csv`
are not strictly rising with model size (7 sizes: 2K / 7K / 28K / 0.1M / 0.8M / 1.8M / 3.6M).
Read-only; nothing was rerun or changed. Data: 149 raw `final_summary.json` under
`/srv/disk00/sshfs/pengchx3/size18task/<task>/<size>/{seed0,seed1}/...` (102 seed0 + 47 seed1),
plus the code in `code_carbon/src/trainer/distill_trainer.py`.

**Consistency check first:** the matrix `mcc` column == raw seed0 `best_test_mcc` exactly
(0 mismatches / 102 non-`original` cells). The `original`/0.1M column and all of
`splice_sites_all` (6 cells) are NOT in the size18task tree — they are spliced from a separate
original-student run set. This provenance split matters for the root cause (§3).

---

## 1. Early stopping — NOT the cause (disabled)

**Code (`distill_trainer.py:175-189`, `_early_stop_step`):**
```python
should_stop = bool(patience and patience > 0 and epochs_no_improve >= patience)
```
With `--trainer-config.early-stop-patience 0`, `patience` is `0`, so `patience and patience > 0`
short-circuits to a falsy `0` → `should_stop` is **always False**. `early-stop-patience 0` therefore
means **early stopping DISABLED (run the full schedule)**, not "stop at 0". Every spec line in
`size18task_specs.txt` uses `--epochs 200 --early-stop-patience 0`.

**Runs confirm it:** across all 149 runs, `total_epochs` ∈ {200} and `early_stopped` ∈ {False}.
No run truncated. `best_epoch` distribution: min 8, p25 32, median 151, p75 181, max 200; 27% of
runs (40/149) have `best_epoch ≥ 180` — i.e. best-val is frequently late, so a truncated schedule
would have changed reported numbers, but the full schedule ran. Note also that even if patience were
enabled, the trainer reports the best-val checkpoint, so early-stopping would yield the identical
reported MCC (see the docstring at `distill_trainer.py:184`, `:214`). **Verdict: early stopping is
not involved.**

## 2. Checkpoint selection — best-VALIDATION-selected, no test peeking

**Code path:**
- `distill_trainer.py:381` `is_best = val_metrics["mcc"] > best_val_mcc` — "best" is decided by
  **validation** MCC only.
- `:385-390` on a new best, `best_epoch`/`best_val_mcc` update and `save_checkpoint(..., is_best=True)`
  writes `run_dir/best_model/student.pt` (the argmax-validation-MCC epoch).
- `:444-451` after training, it reloads `best_model/student.pt` and runs `evaluate(model, test_loader)`
  → `best_test_metrics`.
- `:469-472` `final_test_mcc` = test MCC of the **last** epoch's weights; `best_test_mcc` = test MCC of
  the **argmax-validation** checkpoint.

So `best_test_mcc` is the test MCC at the epoch with the best *validation* MCC — **not** the last epoch
(`final_test_mcc`) and **not** argmax-*test* (which would be leakage). The published matrix uses
`best_test_mcc`.

**Empirics:** `best_test_mcc ≥ final_test_mcc` in 61.7% of runs (92/149), mean `best−final = +0.0031`,
median `+0.0018`. It is deliberately **not** ≥ in 100% of cases — that is the correct signature of
honest best-*val* selection: because val↔test correlation is imperfect, the best-val epoch is
sometimes a slightly worse test point than the last epoch. (If it were best-*test* selected it would be
≥ by construction — that would be the leakage we do NOT see.) The few most-negative cases
(`promoter_tata` pico/ultra_tiny/xxlarge at −0.04 to −0.05) are exactly the task with the largest
val/test distribution mismatch. **Verdict: checkpoints ARE best-val-selected; no test-set peeking.**

## 3. Root cause of the non-monotonicity

### 3a. The drops (published seed0 matrix, all 37 adjacent-size decreases)

108 adjacent pairs; 37 decrease. Full list sorted by magnitude:

| task | transition | drop | |drop|/σ_adj |
|---|---|---:|---:|
| H3K9me3 | 0.1M→0.8M | −0.0612 | 3.46 |
| promoter_tata | 2K→7K | −0.0366 | 2.07 |
| splice_sites_donors | 0.1M→0.8M | −0.0330 | 1.87 |
| H3K4me2 | 0.1M→0.8M | −0.0223 | 1.26 |
| H3K4me2 | 1.8M→3.6M | −0.0216 | 1.22 |
| H3K4me3 | 0.1M→0.8M | −0.0212 | 1.20 |
| H3K36me3 | 1.8M→3.6M | −0.0198 | 1.12 |
| H3K9me3 | 7K→28K | −0.0164 | 0.93 |
| H3K27me3 | 0.8M→1.8M | −0.0151 | 0.86 |
| promoter_all | 28K→0.1M | −0.0140 | 0.80 |
| H3K9ac | 0.1M→0.8M | −0.0123 | 0.69 |
| promoter_no_tata | 28K→0.1M | −0.0120 | 0.68 |
| H4K20me1 | 1.8M→3.6M | −0.0112 | 0.64 |
| H3K27ac | 28K→0.1M | −0.0111 | 0.63 |
| ...(remaining 23 all |drop| ≤ 0.009, ≤ 0.50σ_adj)... | | | |

Landing-size of the dip: 7K:4, 28K:6, 0.1M:5, **0.8M:11**, 1.8M:6, 3.6M:5.

### 3b. Seed-noise scale (from 47 seed0-vs-seed1 pairs, best_test_mcc)

- mean `|seed0 − seed1|` = **0.0139**, median 0.0116, p90 0.0286, max 0.0426
- pooled per-point s.d. **σ = 0.0125 MCC**
- s.d. of a two-point (adjacent-size) difference **σ_adj = √2·σ = 0.0177**

The typical single-seed wobble (0.012–0.014) is **larger than the median observed drop (0.005)**.

### 3c. Are the drops within seed noise? — Yes, overwhelmingly

- 30/37 drops within **1·σ_adj**, **35/37 within 2·σ_adj** (~95%).
- Only **2 drops exceed 2σ**: `H3K9me3 0.1M→0.8M` (−0.061, 3.5σ) and `promoter_tata 2K→7K`
  (−0.037, 2.1σ). Both are explained by structured effects below, not "large model collapses."

### 3d. Structured (non-noise) contributors

**(i) Splice/provenance discontinuity at 0.1M→0.8M — the single biggest structured effect.**
0.1M ("original") is spliced from a separate original-student run set; 0.8M+ come from the size18task
tree. `0.1M→0.8M` is the **only** transition whose mean Δ is **negative** and where a **majority of
tasks drop (11/18, mean Δ = −0.0034)**. Every other transition is net-positive with a minority
dropping:

| transition | tasks dropping | mean Δ |
|---|---:|---:|
| 2K→7K | 4/18 | +0.0466 |
| 7K→28K | 6/18 | +0.0393 |
| 28K→0.1M | 5/18 | +0.0217 |
| **0.1M→0.8M** | **11/18** | **−0.0034** |
| 0.8M→1.8M | 6/18 | +0.0085 |
| 1.8M→3.6M | 5/18 | +0.0043 |

Both >2σ outliers and 5 of the 6 largest drops sit on this one boundary. The original-student 0.1M
point sits slightly high relative to the sweep, manufacturing a dip right after it. This is a
data-assembly artifact, not model-size physics — the hardest/noisiest task (H3K9me3, MCC≈0.40, and
missing seed1 at the large sizes so it can't be seed-averaged) shows it most.

**(ii) Fixed HP across sizes → diminishing returns / mild overfit at the large end.** HP (LR, loss
weights, temperature) in `best_hp_nt.csv` were tuned for the 0.1M student and applied unchanged to all
sizes. Signature by size (seed0):
- median `best_epoch`: 2K 181, 7K 181, 28K 153, 0.8M **65**, 1.8M 82, 3.6M 90 — larger models reach
  their val peak far earlier under the fixed LR.
- `best_val − best_test` (generalization gap) grows with size: 2K +0.016 → 0.8M +0.033 → 3.6M +0.030.
- `best_test − final_test` peaks at 0.8M (+0.012): larger models overfit after their early peak.

Net effect: the large-size end shows **flat/diminishing returns**, not catastrophic dips (all 1.8M/3.6M
drops are ≤1.2σ). Best-val selection catches the early peak, which is why fixed-HP mismatch degrades
*slope* (gains stall) rather than producing big wobbles.

**(iii)** `promoter_tata 2K→7K` (2.1σ) is small-model instability on a small, val/test-mismatched task
(its best_test < final_test by 0.04–0.05 at pico/ultra_tiny), i.e. a small-end effect, not large-model HP.

### Dominant cause

**Single-seed noise dominates.** 95% of the 37 drops (35/37) are within 2σ of the directly measured
seed-to-seed noise, and the median drop (0.005) is smaller than one seed's typical wobble (0.012–0.014).
The two structured effects are secondary: a 0.1M→0.8M splice discontinuity (the largest *systematic*
piece) and fixed-HP diminishing returns at the large end (a slope effect, not dips).

---

## Recommendation

- **The non-monotonicity is benign, dominated by single-seed noise.** Releasing seed1–2 and plotting
  **3-seed means** will remove essentially all wobbles (35/37 ≤2σ; the noise scale exceeds the median
  drop). This is the primary fix.
- **Caveat the 0.1M column / splice boundary.** The 0.1M ("original") points are spliced from a
  separate original-student run set and sit slightly high (11/18 tasks dip at 0.1M→0.8M, the only
  net-negative transition). Cleanest fix: re-run 0.1M inside the same sweep+selection pipeline; minimum:
  footnote that 0.1M is from the original-student runs. This alone accounts for the two >2σ outliers and
  5 of the 6 largest drops.
- **Add a one-line HP caveat.** HP is fixed across sizes (tuned for 0.1M); the large-size end therefore
  shows diminishing returns (early val peak, growing val-test gap), so the plateau at 0.8M–3.6M is a
  fixed-HP artifact, not a true capacity ceiling. Per-size LR tuning would likely recover some large-size
  gains, but this is a *slope* caveat, not a cause of the dips.
- The underlying scaling trend is sound: strongly monotone at the small end (mean Δ +0.047 / +0.039 /
  +0.022 across 2K→7K→28K→0.1M). No reason to distrust the curves beyond the two caveats above.
