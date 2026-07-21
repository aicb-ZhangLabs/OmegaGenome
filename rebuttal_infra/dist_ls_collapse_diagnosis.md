# DIST / Logit-Standardization KD Collapse — Diagnosis

**Date:** 2026-07-19
**Scope:** READ-ONLY diagnosis. No training code changed, nothing rerun. Where a fix is
implied, it is *proposed*, not applied.

**Question:** Why do DIST and Logit-Standardization (LS) KD collapse to near-zero MCC on
specific NT classification tasks in the 3-method x 18-task sweep, while OmegaGenome (vanilla)
and DKD train fine on all 18? Is it (a) a real method property, (b) an unfair HP-mismatch
artifact, or (c) an implementation bug — and do the numbers belong in the paper?

**Bottom line up front:** It is **(a) + (b), NOT (c).** The DIST and LS KD terms are
*mathematically degenerate on 2-class (binary) tasks* — this is a real property of the
methods (they were designed/validated on 100–1000-class vision). The *catastrophic* collapse
(MCC -> 0.03) on top of that degeneracy is an **HP-mismatch artifact**: the sweep used
vanilla-tuned HP with `weight_kl >= weight_ce` and no method-specific tuning, so the
degenerate KD gradient overwhelms CE and traps the student in a constant-prediction basin.
The implementations are **faithful** to the official references (no correctness bug).
**These numbers should NOT be reported as DIST/LS "performance" without a method-specific-HP
rerun**, or they must be framed explicitly as "off-the-shelf DIST/LS with vanilla HP."

---

## 1. What actually happened in each collapsed run (empirical)

Reconstructed from per-epoch checkpoint dirs (`epoch_<n>_valmcc_<v>`), `final_summary.json`,
and `summary.csv` under `/srv/disk00/sshfs/pengchx3/method3_18task/{dist,logit_standard}/<task>/...`.
All confirmed across the 3 seeds (aggregates in `plot_repo/data/method_comparison_18task.csv`).

| Run | val-MCC trajectory (epoch:MCC) | Diagnosis |
|---|---|---|
| **DIST / H3K4me1** (0.03) | 1–200 all **0.0000**; best 0.149@ep28 then falls back to 0 | **Constant-class prediction the entire run.** No NaN/Inf. Metastable: briefly wiggled off ~ep28, reabsorbed. |
| **DIST / splice_acceptors** (0.10) | 1–200 all **0.0000** | Constant prediction, never escaped. |
| **DIST / enhancers** (0.11) | ~0.0000 throughout (blip 0.018@ep5) | Constant prediction, never escaped. |
| **DIST / H3K9me3** (0.15) | 0.0000 for the shown seed; CSV 0.149+/-0.132 | One seed partially escaped; others stuck at 0. |
| **DIST / splice_donors** (0.95, HEALTHY contrast) | 0.00 for ep 1–~65, **jumps to 0.85@ep80 -> 0.95** | **Same basin, but escaped** at ~ep70. Proves 0-MCC is a metastable trap, not a hard law. |
| **LS / splice_acceptors** (0.36+/-0.19) | noisy 0.10–0.20, occasional 0, no convergence | **Not** stuck-at-exactly-0. Weak/degraded learning, high seed variance. |
| **LS / H3K9me3** (0.23+/-0.18) | noisy low; CSV 0.23+/-0.18 | Degraded + high variance; one seed better. |

**Key facts:**
- **No NaN/Inf.** Losses stayed finite. This is *not* numerical divergence.
- **DIST collapse = exact `val_mcc = 0.0000` = constant single-class prediction** (`final_test_f1 ~ 0.33`,
  the macro-F1 of an all-one-class predictor on binary). It is a **metastable constant-prediction
  basin**, entered at init.
- **Escape is stochastic and late** (splice_donors escapes ~ep70; H3K9me3/H3K27me3 escape in
  *some* seeds only -> the large 3-seed std: DIST H3K27me3 0.36**+/-0.31**, H3K27ac 0.32**+/-0.20**,
  H2AFZ 0.43**+/-0.07**). High variance = signature of a metastable trap, not a stable optimum.
- **LS is milder:** most binary LS tasks land slightly *below* vanilla (e.g. H3K4me1 0.40 vs
  vanilla 0.475); only 2 tasks collapse hard. LS degrades rather than hard-traps.

---

## 2. #classes and the binary correlation

`num_labels` read from each run's `hyperparameters.json`:

| #classes | tasks | DIST outcome | LS outcome |
|---|---|---|---|
| **2 (binary)** | all 10 histone marks, enhancers, promoter_all/no_tata/tata, splice_acceptors, splice_donors | **all collapses live here** (and all healthy binaries too) | both collapses live here |
| **3** | enhancers_types, splice_sites_all | **healthy** (0.46, 0.85) | **healthy** (0.47, 0.84) |

**Every collapse is binary; the two 3-class tasks are healthy for both methods.** So binary is
**necessary but not sufficient** — many binary tasks train fine (splice_donors 0.95, promoters
0.65–0.73, H3K4me3 0.58). This rules out a clean "2-class law" and matches the
splice_donors(0.95) vs splice_acceptors(0.10) puzzle: both binary, opposite outcomes.

The NT tasks are **not** severely class-imbalanced (NT-benchmark binary tasks are ~balanced;
constant-prediction F1~0.33 and MCC=0 is consistent with a balanced 2-class set), so imbalance
is not the driver — the driver is the **2-class degeneracy of the KD terms** plus HP.

### Why binary is degenerate (proved numerically — see `scratchpad/check.py`)

- **DIST inter-class-relation term** = per-sample Pearson correlation of the softmax vectors
  *across classes*. For a 2-vector, any mean-centered vector is proportional to `[1,-1]`, so the
  correlation is **exactly +/-1** for every sample — verified: for C=2, `|corr| == 1.0000` (2 unique
  values); for C=3, `|corr|` mean 0.64 with 571 distinct values (smooth). So on binary the inter
  term collapses to a **sign-agreement step function** `1 - sign(p_s-0.5)*sign(p_t-0.5)` in {0, 2} with
  a **singular gradient at p=0.5** — no smooth learning signal, and an unstable push exactly where a
  freshly-initialized student sits.
- **LS standardization** of a 2-logit vector `[z1,z2]` -> `+/-[1,-1]` **regardless of magnitude**
  (verified: logits `[0.1,0]`, `[1,0]`, `[10,0]` all normalize to `+/-[1,-1]`). So the teacher's
  *confidence is entirely erased*: teacher logits `[3,0]` and `[0.3,0]` yield the **identical**
  KD target `[0.731,0.269]` at T=2. LS on binary transmits only the sign of the teacher's
  preference at a fixed confidence — a weak, partly-misleading target.

Both degeneracies are **inherent to the methods on 2 classes** — they are designed for and
validated on many-class vision (CIFAR-100, ImageNet-1000). -> this component is **(a) real property**.

---

## 3. Exact HP used, and why it makes the degeneracy catastrophic (not just mild)

HP came from `best_hp/best_hp_nt.csv` = **vanilla-KD best per task, method toggled on, NO
method-specific tuning** (confirmed in `method3_18task_specs.txt`: DIST/LS runs reuse the exact
`weight_ce/weight_kl/temperature` of the vanilla row).

DIST, binary tasks, sorted by outcome:

| task | weight_ce | weight_kl | T | DIST MCC (+/-std) | outcome |
|---|---|---|---|---|---|
| H3K4me1 | 0.5 | **1.0** | 1.5 | 0.03+/-0.06 | **collapse** |
| splice_acceptors | 0.5 | **1.0** | 2.0 | 0.10+/-0.02 | **collapse** |
| enhancers | 0.5 | **1.0** | 4.0 | 0.11+/-0.05 | **collapse** |
| H3K9me3 | 0.5 | 0.5 | 2.0 | 0.15+/-0.13 | collapse (hardest task; vanilla only 0.43) |
| H3K27ac | 0.5 | **1.0** | 4.0 | 0.32+/-0.20 | partial / seed-dependent |
| H3K27me3 | 0.5 | **1.0** | 1.5 | 0.36+/-0.31 | partial / seed-dependent |
| H2AFZ | 0.5 | **1.0** | 4.0 | 0.43+/-0.07 | partial |
| H3K9ac | 0.5 | **1.0** | 2.0 | 0.43+/-0.01 | partial |
| H3K4me2 | 0.5 | 0.5 | 2.0 | 0.51 | healthy |
| H3K36me3 | 0.5 | 0.5 | 4.0 | 0.55 | healthy |
| H3K4me3 | 0.5 | 0.5 | 2.0 | 0.58 | healthy |
| H4K20me1 | 0.5 | **1.0** | 4.0 | 0.58 | healthy (exception) |
| promoter_tata | 0.5 | 0.5 | 1.5 | 0.65 | healthy |
| promoter_all | 0.5 | 0.25 | 0.5 | 0.71 | healthy |
| promoter_no_tata | 0.5 | **1.0** | 4.0 | 0.73 | healthy (exception) |
| splice_donors | 0.5 | 0.5 | 4.0 | 0.95 | healthy |

**Pattern:** collapse/partial concentrates where **`weight_kl = 1.0` (KD outweighs CE 2:1)**
and/or the task is intrinsically hard/low-separability. When `weight_kl <= 0.5` (CE >= KD), tasks
are almost all healthy (only exception: H3K9me3, the single hardest task, vanilla 0.43).

**This directly resolves the splice_donors vs splice_acceptors asymmetry** — the headline
"it's not a 2-class law" puzzle:
- Both are binary and easy for the teacher (vanilla 0.93 / 0.88).
- splice_**donors** inherited vanilla HP `kl=0.5, T=4` (CE:KD = 1:1, soft) -> CE escapes the basin -> 0.95.
- splice_**acceptors** inherited vanilla HP `kl=1.0, T=2` (CE:KD = 1:2, sharper) -> the degenerate
  KD term dominates and traps -> 0.10.
The difference is **the inherited HP, not the task.** That is the definition of an HP-mismatch
artifact -> **(b).**

Mechanistically the degenerate DIST loss magnitude (bounded in [0,1] after our /2, ~O(1) at
init) with `weight_kl=1.0` exceeds `weight_ce*CE ~ 0.5*ln2 ~ 0.35`, so the singular /
sign-step gradient at p~0.5 dominates the smooth CE gradient and pins the student at the
constant-prediction fixed point until (sometimes) CE stochastically wins. A method-appropriate
weight (lower `weight_kl`, higher T) would plausibly restore most of these — evidenced by every
`weight_kl<=0.5` binary DIST task training fine.

---

## 4. Implementation vs. reference — faithfulness verdict: **FAITHFUL (no bug)**

Code: `code_carbon/src/model/distillation.py`.

### DIST — `_dist_loss` (lines 531–554) vs Huang et al. NeurIPS'22 (github.com/hunto/DIST_KD)
Official structure = `inter_class_relation` (per-sample Pearson corr across classes) +
`intra_class_relation` (same on the batch-transpose), `loss = beta*inter + gamma*intra`, beta=gamma=1
(verified against `segmentation/losses/dist_kd.py`: mean/norm over `dim=1`, `beta=gamma=1.0`).
Our code computes **both terms with the correct axes**:
- our "inter" block operates on `s_probs.t()` [C,B] -> correlation across the **batch** per class
  = official **intra_class_relation**;
- our "intra" block operates on `s_probs` [B,C] -> correlation across **classes** per sample
  = official **inter_class_relation**.

-> **The two term *labels are swapped*, but both terms are present and summed, so the total is
identical** (the sum is symmetric in the labels). Two benign scale deviations:
1. `(inter+intra)/2` (== beta=gamma=0.5).
2. **No `T^2` multiplier** (classification-DIST scales each term by tau^2; the segmentation ref does
   not — both variants exist upstream).

Both deviations only **rescale** the KD magnitude, and with T>1 they make our DIST *weaker*
relative to CE than the official version — i.e. they would **reduce** collapse, not cause it. The
`eps=1e-8` guards on the norms are present, so there is no div-by-zero. **No correctness bug; axes correct.**

### LS — `_logit_standard_kl` (lines 371–388) vs Sun et al. CVPR'24 (github.com/sunshangquan/logit-standardization-KD)
Official (verified from `mdistiller/distillers/KD.py`): `normalize(z) = (z - mean)/(1e-7 + std)`,
then `/T`, `KL(logsoftmax(s), softmax(t))`, `* T^2`. Our code: mean & std over `dim=-1`,
`/sigma /T`, `KL(...) batchmean`, `* T^2`. Matches. **One minor deviation:** we use
`std(unbiased=False)` (population std, = paper Algorithm 1); the official *code* uses PyTorch
default `unbiased=True` (sample std). For K=2 this only rescales the fixed `+/-[1,-1]` vector by
`1/sqrt(2)` (both still magnitude-independent) — **does not change the binary degeneracy and does not
cause the collapse.** `batchmean` == official `sum(1).mean()`. **Faithful.**

**Conclusion: no implementation bug (rules out (c)).** The file header comments ("DEBUGGING
VERSION", "POTENTIAL BUG SOURCES") are historical; the production paths (`_dist_loss`,
`_logit_standard_kl`) are correct.

---

## 5. Verdict and recommendation

**(a) real method property AND (b) HP-mismatch artifact. NOT (c) a code bug.**

- **(a)** DIST's inter-class-relation and LS's logit standardization are *mathematically
  degenerate on 2-class tasks* (Pearson corr of 2-vectors == +/-1; standardized 2-logits erase
  confidence). Real, and worth one honest sentence in the paper: off-the-shelf DIST/LS are
  ill-matched to binary genomic classification (they target many-class vision).
- **(b)** The *catastrophic* MCC->0.03 collapse (vs. mild degradation) is produced by
  **vanilla-tuned HP applied to un-tuned methods** — specifically `weight_kl >= weight_ce`
  driving a constant-prediction basin. The splice_donors(0.95) vs splice_acceptors(0.10)
  asymmetry is decided entirely by inherited HP (`kl=0.5,T=4` vs `kl=1.0,T=2`), not the task.
  Reporting these as DIST/LS's achievable numbers would be **unfair** — a method must not be
  scored on HP tuned for a different method.

### Do the numbers go in the paper?
**Not as-is.** Two acceptable options:
1. **Fair rerun (preferred):** re-sweep DIST/LS with **method-specific HP** — at minimum a small
   grid over `weight_kl in {0.1,0.25,0.5}` (i.e. CE >= KD) and `temperature in {2,4}` per task,
   picking best-val like every other method. Expected: the hard collapses recover toward the
   0.4–0.6 band (evidenced by every `weight_kl<=0.5` binary DIST task already training fine).
   Optionally add the official `*T^2` scaling to `_dist_loss` for exact-reference parity (a
   *proposed* one-line change — do **not** apply yet).
2. **Report honestly as-is** *only if* explicitly framed: "DIST/LS with **vanilla-transferred
   HP (no method-specific tuning)**; DIST/LS are degenerate on 2-class targets and this table
   shows their off-the-shelf brittleness, not their tuned ceiling." Then it is a legitimate
   robustness/parity point for OmegaGenome (which is *not* HP-brittle here), but the reader must
   be told the HP were not tuned for DIST/LS.

**Do not** present the 0.03/0.10/0.11 cells as the methods' performance without one of the above.

### Proposed fix (NOT applied)
- Primary: method-specific HP sweep for DIST/LS (config-only; no code change). This is the fair fix.
- Optional code parity: multiply `_dist_loss` inter/intra terms by `temperature**2` to match the
  classification-DIST reference. This *raises* DIST's weight and would worsen collapse under the
  current vanilla HP, so it must be paired with the HP re-tune, not done alone.

---

### Evidence artifacts
- Aggregates: `plot_repo/data/method_comparison_18task.csv`
- Runs: `/srv/disk00/sshfs/pengchx3/method3_18task/{dist,logit_standard}/<task>/.../{final_summary.json,summary.csv,epoch_*_valmcc_*}`
- HP source: `code_carbon/best_hp_all_teachers/best_hp/best_hp_nt.csv`; sweep specs `rebuttal_infra/method3_18task_specs.txt`
- Impl: `code_carbon/src/model/distillation.py` (`_dist_loss` L531–554, `_logit_standard_kl` L371–388)
- Numerical degeneracy check: `scratchpad/check.py` (C=2 -> |corr|==1.0; LS erases teacher confidence)
- References: DIST github.com/hunto/DIST_KD (`segmentation/losses/dist_kd.py`); LS github.com/sunshangquan/logit-standardization-KD (`mdistiller/distillers/KD.py`)
