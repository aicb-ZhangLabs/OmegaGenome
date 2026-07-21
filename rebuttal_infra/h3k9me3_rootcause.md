# Why H3K9me3 uniquely resists DIST/LS recovery under kl≤0.5 — root cause

**Date:** 2026-07-20
**Scope:** READ-ONLY analysis. No code changed, nothing rerun. A `kl=0.25` escalation for
DIST/LS on H3K9me3 (`loom-hpfix-kl025`) is running separately; this doc predicts its outcome.
**Builds on:** `rebuttal_infra/dist_ls_collapse_diagnosis.md` (the constant-prediction-basin +
2-class-degeneracy + HP-mismatch story). This doc answers the follow-up: of the 9 collapsed
cells, why did **only H3K9me3** fail to stabilize when the kl-fix (kl≤0.5) rescued every other.

## Bottom line up front

The fundamental reason is **(b) lowest task/teacher signal, NOT (a) class imbalance.**
H3K9me3 is **exactly 50/50 balanced** (verified from the raw dataset — identical to every
recovered task), so imbalance is definitively ruled out. What makes it unique is that it is the
**single lowest-signal task in the entire 18-task suite** — lowest teacher MCC, lowest student
ceiling, lowest baseline. DIST/LS have a constant-prediction basin on binary tasks (the known
degeneracy); escaping it requires the CE (true-label) gradient to overpower the degenerate KD
pull, and the strength of that CE escape signal *is* the task's label separability. H3K9me3 has
the weakest separability of all 18 tasks, so it sits **below the escape threshold** that kl=0.5
clears everywhere else. It is the interaction **binary KD degeneracy × minimum-signal task**, not
imbalance and not a bug.

---

## 1. Task properties — H3K9me3 vs the tasks that recovered

### 1a. Class balance (parsed from the actual dataset, not assumed)
`InstaDeepAI/nucleotide_transformer_downstream_tasks_revised`, train + test splits:

| task | n_train | n_test | pos-frac (train) | pos-frac (test) | recovered under kl≤0.5? |
|---|---|---|---|---|---|
| **H3K9me3** | 27,438 | **850** | **0.5000** | **0.5000** | **NO** |
| H3K4me1 | 30,000 | 3,000 | 0.4993 | 0.5017 | yes (DIST 0.454) |
| enhancers | 30,000 | 3,000 | 0.5002 | 0.4843 | yes (DIST 0.491) |
| splice_sites_acceptors | 30,000 | 3,000 | 0.5010 | 0.4917 | yes (DIST 0.921) |
| H3K27ac | 30,000 | 1,616 | 0.5002 | 0.5000 | yes (DIST 0.446) |

**H3K9me3 is perfectly balanced — the least distinguishing feature possible.** The constant-
prediction fingerprint we see in the logs (`final_test_f1 = 0.33333`) is itself the macro-F1 of an
all-one-class predictor on a *balanced* binary set, confirming balance independently. **Imbalance
is not the driver.** (Two minor, non-causal differences: H3K9me3 has a slightly smaller train set
and, notably, the **smallest test set, n=850**, which inflates the *measured* seed-to-seed MCC
variance but does not cause the collapse.)

### 1b. Base difficulty — H3K9me3 is the weakest-signal task, period
From `plot_repo/data/model_comparison_5teacher_formal.csv` and `method_comparison_18task.csv`:

| metric (H3K9me3) | value | rank among 18 tasks |
|---|---|---|
| NT-2.5B teacher MCC (this is the KD teacher) | **0.4636** | **lowest** (next: enhancers_types 0.473, H3K4me1 0.479) |
| Carbon-3B teacher MCC | **0.3502** | **lowest** |
| BPNet baseline (= the student architecture here) | **0.3295** | **lowest** |
| OmegaGenome student MCC | **0.4454** | **lowest** (next: enhancers_types 0.473) |
| DKD student MCC | **0.4439** | **lowest** |

H3K9me3 is the floor on **every** row — teacher, baseline, and both robust students. The soft
labels the KD terms transmit are the least informative available (teacher barely above chance),
and the hard labels are the least separable (BPNet, the actual student architecture, reaches only
0.33 raw). Sequence length is **not** a factor: H3K9me3 is 1000 bp, the same as H3K4me1 which
recovered.

**Biological why:** H3K9me3 marks *constitutive heterochromatin* — broad, repeat-rich, gene-poor
domains defined by megabase-scale chromatin context, not by a sharp local motif (unlike promoters
or splice donor/acceptor sites, which have crisp consensus sequences and reach MCC 0.7–0.97). The
sequence→label mapping is intrinsically diffuse, so the weak-signal ranking above is real, not a
data artifact.

---

## 2. Training dynamics of the collapsed runs (kl=0.5 hpfix)

From `/srv/disk00/sshfs/pengchx3/method_hpfix/{dist,logit_standard}/H3K9me3/seed*/…/{training_history.json,final_summary.json}`.
HP (all seeds): `weight_ce=1.0, weight_kl=0.5, T=4, lr=1e-4, 200 epochs, student=BPNet`.

### The tell-tale basin signature
| method / seed | best_test_mcc (reported) | **final**_test_mcc | **final**_test_f1 | % epochs val_mcc<0.05 | % epochs val_mcc>0.30 |
|---|---|---|---|---|---|
| DIST seed0 | 0.319 | **0.000** | **0.3333** | **97.5%** | 0.5% |
| DIST seed1 | 0.170 | **0.000** | **0.3333** | **98.0%** | 0.0% |
| DIST seed2 | 0.215 | **0.000** | **0.3333** | **97.0%** | 0.0% |
| LS seed0 (healthy) | 0.426 | 0.254 | 0.627 | 8.0% | 56.5% |
| **LS seed1 (COLLAPSED)** | 0.096 | **0.000** | **0.3333** | **68.5%** | 0.0% |
| LS seed2 (healthy) | 0.385 | 0.275 | 0.638 | 12.5% | 56.5% |

**DIST — never learns on any seed.** `train_loss` is flat at ~0.94 for the entire 200 epochs (no
descent), `val_mcc≈0` throughout, `val_f1` pinned at 0.333. All three seeds spend **97–98% of
epochs in the constant-prediction basin**, and **all three final states are the exact
constant-predictor** (`mcc=0.0, f1=0.33333`). The reported DIST means (0.319/0.170/0.215, agg
0.235) are **not genuine performance** — they are best-val checkpoint selection catching a single
transient noise spike off a flat-zero baseline (e.g. seed0's "best" is epoch 4, `val=0.362`, a
fluke before the run settles into the basin for the remaining 196 epochs; seed2's is one epoch,
175). DIST on H3K9me3 at kl=0.5 **did not recover at all** — it just got a lucky checkpoint.

**LS — bifurcation, 2 escape / 1 traps.** seed0 and seed2 show *real* optimization: `train_loss`
descends smoothly 1.21→0.27, `val_mcc` climbs to ~0.40 and holds, 56.5% of epochs above 0.30 →
genuine learning (test 0.426, 0.385). seed1 is the collapsed seed: `train_loss` never descends
(stays 1.15–1.2 and **spikes to 8.96** at ep91, 2.7 at ep136 — the LS gradient destabilizing),
`val_f1` repeatedly snaps back to 0.333, 68.5% of epochs in the basin, final state = constant
predictor (test 0.096). H3K9me3 sits **right on LS's bifurcation boundary**: same HP, same data,
different seed → escape or trap. That is the ~0.18 std and the "one seed collapses to ~0.1."

**High seed variance is the signature of a metastable trap near its escape threshold, not a
stable optimum** — exactly consistent with the prior diagnosis.

---

## 3. Mechanism synthesis — why kl=0.5 escaped everywhere but here

The DIST inter-class term (per-sample Pearson corr → a singular sign-step at p=0.5 on 2 classes)
and LS's 2-logit standardization (erases teacher confidence) both create a **constant-prediction
fixed point** at the student's initialization. Training is a race:

> **escape ⇔ CE gradient (toward true labels) > degenerate-KD pull holding p at the basin.**

Lowering `weight_kl` shrinks the right-hand side. But the **left-hand side is the task's label
separability** — a low-separability task produces a weak CE gradient (that is *definitionally* why
its ceiling MCC is low). So escape probability at a fixed kl scales with task signal.

- On H3K27ac, H3K4me1, enhancers, splice_acceptors (teacher 0.48–0.83), kl=0.5 leaves enough CE
  headroom → they escape. ✔
- **H3K9me3 is the minimum of the signal distribution** (teacher 0.46, student ceiling 0.44,
  BPNet 0.33). Its CE gradient is the weakest of all 18, so at kl=0.5 the degenerate KD still
  wins in some seeds (LS) or all seeds (DIST). It is the **last task to clear the escape
  threshold** — precisely because it has the least CE signal to clear it with.

**Root cause verdict: (b) low teacher/task signal, interacting with the DIST/LS binary
degeneracy. NOT (a) imbalance (ruled out — 50/50), NOT (d) anything else, NOT (c) a bug** (the
implementations were already verified faithful). H3K9me3 is special *only* because it is the
extreme point of the difficulty axis; the degeneracy is the same one that afflicts every binary
task, but H3K9me3 is where the counteracting CE signal is weakest.

**DIST vs LS asymmetry** falls straight out of this: DIST's degeneracy (a *singular* sign-step
gradient sitting exactly at the p=0.5 init point) is harsher than LS's (mere confidence erasure).
So on the weakest-signal task DIST fails on 0/3 seeds while LS survives on 2/3.

---

## 4. Will kl=0.25 fix it? — prediction + recommendation

**LS at kl=0.25: likely fixed.** LS already escapes 2/3 seeds at kl=0.5 (0.426, 0.385, both real
learning). Halving the KD weight (CE:KD = 1:0.25 = 4:1) further widens the CE headroom that the
collapsed seed1 narrowly missed. Expect all 3 seeds to land in the ~0.35–0.42 band, i.e. ≈
OmegaGenome/DKD's 0.44, with the variance collapsing. Good chance of a clean recovery.

**DIST at kl=0.25: uncertain — do not assume it.** At kl=0.5 DIST is in the basin 97–98% of the
time on **every** seed; there is not a single stable escape to build on, and its singular gradient
sits exactly where the balanced-task student initializes. kl=0.25 may finally tip CE over on some
seeds, but there is real risk DIST stays metastable / high-variance / partly collapsed on this
one task. Treat a DIST recovery as plausible-but-not-guaranteed.

### Recommendation
1. **Let `loom-hpfix-kl025` finish and read the *final*-state MCC, not just best-val** — for
   DIST specifically, `best_test_mcc` off a flat-zero run is a checkpoint-selection artifact and
   will overstate stability. Judge recovery by `final_test_mcc` and by % epochs with val_mcc>0.30.
2. **If kl=0.25 cleanly rescues both** → fold H3K9me3 into the recovered set; done.
3. **If DIST (and/or LS) still collapses or needs ever-lower kl** → this is the recommended
   framing, and it is the **stronger paper point**: report H3K9me3 as a *genuine, mechanistic
   finding*, not a loose end —

   > *"DIST and Logit-Standardization are irreducibly unstable on the lowest-signal, most
   > weakly-separable binary task (H3K9me3, constitutive heterochromatin; teacher MCC 0.46,
   > baseline 0.33 — the floor of the suite). Their 2-class KD degeneracy induces a
   > constant-prediction basin whose escape requires a strong CE signal; when label separability
   > is minimal, no seed-robust escape exists and the required kl weight must be tuned per task
   > downward without bound. OmegaGenome and DKD are robust across all 18 tasks with a single,
   > untuned HP setting."*

   This turns H3K9me3 from an awkward outlier into a **clean demonstration that the DIST/LS
   fragility is signal-dependent and predictable**, and that OmegaGenome/DKD's HP-robustness — no
   per-task kl surgery — is a real advantage. H3K9me3 is not a failure to explain away; it is the
   controlled worst case that proves the mechanism.

---

## Evidence artifacts
- Label balance (this analysis): `datasets.load_dataset("InstaDeepAI/nucleotide_transformer_downstream_tasks_revised")`, offline cache `/home/pengchx3/.cache/huggingface/datasets/…` — H3K9me3 = 13,719/13,719 (exactly 50/50).
- Difficulty ranking: `plot_repo/data/model_comparison_5teacher_formal.csv`, `plot_repo/data/method_comparison_18task.csv`.
- Dynamics: `/srv/disk00/sshfs/pengchx3/method_hpfix/{dist,logit_standard}/H3K9me3/seed{0,1,2}/…/{training_history.json,final_summary.json,best_model_info.txt}` (HP `kl=0.5,T=4`; `hyperparameters.json`).
- Mechanism/degeneracy + faithfulness: prior `rebuttal_infra/dist_ls_collapse_diagnosis.md`.
