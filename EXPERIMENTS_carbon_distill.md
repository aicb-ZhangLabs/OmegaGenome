# Carbon-3B → BPNet Distillation Experiments

Distillation phase (Carbon-3B LoRA teacher → 0.12M BPNet student, 18 NT-benchmark tasks).
Companion to the teacher-finetune results (EXPERIMENTS_carbon_teacher.md). Working dir: `code_carbon`.
Regenerate the results table anytime: `python slurm/aggregate_carbon_results.py`.

## 2026-06-18 — distillation pipeline brought up + 18-task raw/l2norm campaign

### Critical bug fixed: transformers `output_hidden_states` capture-wrapper leak (OOM)
Carbon-3B distillation OOM'd at a fixed ~47 GB regardless of GPU size (47 GB laniakea **and** 92 GB
voyager both filled) — a **memory leak, not capacity**. Root cause (forensically traced: tensor census
→ referrer trace → closure scan): transformers' `check_model_inputs` (`transformers/utils/generic.py`)
monkey-patches each `LlamaDecoderLayer.forward` with a `wrapped_forward` closure that collects the
layer `hidden_states`; across the many sequential teacher forwards (precompute over 15–27 k seqs **and**
the teacher eval) the wrappers aren't freed, so each call's 31-layer tuple (~128 MB) accumulates → OOM
~batch 350.
- **Fix:** `_free_capture_wrappers(model)` in `src/trainer/utils.py` — after each batch, delete the
  instance-level `forward` whose `__code__.co_name=='wrapped_forward'` (file `generic.py`), restoring the
  class method. Applied in **both** `precompute_teacher_logits` and `evaluate_teacher_mcc` (the eval
  leaks identically; fresh/uncached tasks OOM in eval before precompute).
- Carbon-3B is Llama-arch so it triggers this; the fix is a precise no-op for NT/Caduceus/Enformer.

### Other fixes this round
- bf16 actually applies: `dtype=` not deprecated `torch_dtype=`, plus explicit `model.to(bf16)` (Carbon
  `trust_remote_code` ignored the load dtype) → 12 GB→6 GB weights.
- `teacher_batch_size=4` (3B forward; student trains at 16); pad_token_id set independently of the
  tokenizer-pad patch; precompute + cache → SSD (`cache_base_dir`), off the degraded /extra NFS.
- Atomic cache writes (tmp + `os.replace`) so carbon-raw and carbon-l2norm can't corrupt a shared
  teacher cache.

### Teacher verification (all 4 families work end-to-end with the changes)
| teacher | result |
|---|---|
| Carbon-3B (Llama) | full chain, flat memory, teacher MCC matches finetune |
| NT-2.5B | completed (1h26m), no leak |
| Caduceus | debug smoke done |
| Enformer | debug smoke done |
11/11 unit tests green throughout.

### Results: raw vs L2-norm feature distillation (student best-test MCC) — seed 0
weight_ce 0.5 / weight_kl 0.5 / weight_mse 0.2, temperature 2.0, 200 epochs, BPNet 0.12 M student.
raw = feature MSE on raw hidden; l2norm = L2-normalized features. **raw 18/18 done; l2norm 11/18**
(rest finishing). `*` = student >= teacher. Regenerate: `python slurm/aggregate_carbon_results.py`.

| task | teacher | raw | l2norm |
|---|---|---|---|
| H3K27me3 | 0.6008 | 0.5786 | 0.5763 |
| H3K36me3 | 0.6075 | 0.5633 | 0.5745 |
| H4K20me1 | 0.6699 | 0.6091 | 0.6048 |
| H2AFZ | 0.5105 | 0.4751 | 0.4840 |
| H3K27ac | 0.4065 | 0.4572* | 0.4604* |
| H3K4me1 | 0.4791 | 0.4817* | 0.4754 |
| H3K4me2 | 0.5856 | 0.5452 | — |
| H3K4me3 | 0.6533 | 0.6240 | — |
| H3K9ac | 0.5339 | 0.5108 | 0.4846 |
| H3K9me3 | 0.3591 | 0.4095* | 0.3915* |
| promoter_all | 0.7709 | 0.7098 | — |
| promoter_tata | 0.9717 | 0.7937 | — |
| promoter_no_tata | 0.7805 | 0.7299 | 0.7144 |
| enhancers | 0.5346 | 0.4913 | — |
| enhancers_types | 0.5509 | 0.4572 | — |
| splice_sites_all | 0.9720 | 0.7129 | 0.7227 |
| splice_sites_acceptors | 0.9727 | 0.7807 | 0.8013 |
| splice_sites_donors | 0.9773 | 0.5519 | — |

**Headline:** raw mean over 18 tasks = **0.5823 vs teacher 0.6632 = 87.8% of teacher**. (cf. published
NT->BPNet ~98%, so Carbon distillation currently underperforms — see HP-search plan below.)

**Observations:**
- Student tracks teacher closely on histone marks; on H3K27ac, H3K4me1, H3K9me3 the distilled student
  **beats** the teacher.
- raw vs l2norm are **statistically indistinguishable** (mixed wins within noise; e.g. H3K27me3
  0.5786/0.5763, splice_acceptors 0.7807/0.8013). No clear feature-normalization effect.
- **Large gap on near-saturated high-MCC tasks** (splices/promoters, teacher ~0.97 -> student
  0.55-0.79). This is NOT under-training (see convergence below) — it's distillation efficacy.

### Convergence check (splice tasks — are they trained enough?)
Inspected per-epoch val_mcc trajectories (seed 0, 200 epochs). They are fully trained / converged;
more epochs would not help (the reported MCC is already the best-val checkpoint, not the overfit final):

| task | student(best) | best epoch | epoch-200 val | note |
|---|---|---|---|---|
| splice_sites_all | 0.713 | 60 | plateaued ~0.72 since ep40 | converged |
| splice_sites_acceptors | 0.781 | 171 | 0.76->0.78 (tiny creep) | ~converged |
| splice_sites_donors | 0.552 | 36 | declined to ~0.50 | **overfitting** (val peaks ep36 then drops while loss keeps falling) |

Conclusion: the splice gap is a **distillation-config / teacher-feature** problem, not epoch count.
Carbon is a generative Llama (vs NT's bidirectional encoder), so feature-MSE may transfer less; and the
default knobs (T=2.0, w_mse=0.2, fixed LR, no schedule, no early-stop reg) are untuned for Carbon ->
motivates the HP search.

### Next: HP search -> best params -> 3-seed (pipeline staged)
1. **3-seed (raw, seeds 1&2)** in progress -> mean+-std significance (kills "student>teacher = noise").
2. **HP search** `distill_hyperparam` over a focused grid (T, w_mse, w_kl) per task, single seed.
3. **Auto-extract best-on-val** per task: `slurm/extract_best_hyperparams.py` -> `best_hyperparams.json`
   (selects on validation MCC, never test; handles multiple candidates).
4. **3-seed on best HP** per task -> final per-task numbers.

### Infra
- Per-task jobs via `slurm/carbon_distill.sbatch <config> --task-names <T> --slurm-config.mode run`.
- Cap-respecting auto-submitter `slurm/auto_submit_carbon.sh` (self-tracks job IDs; laniakea ≤6,
  voyager ≤3; absolute slurm binary paths for nohup shells). l2norm auto-chained after raw via
  `slurm/chain_l2norm_after_raw.sh`. Student ckpts + cache on galaxy SSD.

## 2026-06-21 — HP search running (raw grid) + downstream pipeline de-risked

### Status
- **Raw HP grid** (carbon-base-raw): kl{0,.25,.5,1} × mse{0,1,2,5} × T{.5,1,1.5,2,4} = 80/task × 18 = **1440 combos**.
- Progress **430/1440 (30%)** as of 02:37; ~19 combos/hr across 13 slots (lan 7 + voy 3 + gal 3); ETA ~2 days.
- Live caps in `slurm/submit_caps.env` (re-read each loop). Early-stop patience 100 reports best-val ckpt.

### Failures: 16 early jobs FAILED — infra, NOT code (fixed + self-healing)
- Cause: transient **sshfs blips on the shared galaxy SSD** → `torch.save(student.pt)` "File … cannot be
  opened" (15 on laniakea, 1 on voyager). Galaxy never hits it (local SSD). 0 failures since the fix.
- Fix `save_checkpoint` now wraps writes in `_retry_io` (retries ONLY transient FS errors 3/9/27s;
  CUDA/shape errors re-raise immediately). `reconcile_hp.sh` watcher waits for the one-pass submitter
  to drain, then regenerates ONLY missing combos (resume-aware) until 0 holes. Tests: `test_retry_io`.

### Pipeline de-risked on the 430 partial combos (validated before grid completes)
- `extract_best_hyperparams.py` ✅ runs end-to-end (best-on-VAL per task, never test; tracks n_candidates).
  Early signal: on well-covered tasks the grid already BEATS the mse=0.2 baseline on val — e.g. H3K4me3
  raw 0.6845 (kl.5,mse0,T4, n77), H4K20me1 0.6309 (kl1,mse1,T4), H3K27ac 0.4869 (kl1,mse2,T.5).
- **Bug found + prevented in the 3-seed aggregation:** `aggregate_carbon_results.py` log-scrapes and
  defaults a missing `--random-state` to 0, so all ~80 HP-search runs/task (which actually use the
  config default seed **42**) would collide with the best-HP 3-seed **seed-0** cell → wrong headline.
  Fix: `final_summary.json` now records `random_state`; new **`slurm/aggregate_3seed.py`** reads
  final_summaries and selects ONLY runs with `random_state∈{0,1,2}` (HP-search seed-42 excluded, zero
  contamination). Verified on synthetic data (seed-42 decoy 0.999 + old no-seed 0.888 both excluded).

### Final-phase runbook (when `HP_GRID_COMPLETE` appears / reconciler reports 0 holes)
1. `python slurm/extract_best_hyperparams.py --out best_hyperparams.json`  (best-on-val per task)
2. `python slurm/gen_3seed_best_specs.py --best best_hyperparams.json --seeds 0 1 2 --out best_3seed_specs.txt`
3. `bash slurm/auto_submit_specs.sh best_3seed_specs.txt`  (18×3 = 54 jobs, seeds 0/1/2)
4. `python slurm/aggregate_3seed.py --best best_hyperparams.json`  → per-task mean±std (the paper table)

## 2026-06-23 — HP GRID SEARCH COMPLETE (1440/1440) + best-HP per task + 3-seed launched

**Raw HP grid done: 1440/1440 combos, 0 failures** (after the 2026-06-21 regression fix held all the
way). kl{0,.25,.5,1} × mse{0,1,2,5} × T{.5,1,1.5,2,4} × 18 tasks, fixed seed 42, best-on-VAL selection.

**Best-on-val hyperparameters per task** (`best_hyperparams.json`; ~81–91 candidates/task = full grid):

| task | val MCC | test MCC | best (ce/kl/mse/T) |
|---|:-:|:-:|:-:|
| splice_sites_acceptors | 0.849 | 0.823 | 0.5/0.5/0/1.5 |
| splice_sites_all | 0.818 | 0.805 | 0.5/1.0/0/4 |
| promoter_tata | 0.886 | 0.793 | 0.5/0.0/2/1 |
| promoter_all | 0.749 | 0.727 | 0.5/1.0/1/2 |
| promoter_no_tata | 0.754 | 0.718 | 0.5/0.5/1/0.5 |
| splice_sites_donors | 0.732 | 0.691 | 0.5/0.5/0/0.5 |
| H3K4me3 | 0.685 | 0.624 | 0.5/0.5/0/4 |
| H4K20me1 | 0.631 | 0.603 | 0.5/1.0/1/4 |
| H3K36me3 | 0.628 | 0.580 | 0.5/0.5/2/1.5 |
| H3K27me3 | 0.594 | 0.570 | 0.5/0.5/0.2/2 |
| H3K4me2 | 0.588 | 0.532 | 0.5/0.5/0/2 |
| H3K9ac | 0.563 | 0.506 | 0.5/1.0/2/1.5 |
| enhancers | 0.550 | 0.497 | 0.5/1.0/0/1 |
| H2AFZ | 0.523 | 0.477 | 0.5/1.0/0/1 |
| enhancers_types | 0.499 | 0.452 | 0.5/0.5/1/1.5 |
| H3K4me1 | 0.504 | 0.482 | 0.5/1.0/0/4 |
| H3K27ac | 0.487 | 0.465 | 0.5/1.0/2/0.5 |
| H3K9me3 | 0.419 | 0.436 | 0.5/0.25/0/2 |

**Findings:** (1) **`mse=0` (pure logit-KD) wins or ties on the majority of tasks** — feature-matching MSE
rarely helps the 0.12M student; (2) `promoter_tata` best at **`kl=0`** (CE+MSE only); (3) optimal
temperature spans 0.5→4 and KL weight 0→1 per task — task-specific tuning was worth it. Strong tasks:
splice (0.80–0.82 test), promoters (0.72–0.79); hardest: H3K9me3/H3K27ac/enhancers (~0.44–0.50).

**Artifacts:**
- `results/carbon_grid_results.csv` — full per-run table (one row per task×HP×seed → val/test MCC, etc.).
- `best_hyperparams.json` — winners above; `best_3seed_specs.txt` — 54 seed specs.

### 3-SEED-ON-BEST — ✅ DONE (2026-06-24, 18/18 tasks × seeds {0,1,2})

Best-HP per task re-run at 3 seeds (random_state 0/1/2; excludes the seed-42 HP-search runs, so no
selection-optimism). The significance-backed headline: **mean best-test MCC 0.594 ± (per-task std
≤0.033) = 89.6% of the Carbon-3B teacher (0.663)** — confirms the 90% retention is real, not seed luck.

| task | 3-seed mean ± std | teacher¹ | % | best HP (ce/kl/mse/T) |
|---|:-:|:-:|:-:|:-:|
| splice_sites_acceptors | 0.7888 ± 0.0167 | 0.9727 | 81% | 0.5/0.5/0/1.5 |
| splice_sites_all | 0.7843 ± 0.0207 | 0.9720 | 81% | 0.5/1.0/0/4 |
| promoter_no_tata | 0.7361 ± 0.0074 | 0.7805 | 94% | 0.5/0.5/1/0.5 |
| promoter_all | 0.7294 ± 0.0091 | 0.7709 | 95% | 0.5/1.0/1/2 |
| promoter_tata | 0.8243 ± 0.0225 | 0.9717 | 85% | 0.5/0.0/2/1 |
| splice_sites_donors | 0.6266 ± 0.0325 | 0.9773 | 64% | 0.5/0.5/0/0.5 |
| H3K4me3 | 0.6172 ± 0.0118 | 0.6533 | 94% | 0.5/0.5/0/4 |
| H4K20me1 | 0.6071 ± 0.0075 | 0.6699 | 91% | 0.5/1.0/1/4 |
| H3K27me3 | 0.5739 ± 0.0049 | 0.6008 | 96% | 0.5/0.5/0.2/2 |
| H3K36me3 | 0.5732 ± 0.0072 | 0.6075 | 94% | 0.5/0.5/2/1.5 |
| H3K4me2 | 0.5272 ± 0.0163 | 0.5856 | 90% | 0.5/0.5/0/2 |
| H3K9ac | 0.5066 ± 0.0180 | 0.5339 | 95% | 0.5/1.0/2/1.5 |
| enhancers | 0.5055 ± 0.0007 | 0.5346 | 95% | 0.5/1.0/0/1 |
| H2AFZ | 0.4934 ± 0.0098 | 0.5105 | 97% | 0.5/1.0/0/1 |
| H3K4me1 | 0.4678 ± 0.0081 | 0.4791 | 98% | 0.5/1.0/0/4 |
| H3K27ac | 0.4648 ± 0.0033 | 0.4065 | **114%** | 0.5/1.0/2/0.5 |
| enhancers_types | 0.4579 ± 0.0125 | 0.5509 | 83% | 0.5/0.5/1/1.5 |
| H3K9me3 | 0.4137 ± 0.0169 | 0.3591 | **115%** | 0.5/0.25/0/2 |
| **mean (18)** | **0.5943** | **0.6632** | **89.6%** | — |

- Student (0.12M BPNet) retains **89.6%** of Carbon-3B (~3B, ~25,000× smaller); **beats the teacher**
  on the two noisiest tasks (H3K27ac 114%, H3K9me3 115%). Per-task std small (≤0.033; most ≤0.02).

¹ **Teacher column is a single LoRA finetune per task** (deterministic, `seed=42`; LoRA r16→re-finetuned
  variants noted separately), so it carries no ±std. A teacher mean±std would require 3-seeding all 18
  teachers (54 Carbon-3B LoRA runs) — deferred: LoRA finetunes are low-variance and the **student**
  3-seed already supplies the significance for the retention headline. Re-finetune of the underfit
  H3K27ac teacher (40ep, r32/α64; job 241907) is in flight and will update that one teacher cell +
  cascade a re-distill.
- Reproduce: `python slurm/aggregate_3seed.py --best best_hyperparams.json`. Per-run rows (all
  seeds + the full grid) in `results/carbon_grid_results.csv`.

### 2026-06-24 — canonical from-scratch BPNet baseline vs our distilled student (apples-to-apples)

**Baseline provenance:** the **0.12M BPNet from-scratch** baseline (`model_size="original"` — also ~0.12M,
121,159 params; near-identical to `deploy_120k`'s 121,423) was trained in the **original OmegaGenome repo**
(`OmegaGenome_1-27-clean`), **not** this revise repo, on the **same nt_revised benchmark**. Numbers fetched
from the figure-pack repo `explcre/dna-llm-distillation-plot` and versioned here at
`results/baselines_nt_revised/` (`model_comparison_{long,wide}.csv`).

**Same model size (0.12M), same benchmark → clean test of "does distillation help?":**

| task | from-scratch baseline (0.12M) | our distilled (0.12M, 3-seed) | Δ |
|---|:-:|:-:|:-:|
| splice_sites_donors | **0.8527** | 0.6266 | **−0.2261** ⚠ |
| splice_sites_acceptors | 0.8280 | 0.7888 | −0.0392 |
| splice_sites_all | 0.7571 | 0.7843 | +0.0272 |
| promoter_tata | 0.7833 | 0.8243 | +0.0410 |
| promoter_no_tata | 0.6983 | 0.7361 | +0.0378 |
| promoter_all | 0.6768 | 0.7294 | +0.0526 |
| H3K4me3 | 0.6238 | 0.6172 | −0.0066 |
| H4K20me1 | 0.6061 | 0.6071 | +0.0010 |
| H3K36me3 | 0.5546 | 0.5732 | +0.0186 |
| H3K27me3 | 0.5497 | 0.5739 | +0.0242 |
| H3K4me2 | 0.5366 | 0.5272 | −0.0094 |
| H3K9ac | 0.5213 | 0.5066 | −0.0147 |
| enhancers | 0.4727 | 0.5055 | +0.0328 |
| H3K4me1 | 0.4606 | 0.4678 | +0.0072 |
| H2AFZ | 0.4568 | 0.4934 | +0.0366 |
| enhancers_types | 0.4530 | 0.4579 | +0.0049 |
| H3K27ac | 0.4275 | 0.4648 | +0.0373 |
| H3K9me3 | 0.3295 | 0.4137 | +0.0842 |
| **mean (18)** | **0.5882** | **0.5943** | **+0.0061** |

- **Distillation matches/beats the same-size from-scratch baseline on 13/18 tasks** (biggest wins
  H3K9me3 +0.084, promoter_all +0.053, H3K27ac +0.037) → the **distilled 0.12M student slightly
  exceeds the 0.12M baseline overall (0.5943 vs 0.5882)** while also being the deployable artifact.
- **⚠ splice_donor is the one real failure: 0.627 distilled vs 0.853 baseline (−0.226).** Same size,
  same nt_revised data, and the Carbon-3B teacher scores **0.977** on donor → the task is fully
  learnable. **ROOT CAUSE FOUND (code, not data/training):** the `deploy_120k` student caps dilation at
  `2**min(i,6)=64`, while the canonical baseline's `original` backbone uses `2**i` → **512**. That's an
  ~8× smaller receptive field. Splice **donor** needs long-range exon/intron context, so the cap craters
  it; **acceptor** is mildly hit (−0.039); **splice_all is fine** (+0.027). The `deploy_120k` docstring's
  "full dilation" claim was wrong.
- **FIX = use the `original` BPNet student** (full receptive field, dilation→512) — the SAME student the
  NT/Enformer/Caduceus/DNABERT-2 distillations use (so Carbon is now consistent), 0.12M-deployable
  (teacher-projection is training-only). New `carbon-raw-original` config → SEPARATE `.../original` output
  leaf (cannot collide with the deploy_120k grid), shared teacher cache. (An earlier bespoke
  `deploy_120k_fulldil` idea was dropped in favour of `original` for consistency.)
- **✅ CONFIRMED (2026-06-24, jobs 241920/241921, from-scratch ce0.5/kl0/mse0, 200ep no early stop):**

  | task | `original` (test) | your table baseline | deploy_120k (capped) |
  |---|:-:|:-:|:-:|
  | splice_sites_acceptors | **0.8907** (val 0.906) | 0.828 | 0.81 |
  | splice_sites_donors | **0.9025** (val 0.934) | 0.853 | **0.63** |

  Donor **0.63 → 0.90** purely from the receptive field — the dilation cap was the bug, decisively. The
  `original` student reproduces *and exceeds* the published BPNet-from-scratch baseline. Baselines fetched
  from `explcre/dna-llm-distillation-plot`, versioned at `results/baselines_nt_revised/`.
- **Next (in flight):** vanilla 0.5/0.5/0.2 distillation on `carbon-raw-original` — acceptor+donor first
  (241938/241939), then the other 16, then the staged HP grid (mse [0,1]→[0.25]→[2,5]).

### 2026-06-24 — vanilla 0.5/0.5/0.2 distillation, all 18 tasks, `original` student ✅ DONE

Single fixed config (ce0.5 / kl0.5 / mse0.2 / T2.0, 200ep no early stop) on the **`original`** student
— a sanity pass before the per-task HP grid. **mean test MCC = 0.6195 = 93.4% of the Carbon-3B teacher
(0.6632)**, already **above the old deploy_120k best-HP 3-seed (0.5943)** — switching the student
(capped→full receptive field) lifts the mean ~0.59→0.62 *before any per-task tuning*.

| task | vanilla test | val | | task | vanilla test | val |
|---|:-:|:-:|---|---|:-:|:-:|
| splice_sites_donors | 0.8921 | 0.930 | | H3K4me2 | 0.5473 | 0.586 |
| splice_sites_all | 0.8779 | 0.885 | | H3K9ac | 0.5122 | 0.550 |
| splice_sites_acceptors | 0.8601 | 0.886 | | enhancers | 0.4975 | 0.515 |
| promoter_tata | 0.8308 | 0.854 | | H2AFZ | 0.4919 | 0.523 |
| promoter_no_tata | 0.7305 | 0.749 | | H3K4me1 | 0.4713 | 0.490 |
| promoter_all | 0.7286 | 0.743 | | enhancers_types | 0.4688 | 0.483 |
| H4K20me1 | 0.6224 | 0.616 | | H3K27ac | 0.4622 | 0.488 |
| H3K4me3 | 0.6143 | 0.676 | | H3K9me3 | 0.3939 | 0.402 |
| H3K36me3 | 0.5757 | 0.614 | | **mean (18)** | **0.6195** | — |
| H3K27me3 | 0.5736 | 0.587 | | | | |

The **3-stage HP grid** (`carbon-raw-original`, mse [0,1]→[0.25]→[2,5]) is now running to optimize each
task; stage-1 best-per-task will be appended here. Vanilla is a floor — the grid only goes up.

### 2026-06-24 — splice from-scratch (within this repo, distill path) + "is distillation helping splice?" (200 epochs)

The two splice tasks were flagged as low (acceptor table 0.788, donor 0.627). Pulled the **from-scratch**
(`ce0.5 / kl0 / mse0`, no teacher) runs already present in the 200-epoch grid (no rerun needed), and
compared against the val-selected best **distill** config. All seed-42 (the HP-search seed):

| task | from-scratch 0.5/0/0 (test · val) | best distill (config) test | does distill help? |
|---|:-:|:-:|:-:|
| splice_sites_acceptors | **0.8126** · 0.8349 | **0.8233** (0.5/0.5/0/1.5) | +0.011 (marginal) |
| splice_sites_donors | **0.6313** · 0.6527 | **0.6906** (0.5/0.5/0/0.5) | +0.059 ✅ |

- **Acceptor reproduces ~0.81–0.83** from scratch at 200 ep — the table's 0.788 was the conservative
  **3-seed mean** (seeds 0/1/2), which sits *below* the seed-42 grid peak (0.8233): splice has **high
  seed variance** for a 0.12M student. From-scratch 3-seed (0/1/2) launched (jobs 241908–241913) for a
  matched mean±std; seed-42 already in grid.
- **Donor caps at ~0.63 from scratch even at 200 ep** (val 0.6527, converged). A 0.12M BPNet **cannot
  reach 0.85** on donors — that target is a larger-model/CNN number, not this student's ceiling.
  Distillation *does* lift donor (+0.059 → 0.69), the largest distill gain among splice tasks.
- **kl=0 × temperature is GPU-nondeterminism, not a T effect** — confirmed empirically here: the 5 T
  values at kl=0 collapse into **2 bit-identical clusters** (acceptor: T∈{0.5,1.5}=0.8126 vs
  T∈{1,2,4}=0.7960; same `best_epoch` within a cluster), exactly the cuDNN run-to-run pattern. A real
  T effect would give 5 distinct monotonic values. This is why `gen_hp_specs.grid_combos` keeps only one
  canonical T per kl=0 family (1440→1152 runs; `test_grid_kl0_temp_skip`).

**Infra notes this round (cluster contention 6/21–6/23):**
- Fair-share bottomed out (RawUsage ~95M, factor ~1e-4) from the 1440-run campaign → long Priority-pending
  stalls. Caps raised to LAN 8 / VOY 4 / GAL 5. Jobs right-sized 8→5 CPU, 96G→24G, 12h→3h so they
  **backfill** into short gaps (the old 12h/96G footprint was locked out of every gap). Grid scans run on
  galaxy-local disk (`slurm/grid_progress.py`) to avoid sshfs-walk timeouts.

## GOAL-B status (2026-06-26): ORIGINAL-model 3-stage HP search — STAGE 1 only, in progress
**Critical correction (user-flagged):** the prior `carbon_grid_results.csv` (1546 rows) is the **deploy_120k**
(dilation-CAPPED) student, NOT the original model. Rebuilt for the **original** model →
`results/carbon_grid_results_original.csv` (324 runs) via `collate_runs.py --base .../original`.
**Original model coverage = STAGE 1 ONLY** (weight_mse ∈ {0,1}; 11 of 18 tasks, some partial); **stages 2
(0.2) and 3 (2,5) NOT yet run** on original. The complete 3-stage search exists only for deploy_120k (wrong model).
**Generated CSVs (new filenames, model-tagged):**
- `results/carbon_hp_stage1_3_original.csv` (+ `_best.csv`) — base config (mse 0,1,2,5); currently == stage-1
  only since stages 2/3 absent for original.
- `results/carbon_hp_stage1_2_3_original.csv` (+ `_best.csv`) — full (adds 0.2); IDENTICAL to above until
  stage 2 runs. Generator: `slurm/gen_stage_csvs.py --grid <csv> --tag original`.
**To FINISH GOAL-B (remaining work):** (1) complete original stage 1 (auto-submitter running `hp_original_stage1.txt`,
574 jobs); (2) submit original **stage 2** (weight_mse 0.2) + **stage 3** (weight_mse 2,5); (3) regenerate the two
CSVs; (4) 3-seed the best base-config params/task + any stage-2-improved params. deploy_120k CSVs kept for
reference but are the capped model.

## GOAL-B on rented vast.ai H100 — laggard-task batch mode (2026-06-28)
**Setup:** the 7 stage-1 laggard tasks (the lab finished 11/18) are run on a rented 192-core H100 box via
**batch mode** (`distill_task.py`: teacher loaded ONCE per task, then loops the ~32 HP configs — vs the lab's
old per-config submitter that reloaded the 3B teacher every config). Box dirs are namespaced (`/root/carbon_out`,
`/root/batch_<task>.log`) so they never collide with the lab's `$SSD/carbon_distillation/`. Teacher =
`HuggingFaceBio/Carbon-3B` + per-task LoRA adapter; student = BPNet-original (317K). WANDB_MODE=offline.

**ROOT-CAUSE FINDING (why the 7 laggards lagged everywhere, lab AND H100):** it was **NOT** GPU starvation —
**5 of the 7 laggard tasks had missing/incomplete Carbon-3B LoRA teacher checkpoints** on the box. The seeding
rsync was truncated (~128M arrived = only H3K4me1 + H3K27ac complete; `splice_sites_all/acceptors/donors` +
`enhancers_types` had EMPTY teacher dirs; `enhancers` had only README+adapter_config, no `adapter_model.safetensors`).
With no teacher, `find_teacher_checkpoint`→None → `prepare_task` returns None → batch exits `BATCH_EXIT=0` with
**0 epochs, 0 students** (a silent no-op, no error logged). So those tasks never trained. **Fix:** pulled the 5
complete adapters (98MB each) from HF `explcre/carbon-3b-lora-teachers-nt18` directly onto the box via
`snapshot_download(allow_patterns=['{task}_finetuned/*'...])`, verified `adapter_model.safetensors` present per
task, relaunched. All 7 batches now train with real teachers. **Lesson:** always validate `adapter_model.safetensors`
exists per task before launching a carbon batch (added to the teacher-ckpt memory note).

**PERF FINDING — carbon student training is CPU/Python-bound, not GPU-bound.** On the H100 box: GPU util **0%**,
15GB mem (just the 2 resident teachers), 192 cores but only ~7 in use (one config per task-batch, serial within a
task). The tiny BPNet student trains on a single core; per-epoch wall time on the H100 box is **~5× slower per
config than a lab GPU node** (lab ~20–37 min/config vs H100 ~2–3 h/config) purely from slower per-core throughput —
the GPU sits idle on both. Running 7 task-batches serially uses ~4% of the rented box. At that rate stage-1 across
7×~32 configs would take days.

**SPEEDUP — config-parallel batch mode (implemented + tested, deploying):** added an opt-in `--parallel K` to
`distill_task.py` / `_distill_task_batch_parallel` in `distill.py`. Design: run config 0 once with the live teacher
to warm the HP-independent teacher caches (logits/features + `teacher_evaluation.json`, keyed by
`md5(X_train,teacher_ckpt,max_len)`, NOT by HP), then **free the 3B teacher** (`del`+`empty_cache`), serialize the
`TaskContext` tensors (teacher_hidden, num_labels, data splits — everything except the live model) to
`{out}/_parallel_ctx_cache/`, and fan the remaining configs out to K **spawn** workers (NOT fork — fork-after-CUDA
corrupts the context). Each worker loads its own ctx copy from cache (a `_NoOpTeacher` stub stands in for the
freed model; raises if ever invoked), so dozens of tiny students share the idle GPU and saturate the 192 cores.
**Faithful:** per-config `set_seed` + identical teacher-cache values → results match serial; only concurrency
differs (first config is serial for cache-warmup, speedup applies to configs 1..N-1). Default `--parallel 1` is the
byte-identical audited serial path. **Tests:** 86/86 pass (existing 60 + new `test_distill_task_parallel.py` 26:
serial-path-at-1, ctx-cache round-trip, K-way shard/collect, worker error isolation, teacher-freed-before-pool, CLI
parse). Plan to deploy with `--parallel 8` per task. **No numeric distill results from the box yet** — 0 students
finished at the time of this entry (the teacher blocker had stalled 5/7 tasks; H3K4me1/H3K27ac were at ~32 epochs,
first configs not yet early-stopped).

## ★ GOAL-B carbon ON vast.ai H100 — WORKING recipe [2026-06-28]
After a long fight, the box runs the HP search correctly. The winning recipe (all needed together):
1. **skip-load**: teacher logit/feature caches transferred from lab ($SSD/data/cache/carbon_3b_lora) +
   metadata `cache_key` rewritten to the BOX teacher path (`_compute_cache_key` hashes len+first+last-seq+
   teacher_path+max_len). `prepare_task` then skips the 3B load entirely (`teacher cache valid — skipping`).
   ZERO teacher loads = no 10-min LoRA-merge, no thrash.
2. **bash-parallel, NOT the Python spawn pool**: each config is its own `python -m src.train.distill <line>`
   (single-config entry, also skip-loads). Run via a `wait -n` semaphore at N=48 (run_all_v3.sh). The
   config-parallel spawn pool fork-bombed (pids.max=**5888**; torch default = 1 thread/core = 192/worker).
3. **OMP_NUM_THREADS=1** (+ MKL/OPENBLAS) → each worker ~1 thread not 192 → 48 workers ≈ 48 cores, pids safe.
4. **launch inside tmux with a MINIMAL command**: the box's ssh is so slow that complex commands
   (pkill+sleep+launch) time out before completing; a bare `tmux new-session -d -s carbon_run "bash
   run_all_v3.sh 48"` returns instantly and persists past ssh disconnects. Monitor with minimal grep cmds.
Result: 48 concurrent configs, skip=47/48, teacher-loads=0, fork-errors=0, GPU ~61% util. 222 laggard configs
churning. Code on `precompute_logits_cache` (jhliu17/OmegaGenome). Outputs at /root/carbon_out -> rsync to
lab $SSD/carbon_distillation/original/ then `collate_runs.py --base .../original` refreshes the CSV.
**Latest CSV refresh: 413 runs (was stale at 324); stage1+2 done for most; STAGE 3 (wmse 2,5) = 0 runs (gap).**

---
### 2026-06-28 — Box throughput root-cause + N retune + per-step debug-sync fix

**Why the box looked "stuck" (done=0 for many hours):** NOT a bug. Two compounding causes:
1. **64-way oversubscription THRASH.** 64 configs time-sliced on ONE H100: ~59/80 GB + 64 CUDA contexts
   context-switching → pathological slowdown, and 200-epoch lockstep means NO completions until the whole
   wave nears the end. Extrapolated ~22 days. Killed it.
2. **Single config is overhead/IO-bound, not GPU-bound.** Measured SOLO: ~8.4 s/epoch at only **31% GPU**.
   The BPNet student is tiny; per-epoch cost is dataloader (num_workers=0) + val-eval (early-stop) +
   per-epoch checkpoint save + per-step `.item()` syncs — GPU math is trivial. So one config can't be made
   much faster without changing the recipe (batch size → would break lab-comparability).

**Fix that actually matters = PARALLELISM tuned to GPU saturation.** One config ≈31% GPU → ~3 saturate it.
Re-tuned 64 → **N=6**: GPU pinned **99–100%**, mem ~6 GB (no thrash). Measured completion rate (new code):
first wave at ~30 min, then **~12–15 configs/hour**. **Stage-1-laggard ETA ≈ ~14 h** for all 187 (vs
effectively-never at 64-way). run_all_v3.sh arg = N; launched `bash run_all_v3.sh 6` in tmux.

**Code fix (commit a06935f, deployed to box + lab):** `distillation_loss`/`kl_term` built debug f-strings
whose `.item()`/`.tolist()` + `per_sample()`×2/`abs().mean()`/`torch.unique()` were evaluated EAGERLY every
step even with `DISTILL_DEBUG=0` (default). Guarded the block behind `if DEBUG:` (loss math hoisted above,
unchanged). **Result-preserving: verified bit-identical (loss/metrics/grad) DEBUG on-vs-off and vs an
independent reference across vanilla/logit_standard × {kl,mse}** (test_distill_guard.py, all PASS).
HONEST magnitude: a contended micro-bench showed 15×/step, but that was inflated by GPU contention (syncs
~600 µs contended vs ~20 µs solo); real SOLO gain was only 8.4→7.4 s/epoch (~12%). It helps the *parallel*
(contended) regime modestly and is harmless. Box file md5-verified == lab before deploy.

**Takeaway:** box throughput is parallelism-bound (N=6 = sweet spot), ~14 h for stage-1 laggards. The
**critical path remains the LAB (stage 3, ~720 configs)** — the box is a bonus engine, not the bottleneck.

**N retune follow-up [2026-06-28]:** Tested N=16 vs N=6. N=16: GPU 100%, mem 14.9GB, but 0 completions in 24min
(vs N=6's 3 by 30min) — at 16-way each config is ~5x slower than solo so all 16 finish in one late burst.
Steady-state throughput **identical (~12/hr)** because the GPU is already TRULY saturated at N=6 (~3 configs).
Higher N only worsens completion latency + memory, no throughput gain. **N=6 is the ceiling; reverted to it.**
Box is at its throughput limit (~12-15 configs/hr, ~14h stage-1 laggards); no further parallelism lever exists.

**N-sweep (measured aggregate epochs/min) [2026-06-28]:** N=6→15.3, **N=8→16.4 (peak)**, N=10→16.1, N=12→15.1.
Throughput PLATEAUS flat (~15-16) across N=6-12 and declines past 8 (oversubscription) — the GPU is genuinely
SATURATED, not overhead-bound. **Set N=8 (the measured optimum, +7% over N=6).** Because throughput is flat
(adding configs doesn't raise the ceiling), per-config sync-removal (deferring the 4 metric .item()s +
total_loss) would NOT raise the ceiling either — the GPU is the limit. Box ceiling ≈16 epochs/min ≈ ~12-15
configs/hr; no further result-preserving speedup exists (batch-size/torch.compile/AMP would change numerics →
break lab-comparability, so off-limits). Box ETA ~13h for stage-1 laggards. Relaunched at N=8.

**torch.compile test [2026-06-28] — NEGATIVE.** Added env-gated `torch.compile(model)` (CARBON_COMPILE=1,
default off) and ran a compiled solo on the box. Result: **crashes in the Inductor backend** —
`torch/_inductor/compile_fx.py: AttributeError: 'NoneType' object has no attribute 'data'` — and sat >9 min
in compilation before failing (never reached epoch 1). So on this BPNet + box torch, compile is non-functional
AND carries huge per-process compile overhead. Reverted the flag (box+lab back to clean HEAD). Combined with the
N-sweep saturation result, this closes the optimization search: **no result-preserving speedup beyond N=8 exists**
(compile broken; sync-removal can't move a saturated ceiling; batch/AMP change numerics → off-limits). Box
ceiling = ~16 epochs/min ≈ ~13h for stage-1 laggards at N=8.

---
### 2026-06-29 — De-duplicate box vs lab + box→STAGE 3 (the gap) + accelerate stage 1

**Found duplication:** a 4-day background submitter (`auto_submit_specs.sh hp_original_stage1.txt`, PID 478193) was
running the lab's STAGE-1 grid (galaxy), AND the box was running stage-1 laggards — the SAME configs (e.g.
enhancers wmse0/1). Wasted compute + duplicate CSV rows.

**Resolution — disjoint engines:**
- **Lab submitter → stays on STAGE 1** (resume-aware, ~72-90% done). Caps raised (user-authorized) in
  `submit_caps.env`: LAN 2→7, VOY 0→4, GAL 4→6 (count_node is total-aware → respects GPU counts, no
  oversubscribe) → stage-1 finishes much faster across all 3 nodes.
- **Box → STAGE 3** (wmse{2,5}, the ~720-config gap nothing was filling). Generated `hp_original_stage3.txt`
  (576 configs = 18 tasks × kl{0,.25,.5,1} × mse{2,5} × T, kl=0 collapses T), resume-aware (0 done).
- **Enabled box for all 18 tasks:** box only had 7 tasks cached. Copied the 11 missing tasks':
  (a) logit+feature caches (3.1GB) → rewrote cache_key to box teacher path (fix_keys_11.py, 11/11 ALIGNED);
  (b) teacher dirs MINUS the 103MB safetensors (skip-load never loads weights) → rewrote the 11
  teacher_evaluation.json paths to /root. Verified: H2AFZ (new task) "✓ Loaded teacher outputs from cache",
  0 "No teacher checkpoint" skips, GPU 100%. Box now runs full 18-task stage-3 @N=8.

**Clean recording (no overwrite):** box stage-1 results (72) collated to a SEPARATE
`hp_search_tables/carbon_grid_results_box_stage1_20260629.csv` (canonical CSVs untouched). Box stage-3 results
will rsync into the canonical `original/` tree (new unique configs → no replacement) where the resume-aware
tooling sees them, so the lab submitter never re-submits box work. **No duplication; everything recorded cleanly.**

**Submitter resume-fix [2026-06-29]:** the lab submitter's specs file was the FULL 562-config stage-1 grid
(never resume-filtered), so it would re-run 468 already-done configs — incl the box's 72 stage-1. Fix:
(1) merged box `original/` results into the canonical lab `original/` tree (done_combos reads there);
(2) regenerated resume-filtered list = **108 truly-remaining** (drops 454 redundant re-runs); (3) restarted the
submitter on the 108-config file (backup: hp_original_stage1_full.bak). Submitter now skips all done (box+lab).
No per-config skip exists in the training path — resume is ONLY at specs-gen time, so the specs file MUST be
regenerated to reflect done work. Box stage-3 results also flow into `original/` → future regen skips them too.

**Run-time resume-skip added [2026-06-29, commit d6d0aac]:** root-caused the 454 re-runs to a MISSING
defensive guard — resume was generation-time only (gen_hp_specs.done_combos); the run path had NO per-config
skip, and output dirs are randomized (timestamp+uuid) so nothing could stat a fixed path. Added opt-in
`_config_already_done` + `CARBON_SKIP_IF_DONE=1` gate at the top of train_student: globs output_dir/<task>/**
for a final_summary.json matching the FULL config identity (task,ce,kl,mse,temp,random_state) and skips if found.
Opt-in/default-off (no behavior change); seed-aware (never skips 3-seed re-runs); l2norm-excluded; output_dir-scoped
(matches done_combos). Subagent-audited, 20/20 tests pass. Enabled on box run_all_v3.sh (verified LIVE: 8 done
stage-3 configs skipped on restart) + run_carbon_pipeline.sh grid phase (NOT 3-seed). Box+lab now resume-safe
against stale specs / restarts permanently.

---

## 2026-07-21 — Rebuttal landed results: classification finalized, KD-method sweep, size sweep, BPNet baseline; NTv3 regression ladder IN PROGRESS

Consolidated record of the rebuttal experiments that have completed since 2026-06-29. All numbers below are copied
verbatim from the authoritative source files cited per subsection (no rounding beyond what the source shows). The
NTv3 regression size ladder (F/E) is **explicitly IN PROGRESS — no final regression numbers exist yet.**

### A. Classification rerun finalization — DNABERT-2 @200ep 3-seed + below-baseline reruns
**Source:** `plot_repo/data/model_comparison_5teacher_formal.csv` (Type=Student rows; Type=Baseline BPNet rows).

The 18-task 3-seed reruns (DNABERT-2 at 200 epochs, plus the students that had sat below the from-scratch BPNet
baseline) are finalized. Net effect: the count of student cells below the BPNet baseline dropped **14 → 11**
across the 5 teachers. Per-teacher below-BPNet counts now (18 tasks each):

| student | 18-task mean MCC | cells below BPNet |
|---|:-:|:-:|
| Distilled Nucleotide Transformer | 0.6271 | 0 / 18 |
| Distilled Enformer | 0.6202 | 1 / 18 |
| Distilled Carbon-3B | 0.6197 | 2 / 18 |
| Distilled Caduceus | 0.6084 | 4 / 18 |
| **Distilled DNABERT-2** | **0.6067** | 4 / 18 |
| — total below | — | **11** |

- **DNABERT-2 18-task mean = 0.6067** (~0.607). Its 4 remaining below-BPNet cells: H3K27me3 (0.5493 vs 0.5497),
  H3K4me3 (0.6119 vs 0.6238), H3K9ac (0.5026 vs 0.5213), H4K20me1 (0.6053 vs 0.6061) — all marginal/within noise.
- These reruns are what **paper Table S3 and Figs 2/4** now reflect.

### B. KD-method comparison — 18-task, NT-2.5B teacher, 3 seeds, method-appropriate HP
**Sources:** `plot_repo/data/method_comparison_18task.csv` (aggregates); diagnosis
`code_carbon/rebuttal_infra/dist_ls_collapse_diagnosis.md` (2026-07-19); root cause
`code_carbon/rebuttal_infra/h3k9me3_rootcause.md` (2026-07-20).

18-task mean MCC (± across-task sample s.d.), best method per task counted as a "win":

| method | 18-task mean ± across-task s.d. | wins (best on task) |
|---|:-:|:-:|
| **OmegaGenome** | **0.6271 ± 0.1549** | **10** |
| DKD | 0.6218 ± 0.1499 | 4 |
| DIST | 0.5932 ± 0.1737 | 2 |
| LS | 0.5879 ± 0.1581 | 2 |

**Key finding — DIST/LS "collapse" on binary tasks is a 2-class degeneracy artifact, NOT a bug and NOT true
method performance.** With OmegaGenome's KL-heavy vanilla HP (`weight_kl ≥ weight_ce`), the DIST inter-class
term (per-sample Pearson corr of a 2-vector = exactly ±1, a singular sign-step gradient at p=0.5) and LS's 2-logit
standardization (erases teacher confidence) trap the student in a constant-prediction basin (`val_mcc=0.0000`,
`final_test_f1=0.3333`; no NaN/Inf). Implementations verified **faithful** to the official references (DIST
hunto/DIST_KD, LS sunshangquan/logit-standardization-KD) — no correctness bug. The catastrophic magnitude is an
**HP-mismatch artifact**: vanilla-tuned HP applied to un-tuned methods.

**Recovery (method-appropriate kl):** all **9 collapsed cells recovered** with a per-method kl re-tune — **7 cells
@ kl=0.5** and the **2 lowest-signal H3K9me3 cells @ kl=0.25** (tagged in the CSV `hp_note` column):

- kl=0.5 fix (7): DIST {H3K27ac, H3K27me3, H3K4me1, enhancers, splice_sites_acceptors}; LS {splice_sites_acceptors, splice_sites_donors}.
- kl=0.25 recovered (2): DIST H3K9me3 (0.3850 ± 0.0476), LS H3K9me3 (0.3530 ± 0.0256).

**H3K9me3 root cause = lowest-signal task, NOT class imbalance.** H3K9me3 is **exactly 50/50 balanced** (13,719/13,719,
parsed from the raw dataset), ruling out imbalance. It is the single **lowest-signal task in the whole suite** —
NT-2.5B teacher MCC 0.4636 (lowest), Carbon-3B teacher 0.3502 (lowest), BPNet baseline 0.3295 (lowest), OmegaGenome
student 0.4454 (lowest) — because it marks broad, repeat-rich constitutive heterochromatin with no sharp local motif.
Escaping the KD basin requires the CE (true-label) gradient to overpower the degenerate KD pull; H3K9me3 has the
weakest label separability of all 18 tasks, so it clears the escape threshold only at the lower kl=0.25. This turns
H3K9me3 from an outlier into a controlled worst case: DIST/LS fragility is **signal-dependent and predictable**, and
OmegaGenome/DKD are robust across all 18 tasks under a single untuned HP setting.

### C. Classification student-size sweep (Fig 6B) — 18-task mean, single-seed (seed 0)
**Sources:** `plot_repo/data/size_18task_seed0_matrix.csv`; diagnosis
`code_carbon/rebuttal_infra/size_monotonicity_diagnosis.md`.

7-point student-size curve (18-task mean MCC, seed 0):

| size | 2K | 7K | 28K | 0.1M (deployed) | 0.8M | 1.8M | 3.6M |
|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| 18-task mean MCC | 0.5214 | 0.5680 | 0.6073 | **0.6289** | 0.6255 | 0.6340 | 0.6383 |

- **Plateau:** for **13/18 tasks `|Δ(3.6M − 0.1M)| < 0.02`** — the deployed 0.1M student is at/near the capacity
  plateau; scaling 36× to 3.6M lifts the 18-task mean only 0.6289 → 0.6383 (+0.0094).
- **Diagnosis (read-only, nothing rerun):** the mild non-monotonicity (e.g. the 0.1M→0.8M dip) is **single-seed
  noise** — pooled seed-to-seed σ = 0.0125 MCC (σ_adj = 0.0177), and 35/37 adjacent-size drops are within 2σ; the
  median drop (0.005) is smaller than one seed's typical wobble (0.012–0.014). Confirmed **best-VALIDATION-selected**
  checkpoints (no test peeking; `best_test_mcc ≥ final_test_mcc` in only 61.7% of runs — the correct signature of
  honest best-val, not leakage). **Early stopping is NOT involved** (`--early-stop-patience 0` = disabled; all 149
  runs ran the full 200 epochs, `early_stopped=False`). Secondary structured effects: a 0.1M "original"
  provenance/splice discontinuity (0.1M spliced from a separate original-student run set, sits slightly high) and
  fixed-HP diminishing returns at the large end. 3-seed means will remove essentially all wobbles.

### D. BPNet from-scratch baseline — 3-seed
**Source:** `code_carbon/rebuttal_infra/bpnet_baseline_comparison.md` (runs at `/srv/disk00/sshfs/pengchx3/bpnet_scratch_3seed/`).

3-seed from-scratch BPNet, 200 epochs, best-val-selected: **18-task mean = 0.6085** (vs current single-seed paper
baseline **0.5882**, Δ = **+0.0203**). Direction: 16/18 tasks up, 2 marginally down (H3K4me3 −0.0126, H3K9ac −0.0170,
both within seed std). Biggest movers: H3K9me3 +0.0810 (single-seed 0.3295 was an unlucky n=1 low draw; all three
3-seed runs 0.377/0.423/0.432 sit above it), splice_sites_all +0.0700. 100ep best-val = 0.6040 (best-val is
budget-invariant: bit-identical to 200ep on 14/18 tasks). No collapsed/failed runs (min best_test_mcc 0.377). The
current single-seed paper value already tracks best-val (mean|single − 200best| = 0.0236 vs |single − last-epoch| =
0.0879). **Recommendation = adopt the 200ep 3-seed best-val (mean±std) as the BPNet baseline** (like-for-like n=1→n=3
upgrade, now with an honest error bar). **NOT yet swapped into the paper tables** per user.

### E. NTv3 regression size ladder — COMPLETE ORGANIZED MASTER RECORD ✅ (all 20 tiers, 2026-07-25)
**Sources (all numbers below re-verified against these artifacts; final 8 cells re-verified 2026-07-25, the
rest 2026-07-24):** per-run
`ntv3_finetune_result.json` under `/srv/disk00/sshfs/pengchx3/ntv3_targets/size_sweep/` (joint arms),
`.../size_sweep_pertrack/t12_ENCSR325NFE/` (per-track arm), and the reference runs
`.../ntv3_ft_faithful_s{0,1,2}`, `.../ntv3_8m_baseline`, `.../ntv3_100m_baseline`, `.../ntv3_8m_kd_dist`.
Design + audit: `code_carbon/rebuttal_infra/ntv3_size_ladder_design.md`, `.../ntv3_size_audit.md`.

#### E.0 — Overview
This ladder fills out the **regression** (34-track per-bp bigWig, NTv3-650M teacher) size-scaling curve to
match the classification one. It runs **four from-scratch arms** across five sizes, plus reference points,
all **random-init `from_config`** students from the `NTV3_SCALED_SIZES` registry (param counts verified by
CPU instantiation), trained with the **same recipe** (seed 0, 19,932 steps / 598 warmup, seq_len 32768,
`poisson_multinomial` GT loss) and **best-val checkpoint selection via the repo's own `finetune_ntv3.py`**
(test is evaluated on the best-val checkpoint, never the last epoch):

1. **Joint scratch+KD** — random-init + cached NTv3-650M joint-34 teacher logits (`w_ce=0.5`/`w_kl=0.5`/
   `w_mse=0`); the from-scratch distilled curve.
2. **Joint no-KD (labels-only, data-matched)** — random-init trained on the ground-truth bigWig labels
   with **zero teacher** (`w_ce=1.0`/`w_kl=0`/`w_mse=0`), on the **identical training stream** as arm 1
   (matched ablation → only the teacher differs).
3. **Per-track t12 specialist (KD)** — random-init + KD on a **single** track (ENCSR325NFE ATAC-seq, joint
   channel 12); same recipe/seed.
4. **Per-track t12 no-KD (labels-only, data-matched)** — random-init trained on the single-track
   (ENCSR325NFE) ground-truth bigWig with **zero teacher** (`kd_w_ce=1.0`/`kd_w_kl=0`/`kd_w_mse=0`), on the
   identical 63,707-window stream as arm 3 → the matched specialist ablation (teacher signal ON vs OFF on
   one track). Sbatch `slurm/ntv3_size_sweep_pertrack_t12_nokd.sbatch` (commit `6fd7970`).

**Reference points (not part of the from-scratch arms):** the **650M teacher** (`ntv3_ft_faithful`,
pretrained full-FT, 3-seed ≈ 0.606) and the two **pretrained-init** native tiers `ntv3_8m_baseline` (0.475)
and `ntv3_100m_baseline` (0.518) — InstaDeep publishes NTv3 checkpoints only at 8M/100M/650M, so pretrained
references exist only at those sizes.

**Scientific questions this ladder answers:**
1. Does **from-scratch + KD** scale up to (and match) a **pretrained-init** student at matched size?
2. What does **KD add at each size** — the clean matched no-KD ablation (teacher signal ON vs OFF, data held
   identical)?
3. Does a **per-track specialist** beat the **joint model's slice** on the same track, and how does that gap
   move with capacity?

**Tiers + verified param counts** (CPU-instantiated; native tiers reproduce the true 8M/100M/650M):

| tier | embed / layers / heads / ffn / key | measured params | provenance |
|---|---|:-:|---|
| ntv3-4m | 192 / 2 / 6 / 768 / 32 | 4.34M | random-init |
| ntv3-8m | 256 / 2 / 8 / 1024 / 32 | 7.69M | random-init (native ckpt exists → used as pretrained ref) |
| ntv3-30m | 448 / 4 / 8 / 1792 / 56 | 29.87M | random-init |
| ntv3-100m | 768 / 6 / 12 / 3072 / 64 | 106.46M | random-init (native ckpt exists → used as pretrained ref) |
| ntv3-300m | 1152 / 9 / 18 / 4608 / 64 | 303.05M | random-init |
| ntv3-650m (teacher) | 1536 / 12 / 24 / 6144 / 64 | 651.83M | pretrained full-FT |

#### E.1 — MASTER RESULTS TABLE (all arms × all sizes, seed 0)
Cells are seed-0 **test** mean Pearson with **(best-val)** in parentheses; all 20 tiers landed (no pending).
4dp cells use round-half-up; deltas are computed from full-precision values, not from the rounded cells.

**(1) Joint arms (34-track mean Pearson) + KD ablation + pretrained-init reference**

| size | params | joint scratch+KD | joint no-KD (matched) | **KD Δ** (KD − no-KD) | pretrained-init ref |
|---|:-:|:-:|:-:|:-:|:-:|
| 4M | 4.34M | 0.4606 (0.4857) | 0.4349 (0.4614) | **+0.0257** | — |
| 8M | 7.69M | 0.4769 (0.5031) | 0.4483 (0.4781) | **+0.0286** | 0.4748 (0.5010) |
| 30M | 29.87M | 0.4958 (0.5264) | 0.4681 (0.4909) | **+0.0277** | — |
| 100M | 106.46M | 0.5188 (0.5494) | 0.4848 (0.5140) | **+0.0341** | 0.5178 (0.5441) |
| 300M | 303.05M | **0.5309 (0.5572)** | 0.4969 (0.5283) | **+0.0341** | — |
| **650M teacher** | 651.83M | **0.606** (3-seed 0.6059/0.6071/0.6061) | — | — | pretrained full-FT |

**(2) Per-track t12 = ENCSR325NFE specialist (single-ATAC-track Pearson) vs the joint model's slice**
⚠ **Footnote — NOT level-comparable to the joint columns above:** these are a **single ATAC track**
Pearson, not the 34-track mean; the specialist columns are included only so the two single-track arms sit
in one view.

| size | specialist KD | specialist no-KD (matched) | **per-track KD Δ** (KD − no-KD) | joint ENCSR325NFE slice | Δ (specialist − joint slice) |
|---|:-:|:-:|:-:|:-:|:-:|
| 4M | 0.7322 (0.7488) | 0.7160 (0.7379) | **+0.0162** | 0.6882 | **+0.0440** |
| 8M | 0.7362 (0.7672) | 0.7183 (0.7389) | **+0.0179** | 0.7060 | **+0.0302** |
| 30M | 0.7521 (0.7704) | 0.7396 (0.7660) | **+0.0126** | 0.7255 | **+0.0266** |
| 100M | 0.7631 (0.7830) | 0.7448 (0.7725) | **+0.0183** | 0.7472 | **+0.0159** |
| 300M | 0.7681 (0.7907) | 0.7562 (0.7795) | **+0.0119** | 0.7585 | **+0.0095** |
| **650M teacher slice** | — | — | — | 0.832 (3-seed 0.8326/0.8327/0.8322) | — |

**per-track KD Δ** = single-track KD − single-track no-KD (matched 63,707-window stream), computed from
full precision (see E.2.4). Reference-point slices for context: pretrained `ntv3_8m_baseline` ENCSR325NFE
slice = 0.6850, `ntv3_100m_baseline` slice = 0.7379.

#### E.2 — PER-ARM DETAIL (protocol, config, per-tier numbers)

**Shared recipe (all three arms):** random-init `from_config` (`NTV3_SCALED_SIZES`); **seed 0**;
**19,932 steps** training / **598 warmup**; **seq_len 32768**; GT loss `poisson_multinomial`
(`multinomial_weight=5`, `cwd_temperature=4`); `train_overlap=0.0`; `keep_target_center_fraction=0.375`;
`mini_batch_size=4 × grad_accum 8`; LR 1e-5→5e-5; best-val selection (`best_model.pth`), test on best-val
checkpoint. Distillation/validation/checkpoint-selection all run through the repo's own `finetune_ntv3.py`
(`track_kd_loss`, `poisson_multinomial_loss`); our changes are additive (size registry, RAM/memmap teacher
cache, no-KD flags, data-match, mount guards).

##### E.2.1 — Joint scratch+KD (34-track) — ✅ COMPLETE (all 5 tiers)
**Config delta vs shared recipe:** `kd_w_ce=0.5`, `kd_w_kl=0.5`, `kd_w_mse=0.0`, `distill_loss=standardized_mse`,
`cached_teacher_logits` = joint-34 NTv3-650M cache (`teacher_cache_joint34`), `teacher_specialist=false`,
`track_subset=null` (all 34 tracks).

| tier | test | best-val | git_commit | run-dir (`.../size_sweep/…`) |
|---|:-:|:-:|:-:|---|
| ntv3-4m | 0.4606471669063459 | 0.48570857764609854 | `9299f63` | `ntv3-4m_scratch_kd_s0` |
| ntv3-8m | 0.4768708367942698 | 0.5031196294784239 | `9299f63` | `ntv3-8m_scratch_kd_s0` |
| ntv3-30m | 0.49582419754634266 | 0.5263541463030401 | `bb281a7` | `ntv3-30m_scratch_kd_s0` |
| ntv3-100m | 0.5188377911260873 | 0.549350867372519 | `9299f63` | `ntv3-100m_scratch_kd_s0` |
| ntv3-300m | 0.5309141744530618 | 0.5571820520990387 | `bae128d` | `ntv3-300m_scratch_kd_s0` |

**Per-assay test Pearson (from the JOINT 4M `result.json`, grouped from the 34 tracks):**

| assay | #tracks | 4M | 8M | 30M | 100M | 300M |
|---|:-:|:-:|:-:|:-:|:-:|:-:|
| ATAC-seq | 5 | 0.5665 | 0.5874 | 0.6206 | 0.6497 | 0.6655 |
| Histone ChIP-seq | 4 | 0.5626 | 0.5819 | 0.6021 | 0.6271 | 0.6404 |
| total RNA-seq | 3 | 0.4784 | 0.5134 | 0.5280 | 0.5494 | 0.5636 |
| eCLIP | 10 | 0.4421 | 0.4591 | 0.4731 | 0.4866 | 0.4950 |
| polyA plus RNA-seq | 2 | 0.4003 | 0.4407 | 0.4631 | 0.4832 | 0.4982 |
| PRO-cap | 10 | 0.3922 | 0.3936 | 0.4105 | 0.4403 | 0.4525 |

(Per-assay ordering — ATAC/histone easy, PRO-cap/eCLIP hard — is biologically expected and holds at every
tier. Full 34-track breakdowns are in each `result.json`; the 4M per-track list, for the record: ENCSR325NFE
0.688, ENCSR962OTG 0.828, ENCSR814RGG 0.665 the strongest; ENCSR114HGS_M/P ~0.26–0.27 the weakest.)

##### E.2.2 — Joint no-KD (labels-only) ablation — matched all 5 tiers ✅ COMPLETE
**Config delta vs E.2.1:** `kd_w_ce=1.0`, `kd_w_kl=0.0`, `kd_w_mse=0.0`, **no `--teacher`**, **no
`--cached_teacher_logits`** — the label loss runs at full weight and each `result.json` self-documents the
labels-only condition. Two variants; **do not conflate them**:

**(a) DATA-MATCHED (the headline ablation — only the teacher differs).** Reads the **identical
63,707-window** training stream as the KD arm at the same size (`train_overlap=0.0`; the no-KD `result.json`
records `n_train_windows=63707`), so init, data, steps, and schedule are all held fixed.

| size | scratch+KD (E.2.1) | scratch no-KD (matched) | **KD Δ** | git | run-dir |
|---|:-:|:-:|:-:|:-:|---|
| 4M | 0.4606 | 0.4349191411066653 (val 0.4613544748069352) | **+0.0257** | `bae128d` | `ntv3-4m_scratch_nokd_matched_s0` |
| 8M | 0.4769 | 0.4483069102471927 (val 0.47814321235315793) | **+0.0286** | `bae128d` | `ntv3-8m_scratch_nokd_matched_s0` |
| 30M | 0.4958 | 0.46807474264756777 (val 0.4909315548558588) | **+0.0277** | `bae128d` | `ntv3-30m_scratch_nokd_matched_s0` |
| 100M | 0.5188 | 0.48478547398375205 (val 0.5139648434186777) | **+0.0341** | `6fd7970` | `ntv3-100m_scratch_nokd_matched_s0` |
| 300M | 0.5309 | 0.49685810910058640 (val 0.5283428630936765) | **+0.0341** | `6fd7970` | `ntv3-300m_scratch_nokd_matched_s0` |

Deltas computed, not copied: 0.4606471669 − 0.4349191411 = **0.0257280258**; 0.4768708368 − 0.4483069102 =
**0.0285639265**; 0.4958241975 − 0.4680747426 = **0.0277494549** (→ +0.0277 at 4dp, NOT 0.0278 — single-round
of 0.0277495); 0.5188377911 − 0.4847854740 = **0.0340523171** (→ +0.0341); 0.5309141745 − 0.4968581091 =
**0.0340560654** (→ +0.0341). Each matched run verified `kd_w_ce=1.0`, `kd_w_kl=0.0`, `kd_w_mse=0.0`,
`teacher=None`, `n_train_windows=63707` (100M/300M jsons re-checked 2026-07-25: all four fields confirmed).
Note the 4dp display cells round half-up (100M test 0.4847854→**0.4848**, 300M 0.4968581→**0.4969**), matching
the round-half-up convention used throughout E.1/E.2; the KD Δ is computed from full precision, not the cells.

**(b) SUPERSEDED UNMATCHED run (kept for transparency — NOT the clean ablation).** The original no-KD 4M run
used `train_overlap=0.999` → a **65,051,340-window** stream (~1021× more diverse than the KD arm's
63,707-window stream), so it differs from the KD arm in **BOTH** teacher AND data diversity.

| size | scratch no-KD (UNMATCHED) | train_overlap / stream | scratch+KD | Δ vs KD | git | run-dir |
|---|:-:|:-:|:-:|:-:|:-:|---|
| 4M | 0.43931571217552756 (val 0.4595010140270073) | 0.999 / 65,051,340 | 0.4606 | +0.0213 | `9299f63` | `ntv3-4m_scratch_nokd_s0` |

The unmatched run had a data-diversity **advantage** yet still **LOST to KD by +0.0213**, so +0.0213 was a
conservative *lower bound* on the true teacher contribution at matched data. The now-landed **matched delta
(+0.0257) is indeed larger**, directly confirming that lower-bound reasoning. (The `_scratch_nokd_s0` dirs at
8M/30M/100M/300M hold checkpoint files but **no `result.json`** — those unmatched runs are superseded and not
reported.)

##### E.2.3 — Per-track t12 = ENCSR325NFE (ATAC-seq) specialist (KD) — all 5 tiers ✅ COMPLETE
**Config delta vs E.2.1:** `track_subset=12` (single track ENCSR325NFE), `cached_teacher_logits` =
`ntv3_cache/teacher_logits/gen/t12.pt` — the **byte-identical channel-12 slice** of the joint-34 teacher
cache (no re-forward of the 650M teacher; specialist and joint see numerically identical targets on this
track). Path scheme `size_sweep_pertrack/t12_ENCSR325NFE/`.

| tier | test | best-val | git_commit | run-dir |
|---|:-:|:-:|:-:|---|
| ntv3-4m | 0.7322410774399275 | 0.7487546384587935 | `9299f63` | `ntv3-4m_scratch_kd_s0` |
| ntv3-8m | 0.7361891537787985 | 0.7671725292651699 | `9299f63` | `ntv3-8m_scratch_kd_s0` |
| ntv3-30m | 0.7521458578345879 | 0.7704396083646101 | `bb281a7` | `ntv3-30m_scratch_kd_s0` |
| ntv3-100m | 0.7631253286436455 | 0.7829635845309715 | `bb281a7` | `ntv3-100m_scratch_kd_s0` |
| ntv3-300m | 0.768078039219381 | 0.7907219401730075 | `6fd7970` | `ntv3-300m_scratch_kd_s0` |

The 300M row landed 2026-07-25 (`result.json` written 19:41; test 0.768078039219381 → **0.7681** at 4dp
round-half-up, val 0.7907219401730075), closing the specialist KD ladder.

Joint slices `per_track_pearson["ENCSR325NFE"]` pulled + verified from the joint runs: 4M 0.6882138349565987,
8M 0.705974003088749, 30M 0.7254992718402737, 100M 0.7471823661275808, 300M 0.7585421291297799.

##### E.2.4 — Per-track t12 = ENCSR325NFE no-KD (labels-only, data-matched) — all 5 tiers ✅ COMPLETE
**NEW ARM.** The single-track counterpart of E.2.2: the matched no-KD ablation for the specialist, isolating
what the teacher adds on one track (ENCSR325NFE ATAC-seq) at each capacity. **Config delta vs E.2.3:**
`kd_w_ce=1.0`, `kd_w_kl=0.0`, `kd_w_mse=0.0`, **no `--teacher`**, **no `--cached_teacher_logits`** (each
`result.json` self-documents `"kd": null`, `teacher: null`); `track_subset=[12]` → **single track
ENCSR325NFE** (`n_tracks=1`, `track_ids=['ENCSR325NFE']`). Data-matched to the specialist KD arm:
`train_overlap=0.0` → **`n_train_windows=63707`** (identical stream; only the teacher signal differs).
Sbatch `slurm/ntv3_size_sweep_pertrack_t12_nokd.sbatch`, commit `6fd7970`. Path scheme
`size_sweep_pertrack/t12_ENCSR325NFE/ntv3-<size>_scratch_nokd_matched_s0/`.

| tier | test | best-val | git_commit | run-dir |
|---|:-:|:-:|:-:|---|
| ntv3-4m | 0.7160343804293897 | 0.7378933486018797 | `6fd7970` | `ntv3-4m_scratch_nokd_matched_s0` |
| ntv3-8m | 0.7182608479411978 | 0.7388721768919715 | `6fd7970` | `ntv3-8m_scratch_nokd_matched_s0` |
| ntv3-30m | 0.7395822453215777 | 0.7660014435196796 | `6fd7970` | `ntv3-30m_scratch_nokd_matched_s0` |
| ntv3-100m | 0.7448439142014617 | 0.7724969704651115 | `6fd7970` | `ntv3-100m_scratch_nokd_matched_s0` |
| ntv3-300m | 0.7562202260856036 | 0.7794590736694671 | `6fd7970` | `ntv3-300m_scratch_nokd_matched_s0` |

All five re-verified 2026-07-25 against `result.json`: `n_tracks=1`, `track_ids=['ENCSR325NFE']`,
`kd_w_ce=1.0`, `kd_w_kl=0.0`, `kd_w_mse=0.0`, `teacher=None`, `n_train_windows=63707` — matched to the
specialist KD arm at every size.

**Per-track KD Δ (specialist KD − specialist no-KD, from full precision, single-round 4dp):**

| size | specialist KD | specialist no-KD | **KD Δ** | full-precision diff |
|---|:-:|:-:|:-:|:-:|
| 4M | 0.7322410774399275 | 0.7160343804293897 | **+0.0162** | 0.0162066970 |
| 8M | 0.7361891537787985 | 0.7182608479411978 | **+0.0179** | 0.0179283058 |
| 30M | 0.7521458578345879 | 0.7395822453215777 | **+0.0126** | 0.0125636125 |
| 100M | 0.7631253286436455 | 0.7448439142014617 | **+0.0183** | 0.0182814144 |
| 300M | 0.768078039219381 | 0.7562202260856036 | **+0.0119** | 0.0118578131 |

⚠ **Precision note (verified, no fabrication).** These KD Δ's are single-rounds of the full-precision
differences (the doc-wide E.2.2 convention). Subtracting the *rounded* 4dp display cells instead would give
+0.0180 (8M: 0.7362−0.7182) and +0.0118 (300M: 0.7680−0.7562) — off by one in the last digit at those two
sizes because 8M no-KD (0.71826→0.7183) and the KD 300M (0.76807→0.7681) round up. The full-precision Δ's
above (+0.0179, +0.0119) are the correct ones.

#### E.3 — FINDINGS (with evidence and full revision history preserved)

**Finding 1 — from-scratch+KD ≈ pretrained-init at matched size; KD closes the pretraining gap.**
The from-scratch+KD tiers land on top of their pretrained-init baselines:
- **8M:** from-scratch+KD **0.4769** vs pretrained-8M baseline **0.4748** (`ntv3_8m_baseline`) — Δ ≈ +0.002.
- **100M:** from-scratch+KD **0.5188** vs pretrained-100M baseline **0.5178** (`ntv3_100m_baseline`) — Δ ≈ +0.001.

Distilling from the 650M teacher lets a *randomly-initialized* student match a *pretrained* one at matched
size — the pretraining bonus is fully recovered by KD on this 34-track regression.

**Finding 2 — the KD contribution (matched ablation) is ~~GROWS with size~~ ~~roughly constant at ~+0.027
across 4M–30M~~ flat (~+0.027) through 30M, then a step up to ~+0.034 at 100M–300M [COMPLETE, all 5 tiers,
2026-07-25].** With data matched (identical 63,707-window stream, teacher the only difference), the full
five-point KD-delta curve is **+0.0257 (4M) → +0.0286 (8M) → +0.0277 (30M) → +0.0341 (100M) → +0.0341
(300M)**. The shape is **flat-then-step**: essentially constant at ~+0.026–0.029 across 4M/8M/30M, then a
modest, clearly-separated lift to **+0.0341 at both 100M and 300M** (identical to 4dp). So KD's benefit is
**modestly larger at the top of this size range**, not smoothly growing and not flat throughout — the honest
read is the five measured values and the flat-then-step shape, **not** an extrapolated trend line. This is the
first *clean, complete* measurement of the teacher's contribution as a function of student capacity on the
34-track regression.

- **[REVISED 2026-07-25 — completed with the landed 100M/300M matched cells.]** This refines (does not
  contradict) the 2026-07-24 "~stable +0.027" read, which was **explicitly conditioned** on the then-pending
  100M/300M tiers ("whether the delta trends at all awaits the pending 100M/300M matched cells"). Those cells
  now land at **+0.0341 each** — above the +0.026–0.029 of 4M–30M — so the complete picture is flat through
  30M then a step up, not flat throughout. Deltas verified from full precision (0.5188377911 − 0.4847854740 =
  0.0340523171; 0.5309141745 − 0.4968581091 = 0.0340560654; both → +0.0341). Do **not** overclaim a smooth
  monotonic climb: 8M (+0.0286) sits above 30M (+0.0277), and the top two tiers are tied — it is a two-level
  step, reported as five measured values.

- **~~Superseded read — "the KD contribution GROWS with size".~~ [REVISED 2026-07-24]** Original text kept
  for the record: *"the KD contribution (matched ablation) GROWS with size. … KD adds +0.0257 at 4M and +0.0286
  at 8M."* That read came from **only the 4M→8M pair**; the landed 30M point (+0.0277) sits *below* 8M, breaking
  the monotonic-growth story. The three points show a ~stable +0.027 with **no clear trend** — and this is
  **not** a claim that the delta is proven flat either (three points cannot establish flatness), only that
  there is no monotonic climb.
- ⚠ **THIRD 2-point-extrapolation overreach on record (reinforces, not replaces, the caution under Finding 3).**
  This is now the third claim in section E where a partial-data extrapolation overstated: the per-track
  **plateau** claim and the ~100M **crossover** claim (both Finding 3, both overturned as tiers landed), and now
  the KD-delta **"grows with size"** read (overturned by the 30M point). Same lesson already on record under
  Finding 3: with 2–3 points per curve, trend extrapolation has no track record here — state measured pairs
  only, and nothing about a KD-delta trend goes in the paper ahead of the 100M/300M matched tiers that would
  actually test it.

**Finding 3 — the per-track specialist beats the joint slice at EVERY size tested, with DIMINISHING
returns; and both prior extrapolations from this arm were overturned.** Measured pairs (specialist − joint
slice): **+0.0440 (4M) → +0.0302 (8M) → +0.0266 (30M) → +0.0159 (100M)** — positive at all four sizes and
**shrinking monotonically** with scale. Both curves rise with capacity (specialist 0.7322→0.7362→0.7521→
0.7631; joint slice 0.6882→0.7060→0.7255→0.7472); the specialist stays ahead throughout 4M→100M and the gap
narrows. A crossover at *larger* scale (300M+) is **not excluded** — the Δ is still shrinking at the top of
the measured range — and the remaining test is the **pending per-track 300M vs joint-300M** pair (joint-300M
slice already landed at 0.7585). No crossover point is predicted here.

- **~~Superseded claim A — "per-track ATAC saturates early (plateau)".~~ [REVISED 2026-07-23 — SUPERSEDED]**
  Original text kept for the record: *"The specialist curve for ENCSR325NFE is nearly flat: 4M 0.7322 → 8M
  0.7362 (+0.004). ENCSR325NFE is a high-signal track that a 4M specialist already maxes out, so its
  per-track scaling curve is a plateau rather than a climbing law."* **Refuted by the 30M point:** 8M→30M is
  +0.0159 (an order of magnitude larger than the 8M step it was built on) — the specialist is still climbing.
- **~~Superseded claim B — "a specialist→joint CROSSOVER is expected at ~100M".~~ [REFUTED 2026-07-24 by the
  landed pt12-100M run]** Original text kept for the record: *"the joint model's ENCSR325NFE slice at 100M
  (0.7472) is already above the saturated specialist level (~0.736). If the specialist stays on its plateau,
  the two curves cross somewhere around ~100M."* The decisive test landed: **pt12-100M = 0.7631 > joint-100M
  slice 0.7472** — the specialist climbed past the slice, no crossover.
- ⚠ **METHODOLOGICAL CAUTION ON RECORD (this is the SECOND correction in this arm).** The plateau claim
  (extrapolated from the 4M/8M pair) and the ~100M-crossover claim (extrapolated from 2 specialist + 3 joint
  points) were **both overturned** as the ladder filled in. Two successive predictions from partial data, both
  wrong: with only 2–3 points per curve, trend extrapolation has no track record here. Future reads should
  state measured pairs only, and nothing goes in the paper ahead of the tier that would test it.

**Finding 4 — joint scaling is monotonic and approaches the teacher.** The joint scratch+KD 34-track mean
rises at every capacity step: **4M 0.4606 → 8M 0.4769 → 30M 0.4958 → 100M 0.5188 → 300M 0.5309**, climbing
toward the **650M teacher 0.606**. The 300M point (test 0.5309, best-val 0.5572, git `bae128d`) closes the
ladder and sits cleanly below the teacher, so the from-scratch scaling-law fit is computable end-to-end.

**Finding 5 — KD helps the 34-track JOINT model ~2× more than the 1-track SPECIALIST, at EVERY size.**
With both matched no-KD ablations now complete across all five sizes, the clean teacher-contribution (KD −
matched no-KD) can be compared arm-for-arm:

| size | joint KD Δ (34-track) | per-track KD Δ (1-track) | ratio (joint / per-track) |
|---|:-:|:-:|:-:|
| 4M | +0.0257 | +0.0162 | 1.6× |
| 8M | +0.0286 | +0.0179 | 1.6× |
| 30M | +0.0277 | +0.0126 | 2.2× |
| 100M | +0.0341 | +0.0183 | 1.9× |
| 300M | +0.0341 | +0.0119 | 2.9× |
| **range / mean** | **+0.0257 … +0.0341 (mean ~+0.030)** | **+0.0119 … +0.0183 (mean ~+0.015)** | **~2×** |

At **every one of the five sizes** the joint KD delta exceeds the per-track KD delta, by roughly 2× on average
(joint mean ~+0.030 vs per-track mean ~+0.015). **Interpretation:** the teacher's value is largely
**cross-track structure** that a joint 34-track head can absorb but a single-track specialist cannot — a
specialist already has the full label signal for its one track, so the teacher mostly adds correlations *across*
tracks (co-regulation shared across assays/loci), which only the joint model can exploit. This is supported by
**all 5 sizes** (not a 2-point extrapolation), and the gap holds throughout the ladder. Both deltas are
computed from full precision (E.2.2 / E.2.4); per-track means/ranges use the corrected full-precision values
(300M per-track Δ = +0.0119, not the +0.0118 that subtracting rounded cells would give).

#### E.4 — Related classification experiments (cross-links, recorded earlier in this doc)
The regression ladder above is the companion to the **classification** experiments recorded in the
**2026-07-21 block, sub-sections A–D** of this doc (not duplicated here):
- **KD-method comparison** (§B, NT-2.5B teacher, 18-task, 3-seed): OmegaGenome **0.6271** / DKD 0.6218 /
  DIST 0.5932 / LS 0.5879; DIST/LS "collapse" is a 2-class degeneracy artifact recovered by per-method kl
  re-tune, H3K9me3 is the lowest-signal worst case (see §B).
- **Classification student-size sweep** (§C, Fig 6B, 18-task mean, seed 0): plateau **0.5214 → 0.6289 (0.1M
  deployed) → 0.6383 (3.6M)**; 13/18 tasks within |Δ|<0.02 across 36× scale.
- **BPNet from-scratch baseline** (§D, 3-seed): 18-task mean **0.6085** (vs single-seed 0.5882).
- **Classification finalization** (§A): DNABERT-2 @200ep 3-seed **0.6067**; students-below-BPNet **14 → 11**.

#### E.5 — Reproducibility & artifacts
- **Doc commits:** on `precompute_logits_cache` (nested repo `code_carbon`, remote `jhliu17/OmegaGenome`);
  prior incremental section-E entries at `9d4cd3c` (30M tiers + no-KD 4M; revise plateau), `67c4c03`
  (per-track 100M; refute crossover), `0f8a8d6` (joint 300M complete; matched no-KD 4M/8M; reorganize E.3).
- **Code / sbatch:** on branch `ntv3-size-ladder` — key commits: data-match fix `9d7942c` (record
  `n_train_windows`, `train_overlap` matching), hang-watchdog `bae128d`, TMPDIR/node-local fix `bb281a7`,
  per-track no-KD baseline sbatch `slurm/ntv3_size_sweep_pertrack_t12_nokd.sbatch` = commit `6fd7970` (the
  E.2.4 arm, and the git-stamp on the last-landed 100M/300M joint no-KD + 300M per-track KD runs).
  (Run `git_commit` stamps across the `result.json`s: `9299f63`, `bb281a7`, `bae128d`, `6fd7970`.)
- **HF:** `explcre/omegagenome-distilled-students/regression/size_ladder/{joint34_scratch_kd,
  joint34_scratch_nokd, pertrack_t12_ENCSR325NFE_kd, pertrack_t12_ENCSR325NFE_nokd}` — each tier carries
  `best_model.pth` + `result.json` + logs + README. ✅ All 20 tiers synced (21 dirs incl. legacy unmatched 4M;
  86 files; 6.04 GB newly uploaded, sha256-verified; README = full matrix + KD-delta table) [2026-07-25].
- **SSD run-dirs:** `/srv/disk00/sshfs/pengchx3/ntv3_targets/size_sweep/ntv3-<size>_scratch_{kd,nokd,
  nokd_matched}_s0/` (joint), `.../size_sweep_pertrack/t12_ENCSR325NFE/ntv3-<size>_scratch_{kd,nokd_matched}_s0/`
  (per-track), `.../ntv3_ft_faithful_s{0,1,2}`, `.../ntv3_{8m,100m}_baseline`, `.../ntv3_8m_kd_dist` (refs).
- **Repo-native pipeline:** distillation, validation, and best-val checkpoint selection all use the repo's
  own `finetune_ntv3.py` (`track_kd_loss`, `poisson_multinomial_loss`, `best_model.pth` selection, test
  evaluated on the best-val checkpoint). Our changes are strictly **additive**: the size registry
  (`NTV3_SCALED_SIZES`), the RAM/memmap teacher cache, the no-KD flags (`--kd_w_ce/kl/mse`, no `--teacher`),
  the data-match (`train_overlap=0.0` + 63,707-window cap), and the mount/hang guards.

#### E.6 — Methodology / infra notes
- **Data-match fix (why + the confound).** The clean KD ablation requires the no-KD arm to see the *same*
  training stream as the KD arm. The original no-KD 4M run used `train_overlap=0.999` → **65,051,340**
  windows, vs the KD arm's `train_overlap=0.0` → **63,707** windows (a ~1021× data-diversity confound). The
  **matched** no-KD arm sets `train_overlap=0.0` back to the KD arm's value. Stream identity is now provable
  from artifacts: `n_train_windows` is written into the no-KD `result.json`s (as of code commit `9d7942c`) and
  reads **63,707** in both matched 4M/8M runs. **Honest note on verifiability:** the joint scratch+KD
  `result.json`s do **not** emit `n_train_windows` (the field predates them / is written by the no-KD path),
  so KD-arm stream identity is established **by construction** — the KD runs carry the identical
  `train_overlap=0.0`, `data_dir`, and window logic — rather than by a recorded window count in the KD json.
  The unmatched 4M run predates the field entirely; its 65,051,340-window stream is recorded in its run log.
- **Self-healing mount guard + hang watchdog.** Covers two failure modes seen on this ladder: (1) a **voyager
  node-local `/tmp` overflow** (49 GB) when staging teacher caches — fixed by RAM/memmap-loading the caches
  from the SSD mount instead of staging to node-local `/tmp`; (2) **sshfs concurrency saturation** when
  concurrent NTv3-ladder launches hit the shared galaxy SSD on laniakea → job failure — fixed with an
  **sshfs-retry wrapper** on the loader plus **throttle=1 staggering** so launches don't hit the mount
  simultaneously.
- **One documented KD-vs-plain-label asymmetry (NOT a data confound).** The KD arm's ground-truth term
  carries `w_ce=0.5` (the other 0.5 is the KL/distill term); the no-KD arm runs the full `poisson_multinomial`
  label loss at `w_ce=1.0`. This is the intrinsic difference between a KD *total* loss and a plain *label*
  loss (called out in the sbatch header) — the training **stream** is identical between the two matched arms.

#### E.7 — Status: ✅ COMPLETE (all 20 tiers = 4 arms × 5 sizes, seed 0, verified `result.json`) [2026-07-25]
**All four from-scratch arms landed across all five sizes (4M/8M/30M/100M/300M):**
- Joint scratch+KD (34-track): **all 5 tiers** ✅.
- Joint no-KD matched (34-track): **all 5 tiers** ✅ (100M/300M landed 2026-07-24/25).
- Per-track t12 specialist KD (1-track): **all 5 tiers** ✅ (300M landed 2026-07-25).
- Per-track t12 no-KD matched (1-track): **all 5 tiers** ✅ (NEW arm, E.2.4).
- Reference points: 650M teacher (3-seed), pretrained-init 8M/100M, `ntv3_8m_kd_dist`.

Every one of the 20 cells re-verified against its `ntv3_finetune_result.json` on 2026-07-25 (test + best-val
+, for the no-KD arms, `n_train_windows=63707` / `kd_w_kl=0` / `teacher=None` / correct `n_tracks` &
`track_ids`). No pending training. The KD-delta-vs-size curve (both joint 34-track and per-track 1-track) and
the specialist-vs-joint-slice curve are now complete end-to-end.

**Packaging deliverables — ✅ DONE [2026-07-25]:**
- **Plots:** ✅ 3 figures generated from the complete 20-tier data (verified against `result.json`), committed
  `ab34ba2` — `figs_size_ladder/{fig1_joint34_scaling,fig2_pertrack_t12_scaling,fig3_kd_delta}.png` + the
  reproducible `plot_size_ladder.py`.
- **Final HF sync:** ✅ all 20 tiers on `explcre/omegagenome-distilled-students/regression/size_ladder/`
  (21 tier dirs incl. the legacy unmatched 4M; 86 files; the newly-landed cells = 6.04 GB, sha256-verified),
  README rewritten for the full 4-arm × 5-size matrix + KD-delta table.
