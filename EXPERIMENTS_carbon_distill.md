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
