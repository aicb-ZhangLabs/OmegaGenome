# NTv3 multi-track (per-bp) benchmark — understanding, cost, and integration design

Branch: `ntv3-multitrack` (isolated git worktree `code_ntv3/`, branched from `carbon-teacher`).
Goal: add NTv3's per-base-pair multi-track regression benchmark as a new task type in the
OmegaGenome distillation codebase. This directly answers reviewer **R1.1a** (the "all tasks are
classification, none predict numeric/track values" criticism).

## 1. What NTv3 and its benchmark are (understanding)

**NTv3** (InstaDeep, Dec 2025; bioRxiv 2025.12.22.695963): a multi-species genomics FM —
U-Net + transformer, **single-nucleotide tokenization, context up to 1 Mb**, pretrained on
9T bp (OpenGenome2) with base-resolution MLM, then **post-trained on ~16,000 functional tracks
+ annotation labels** across 24 animal/plant species. Species-conditioned.

**The NTv3 Benchmark**: **106 tasks**, **32 kb natural inputs → base-pair-resolution outputs**
(multi-track functional readouts + genome annotation), strict data-leakage controls. It's a
sequence-to-function **regression / per-bp** benchmark (Pearson-style track correlation), unlike
our current 18 classification tasks.

**Models / local availability** (`/extra/zhanglab0/INDV/pengchx3/ntv3_local/`):
- `8m_pre` (8M, masked-LM only), `100m_post` (**100M post-trained, 459 MB, HF/PyTorch**), `generative`.
- 650M pre/post exist upstream (not local).
- **Crucially HF-native**: loads with `AutoModel.from_pretrained(repo, trust_remote_code=True)`
  + `AutoTokenizer` — no JAX needed. The post model returns `bigwig_tracks_logits` (per-bp
  tracks), `bed_tracks_logits` (annotation), and `embedding`. (Confirmed in your biomodel repo's
  `adapters/ntv3_adapter.py::NTv3PostAdapter` and `benchmark/backends/ntv3.py` — **reuse these**.)

So NTv3-100M-post is the natural **teacher**: feed a 32 kb sequence (+ species id) → get per-bp
`bigwig_tracks_logits [B, L, T]`. We distill that into a small per-bp student.

## 2. Fine-tune / distill cost (estimate + how we'll measure)

No published NTv3 fine-tune cost figure, so — same approach as the full-FT probe — **measure on
one task, then extrapolate**. First-order reasoning:

- **Teacher (NTv3-100M-post) inference** on 32 kb base-res: 100M params, 32k positions. On an
  H100 this is moderate (~order 0.1–0.5 s/window). Caching teacher tracks over a task's windows
  is a one-time pass. The 650M post model would be ~3–6× heavier (use 100M for the revision).
- **Student training** (small per-bp regression CNN): cheap, like the BPNet students.
- **Dominant cost** = teacher track caching × number of 32 kb windows × number of tasks.
- Per task (cache teacher + train student) ballpark **~1–3 H100-hours**; for the paper we use a
  **representative subset (≈3–6 tasks)**, not all 106 — enough to demonstrate per-bp regression
  distillation. Full 106 is out of scope for the revision.
- **Action:** a probe (cache NTv3 tracks + train one student) on one task to get real numbers,
  exactly like `carbon_3b_fullft_debug`.

## 3. Integration design (clean, minimal, reuse)

The task type is different (per-bp multi-track regression vs scalar classification), so a few new
pieces are unavoidable — but each reuses existing structure:

| Piece | Reuse / change |
|---|---|
| **Teacher loader** | New `NTv3TeacherConfig` + loader, modeled on `src/model/glm.py::build_glm` and your `ntv3_adapter.py`. `AutoModel(..., trust_remote_code=True)`; expose `bigwig_tracks_logits`. Reuse `GLMConfig.input_prefix`/species handling. |
| **Student** | `BPNetRegressor` = the existing `BPNetClassifier` **stem unchanged** (dilated-conv → `[C, L]`), swap the global-pool+linear head for a **1×1 conv head → `[T, L]`** (per-bp, T tracks). Minimal diff to `bpnet_classifier.py`. |
| **Dataset** | New per-bp loader: 32 kb sequence + per-bp track target array `[L, T]` (+ species). Mirror `src/data/dataset.py` structure; add a `TrackDataset` alongside `SeqDataset`. |
| **Loss** | Per-bp regression: Poisson (track counts) or MSE, + optional distillation term matching the teacher's per-bp tracks. Reuse the 3-term `L_task + β·L_KD + γ·L_feat` shape from `distillation.py`, regression variants. |
| **Metric** | Per-track **Pearson/Spearman** (standard for track prediction) instead of MCC. New `compute_track_corr`. |
| **Config/CLI** | New `ntv3_multitrack` experiment config + registry entry; same tyro + `@slurm_fn` entrypoint pattern as `finetune_teacher.py` / `distill.py`. |

Net new files (small): `config/.../ntv3.py` (teacher cfg), `src/model/bpnet_regressor.py`,
`src/data/track_dataset.py`, a regression metric helper, one experiment config. Everything else
(trainer loop, SLURM scripts, env, run-dir/output handling) is reused.

## 4. Open items before coding
1. **Benchmark data access** — locate the NTv3 Benchmark dataset (HF dataset id / the InstaDeep
   "Try NTv3 Benchmark" HF Space; your biomodel repo may already reference the windows/tracks).
   This is the one true blocker for "make it a task dataset."
2. **Species conditioning** — NTv3 needs a species id; default to human for the NT-style tasks.
3. **Pick the representative subset** of NTv3 tasks for the revision (e.g. a few expression /
   chromatin tracks) rather than all 106.

## 4b. Confirmed interface (probe, 2026-06-16)
Loaded NTv3-100M-post locally via the gated-snapshot bypass (`prepare_local_snapshot` strips the
`InstaDeepAI/ntv3_base_model--` auto_map prefix → loads the snapshot's local `.py` with
`local_files_only`). One CPU forward on 1024 bp returned:
- `bigwig_tracks_logits` = **`[B, L_out, T]` = `(1, 384, 7362)`** — tracks are the **last dim**
  (7362 for human); subsetting via `index_select(-1, idx)` is correct.
- **Resolution**: 1024 bp → 384 bins (U-Net downsampling), so align the student to the teacher's
  `L_out` in the loss (`AdaptiveAvgPool1d`/interpolate), not a fixed `out_resolution`.
- 650M default needs an HF token (gated) or its own local snapshot; runs on H100.

## 4c. Track subset — recommendation (chosen to fix the major revision)

Picked to answer specific reviewer points, not just "interesting biology." One NTv3 multi-track
distillation experiment hits four concerns at once:

| Reviewer point | How this answers it |
|---|---|
| **R1.1a** "classification, not numeric inference (e.g. histone mark signals)" | per-bp **quantitative signal** regression, led by the **same histone marks** as our classification benchmark |
| **R1.1b** "resolution not very high" | NTv3 + student are **base-resolution** |
| **R1.1c** "one task per data type, not per cell type" | same assays across **multiple cell types** |
| **R2.1a** "use strong billion-scale gLM teachers" | teacher is **NTv3** (SOTA seq-to-function FM) |

Track families mirror the canonical Enformer/Borzoi readouts (CAGE / DNase / ATAC / histone ChIP
/ RNA-seq), scored by **per-track Pearson** — the field-standard metric reviewers know.

**Recommended subset (tiered):**
- **v1 (proof, K562 only):** the 6 canonical histone marks H3K4me3 (active promoters), H3K27ac
  (active enhancers), H3K4me1 (primed enhancers), H3K27me3 (Polycomb), H3K36me3 (gene bodies),
  H3K9me3 (heterochromatin) — *the same marks as our classification tasks, now as signal* — plus
  DNase-seq + CAGE. ~8 tracks. Direct R1.1a rebuttal.
- **v2 (full, per cell type):** same panel across K562 / GM12878 / HepG2 (+ ATAC, RNA-seq where
  available). ~25-35 tracks. Adds the per-cell-type axis (R1.1c).

Enabled rebuttal: *"Addressing the concern that our tasks were classification rather than
quantitative inference, we extend OmegaGenome to base-resolution, multi-cell-type signal
prediction by distilling NTv3 across histone-mark, accessibility, and transcription tracks;
compact students retain X% of NTv3's per-track Pearson at Y x fewer parameters."*

**Optional complementary task:** NTv3's 21 named BED annotation tracks (protein_coding_gene, exon,
splice donor/acceptor, CTCF-bound, enhancer/promoter tissue-specific, UTRs, ...) — readable, no
metadata join needed — a per-bp genome-annotation task for breadth.

**Impl note:** NTv3's 7362 human bigwig tracks are named only by ENCODE accession (`ENCSR...`), so
selecting "H3K4me3 in K562" needs an accession->(assay, biosample) join via the ENCODE API (or
NTv3's `ntv3_tracks_pipeline.py`). That join is the one data step before training the subset.

## 5. Status
- Branch `ntv3-multitrack`: full pipeline + tests committed (regressor, teacher loader w/ gated
  bypass + HF token, UCSC window fetch, target-gen, trainer w/ per-track z-norm, per-track Pearson;
  44 test assertions). v1 K562 9-track manifest ENCODE-joined.
- Bugs found & fixed via runs: (1) bf16 dtype crash in NTv3-650M -> fp32 default + autocast;
  (2) `build_teacher_targets` equal-length guard.

### v1 first result (2026-06-16) — CPU 100M proof: FAILED, root-caused, fixed
- Tiny CPU proof (100M teacher, 48 train/16 test windows x 4kb): **mean test Pearson -0.045**
  (negative). NOT a code bug.
- Diagnosed via train-vs-test curve: TRAIN Pearson rises 0.24->0.47 while TEST stays ~0 ->
  **overfitting from data starvation** (48 windows can't generalize across genomic regions).
  Alignment verified correct (student emits L=4096 -> interp to teacher L=1536, no transpose).
  Per-track z-norm added (heterogeneous scales: DNase std 8.5 vs H3K9me3 std 0.43) but the real
  fix is window count.
- **Fix = scale up windows** + per-track z-norm. Confirmed directionally:

  | run | teacher | windows (tr/te) | student | mean test Pearson |
  |---|---|---|---|---|
  | CPU proof | 100M | 48 / 16 | small | **-0.045** (fail) |
  | 235979 | 100M | 2000 / 400 | medium | **+0.172** |
  | 235981 | 100M | 8000 / 1500 | medium | **+0.221** |
  | 235964 | **650M** | 2000 / 400 | medium | **+0.169** |
  | 236263 | 100M | 8000 / 1500 | **dilated** | cancelled mid-run (rerun pending) |

  **Teacher size does NOT help**: 650M@2k (0.169) ~ 100M@2k (0.172). **Data is the lever**:
  100M@8k (0.221) >> both @2k. Consistent per-track pattern (650M): broad marks learn (H3K27me3
  0.364, H3K4me1 0.208, H3K9me3 0.148) but sharp marks don't (H3K4me3 0.016, H3K9ac 0.021) ->
  motivates the dilated/long-RF student (rerun pending, see RESUME_JOBS.md).

### Loss x architecture ablation (cached targets, 800 windows, 30 ep)
  | loss | arch | params | best test Pearson |
  |---|---|---|---|
  | mse | medium | 0.46M | 0.202 |
  | mse | **large** | 1.8M | **0.240** |
  | poisson | medium | 0.46M | 0.213 |
  | pearson | medium | 0.46M | 0.155 |
  | mse+pearson | medium | 0.46M | 0.168 |

  **Capacity is the main lever** (large > medium, +0.04). **Loss is minor**: poisson ~ mse (slight
  poisson edge, matches count-like data); the correlation loss *hurt* (refuted the hypothesis that a
  metric-aligned loss would help). => the new **DilatedTrackNet** (RF ~32kb, full-window context)
  is the principled next step beyond just adding channels; job 236263 tests it on the 8k targets.
- **Still preliminary, not paper-grade.** Levers left: more windows (genome affords ~190k 16kb
  windows), bigger student, 650M teacher, and student-vs-*ground-truth* eval (needs ENCODE bigwigs +
  pyBigWig) rather than student-vs-teacher fidelity.
- Lesson: the 30-min CPU proof caught the failure before burning a voyager H100 — keep proofs small-first.

### NTv3 Benchmark fine-tuning reproduction (2026-06-18) — 34 held-out human tracks, their HF data

Fine-tune NTv3-650M-post on the **paper's own benchmark dataset** (`InstaDeepAI/NTv3_benchmark_dataset`,
34 functional tracks, their `splits.bed` train/val/test), val-select / test-report, paper log1p PCC.
n_train=2000, n_test=400, 16 kb window, 20 epochs, Poisson loss. Jobs 237440 (full-FT) / 237441 (LoRA r64).

| config | best_val | **TEST mean** | ATAC | Histone | PRO-cap | eCLIP | polyA RNA | total RNA |
|---|---|---|---|---|---|---|---|---|
| **full-FT lr1e-5 FAITHFUL — FULL data, 19932 steps — seed 0** ⭐ | 0.629 | **0.606** | **0.758** | **0.723** | **0.514** | 0.559 | 0.613 | 0.653 |
| full-FT (lr 1e-5, 2000-window) | 0.408 | 0.440 | 0.543 | 0.572 | 0.445 | 0.289 | 0.602 | 0.469 |
| LoRA r=64 (lr 1e-4, 2000-window) | 0.402 | 0.423 | 0.559 | 0.566 | 0.417 | 0.235 | 0.672 | 0.482 |
| full-FT (lr 1e-4, 2000-window) | 0.105 | 0.157 | 0.418 | 0.424 | 0.124 | 0.032 | -0.00 | -0.00 |
| *paper NTv3-650M-post (target)* | — | — | *0.759* | *0.717* | *0.508* | *0.584* | *0.695 (combined RNA)* | *↤* |
| *paper BPNet-6M (baseline)* | — | — | *0.571* | *0.520* | *0.398* | *0.283* | *0.330 (combined RNA)* | *↤* |

(Paper numbers per category from `InstaDeepAI/ntv3_benchmark` -> `ntv3_benchmark_results.csv`, see
NTV3_DATASET_SUMMARY.md §5d. The paper reports a SINGLE "RNA-seq" PCC (0.695); this benchmark split it
into polyA + total RNA, so those two columns aren't 1:1 with the paper's combined value — ↤ = same cell.)

**Findings (honest):**
1. **LoRA ≫ full-FT @ lr 1e-4 (0.423 vs 0.157), CONFIRMED as an LR problem.** `lr=1e-4` is fine for
   low-rank adapters but far too high for the full 650M backbone (destabilized features; RNA/eCLIP→0).
   **Dropping to lr 1e-5 recovered full-FT 0.157 → 0.440 (+0.28)** — RNA/eCLIP collapse gone, and full-FT
   now edges LoRA (0.440 vs 0.423). LoRA passes the *same* data/eval path, so the pipeline was always
   sound; the LoRA>full-FT inversion was pure optimization. Lesson: full-FT of a large FM needs ~1e-5.
2. **Not yet a paper match.** Best config (full-FT lr 1e-5) reaches **ATAC 0.543 / Histone 0.572 ≈ the
   BPNet-6M baseline** (0.571/0.520) but still **~0.15–0.18 below NTv3's reported fine-tuning**
   (0.759/0.717). LoRA & full-FT(1e-5) are now within 0.02 of each other — the LR was the big lever, not
   the adapter-vs-full choice.
3. **Remaining gap driver = data/compute scale.** With LR fixed, the residual ~0.15 gap most plausibly
   reflects training budget: we use 2000 windows / 20 epochs; the paper fine-tunes over the full
   benchmark training regions with a longer schedule. Testing this next.
4. **Next experiments:** scale n_train (2000→8000) + epochs for full-FT lr 1e-5 (the now-best config)
   to probe the data-scale hypothesis toward the paper's 0.72–0.76. Tracking as they run.
5. **✅ PAPER MATCH ACHIEVED (2026-06-19) — full-data faithful run, seed 0.** The faithful port of the
   official notebook 03 (FULL benchmark training regions, official 19932-step schedule, eff-batch 32,
   Poisson-multinomial loss, lr 5e-5 square-decay) **closes the gap to the paper** — confirming the
   data-scale hypothesis (#3). Per-category TEST PCC (n_tracks=34), paper-vs-ours:
   | category | paper NTv3-650M | seed 0 | |
   |---|---|---|---|
   | ATAC-seq | 0.759 | **0.758** | ✅ match |
   | Histone ChIP-seq | 0.717 | **0.723** | ✅ above |
   | PRO-cap | 0.508 | **0.514** | ✅ above |
   | eCLIP | 0.584 | 0.559 | −0.025 |
   | RNA-seq (combined) | 0.695 | 0.613 / 0.653 (polyA / total) | ~−0.05 |
   | **mean over 34 tracks** | — | **0.606** | |
   Match/exceed on 3/5 categories; within ~0.03–0.06 on eCLIP and RNA. **All 34 per-track PCCs are saved
   in `ntv3_targets/ntv3_ft_faithful_s0/ntv3_finetune_result.json`** (`per_track_pearson`; range 0.341
   [ENCSR114HGS_M] → 0.913 [ENCSR962OTG], with `test_pearson_by_assay` = the 6 category means above).
   Resume bug fixed en route: `load_state_dict(strict=False)` to skip derived rotary cos/sin cached
   buffers (a fresh model doesn't register them). **Seeds 1 & 2 in flight for 3-seed error bars.**

## 6. Student architecture: BPNet → DilatedTrackNet (design logic)

Code: `src/model/bpnet_regressor.py` (v1 student) and `src/model/dilated_track_net.py` (the upgrade).
Tests: `tests/test_dilated_track_net.py` (10 angles / 24 asserts, incl. empirical receptive field).

**v1 student (BPNetRegressor) and why it caps out.** It reuses the BPNetClassifier dilated-conv
stem and swaps the global-pool+linear head for a **1×1 conv head** → per-bp `[T, L]`. Because the
head is 1×1, each output position's tracks are predicted from **only the stem's receptive field (RF)**:
- `medium`: dilations capped at 2⁶=64 → RF ≈ **a few hundred bp**
- `large`: dilations capped at 2⁸=256 → RF ≈ **1–2 kb**

Both are **far below the 16 kb window**, so marks that depend on long-range context can't be modeled.
The data agree: across every run, **broad marks learn** (H3K27me3 ~0.36, H3K4me1 ~0.21) while
**sharp/locally-defined marks stay flat** (H3K4me3 ~0.02, H3K9ac ~0.02). The ablation also showed
capacity helps (large 0.240 > medium 0.202) but that's just more channels — it does **not** extend RF.

**The change — DilatedTrackNet.** Keep the base-resolution, per-bp, 1×1-head design (so it stays a
drop-in: `input_ids [B,L]` → `[B,T,L]`), but replace the shallow stem with a **deep stack of dilated
residual blocks** whose dilation doubles each layer:

    one-hot(4) → stem(conv k=15) → [ResBlock(dilation 2^i) for i in 0..13] → 1×1 track head
    ResBlock = Conv(k=3,dilation d) → BN → GELU → Conv(1×1) → BN → (+ residual) → GELU

Receptive field (the whole point):

    RF = 1 + (stem_k − 1) + Σ_blocks (k − 1)·dilation
       = 1 + 14 + 2·(1+2+4+…+8192) = 1 + 14 + 2·16383 ≈ **32.8 kb  ⟹ covers the 16 kb window**

**Why this design (vs alternatives).**
- *Just a bigger BPNet* — adds channels, not RF; ablation showed capacity alone plateaus.
- *Dilated residual tower (chosen)* — RF grows **exponentially with depth** at base resolution,
  cheap, residual makes the depth trainable; proven lineage (WaveNet/BPNet/Borzoi trunk). Keeps
  base-res output to match NTv3's per-bp targets, and the same I/O contract → swappable via `--student`.
- *U-Net* — also viable (NTv3 itself is one) and more memory-efficient at long range; deferred as a
  later option if base-res memory becomes the bottleneck.
- *Transformer (Enformer-style)* — best long-range but heavier and outputs coarse bins; overkill for
  a compact student.

**Hypothesis it tests (not yet confirmed — job was cancelled mid-run):** full-window RF should lift
the *sharp* marks (H3K4me3/H3K9ac) that the small-RF BPNet can't localize, while holding the broad
marks. Result pending rerun (`sbatch slurm/ntv3_distill_dilated.sbatch`, reuses cached 8k targets).
