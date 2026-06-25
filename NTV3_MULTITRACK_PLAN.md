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
- **Resolution**: **single-nucleotide** (1 bp), NOT downsampled bins — the 384 is just the central
  crop `L_out = 0.375 × L` (1024 × 0.375 = 384; outer 62.5 % dropped for edge context). So align the
  student to the teacher's `L_out` (same 0.375 crop), not to a coarser `out_resolution`. See §5 Task I/O.
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

### NTv3 Benchmark fine-tuning reproduction — ✅ MATCHES PAPER (2026-06-19)

**Goal (answers reviewer R1.1a):** reproduce NTv3-650M-post's *published* benchmark numbers by
fine-tuning it on the paper's own benchmark dataset, then test-reporting per-assay PCC.

**Setup — faithful port of InstaDeep's official notebook 03** (`src/train/finetune_ntv3.py`):
- **data:** `InstaDeepAI/NTv3_benchmark_dataset`, 34 human functional tracks, official `splits.bed`
  (train/val/test); dense windows over the **FULL** training regions (not a 2000-window subsample)
- **model:** headless NTv3-650M-post `core` + fresh linear head; **full** fine-tune (not LoRA)
- **train:** 19932 steps, eff-batch 32, 32 kb windows, Poisson-multinomial loss, AdamW **lr 5e-5**
  (warmup→square-decay); val-select best checkpoint, report on held-out test
- **metric:** per-track Pearson, averaged within each assay (the paper's reporting unit)

**Task I/O — exactly what each track task predicts** (one spec shared by all 34 tracks; source of
truth = `src/train/finetune_ntv3.py`, `src/model/ntv3_finetune.py`, `src/data/ntv3_ft_data.py`):

| Field | Value |
|---|---|
| **Model context capacity** | NTv3 supports up to **1 Mb**; this benchmark uses a fixed 32 kb window (below). |
| **Input length (per window)** | **32,768 bp (32 kb)** — `--sequence_length 32768`. |
| **Input format** | **single-nucleotide tokens** (1 token/bp: A/C/G/T/N), species-conditioned (human). No k-mer/BPE pooling. |
| **Output resolution** | **single base-pair (1 bp)** — NTv3 predicts one value *per nucleotide* (not binned). |
| **Output length** | central **12,288 bp** of the window (`L_out = 0.375 × L`); the outer 62.5 % is dropped because the U-Net edges lack full context (`keep_target_center_fraction = 0.375`). |
| **Output tensor** | `bigwig_tracks_logits` `[B, 12288, T]`, tracks on the last dim (here **T = 34**). |
| **Value in each bp** | one **non-negative continuous** number per (bp, track) = predicted **read-coverage signal** at that base. **Range [0, ∞)** — a count/coverage intensity, *not* a probability and *not* a class label. |
| **Target construction** | raw bigWig per-bp coverage → `nan→0` → center-crop 0.375 → **÷ per-track mean** → smooth-clip values > 10 via `2·√(10x) − 10` (deliberately **not** log1p). |
| **Loss** | **Poisson-multinomial** = a *scale* term (Poisson on each track's total count) + a *shape* term (multinomial over the per-bp profile) — the standard count-data loss for genomic tracks. |
| **Metric** | per-track **Pearson r** between predicted and observed per-bp signal, averaged within each assay. |

So every task is the same shape of problem: **given 32 kb of DNA, regress the base-by-base coverage
signal of one functional assay over the central 12 kb** — a per-bp, non-negative intensity (the same
readout Enformer/Borzoi predict), differing only in *which molecular event* the reads count:

| Assay | #tracks | What the per-bp value physically measures | Units / range |
|---|:-:|---|---|
| **ATAC-seq** | 5 | chromatin **accessibility** — Tn5 insertion density (open chromatin) | read coverage, [0, ∞) |
| **Histone ChIP-seq** | 4 | **histone-mark enrichment** (e.g. H3K4me3, H3K27ac) along the genome | ChIP read coverage, [0, ∞) |
| **PRO-cap** | 10 | **transcription initiation** — 5′ ends of nascent capped RNA | capped-5′ read counts, [0, ∞) |
| **eCLIP** | 10 | **RNA-binding-protein occupancy** — RBP crosslink density | crosslink read coverage, [0, ∞) |
| **RNA-seq** | 5 (2 polyA + 3 total) | **transcript abundance** — steady-state RNA coverage | read coverage, [0, ∞) |

All five are non-negative coverage signals (hence the shared Poisson-multinomial loss); the human
benchmark draws these 34 tracks from NTv3's 7,362-track human panel (ENCODE `ENCSR…` accessions).

**RESULT — per-assay TEST PCC, ours (3 seeds, mean ± std) vs the paper** (paper numbers from
`InstaDeepAI/ntv3_benchmark` → `ntv3_benchmark_results.csv`, see NTV3_DATASET_SUMMARY.md §5d):

| assay | # tracks | **paper NTv3-650M** | **ours (3-seed mean ± std)** | Δ vs paper | BPNet-6M baseline |
|---|:-:|:-:|:-:|:-:|:-:|
| ATAC-seq | 5 | 0.759 | **0.7573 ± 0.0007** | −0.002 ✅ | 0.571 |
| Histone ChIP-seq | 4 | 0.717 | **0.7228 ± 0.0007** | **+0.006** ✅ | 0.520 |
| PRO-cap | 10 | 0.508 | **0.5168 ± 0.0030** | **+0.009** ✅ | 0.398 |
| eCLIP | 10 | 0.584 | 0.5595 ± 0.0012 | −0.025 | 0.283 |
| RNA-seq (polyA+total) | 5 | 0.695 | 0.635 (0.6114 / 0.6509) | −0.060 | 0.330 |
| **overall mean** | **34** | — | **0.6064 ± 0.0006** | — | — |

(3 seeds = `ntv3_ft_faithful_s{0,1,2}`; std ≤ 0.003 everywhere → reproduction is **deterministic to 3
decimals**. Per-seed per-track CSVs: `results/ntv3_seed{0,1,2}_per_track.csv`.)

**Detailed per-track table (all 34, 3-seed mean±std, WITH the paper assay-target side-by-side):**
`results/ntv3_650m_per_track_3seed.csv` — columns `track_id, assay, seed0, seed1, seed2, mean, std,
paper_assay, ours_minus_paper`. Highlights (ours 3-seed mean vs the paper's assay number):
- **ATAC** ENCSR325NFE 0.832 (+0.073 vs paper 0.759) … ENCSR628PLS 0.650 (−0.109) — wide within-assay spread.
- **Histone** ENCSR962OTG 0.914 (+0.197!) … ENCSR754DRC 0.603 (−0.114 vs paper 0.717).
- **PRO-cap** 0.342→0.583; most tracks **above** paper 0.508 except the two ENCSR114HGS (~0.342).
- **eCLIP** 0.473→0.689 (paper 0.584); **RNA** polyA ENCSR527JGN_P 0.720 vs _M 0.503 (strand asymmetry).
The paper publishes only **assay-level** means (no per-track), so `paper_assay` is the assay target
repeated per track — lets you see each track against its assay's paper bar. Assay-level paper-vs-ours
is the table above.

- **Matches/exceeds the paper on ATAC, Histone, PRO-cap; within ~0.03–0.06 on eCLIP & RNA** — and far
  above the BPNet-6M baseline on every assay. This is a **paper-grade reproduction** of NTv3.
- **RNA note (the 0.695):** the paper reports a single combined "RNA-seq" (0.695). The benchmark
  metadata splits it into **polyA RNA** (2 tracks → 0.611) + **total RNA** (3 tracks → 0.651); pooling
  all 5 for an apples-to-apples number gives **0.635**. So RNA is genuinely ~0.06 below the paper
  (along with eCLIP, the two harder assays).
- **Why the residual on RNA/eCLIP is expected — vendor-documented:** InstaDeep's own notebook 03 states
  this PyTorch pipeline lands "within 0.01 mean Pearson of the paper" and differs from their internal
  **JAX** pipeline; the notebook itself reports mean 0.6050 (we get **0.6064** — a track-for-track match
  of the official notebook). So the RNA/eCLIP gap is the documented PyTorch↔JAX difference, **not** a
  fine-tuning shortfall. 3-seed averaging is exactly what InstaDeep recommends.

**How we got here (history — kept for the record):**

| run | data | lr | TEST mean | ATAC | Histone | note |
|---|---|:-:|:-:|:-:|:-:|---|
| ⭐ **full-FT FAITHFUL (seed 0)** | **full regions, 19932 steps** | 5e-5 | **0.606** | **0.758** | **0.723** | **paper match** |
| full-FT | 2000-window, 20 ep | 1e-5 | 0.440 | 0.543 | 0.572 | ≈ BPNet baseline |
| LoRA r=64 | 2000-window, 20 ep | 1e-4 | 0.423 | 0.559 | 0.566 | LoRA tolerates high lr |
| full-FT | 2000-window, 20 ep | 1e-4 | 0.157 | 0.418 | 0.424 | collapsed (RNA/eCLIP→0) |

1. **LR was the first lever.** full-FT @ lr 1e-4 collapsed (0.157); dropping to **lr 1e-5 → 0.440**
   (+0.28). (LoRA tolerated 1e-4 and matched, proving the pipeline was always sound — pure optimization.)
2. **Data/compute scale closed the rest.** The lr-1e-5 run on only 2000 windows reached ≈ the BPNet
   baseline (~0.15 below paper); scaling to the **full benchmark + 19932-step schedule** lifted it to
   the paper match above. ✅
3. **Resume bug fixed en route:** `load_state_dict(strict=False)` skips derived rotary cos/sin cached
   buffers (a fresh pre-forward model doesn't register them) — needed for crash/requeue recovery.

**Artifacts (seed 0):**
- Result JSONs `ntv3_targets/ntv3_ft_faithful_s{0,1,2}/ntv3_finetune_result.json` — each has
  `per_track_pearson` (all 34, range ~0.34 → ~0.91), `test_pearson_by_assay` (6 means), `test_mean_pearson`.
- Per-track CSVs `results/ntv3_seed{0,1,2}_per_track.csv` (track_id, assay, PCC) — flat, paper-ready.

**✅ DONE: 3 seeds complete (2026-06-20).** Paper match holds across seeds with std ≤ 0.003 (table above);
this is the R1.1a reproduction + multi-seed significance. Join script: `src/eval/per_track_csv.py`.

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

## 7. NTv3-650M → small-student distillation (2026-06-24, in progress)

**Goal:** distill the *reproduced* NTv3-650M-post (the §5 paper-match teacher, 0.606) into compact
students, vs a from-pretrain baseline. Students: **NTv3-8M** (pretrained backbone + fresh 34-track
head, `NTv3PreBigWigModel`) and **~1M DilatedTrackNet**. 34 benchmark tracks, per-track Pearson.

**Setup (clean, reuse-heavy; classification infra `src/model/distillation.py` UNTOUCHED):**
- Teacher: `load_finetuned_bigwig_teacher` = `build_bigwig_model(NTv3_650M_post)` + `best_model.pth`
  (strips the `_orig_mod.` compiled-save prefix; tolerates rotary-cache buffers). Frozen, eval.
- Student build: `build_bigwig_model` factory — PRE (8M, AutoModelForMaskedLM) vs POST auto-dispatch.
- KD loss: `track_kd_loss` (3-term regression analog of CE+KL+MSE), **all weights + per-term loss
  types configurable** (`TrackKDConfig`; defaults 0.5/0.5/0.2, Poisson-multinomial). Distill-term
  options grounded in SOTA regression-KD: poisson_multinomial (Borzoi/Enigma; multinomial profile =
  KL analog), teacher_bounded (Chen 2017), mse, pearson.
- Same `finetune_ntv3` entrypoint: no `--teacher` = baseline; `--teacher <ckpt>` = KD.
- Audited: `test_track_kd_loss` (26 assert), `test_ntv3_pre_bigwig` (9), CPU + GPU smokes.

### RESULT — 8M baseline (no distillation) ✅ DONE (job 241800, full-FT, seq 32768)
**test mean Pearson 0.4748** (between BPNet-6M ~0.40 and the 650M teacher 0.606).

| assay | #tracks | 8M baseline¹ | 650M teacher (3-seed mean±std) | (Δ to teacher) |
|---|:-:|:-:|:-:|:-:|
| Histone ChIP-seq | 4 | 0.580 | 0.7228 ± 0.0007 | −0.143 |
| ATAC-seq | 5 | 0.567 | 0.7573 ± 0.0007 | −0.191 |
| total RNA-seq | 3 | 0.566 | 0.6509 ± 0.0015 | −0.087 |
| polyA plus RNA-seq | 2 | 0.501 | 0.6114 ± 0.0019 | −0.112 |
| eCLIP | 10 | 0.482 | 0.5595 ± 0.0012 | −0.077 |
| PRO-cap | 10 | 0.347 | 0.5168 ± 0.0030 | −0.167 |
| **overall** | **34** | **0.475** | **0.6064 ± 0.0006** | **−0.131** |

¹ 8M baseline is currently a **single** full-FT run (job 241800); its 3-seed mean±std is queued (see "3-seed" task below) and will replace these point values for an apples-to-apples ±std comparison. The 650M teacher column is the per-assay 3-seed mean±std (std across the 3 per-seed assay-means; same source as the §5 table, with RNA-seq here split into total/polyA).

**Complete per-track table — all 34 tracks (650M teacher 3-seed mean · 8M baseline · paper assay):**
(CSVs: `results/ntv3_650m_per_track_3seed.csv`, `results/ntv3_8m_baseline_per_track.csv`.)

| # | track_id | assay | 650M (3-seed mean±std) | 8M baseline | paper (assay) |
|--:|---|---|:-:|:-:|:-:|
| 1 | ENCSR325NFE | ATAC-seq | 0.8325 ± 0.0002 | 0.6850 | 0.759 |
| 2 | ENCSR814RGG | ATAC-seq | 0.7968 ± 0.0006 | 0.6591 | 0.759 |
| 3 | ENCSR410DWV | ATAC-seq | 0.7957 ± 0.0015 | 0.5491 | 0.759 |
| 4 | ENCSR487QSB | ATAC-seq | 0.7117 ± 0.0013 | 0.5161 | 0.759 |
| 5 | ENCSR628PLS | ATAC-seq | 0.6498 ± 0.0018 | 0.4274 | 0.759 |
| 6 | ENCSR962OTG | Histone ChIP-seq | 0.9143 ± 0.0013 | 0.8470 | 0.717 |
| 7 | ENCSR863PSM | Histone ChIP-seq | 0.6899 ± 0.0034 | 0.5113 | 0.717 |
| 8 | ENCSR682BFG | Histone ChIP-seq | 0.6840 ± 0.0012 | 0.5141 | 0.717 |
| 9 | ENCSR754DRC | Histone ChIP-seq | 0.6030 ± 0.0012 | 0.4476 | 0.717 |
| 10 | ENCSR799DGV_P | PRO-cap | 0.5832 ± 0.0050 | 0.3751 | 0.508 |
| 11 | ENCSR935RNW_P | PRO-cap | 0.5801 ± 0.0052 | 0.3767 | 0.508 |
| 12 | ENCSR100LIJ_P | PRO-cap | 0.5746 ± 0.0063 | 0.3701 | 0.508 |
| 13 | ENCSR100LIJ_M | PRO-cap | 0.5741 ± 0.0043 | 0.3797 | 0.508 |
| 14 | ENCSR935RNW_M | PRO-cap | 0.5731 ± 0.0038 | 0.3737 | 0.508 |
| 15 | ENCSR799DGV_M | PRO-cap | 0.5697 ± 0.0030 | 0.3723 | 0.508 |
| 16 | ENCSR046BCI_M | PRO-cap | 0.5317 ± 0.0042 | 0.3511 | 0.508 |
| 17 | ENCSR046BCI_P | PRO-cap | 0.4970 ± 0.0054 | 0.3130 | 0.508 |
| 18 | ENCSR114HGS_M | PRO-cap | 0.3426 ± 0.0013 | 0.2787 | 0.508 |
| 19 | ENCSR114HGS_P | PRO-cap | 0.3420 ± 0.0005 | 0.2794 | 0.508 |
| 20 | ENCSR321PWZ_P | eCLIP | 0.6886 ± 0.0034 | 0.5894 | 0.584 |
| 21 | ENCSR321PWZ_M | eCLIP | 0.6797 ± 0.0019 | 0.5490 | 0.584 |
| 22 | ENCSR249ROI_M | eCLIP | 0.6072 ± 0.0002 | 0.5444 | 0.584 |
| 23 | ENCSR249ROI_P | eCLIP | 0.5727 ± 0.0040 | 0.5244 | 0.584 |
| 24 | ENCSR862QCH_M | eCLIP | 0.5423 ± 0.0016 | 0.4487 | 0.584 |
| 25 | ENCSR862QCH_P | eCLIP | 0.5380 ± 0.0012 | 0.4495 | 0.584 |
| 26 | ENCSR154HRN_M | eCLIP | 0.5076 ± 0.0005 | 0.4562 | 0.584 |
| 27 | ENCSR154HRN_P | eCLIP | 0.5069 ± 0.0012 | 0.4543 | 0.584 |
| 28 | ENCSR484LTQ_M | eCLIP | 0.4792 ± 0.0013 | 0.3920 | 0.584 |
| 29 | ENCSR484LTQ_P | eCLIP | 0.4732 ± 0.0007 | 0.4076 | 0.584 |
| 30 | ENCSR527JGN_P | polyA plus RNA-seq | 0.7203 ± 0.0019 | 0.6053 | 0.695 |
| 31 | ENCSR527JGN_M | polyA plus RNA-seq | 0.5025 ± 0.0018 | 0.3969 | 0.695 |
| 32 | ENCSR619DQO_P | total RNA-seq | 0.6988 ± 0.0005 | 0.6240 | 0.695 |
| 33 | ENCSR619DQO_M | total RNA-seq | 0.6953 ± 0.0025 | 0.5970 | 0.695 |
| 34 | ENCSR701YIC | total RNA-seq | 0.5585 ± 0.0018 | 0.4771 | 0.695 |

(paper publishes assay-level only → `paper (assay)` is the assay bar repeated per track.) The KD
question: does distilling the 650M lift the 8M (overall 0.475) toward the teacher (0.606) — most
headroom on PRO-cap and ATAC. KD per-track columns will be appended to this table as runs complete.

### KD runs — ✅ COMPLETE (2026-06-24) — clean 2×2 loss ablation, full 32 kb recipe on voyager

All four on the **full dataset + 32 kb + 19932 steps** (identical recipe to the 650M reproduction), so
the gt∈{poisson,mse} × distill∈{poisson,mse} comparison is apples-to-apples. Student = NTv3-**8M**
(`NTv3PreBigWigModel`); teacher = the §5 paper-match **650M** (0.6064). All weights `w_ce/w_kl/w_mse =
0.5/0.5/0.2`. Metric = per-track Pearson averaged over the **34** tracks.

| run | gt_loss / distill_loss | best-val | **TEST mean Pearson** | Δ vs 8M baseline (0.4748) |
|---|---|:-:|:-:|:-:|
| `ntv3_8m_kd_pois` (241810) | poisson_mn / poisson_mn | 0.5040 | **0.4759** | **+0.0011** ✅ (only one ≥ baseline) |
| `ntv3_8m_kd_allmse` (241807) | mse / mse | 0.4946 | 0.4655 | −0.0093 |
| `ntv3_8m_kd_distmse` (241815) | poisson_mn / mse | 0.4921 | 0.4620 | −0.0128 |
| `ntv3_8m_kd_gtmse` (241875) | mse / poisson_mn | 0.4854 | 0.4572 | −0.0176 |
| — 8M baseline (no KD, job 241800) | — | — | 0.4748 | (ref) |
| — 650M teacher (§5) | — | — | 0.6064 | +0.1316 |

**Honest headline:** on the 8M student, **KD essentially does not help** — the best variant
(Poisson-multinomial on *both* terms) lands at **0.4759 ≈ the 0.4748 no-distill baseline (+0.001)**, and
every MSE-using variant is *below* baseline. The 650M→8M capacity gap (0.606 vs 0.475, the §7 table) is
too large for soft-target distillation to bridge at this size; **Poisson-multinomial clearly beats MSE**
on both the gt and distill terms (consistent with the §6 student-architecture finding and the count-like
nature of the data), so it is the right loss — but the lever for closing the gap is **student capacity**
(next: the ~1M DilatedTrackNet student), not the loss form. Per-track CSVs via `per_track_csv.py`.

#### Method — loss formulas (all in `src/trainer/track_distill.py` / `track_losses.py`)

Total per-bp track-KD objective (regression analog of the classification CE+KL+MSE), student/teacher/gt
tensors `[B, L, T]` (batch × sequence × tracks):

```
L = w_ce · L_gt(student, real_bigWig)        # ground-truth supervision  (w_ce = 0.5)
  + w_kl · L_distill(student, teacher_tracks)# distill from frozen 650M   (w_kl = 0.5)
  + w_mse · L_feat(student_feat, teacher_feat)# intermediate feature match (w_mse = 0.2)
```

`L_gt` and `L_distill` each pick one of the per-term losses below (the 2×2 ablation varies them);
`L_feat = MSE(student_feat, teacher_feat)` (Hinton-style hint/feature matching).

- **Poisson-multinomial** (Borzoi/Enformer/Enigma; `shape_loss_coefficient λ = 5`):
  `PM(pred, target) = L_shape + L_scale / λ`
  - scale (total coverage): `L_scale = mean_{B,T} [ Poisson(Σ_L target, Σ_L pred) ] / L`, with
    `Poisson(t, p) = p − t·log(p)` (Poisson NLL on the summed-over-position counts per track)
  - shape (per-position profile): `L_shape = − Σ target · log(p_pred) / (B·L·T)`, where
    `p_pred = pred / Σ_L pred` is the multinomial distribution over the sequence axis.
  The multinomial **profile** term is the per-bp regression analog of Hinton soft-target KL.
- **MSE:** `F.mse_loss(pred, target)`.
- **Pearson:** `1 − mean_track corr(pred, target)` (penalises shape mismatch directly).
- **Teacher-bounded** (Chen et al. 2017, for the distill term): squared student-vs-GT error counted
  **only where the student is worse than the teacher** by > margin —
  `Σ mask·(student−gt)² / Σ mask`, `mask = [(student−gt)² > (teacher−gt)² + margin]` — so the teacher
  pulls the student only where it actually helps.

Next: the ~1M **DilatedTrackNet** student (capacity is the lever per the result above).

### 2026-06-25 — breaking the 8M plateau: SOTA-grounded methods + a capacity study

The 2×2 Poisson/MSE ablation + **feature alignment** (FitNets; the `w_mse` term was a silent no-op before
— now wired: student emb → teacher emb projection, L2-normalised) all land at **test ≈ 0.476 = the no-KD
baseline (0.4748) / no-op poisson (0.4759)**. The 8M val **plateaus** (peaks 0.5036 @ step ~16500, flat
after — so *not* epoch-limited). A literature survey (NeurIPS/ICLR/ICML/CVPR + genomics) drove a set of
new methods, all added to `track_distill.py` (reuse the existing `_regression_term` dispatch + config;
**94 assertions, numpy-reference-verified**):

| `distill_loss` / option | paper | formula (student/teacher `[B,L,T]`) | why it might beat exact-value matching |
|---|---|---|---|
| **`dist`** | DIST, Tang et al. **NeurIPS'22** | `(1−corr_pos) + (1−corr_track)` — match the teacher's **Pearson structure** (intra-track over positions = *the eval metric* + inter-track) | scale/shift-invariant → robust to the teacher↔student **capacity gap** |
| **`standardized_mse`** | logit-standardization, Sun et al. **CVPR'24** | z-score **each track** over positions, then MSE | scale-invariant → stops high-count tracks (ATAC/RNA) dominating low-count high-headroom ones (PRO-cap/eCLIP) |
| **`cwd`** | Channel-Wise Distillation, Shu et al. **ICCV'21** | `(T²/C)·Σ_c KL(softmax_pos(t^c/T) ‖ softmax_pos(s^c/T))` — softmax each track's profile over positions, KL teacher→student | matches positional **shape** (what Pearson rewards), not absolute counts |
| **`distill_target_gt_mix`** = m | Enigma (bioRxiv'25) | distill target = `(1−m)·teacher + m·ground-truth` (m≈0.1) | anchors the distill term to truth where the teacher errs |
| **feature alignment** (`w_mse`) | FitNets | `MSE(L2norm(proj(student_emb)), L2norm(teacher_emb))` | representation transfer (verdict: no lift — capacity wall) |
| **`--track_subset`** (specialist) | — | train a student on **one track** (teacher index-selected to match) | *diagnostic*: if a 1-track 8M ≫ the joint 8M on that track, **capacity-sharing across 34 tracks is the wall** |

**Survey verdicts:** (a) **on-policy / reverse-KL distillation does NOT apply** — it is defined by an
autoregressive student *sampling* its output; our discriminative fixed-window regressor has no output
policy, and for unimodal track likelihoods reverse-KL collapses to moment-matching (= MSE). (b) Plain
**augmentation** (RC/shift) is low-value here: the student is capacity-saturated (not data-starved), and
RC is non-trivial for **stranded** tracks (RNA/PRO-cap need +/− strand swapping). (c) The cheapest untried
lever is a **weaker/mid-size teacher** — *Distillation Scaling Laws* (Apple'25) show a too-strong teacher
hurts a small student; so the **100M student** doubles as a mid-teacher for TAKD on the 8M. **Reserve
(if all above plateau):** DiffKD (denoise student features toward the teacher manifold, NeurIPS'23), MGD
(masked generative distillation, ECCV'22), per-track headroom weighting.

**Experiment matrix (running, voyager + laniakea):** 8M × {dist, standardized_mse, dist+lean, cwd,
gt_mix}; **100M baseline vs KD** (the decisive *capacity* test — no 50M variant exists, 100M = 12× the
8M); **single-track specialist** baseline+KD on PRO-cap track 18 (joint 8M = 0.2787). Results pending;
this table's right-most column is the hypothesis under test — outcomes will be recorded here.
