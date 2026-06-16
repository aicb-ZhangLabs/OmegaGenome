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
  | 235979 | 100M | 2000 / 400 | medium | **+0.172** (best; final 0.108) |
  | 235981 | 100M | 8000 / 1500 | medium | queued |
  | 235964 | 650M | 2000 / 400 | medium | queued (voyager) |

  Best per-track at 100M/2000: H3K27me3 0.354, H3K4me1 0.213, H3K9me3 0.100; H3K4me3/H3K9ac ~0.03.
  `best>final` => still mildly data-limited at 2000 windows; 8k run tests the window lever.
- **Still preliminary, not paper-grade.** Levers left: more windows (genome affords ~190k 16kb
  windows), bigger student, 650M teacher, and student-vs-*ground-truth* eval (needs ENCODE bigwigs +
  pyBigWig) rather than student-vs-teacher fidelity.
- Lesson: the 30-min CPU proof caught the failure before burning a voyager H100 — keep proofs small-first.
