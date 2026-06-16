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

## 5. Status
- Branch `ntv3-multitrack` created (worktree `code_ntv3/`). No code written yet — this is the
  understanding + design + cost-estimate plan (per "first understand" before implementing).
- Next: resolve data access (#1), then scaffold the loader + `BPNetRegressor` + one probe task to
  measure real cost, then extrapolate to the chosen subset.
