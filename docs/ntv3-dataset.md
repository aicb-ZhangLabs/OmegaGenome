# NTv3 multi-track dataset & benchmark — detailed summary

Background reference for the `ntv3-multitrack` work. NTv3 = Nucleotide Transformer v3 (InstaDeep,
bioRxiv 2025.12.22.695963). It unifies representation learning, **functional-track + genome-
annotation prediction**, and controllable generation in one backbone.

## 1. The model (context for the data)
- **Architecture:** U-Net + transformer, **single-nucleotide tokenization**, context up to **1 Mb**.
- **Pretraining:** base-resolution masked language modeling on **~9 trillion bp** (OpenGenome2).
- **Post-training (what makes the tracks):** a joint objective adding **supervised learning on
  ~16,000 functional tracks + genome-annotation labels** across **24 animal and plant species**.
- Sizes: 8M / 100M / 650M (each `pre` and `post`). The `post` checkpoints carry the track heads.

## 2. What the supervised data is (~16k tracks across 24 species)
Two output families, both **per-base-pair**:

**(a) Functional signal tracks** — `bigwig_tracks_logits`, the quantitative seq-to-function
readouts (the regression target). For **human: 7,362 tracks**, drawn from multiple consortia:
- **ENCODE** (`ENCSR…` accessions, ~5,600): histone ChIP-seq, TF ChIP-seq, DNase-seq, ATAC-seq,
  RNA-seq, etc., across many cell lines/tissues.
- **FANTOM5 CAGE** (`CNhs…_P/_M`, plus/minus strand): transcription-start-site activity.
- ~222 `kai*` internal tracks.
Other species have fewer (mouse 2,450; arabidopsis 1,899; soybean 590; cotton 319; fly 180; …),
so human is by far the richest and the natural target for our distillation.

**(b) Genome-annotation tracks** — `bed_tracks_logits`, **21 human-readable element types**:
`protein_coding_gene, lncRNA, exon, intron, splice_donor, splice_acceptor, CTCF-bound,
polyA_signal, enhancer_Tissue_specific, enhancer_Tissue_invariant, promoter_Tissue_specific,
promoter_Tissue_invariant, 5UTR±, 3UTR±, skipped_exon, always_on_exon, start_codon, stop_codon,
ORF`. These need no metadata join (already named), so they make a clean per-bp annotation task.

These map to the canonical sequence-to-function assay taxonomy used by **Enformer** and **Borzoi**
(CAGE / DNase / ATAC / ChIP-histone / ChIP-TF / RNA-seq), the readouts those models are evaluated on.

## 3. The "NTv3 Benchmark" (downstream evaluation)
- **106 tasks**, defined on **natural long-range inputs of 32 kb** with **base-pair-resolution
  outputs** (a track or annotation), spanning new assays and species.
- Strict **data-leakage controls** (held-out chromosomes / regions) for realistic genome-wide eval.
- Metric: **per-track/per-position correlation (Pearson)** between predicted and observed signal —
  the field standard (same as Enformer/Borzoi).
- NTv3 reports SOTA on this benchmark vs sequence-to-function and foundation-model baselines.

## 4. How we use it in OmegaGenome (distillation)
- **Teacher:** NTv3-650M-post (or 100M-post) → per-bp `bigwig_tracks_logits` `[B, L_out, T]`
  (1024 bp → 384 bins; tracks last dim).
- **Distillation target:** the teacher's per-bp tracks on a set of 32 kb windows — so **no external
  benchmark labels are needed to train the student**; ground-truth tracks are only needed if we want
  to evaluate the student against measured signal (a later enhancement).
- **Student:** `BPNetRegressor` (BPNet stem + 1×1-conv track head), aligned to the teacher's `L_out`.
- **Metric:** per-track Pearson (`track_metrics.py`).
- **Chosen subset (see NTV3_MULTITRACK_PLAN §4c):** 6 histone marks + DNase + CAGE in K562 (v1),
  expanding to K562/GM12878/HepG2 (v2) — picked for the multi-track regression benchmark.

## 5b. NTv3's OWN reported performance (the teacher ceiling — from the paper)

Source: NTv3 paper PDF (`instadeep.com/.../NT_v3.pdf`, extracted text). These are **NTv3-vs-measured-
signal** PCC (the teacher's quality), NOT our student-vs-teacher fidelity — different targets.
- **DNase-seq, base-resolution PCC:** NTv3-650M-post = **0.753 (HepG2), 0.755 (IMR-90)** vs
  ChromBPNet 0.704 / 0.717. So the per-bp teacher ceiling is **~0.75 for DNase**.
- **vs Borzoi:** at Borzoi's native 32bp regime, NTv3 beats Borzoi by **up to +3%** on CAGE/ChIP/ATAC/
  DNase (RNA-seq −0.5%). At base resolution NTv3 *matches* Borzoi on DNase/ChIP, *beats* on ATAC/CAGE/RNA.
- **Model-size scaling (teacher, fine-tuned per task):** 8M < 100M < 650M < 650M-post, monotonic.
  NOTE: this is the TEACHER's own quality scaling — distinct from our distillation, where 650M-teacher
  ≈ 100M-teacher (the small *student* is the bottleneck, not the teacher).
- **Data regime:** NTv3 fine-tunes **genome-wide** (Borzoi's regime ~whole genome), 32kb→1Mb context;
  longer context helps, especially expression/distal modalities. Benchmark = 106 tasks, 32kb, base-res.

## 5c. OUR independent teacher eval (2026-06-17) — crop-validated, NOT a verified paper reproduction

Eval code: `src/train/eval_teacher.py` + `src/data/ground_truth.py` (OUR code — NTv3 ships no public
eval/benchmark/preprocessing code; their GitHub `notebooks/` has only inference examples).
**Key fix (correct):** NTv3-post crops track outputs to the **central 37.5%** of the input (per the
GitHub v3 doc); aligning ground truth to that region at 1 bp took the mean from **0.197 -> 0.611**.
The jump proves the alignment was the bug and that NTv3 is a strong teacher.

**Second fix (paper metric):** the paper computes Pearson after **log(1+x) on BOTH** prediction and
truth (methods, line ~1471). With log1p, **DNase = 0.742 ≈ the paper's 0.75** anchor.

NTv3-650M-post vs ENCODE "signal p-value", K562, 400 chr8 windows (16 kb -> central 6144 bp @ 1 bp):

| track | raw PCC | **log1p (paper metric)** | track | raw | log1p |
|---|---|---|---|---|---|
| DNase-seq | 0.916 | **0.742** | H3K4me2 | 0.819 | 0.522 |
| H3K27me3 | 0.748 | 0.785 | H3K4me3 | 0.807 | 0.467 |
| H3K9ac | 0.779 | 0.527 | H3K4me1 | 0.413 | 0.356 |
| H2AFZ | 0.481 | 0.291 | H3K36me3 | 0.279 | 0.206 |
| H3K9me3 | 0.257 | 0.243 | **MEAN** | **0.611** | **0.460** |

**CAVEATS — do NOT call this a paper reproduction:**
- DNase **0.92 != paper 0.75** (different cell line: K562 vs their HepG2/IMR-90).
- **Possible chr8 train-leakage** (NTv3 trained genome-wide; held-out split unknown) -> 0.92 may be inflated.
- **Ground-truth target/normalization unverified** — used raw signal-p-value + mean-bin; paper applies
  power-squash/clip transforms not matched here (PCC survives linear scaling, not nonlinear).
- Window selection (400 tiled chr8) != their genome-wide held-out protocol.
- For a defensible number: eval on a confidently held-out chrom, match the GT processing, and compare
  like-for-like (e.g., NTv3 on HepG2 DNase -> compare to their 0.753). Metric aggregation (pooled
  per-track PCC) IS sound. Our BPNet student reaches only ~0.22 student-vs-teacher fidelity regardless.

### 5c-broad. BROADENED zero-shot native eval (2026-06-18) — 22 tracks × 3 cell lines

Same eval, widened from 9 K562 tracks to **22 native tracks across K562 / HepG2 / GM12878** (DNase +
8 histone marks), to characterize the teacher across cell types and the full histone panel.
NTv3-650M-post, 200 chr10 windows (16 kb -> central 6144 bp @ 1 bp), **log1p (paper metric)**.
Manifest: `config/distillation/ntv3_native_broad.json`. Job 237475.

**Overall mean PCC = 0.489** (22 tracks). Note: chr10 (NOT chr8) — less leakage-prone than §5c.

| by cell line | log1p PCC (n) | | by assay | log1p PCC (n) |
|---|---|---|---|---|
| HepG2 | 0.511 (8) | | DNase-seq | **0.711** (1) |
| K562 | 0.494 (7) | | H3K4me1 | 0.624 (3) |
| GM12878 | 0.460 (7) | | H3K27ac | 0.560 (1) |
| | | | H3K4me2 | 0.553 (3) |
| | | | H3K9ac | 0.517 (3) |
| | | | H3K4me3 | 0.484 (3) |
| | | | H3K36me3 | 0.419 (3) |
| | | | H3K27me3 | 0.366 (3) |
| | | | H3K9me3 | 0.302 (2) |

**Reading it:** DNase (0.711) ≈ the paper's ~0.75 anchor; **active/sharp marks** (H3K4me1/2/3, H3K27ac,
H3K9ac: 0.48-0.62) predict well; **broad repressive domains** (H3K27me3, H3K9me3: 0.30-0.37) are the
weakest — expected, they are diffuse/noisy at 1 bp and hardest to localize. Cell lines are comparable
(0.46-0.51), so the teacher generalizes across cell types, not just K562. Best single track HepG2
H3K4me1 = 0.765; worst HepG2 H3K27me3 = 0.124 / GM12878 H3K27me3 = 0.166 / K562 H3K9me3 = 0.130.
Same caveats as §5c (single replicate per track, GT normalization unverified, native-track zero-shot
≠ the held-out *benchmark* which requires fine-tuning — see §5d/§5e).

## 5d. THEIR benchmark dataset IS open-sourced on HF (2026-06-17) — use it for real reproduction

`InstaDeepAI/NTv3_benchmark_dataset` (HF dataset) provides everything needed for a faithful eval,
per species:
- `human/functional_tracks/*.bigwig` — their EXACT ground-truth tracks (34 human, `ENCSR..._M/_P` strands)
- `human/splits.bed` — the EXACT train/val/test split -> **leakage-free** eval (fixes the chr8 risk)
- `human/genome.fasta` — exact sequences
- `benchmark_metadata.tsv` — per-track **mean/std** (for normalization) + assay type
- Results: `InstaDeepAI/ntv3_benchmark` space -> `data/ntv3_benchmark_results.csv` = per-(model,track)
  Pearson/MCC for NTv2-500M, **BPNet arch. 6M**, NTv3 650M (pre/pos), etc. -> the numbers to match.

**Paper eval procedure (methods, line ~1471):** predictions unscaled to raw, then BOTH pred and
truth **log(1+x)** transformed before Pearson. The benchmark is a **fine-tuning suite** (README:
"fine-tuning") -> reproducing its numbers means fine-tuning per task, distinct from NTv3's native
zero-shot track output (what we distill).

**=> Switch our eval to their data**: use `human/splits.bed` test regions + their `functional_tracks`
bigwigs + mean/std + log1p, compare to the CSV. The CSV's **BPNet-6M baseline is directly comparable
to our BPNet student.** My earlier ad-hoc ENCODE/chr8 eval (0.61/0.92) is superseded by this. GPU work
(fine-tune/eval) deferred until LoRA finishes.

## 5e. PAPER BENCHMARK TARGETS (the reproduction goal) — from their results CSV (2026-06-17)

Extracted from `InstaDeepAI/ntv3_benchmark` -> `data/ntv3_benchmark_results.csv` (mean Pearson per
assay over the human functional tracks). This is what our fine-tuned NTv3 must match and what our
distilled student must beat (vs **BPNet-6M**, the architecture closest to our student):

| assay (human) | NTv3-650M (post) | NTv3-650M (pre) | **BPNet-6M** | NTv2-500M |
|---|---|---|---|---|
| ATAC-seq | **0.759** | 0.698 | 0.571 | 0.497 |
| Histone ChIP-seq | **0.717** | 0.659 | 0.520 | 0.532 |
| RNA-seq | **0.695** | 0.677 | 0.330 | 0.452 |
| eCLIP | **0.584** | 0.565 | 0.283 | 0.397 |
| PRO-cap | **0.508** | 0.453 | 0.398 | 0.215 |

- Other baselines in the CSV: Caduceus-7M, Evo2-1B, HyenaDNA-7M, Residual-CNN-44M/700k, NTv3-8M/100M (pre).
- Our targets: fine-tuned NTv3-650M -> ~these numbers; distilled student -> **beat BPNet-6M** (0.52-0.57).
- NB: for our 9 K562 *native* tracks (§5c), the paper only reports DNase precisely (~0.75); the
  benchmark Histone-ChIP 0.717 is a *fine-tuned, held-out-track* number, not directly comparable to
  our zero-shot native histone eval.

## 5. Practical notes
- Track names are bare accessions, so selecting "H3K4me3 in K562" requires an
  **accession → (assay, biosample) join** (ENCODE portal API for `ENCSR…`; FANTOM5 sample table for
  `CNhs…`). This is the one data step before training a chosen subset.
- NTv3 modeling code is gated (`InstaDeepAI/ntv3_base_model`); load via the local-snapshot bypass
  (`prepare_local_snapshot`) or with the HF token. Caches live on `/extra` to spare the home quota.
