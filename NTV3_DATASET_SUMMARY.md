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
  expanding to K562/GM12878/HepG2 (v2) — picked to answer reviewer R1.1a/b/c + R2.1a.

## 5. Practical notes
- Track names are bare accessions, so selecting "H3K4me3 in K562" requires an
  **accession → (assay, biosample) join** (ENCODE portal API for `ENCSR…`; FANTOM5 sample table for
  `CNhs…`). This is the one data step before training a chosen subset.
- NTv3 modeling code is gated (`InstaDeepAI/ntv3_base_model`); load via the local-snapshot bypass
  (`prepare_local_snapshot`) or with the HF token. Caches live on `/extra` to spare the home quota.
