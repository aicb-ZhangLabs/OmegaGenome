# NTv3 multi-track — real input/output example

One real TEST window from `ntv3_benchmark_data` (human, hg38). Long axes truncated with `…`.

## Shapes (per window)
```
INPUT   tokens         : (32768,)    # 32768 single-nt ids (A=6 T=7 C=8 G=9 N=10) — DNA context
OUTPUT  tracks (target): (12288, 34)   # [L_out, num_tracks]; L_out = 0.375*32768 = 12288 bp (central crop)
          per track    : [12288]                 # one non-negative signal value PER BP, for each of 34 tracks
```

## INPUT — DNA (first 60 of 32,768 bp)
```
token ids : [6, 7, 9, 9, 9, 6, 6, 6, 9, 7, 9, 8] …  (len 32768)
as DNA    : ATGGGAAAGTGCACTCAGGTCTTGAAAGAGTCAGGAGTGGCCAGGGCAAAGAACATGAAC …
```

## OUTPUT / TARGET — per-bp signal, one row per track (34 tracks)

Each track's target is a length-12288 per-bp vector. Showing **first 6 bp … peak bp … last 3 bp** of each.

| # | track id | assay | per-bp target  (first6 … max@pos … last3) |
|---|---|---|---|
| 0 | ENCSR046BCI_M | PRO-cap | `-0.00 -0.00 -0.00 -0.00 -0.00 -0.00` … `max@8494=4.0` … `-0.00 -0.00 -0.00` |
| 1 | ENCSR046BCI_P | PRO-cap | `0.00 0.00 0.00 0.00 0.00 0.00` … `max@8624=1.5` … `0.00 0.00 0.00` |
| 2 | ENCSR100LIJ_M | PRO-cap | `-0.00 -0.00 -0.00 -0.00 -0.00 -0.00` … `max@0=-0.0` … `-0.00 -0.00 -0.00` |
| 3 | ENCSR100LIJ_P | PRO-cap | `0.00 0.00 0.00 0.00 0.00 0.00` … `max@8405=0.5` … `0.00 0.00 0.00` |
| 4 | ENCSR114HGS_M | PRO-cap | `-0.00 -0.00 -0.00 -0.00 -0.00 -0.00` … `max@4539=8.0` … `-0.00 -0.00 -0.00` |
| 5 | ENCSR114HGS_P | PRO-cap | `0.00 0.00 0.00 0.00 0.00 0.00` … `max@8566=0.5` … `0.00 0.00 0.00` |
| 6 | ENCSR154HRN_M | eCLIP | `-0.00 -0.00 -0.00 -0.00 -0.00 -0.00` … `max@8410=0.7` … `-0.00 -0.00 -0.00` |
| 7 | ENCSR154HRN_P | eCLIP | `0.00 0.00 0.00 0.00 0.00 0.00` … `max@0=0.0` … `0.00 0.00 0.00` |
| 8 | ENCSR249ROI_M | eCLIP | `-0.00 -0.00 -0.00 -0.00 -0.00 -0.00` … `max@2643=0.0` … `-0.00 -0.00 -0.00` |
| 9 | ENCSR249ROI_P | eCLIP | `0.00 0.00 0.00 0.00 0.00 0.00` … `max@0=0.0` … `0.00 0.00 0.00` |
| 10 | ENCSR321PWZ_M | eCLIP | `-0.00 -0.00 -0.00 -0.00 -0.00 -0.00` … `max@8458=0.2` … `-0.00 -0.00 -0.00` |
| 11 | ENCSR321PWZ_P | eCLIP | `0.00 0.00 0.00 0.00 0.00 0.00` … `max@0=0.0` … `0.00 0.00 0.00` |
| 12 | ENCSR325NFE | ATAC-seq | `0.21 0.21 0.18 0.18 0.18 0.18` … `max@7721=6.5` … `0.10 0.10 0.10` |
| 13 | ENCSR410DWV | ATAC-seq | `0.03 0.03 0.03 0.03 0.03 0.03` … `max@4470=9.4` … `0.02 0.02 0.01` |
| 14 | ENCSR484LTQ_M | eCLIP | `-0.00 -0.00 -0.00 -0.00 -0.00 -0.00` … `max@8435=0.4` … `-0.00 -0.00 -0.00` |
| 15 | ENCSR484LTQ_P | eCLIP | `0.00 0.00 0.00 0.00 0.00 0.00` … `max@0=0.0` … `0.00 0.00 0.00` |
| 16 | ENCSR487QSB | ATAC-seq | `0.10 0.10 0.10 0.10 0.10 0.10` … `max@8521=7.4` … `0.29 0.29 0.29` |
| 17 | ENCSR527JGN_M | polyA plus RNA-seq | `0.00 0.00 0.00 0.00 0.00 0.00` … `max@9491=0.0` … `0.00 0.00 0.00` |
| 18 | ENCSR527JGN_P | polyA plus RNA-seq | `0.00 0.00 0.00 0.00 0.00 0.00` … `max@0=0.0` … `0.00 0.00 0.00` |
| 19 | ENCSR619DQO_M | total RNA-seq | `0.00 0.00 0.00 0.00 0.00 0.00` … `max@0=0.0` … `0.00 0.00 0.00` |
| 20 | ENCSR619DQO_P | total RNA-seq | `0.00 0.00 0.00 0.00 0.00 0.00` … `max@0=0.0` … `0.00 0.00 0.00` |
| 21 | ENCSR628PLS | ATAC-seq | `0.00 0.00 0.00 0.00 0.00 0.00` … `max@8522=10.2` … `0.94 0.94 0.94` |
| 22 | ENCSR682BFG | Histone ChIP-seq | `0.00 0.00 0.00 0.00 0.00 0.00` … `max@8536=1.1` … `0.00 0.00 0.00` |
| 23 | ENCSR701YIC | total RNA-seq | `0.00 0.00 0.00 0.00 0.00 0.00` … `max@545=0.0` … `0.00 0.00 0.00` |
| 24 | ENCSR754DRC | Histone ChIP-seq | `0.13 0.13 0.13 0.13 0.13 0.13` … `max@8556=1.3` … `0.26 0.26 0.26` |
| 25 | ENCSR799DGV_M | PRO-cap | `-0.00 -0.00 -0.00 -0.00 -0.00 -0.00` … `max@8494=6.5` … `-0.00 -0.00 -0.00` |
| 26 | ENCSR799DGV_P | PRO-cap | `0.00 0.00 0.00 0.00 0.00 0.00` … `max@8577=0.5` … `0.00 0.00 0.00` |
| 27 | ENCSR814RGG | ATAC-seq | `0.25 0.25 0.25 0.25 0.25 0.25` … `max@8519=5.9` … `0.22 0.22 0.22` |
| 28 | ENCSR862QCH_M | eCLIP | `-0.00 -0.00 -0.00 -0.00 -0.00 -0.00` … `max@8463=1.1` … `-0.00 -0.00 -0.00` |
| 29 | ENCSR862QCH_P | eCLIP | `0.00 0.00 0.00 0.00 0.00 0.00` … `max@0=0.0` … `0.00 0.00 0.00` |
| 30 | ENCSR863PSM | Histone ChIP-seq | `0.65 0.65 0.65 0.65 0.65 0.65` … `max@9760=3.3` … `0.11 0.11 0.11` |
| 31 | ENCSR935RNW_M | PRO-cap | `-0.00 -0.00 -0.00 -0.00 -0.00 -0.00` … `max@4539=0.5` … `-0.00 -0.00 -0.00` |
| 32 | ENCSR935RNW_P | PRO-cap | `0.00 0.00 0.00 0.00 0.00 0.00` … `max@4524=0.5` … `0.00 0.00 0.00` |
| 33 | ENCSR962OTG | Histone ChIP-seq | `0.13 0.13 0.13 0.13 0.13 0.13` … `max@8670=44.1` … `0.00 0.00 0.00` |

## Reading this

- One window: `tokens[32768]` → model → `pred[12288, 34]` (a per-bp value for every nt in the central 12288 bp, for every track).
- **Single-track** student (`--track_subset i`): same input, output `[12288, 1]` = column *i* only.
- **Distillation** aligns per position: student bp *j* ↔ teacher bp *j* ↔ target bp *j* (element-wise).
- **Pearson** is computed per track across all (window, position) pairs, then averaged over tracks.
- Most tracks are **sparse** (mostly 0 with peaks at regulatory sites) — typical of ATAC/Histone/PRO-cap bigWigs.
