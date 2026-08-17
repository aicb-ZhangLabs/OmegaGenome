# R1.1c per-assay multi-track: distillation vs from-scratch (base-resolution regression)

**Goal (rebuttal R1.1c "multiple contexts").** For each of the 5 ENCODE assay families, train ONE 8M
NTv3 student on *all tracks of that assay* two ways — distilled from the NTv3-650M teacher (KD) and
from-scratch (baseline) — and compare per-track Pearson. Strengthens R1.1c beyond one-track-per-context
by showing distillation helps within each assay context, per assay.

## Setup (reuses `src/train/finetune_ntv3.py` — no new trainer)
- Student: `ntv3-8m` (7.7M), seq 32768, central-0.375 crop, Poisson-multinomial, 19,932 steps, seed {0,1,2}.
- **Distilled**: `--track_subset "<assay idxs>" --cached_teacher_logits <assay .pt>` , kd_w_ce/kl/mse = .5/.5/0,
  gt=poisson_multinomial, distill=standardized_mse (== the per-track KD recipe).
- **Baseline**: same `--track_subset`, NO teacher, **data-matched** `--train_overlap 0 --max_train_samples
  64000` → the SAME 63,707-window stream as KD (only difference = teacher on/off).
- Launcher: `slurm/ntv3_assay_experiment.sbatch` (array idx→(assay,mode,seed): assay=idx%5,
  mode=(idx//5)%2, seed=idx//10). 1 GPU/job, `--mem 64G`, laniakea/voyager (galaxy excluded: seq-32768),
  full mount-guard + hang-watchdog hardening reused verbatim.

## Assays → dataset `--track_subset` indices (verified vs canonical NTV3_HUMAN_TRACKS + cache track_ids)
| assay | idxs | #tracks | cache |
|---|---|---|---|
| atac | 12,13,16,21,27 | 5 | assay/atac.pt (7.8G) |
| histone | 22,24,30,33 | 4 | assay/histone.pt (6.3G) |
| rnaseq | 17,18,19,20,23 | 5 | assay/rnaseq.pt (7.8G) |
| procap | 0,1,2,3,4,5,25,26,31,32 | 10 | assay/procap.pt (15.7G) |
| eclip | 6,7,8,9,10,11,14,15,28,29 | 10 | assay/eclip.pt (15.7G) |

## Teacher caches — sliced from the joint-34 cache (no teacher forward)
`scripts/build_assay_cache_from_joint.py` slices the assay's channels out of
`ntv3_targets/teacher_cache_joint34/logits.npy` (63707×12288×34 fp16); each byte-verified vs the joint
cache. Multi-track generalization of the verified single-track `build_pertrack_cache_from_joint.py`.

## Provenance / restore
The benchmark data + teacher had been cleaned off the galaxy SSD; restored from the HF backup dataset
`explcre/galaxy-ssd-pengchx3-backup` (2026-08-16): `ntv3_benchmark_data/` (34 bigwigs + genome.fasta +
splits) and `ntv3_targets/teacher_cache_joint34/` (53G logits.npy + meta). Restore driver:
`scratchpad/dl_ntv3.py` (per-file hf_hub_download; the `--include` path list-repo-trees the whole huge
backup and hangs). meta: limit_num_samples 64000, num_windows 63707, overlap 0.0.

## Dry-run (both arms, atac) — PASSED
KD: 5 ATAC bigwigs validated, cached teacher (63707,12288,5) aligned to 63,707 windows, TEST Pearson 0.126
in 2 steps. Baseline: same 63,707 windows, TEST Pearson 0.058. Code confirmed correct.

## Launch
`sbatch --array=0-29%8 slurm/ntv3_assay_experiment.sbatch` → array **276399** (2026-08-16), 30 jobs.
Results per run: `$SSD/ntv3_targets/assay_experiment/<assay>/8m_<mode>_s<seed>/ntv3_finetune_result.json`
(`test_mean_pearson`, `test_pearson_by_assay`, `per_track_pearson`). Aggregate KD delta per assay/track
once seed 0 lands.

## Results — per-assay distilled vs from-scratch (8M student)
_Auto-refreshed as each assay's kd+base pair (same seed) completes, via `scripts/aggregate_assay_results.py`. Δ = distilled − from-scratch per-track Pearson (positive = distillation helps); mean±std over completed seed pairs._


| assay | distilled (mean Pearson) | from-scratch | Δ (distilled−baseline) | seed pairs |
|---|---|---|---|---|
| atac | 0.6499 | 0.6277 | **0.0222** | 1 |
| histone | 0.6217 | 0.5834 | **0.0383** | 1 |
| rnaseq | 0.5078 | 0.4761 | **0.0317** | 1 |
| procap | (no complete seed pair yet) | | | 3 |
| eclip | (no complete seed pair yet) | | | 3 |

**atac** — per-track distilled−baseline Δ (mean over 1 seed pair(s)):

| track | Δ Pearson | distilled | baseline |
|---|---|---|---|
| ENCSR325NFE | **0.0146** | 0.7452 | 0.7307 |
| ENCSR410DWV | **0.0291** | 0.6535 | 0.6244 |
| ENCSR487QSB | **0.0266** | 0.6080 | 0.5814 |
| ENCSR628PLS | **0.0279** | 0.5364 | 0.5085 |
| ENCSR814RGG | **0.0128** | 0.7065 | 0.6936 |

**histone** — per-track distilled−baseline Δ (mean over 1 seed pair(s)):

| track | Δ Pearson | distilled | baseline |
|---|---|---|---|
| ENCSR682BFG | **0.0423** | 0.5488 | 0.5065 |
| ENCSR754DRC | **0.0350** | 0.5005 | 0.4656 |
| ENCSR863PSM | **0.0695** | 0.5964 | 0.5268 |
| ENCSR962OTG | **0.0063** | 0.8412 | 0.8348 |

**rnaseq** — per-track distilled−baseline Δ (mean over 1 seed pair(s)):

| track | Δ Pearson | distilled | baseline |
|---|---|---|---|
| ENCSR527JGN_M | **0.0382** | 0.3711 | 0.3329 |
| ENCSR527JGN_P | **0.0378** | 0.5529 | 0.5150 |
| ENCSR619DQO_M | **0.0347** | 0.5799 | 0.5452 |
| ENCSR619DQO_P | **0.0241** | 0.5850 | 0.5609 |
| ENCSR701YIC | **0.0239** | 0.4503 | 0.4264 |

_Still running / not yet paired (21): atac/8m_base_s1, atac/8m_base_s2, atac/8m_kd_s2, eclip/8m_base_s0, eclip/8m_base_s1, eclip/8m_base_s2, eclip/8m_kd_s0, eclip/8m_kd_s1, eclip/8m_kd_s2, histone/8m_base_s1, histone/8m_base_s2, histone/8m_kd_s1, histone/8m_kd_s2, procap/8m_base_s0, procap/8m_base_s1, procap/8m_base_s2, procap/8m_kd_s1, procap/8m_kd_s2, rnaseq/8m_base_s1, rnaseq/8m_base_s2, rnaseq/8m_kd_s2_

**Complete pairs so far (3 assays: atac, histone, rnaseq):** distillation helps on **every track of every completed assay**, under the data-matched setting where the ONLY difference is the 650M teacher's soft targets. Table refreshes as remaining assays/seeds land.
