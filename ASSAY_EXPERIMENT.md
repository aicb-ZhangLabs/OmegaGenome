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

_Progress: 25/30 runs complete (5 running: atac/kd_s2, procap/kd_s1+base_s1+base_s2, eclip/base_s2). Multi-seed means below; full 3-seed table swaps into the rebuttal when 30/30 lands._

## Results — per-assay distilled vs from-scratch (8M student)

| assay | distilled (mean Pearson) | from-scratch | Δ (distilled−baseline) | seed pairs |
|---|---|---|---|---|
| atac | 0.6504±0.0005 | 0.6264±0.0013 | **0.0240±0.0018** | 2 |
| histone | 0.6227±0.0013 | 0.5861±0.0025 | **0.0366±0.0030** | 3 |
| rnaseq | 0.5056±0.0034 | 0.4738±0.0033 | **0.0318±0.0001** | 3 |
| procap | 0.4106 | 0.3827 | **0.0279** | 1 |
| eclip | 0.4636±0.0019 | 0.4505±0.0051 | **0.0131±0.0069** | 2 |

**atac** — per-track distilled−baseline Δ (mean over 2 seed pair(s)):

| track | Δ Pearson | distilled | baseline |
|---|---|---|---|
| ENCSR325NFE | **0.0157±0.0012** | 0.7452 | 0.7307 |
| ENCSR410DWV | **0.0319±0.0029** | 0.6535 | 0.6244 |
| ENCSR487QSB | **0.0277±0.0011** | 0.6080 | 0.5814 |
| ENCSR628PLS | **0.0297±0.0018** | 0.5364 | 0.5085 |
| ENCSR814RGG | **0.0150±0.0022** | 0.7065 | 0.6936 |

**histone** — per-track distilled−baseline Δ (mean over 3 seed pair(s)):

| track | Δ Pearson | distilled | baseline |
|---|---|---|---|
| ENCSR682BFG | **0.0464±0.0059** | 0.5488 | 0.5065 |
| ENCSR754DRC | **0.0289±0.0049** | 0.5005 | 0.4656 |
| ENCSR863PSM | **0.0681±0.0029** | 0.5964 | 0.5268 |
| ENCSR962OTG | **0.0029±0.0063** | 0.8412 | 0.8348 |

**rnaseq** — per-track distilled−baseline Δ (mean over 3 seed pair(s)):

| track | Δ Pearson | distilled | baseline |
|---|---|---|---|
| ENCSR527JGN_M | **0.0238±0.0126** | 0.3711 | 0.3329 |
| ENCSR527JGN_P | **0.0542±0.0120** | 0.5529 | 0.5150 |
| ENCSR619DQO_M | **0.0239±0.0093** | 0.5799 | 0.5452 |
| ENCSR619DQO_P | **0.0351±0.0092** | 0.5850 | 0.5609 |
| ENCSR701YIC | **0.0220±0.0049** | 0.4503 | 0.4264 |

**procap** — per-track distilled−baseline Δ (mean over 1 seed pair(s)):

| track | Δ Pearson | distilled | baseline |
|---|---|---|---|
| ENCSR046BCI_M | **0.0184** | 0.4219 | 0.4035 |
| ENCSR046BCI_P | **0.0502** | 0.3909 | 0.3407 |
| ENCSR100LIJ_M | **0.0176** | 0.4514 | 0.4337 |
| ENCSR100LIJ_P | **0.0463** | 0.4470 | 0.4008 |
| ENCSR114HGS_M | -0.0046 | 0.2817 | 0.2863 |
| ENCSR114HGS_P | **0.0087** | 0.2943 | 0.2856 |
| ENCSR799DGV_M | **0.0201** | 0.4482 | 0.4282 |
| ENCSR799DGV_P | **0.0520** | 0.4562 | 0.4042 |
| ENCSR935RNW_M | **0.0204** | 0.4556 | 0.4352 |
| ENCSR935RNW_P | **0.0496** | 0.4589 | 0.4093 |

**eclip** — per-track distilled−baseline Δ (mean over 2 seed pair(s)):

| track | Δ Pearson | distilled | baseline |
|---|---|---|---|
| ENCSR154HRN_M | **0.0007±0.0094** | 0.4376 | 0.4275 |
| ENCSR154HRN_P | **0.0108±0.0075** | 0.4478 | 0.4295 |
| ENCSR249ROI_M | **0.0033±0.0086** | 0.5203 | 0.5084 |
| ENCSR249ROI_P | **0.0069±0.0099** | 0.5036 | 0.4868 |
| ENCSR321PWZ_M | **0.0203±0.0054** | 0.5259 | 0.5109 |
| ENCSR321PWZ_P | **0.0334±0.0126** | 0.5698 | 0.5238 |
| ENCSR484LTQ_M | **0.0056±0.0022** | 0.3717 | 0.3683 |
| ENCSR484LTQ_P | **0.0138±0.0131** | 0.3967 | 0.3699 |
| ENCSR862QCH_M | **0.0226±0.0002** | 0.4372 | 0.4144 |
| ENCSR862QCH_P | **0.0140±0.0155** | 0.4444 | 0.4149 |

_Still running / not yet paired (5): atac/8m_kd_s2, eclip/8m_base_s2, procap/8m_base_s1, procap/8m_base_s2, procap/8m_kd_s1_
