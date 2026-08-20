# BPNet from-scratch baseline: single-seed vs 3-seed (200ep / 100ep)

Read-only analysis for choosing the fairest BPNet baseline. **No paper table or CSV has been modified.**

## Sources
- **Single-seed baseline (current paper value)** = `Type=Baseline, Model=BPNet` rows in
  `OmegaGenome_Revise_202606/plot_repo/data/model_comparison_5teacher_formal.csv` (Std column is 0.0 = single run). This is the canonical current baseline.
- **200-epoch 3-seed sweep**: `/srv/disk00/sshfs/pengchx3/bpnet_scratch_3seed/` — 54 `final_summary.json` (18 tasks × 3 seeds).
- **100-epoch 3-seed sweep**: `/srv/disk00/sshfs/pengchx3/bpnet_scratch_3seed_100ep/` — 54 `final_summary.json`.
- `best_test_mcc` = test MCC of the **best-VAL-selected** checkpoint; `final_test_mcc` = test MCC of the **LAST-epoch** checkpoint. std is sample std (ddof=1) over 3 seeds.

## Comparison table (18 tasks)

| Task | single-seed (paper) | 200ep best-val | 200ep last-ep | 100ep best-val | 100ep last-ep | Δ(200best−single) |
|---|---|---|---|---|---|---|
| H2AFZ | 0.4568 | 0.4831±0.0184 | 0.3477±0.0404 | 0.4831±0.0184 | 0.3693±0.0216 | +0.0263 |
| H3K27ac | 0.4275 | 0.4693±0.0060 | 0.3182±0.0240 | 0.4693±0.0060 | 0.3222±0.0222 | +0.0418 |
| H3K27me3 | 0.5497 | 0.5506±0.0105 | 0.4262±0.0355 | 0.5506±0.0105 | 0.4351±0.0072 | +0.0009 |
| H3K36me3 | 0.5546 | 0.5586±0.0054 | 0.4419±0.0214 | 0.5586±0.0054 | 0.4470±0.0096 | +0.0040 |
| H3K4me1 | 0.4606 | 0.4645±0.0020 | 0.3201±0.0046 | 0.4645±0.0020 | 0.3277±0.0051 | +0.0039 |
| H3K4me2 | 0.5366 | 0.5367±0.0095 | 0.4225±0.0222 | 0.5367±0.0095 | 0.3781±0.0138 | +0.0001 |
| H3K4me3 | 0.6238 | 0.6113±0.0142 | 0.5090±0.0210 | 0.6113±0.0142 | 0.5407±0.0238 | -0.0126 |
| H3K9ac | 0.5213 | 0.5043±0.0236 | 0.3839±0.0209 | 0.5043±0.0236 | 0.3985±0.0308 | -0.0170 |
| H3K9me3 | 0.3295 | 0.4105±0.0294 | 0.2666±0.0373 | 0.4105±0.0294 | 0.2921±0.0305 | +0.0810 |
| H4K20me1 | 0.6061 | 0.6127±0.0033 | 0.4797±0.0122 | 0.6127±0.0033 | 0.4786±0.0140 | +0.0066 |
| enhancers | 0.4727 | 0.4945±0.0042 | 0.3569±0.0228 | 0.4945±0.0042 | 0.3657±0.0117 | +0.0218 |
| enhancers_types | 0.4530 | 0.4585±0.0100 | 0.3162±0.0182 | 0.4585±0.0100 | 0.3506±0.0209 | +0.0055 |
| promoter_all | 0.6768 | 0.7131±0.0019 | 0.6607±0.0059 | 0.7131±0.0019 | 0.6732±0.0247 | +0.0363 |
| promoter_no_tata | 0.6983 | 0.7233±0.0104 | 0.6708±0.0144 | 0.7233±0.0104 | 0.6702±0.0111 | +0.0250 |
| promoter_tata | 0.7833 | 0.7998±0.0230 | 0.8022±0.0095 | 0.7964±0.0185 | 0.8024±0.0189 | +0.0165 |
| splice_sites_acceptors | 0.8280 | 0.8637±0.0136 | 0.8606±0.0179 | 0.8413±0.0114 | 0.8391±0.0269 | +0.0357 |
| splice_sites_all | 0.7571 | 0.8271±0.0104 | 0.8180±0.0035 | 0.8086±0.0155 | 0.7924±0.0056 | +0.0700 |
| splice_sites_donors | 0.8527 | 0.8716±0.0293 | 0.8762±0.0318 | 0.8341±0.0367 | 0.8422±0.0345 | +0.0189 |
| **18-task mean** | **0.5882** | **0.6085** | **0.5154** | **0.6040** | **0.5181** | **+0.0203** |

## Analysis

### 1. Does 3-seed best-val raise the 18-task mean vs single-seed?
Yes. Single-seed mean **0.5882** → 200ep 3-seed best-val **0.6085** (**Δ = +0.0203**). 100ep best-val = 0.6040 (Δ = +0.0157).
Direction: **16 / 18 tasks up**, 2 down (H3K4me3 −0.0126, H3K9ac −0.0170; both small, within seed std). The single biggest driver is H3K9me3 (+0.0810) and splice_sites_all (+0.0700) — see below.

### 2. 200ep vs 100ep — overfitting and where the single-seed value sits
- **best-val is essentially budget-invariant.** For 14/18 tasks the 200ep and 100ep best-val numbers are *bit-identical* (same 3 seeds; the best-val checkpoint lands within the first 100 epochs). They differ only for the 3 splice tasks and promoter_tata, where 200ep found a later-epoch best-val checkpoint (max per-task gap 0.0375; 200ep ≥ 100ep on all four). So "best-val" as a baseline is robust to the epoch budget.
- **last-epoch overfits, and 200ep overfits slightly more.** Mean best−final gap = **0.0931 (200ep)** vs **0.0859 (100ep)**. The gap is severe on the histone/enhancer marks (~0.10–0.14, e.g. H3K27ac 0.469→0.318) and near-zero on the easy splice/promoter_tata tasks (last-epoch even edges out best-val there because those tasks don't overfit). Longer training = more degradation of the last-epoch checkpoint, exactly as expected.
- **The single-seed paper value tracks best-val, not last-epoch.** mean|single − 200best| = **0.0236** vs mean|single − 200final| = 0.0879 (and |single−100best| = 0.0211). So the current paper baseline was already effectively a best-val-selected number. Adopting a *last-epoch* 3-seed baseline would silently drop the baseline ~0.07–0.09 MCC and is **not** apples-to-apples with how the paper value was produced.

### 3. Recommendation (baseline NOT yet swapped — recommendation only)
**Use the 200-epoch 3-seed `best_test_mcc` (mean±std) as the BPNet baseline.** Justification:
- **Consistent with the existing baseline's selection rule** (best-val), so swapping it in is a like-for-like upgrade from n=1 → n=3, not a redefinition. mean moves only +0.0203 and now carries an honest error bar.
- **Budget-robust**: best-val is identical between 100ep and 200ep on most tasks, and 200ep dominates on the 4 tasks that benefit from longer training — so 200ep best-val is the *upper envelope* of the defensible best-val numbers and can't be accused of under-training BPNet.
- **Fairest to BPNet**: it gives the from-scratch baseline its full training budget and best checkpoint, which is the strongest, least-attackable form of the baseline (a reviewer cannot argue BPNet was hobbled).
- 100ep best-val (0.6040) is an acceptable fallback if a shorter, cheaper protocol is wanted — it's only −0.0045 below 200ep and identical on 14 tasks — but 200ep is strictly more defensible.
- **Avoid** any last-epoch variant as the headline baseline: it is a different (worse) selection rule than the current paper number and would misleadingly deflate BPNet by ~0.09.

### 4. Outlier / failed-run flags
**None.** No run collapsed. Across all 108 runs the minimum `best_test_mcc` = 0.377 (H3K9me3) and minimum `final_test_mcc` = 0.227 (H3K9me3); nothing is near 0 or NaN, and per-seed spreads are modest (largest best-val std = splice_sites_donors 0.037, then H3K9me3 0.029). No seed needs to be dropped; the means above are clean.

**Note on H3K9me3**: single-seed = 0.3295 was an *unlucky low n=1 draw* — all three 3-seed best-val runs (0.377, 0.423, 0.432) sit well above it, so the +0.0810 jump there is the single-seed value being anomalously low, not the 3-seed being inflated. This is the strongest single argument that the current n=1 baseline is noisy and benefits from seed-averaging.
