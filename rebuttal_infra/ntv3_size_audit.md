# NTv3 size-ladder infra — INDEPENDENT audit (adversarial)

Auditor: separate agent (not the implementer). Method: verify from code + real execution, not the
self-report. Env: `code_carbon/.venv_carbon_portable/bin/python` (torch 2.6.0+cu118, transformers 4.57.1).
CPU checks on galaxy login; GPU checks via SLURM on laniakea (RTX 6000 Ada / A6000 49 GB). Nothing
committed; no full sweep or full memmap precompute launched.

Files audited (working-tree, uncommitted), all under `OmegaGenome_Revise_202606/`:
`code_ntv3/src/model/ntv3_finetune.py`, `code_ntv3/src/train/finetune_ntv3.py`,
`code_ntv3/src/trainer/teacher_cache.py`, `code_ntv3/src/train/precompute_teacher_logits.py`.

---

## Verdict per checklist item

### 1. from_config random-init correctness — PASS
Built each tier with `build_bigwig_model(<key>, num_tracks=34)` on CPU.

(a) Param counts (measured backbone, T=34 head excluded):

| tier | measured backbone | design target | err |
|---|---|---|---|
| ntv3-4m | 4.344 M | 4.34 M | 0.09 % |
| ntv3-8m | 7.692 M | 7.69 M | 0.03 % |
| ntv3-30m | 29.869 M | 29.87 M | 0.00 % |
| ntv3-100m | 106.463 M | 106.46 M | 0.00 % |
| ntv3-300m | 303.050 M | 303.05 M | 0.00 % |

All within +/-2%. Non-circular check: the registry `ntv3-8m` config exactly matches the real
`8m_pre/config.json` (embed 256 / L2 / H8 / ffn1024 / ks32 / conv_init 256 / ds7 / token_embed 16 /
alphabet 11 all MATCH) and reproduces the true 8M param count; `ntv3-100m` reproduces the true 100M.
The scaling recipe reproduces the real NTv3 family; no config field affecting param count was missed.

(b) Genuinely fresh random init (NOT loading template weights): two proofs.
  - The `8m_pre` pretrained `model.safetensors` on this fs is a dangling symlink (HF blob archived off),
    yet `build_bigwig_model("ntv3-8m")` builds fine -> the `from_config` path loads ZERO weights.
  - Built `ntv3-8m` twice with seeds 1 vs 2: every stochastic tensor (std>0) diverges across seeds
    (43/43 differ); only the 101 constant-init tensors (zero biases / LayerNorm ones) match, as expected.

(c) Non-native dims/buffers: forward succeeds for embed 192 (ks32), 448 (ks56), 1152 (ks64) -- i.e.
rotary/attention with non-standard head_dim 56 and the wide 1152/18-head config both give finite output.
head_dim==key_size for every tier (192/6=32, 448/8=56, 1152/18=64).

(d) Forward contract: dummy `[B=2, L=32768]` -> `bigwig_tracks_logits [2,12288,34]` (finite, non-neg) +
`features [2,12288,embed]` for 4m/30m/300m. `L_out = 0.375*32768 = 12288` (design/task "384" = 0.375*1024;
real contract at seq 32768 is 12288, correct).

Pretrained `from_pretrained` path byte-unchanged: the scaled branch is purely additive, guarded by
`if model_name in NTV3_SCALED_SIZES` (ntv3_finetune.py:149) and routed only for registry keys
(build_bigwig_model:295); real paths/repo-ids fall through to the unchanged else-branch. Confirmed by
inspection (could not runtime-load a pretrained NTv3 -- 8M/650M-post blobs archived off this fs).

### 2. Memmap vs LIVE teacher faithfulness — PASS (the critical one)
GPU (laniakea A6000). Precomputed a 16-window joint 34-track memmap with the real 650M teacher
(`ntv3_ft_faithful_s0/best_model.pth`), then ran the SAME 16 windows through the LIVE teacher and compared
`cache[idx]` vs live per window (both bf16-autocast, cache stored fp16):

- max_abs_err = **1.94e-3**, max_rel_err = **4.78e-4** (signal magnitude up to 88) -- within fp16/bf16
  rounding. Per-batch cached mean == live mean to 4 dp.
- **idx/coord alignment = 0/5 mismatches**: for windows {0,5,11,15,3}, `|cache[j]-live[j]|max ~ 1.9e-3`
  while `|cache[j]-live[other]|max ~ 78-88` -- cache[idx] unambiguously returns the RIGHT window.

This is the check the implementer did NOT do (they round-tripped the read only); it now passes cleanly.

### 3. Joint vs subset guard — PASS
Built tiny joint (`track_subset=None`) and subset (`[12]`) memmaps via `compute_logits_memmap` (mock
teacher, CPU) and exercised the guard `_cache_subset != idx` (finetune_ntv3.py:263-264):

| cache | student idx | raises? | expected | result |
|---|---|---|---|---|
| joint (None) | None (joint) | no | no | PASS |
| joint (None) | [12] (subset) | yes | yes | PASS |
| subset [12] | None (joint) | yes | yes | PASS |
| subset [12] | [12] | no | no | PASS |
| subset [12] | [13] (wrong) | yes | yes | PASS |

No silent wrong-target training possible. Also: `MemmapLogitCache[idx]` returns the right window's logits
(idx-aligned), upcasts fp16->fp32, correct shape `[b,L_out,C]`; coords fingerprint stored correctly;
write is idempotent. The guard is shared by the memmap DIR and the `.pt` FILE paths (sits after if/else).

### 4. 300M co-residency — reasoned PASS; GPU spot-check PENDING (dry-run C, job 253740)
By construction the cached path holds NO teacher: with a memmap DIR, `teacher` stays `None`
(finetune_ntv3.py:341 loads a live teacher only when `cached_logits is None`); the KD loop reads
`cached_logits[idx]` from the disk memmap (line 460-461). So a 300M cached run holds only the 300M
student + activations -> fits 49 GB. A 300M LIVE-teacher run also holds the 650M teacher resident at seq
32768 -- the OOM risk the design flagged. **Spot-check (dry-run C):** `ntv3-300m --cached_teacher_logits
<memmap>` on the 49 GB A6000 built a 303.1M student, confirmed `teacher=None` (KD log:
`teacher=None`), and completed 2 train steps + val + test **without OOM** (mbs=2). So the cached 300M path
fits; the LIVE-teacher 300M path remains the design-flagged OOM risk (not exercised, mitigations in §6).

### 5. Existing tests — PASS
Ran both suites in the portable venv (fast; the implementer's CPU timeout did not reproduce):
- `test_track_kd_loss.py`: 94 assertions passed.
- `test_teacher_cache.py`: 8/8 passed (via a manual runner; no pytest in the venv).
- Gap (not a regression): these suites predate the NEW memmap functions (`compute_logits_memmap`,
  `MemmapLogitCache`, `open_memmap_logit_cache`, `_probe_logits_shape`) and do not cover them; my
  item-2/3 checks cover that code instead. Adding a memmap unit test is advisable.

---

## Bugs / issues found

**No bug in the audited size-ladder code** (`ntv3_finetune.py` registry/from_config/routing,
`teacher_cache.py` memmap, `precompute_teacher_logits.py`, `finetune_ntv3.py` tokenizer-resolve +
cached-DIR-or-.pt read). All additive changes are correct and validated above.

**ISSUE #1 (GATING for the sweep — infra/cache STATE, not size-ladder code).** The launch template
`code_ntv3/slurm/ntv3_finetune.sbatch` pins `HF_HOME=HF_HUB_CACHE=code_carbon/.hf_cache` with
`HF_HUB_OFFLINE=1`. That cache **no longer contains `InstaDeepAI/NTv3_650M_post`** — it was archived off
to `<SSD>/hf_cache_archive_from_code_carbon/` (verified: `ls code_carbon/.hf_cache/hub` has no NTv3
model). The design's dry-run/sweep command uses `--teacher_base InstaDeepAI/NTv3_650M_post`, so a run
launched via the template as-is will **fail offline at teacher load** (`load_finetuned_bigwig_teacher`
builds the base via `from_pretrained` before overwriting with the finetuned ckpt).
FIX (any one): (a) export `HF_HUB_CACHE=HF_HOME=<SSD>/hf_cache_archive_from_code_carbon` in the sbatch
(what this audit did — verified it loads and matches live); (b) restore the 650M_post snapshot into
`code_carbon/.hf_cache/hub`; (c) pass a local base dir, e.g.
`--teacher_base /extra/zhanglab0/INDV/pengchx3/ntv3_local/generative`. Owner should pick + apply before
the sweep. (The audited size-ladder code is orthogonal to this — students build from the `/extra` 8m_pre
template, which is present.)

**CAVEAT (not a bug).** `train_pearson` prints `nan` on the 2-step dry-runs, and for 300M the tiny
val/test Pearson are also `nan`. This is a **degenerate-metric artifact**: a random-init model over a
1-batch / 2-window slice emits near-constant per-position output → zero variance → Pearson undefined.
Loss is finite and **decreasing** (backprop works), checkpoints + result.json write. It resolves once
weights move in a real run. One consequence to note: because 300M val was `nan`, `best_val` stayed 0.0 →
`best_model.pth` was not saved (only `latest_state.pth`); in a real multi-step run val becomes finite and
best saves normally.

**RECOMMENDATION.** Add a unit test for the new memmap functions (`compute_logits_memmap`,
`MemmapLogitCache`, `open_memmap_logit_cache`) — the existing suites predate them (see item 5).

## Dry-run result (all three exit 0, laniakea A6000, job 253740)

| dry-run | model | params | teacher | train loss (s1→s2) | val/test | outcome |
|---|---|---|---|---|---|---|
| A (design cmd) | ntv3-4m + **LIVE** 650M | 4.4M | live | 4.10 → 3.52 | val 0.021, test 0.0049 | best_model.pth + result.json ✓ |
| B (item3 e2e) | ntv3-4m + **cached** memmap | 4.4M | None | 10.11 → 8.01 | val 0.008, test −0.001 | `cached-teacher alignment verified (16 windows)` ✓ |
| C (item4) | ntv3-300m + **cached** memmap | 303.1M | None | 11.57 → 9.25 | nan (see caveat) | no OOM on 49 GB ✓ |

Dry-run A = the exact design §7 command: builds random-init 4.4M, loads data + live 650M teacher, runs
1 train step + val + test, writes finite-loss result.json with the correct 2-term KD record
(`w_ce/w_kl/w_mse=0.5/0.5/0.0`, gt=poisson_multinomial, distill=standardized_mse). B additionally proves
the joint memmap is consumed by a joint student with coord-alignment verified, teacher-free. C proves the
300M cached path fits.

## GO / NO-GO

**GO for the full sweep — CONDITIONAL on applying the ISSUE #1 fix** (point the HF cache at the archive,
restore the 650M_post snapshot, or pass a local `--teacher_base`). Without it, every run dies at teacher
load; with it (verified in this audit) the pipeline runs end-to-end.

The just-implemented size-ladder infrastructure itself is **correct and validated**: param counts exact,
genuinely fresh random init, forward works at all non-native widths, memmap cache is numerically faithful
to the live teacher and correctly idx-aligned, the joint-vs-subset guard blocks every wrong-target
combination, the 300M cached path holds no teacher and fits, and both existing test suites pass. Science
caveat (orthogonal to infra): the §5 pretraining-confound handling (random-init new tiers vs pretrained
native points) is a real analysis decision the sweep design must honor — not an infra blocker.

Audit artifacts: memmap cache + result.jsons at `<SSD>/ntv3_audit/` (large dry-run .pth removed).
Job 253740 log: `<SSD>/ntv3_audit/slurm-loom_ntv3audit-sizeladder-validate-253740.out`.
