# NTv3 regression student-size ladder — design (READ-ONLY investigation, no code changed)

Goal: fill out the **regression** (34-track per-bp bigWig distillation) size-scaling curve so it is as
solid as the classification one. Existing distilled NTv3 students are **8M** and **100M** from the
**650M** teacher; we want three new tiers — **<8M**, **~30M**, **~300M** — as *correctly scaled members
of the NTv3 student family* (same UNet-conv + transformer arch, scaled width/depth only).

This doc is the plan; nothing is implemented yet.

---

## 0. TL;DR + the one hard constraint

The NTv3 transformer family scales cleanly and I verified param counts by **instantiating** each config
(random init, CPU) — the arithmetic is exact:

| tier | embed_dim | layers | heads | ffn | key_size | **params (measured)** | status |
|---|---|---|---|---|---|---|---|
| **<8M (new)** | **192** | **2** | **6** | **768** | **32** | **4.34M** | proposed |
| 8M native | 256 | 2 | 8 | 1024 | 32 | 7.69M | pretrained ckpt exists |
| **~30M (new)** | **448** | **4** | **8** | **1792** | **56** | **29.87M** | proposed |
| 100M native | 768 | 6 | 12 | 3072 | 64 | 106.46M | pretrained ckpt exists |
| **~300M (new)** | **1152** | **9** | **18** | **4608** | **64** | **303.05M** | proposed |
| 650M native (teacher) | 1536 | 12 | 24 | 6144 | 64 | 651.83M | pretrained ckpt exists |

**THE HARD CONSTRAINT (must be decided before running):** the existing 8M/100M/650M students are
**pretrained InstaDeep checkpoints** (`InstaDeepAI/NTv3_{8M,100M,650M}_*`). InstaDeep publishes NTv3 at
**only** those three sizes ("no 50M variant exists; sizes are 8M/100M/650M" — confirmed in project notes
and by the HF repos we have snapshotted). **There is no pretrained NTv3 at 4M / 30M / 300M.** The three
new tiers can therefore only be **randomly-initialized** scaled configs, then distilled. That is a
different protocol from the pretrained points and **confounds a naive 6-point curve** (pretraining is
worth a lot here — the 8M distilled student is already *capacity-saturated at* ≈0.476, riding on its
pretraining). §5 gives the honest fix (anchor the curve with same-protocol reference points).

Everything else (data, loss, KD, launch, metrics) is **reused unchanged** — the new tiers need only a
~20-line registry + a `from_config` branch (§4).

---

## 1. Where the code lives

Repo: `/home/pengchx3/text-dna/OmegaGenome_Revise_202606/` (NOT a git repo at that level).

| piece | path |
|---|---|
| **Finetune/distill entrypoint** | `code_ntv3/src/train/finetune_ntv3.py` |
| Student model factory | `code_ntv3/src/model/ntv3_finetune.py` → `build_bigwig_model()` |
| NTv3 teacher wrapper | `code_ntv3/src/model/ntv3_teacher.py` |
| Teacher-logit precompute (cache) | `code_ntv3/src/train/precompute_teacher_logits.py` |
| KD loss | `code_ntv3/src/trainer/track_distill.py` (`track_kd_loss`, `TrackKDConfig`) |
| Data loader | `code_ntv3/src/data/ntv3_ft_data.py` (`GenomeBigWigDataset`) |
| Faithful FT sbatch | `code_ntv3/slurm/ntv3_finetune.sbatch` (env `MODEL=`, `RUNTAG=`) |
| Teacher-cache sbatch | `code_ntv3/slurm/ntv3_cache_logits.sbatch` |
| Python env | `code_carbon/.venv_carbon_portable/bin/python` (torch 2.6.0+cu118, transformers 4.57.1) |

**How a student is currently selected** (key finding): there is **no size registry** for the NTv3
transformer. `--model <path-or-repo>` *is* the size — the arch is whatever pretrained checkpoint that
path loads. `build_bigwig_model()` dispatches by inspecting the config:
- PRE checkpoint (8M, 100M-pre) → `NTv3PreBigWigModel` (loads via `AutoModelForMaskedLM.from_pretrained`)
- POST checkpoint (100M-post, 650M) → `NTv3BigWigModel` (headless `core` + head)
- `--student_arch bpnet` (+ `--bpnet_channels/--bpnet_n_dilated`) → `BPNetTrackStudent` (a from-scratch
  dilated CNN — the *only* currently-scalable student, but a **different architecture family**).

Both NTv3 students attach the same head: `LinearHead` = `LayerNorm → Linear(embed_dim, 34) → softplus`
over the central 37.5 % crop. Forward contract: `{"bigwig_tracks_logits": [B, L_out, 34], "features":
[B, L_out, embed_dim]}`, `L_out = 0.375 × 32768 = 12288`.

---

## 2. Exact confirmed architecture configs (from the snapshot `config.json`s)

All four native sizes share: `num_downsamples=7`, `token_embed_dim=16`, `alphabet_size=11`,
`ffn_embed_dim = 4 × embed_dim`, `conv_init_embed_dim = embed_dim`, `head_dim = embed_dim / heads =
key_size`. The arch is a UNet (7 down/up conv stages) with a `num_layers`-deep transformer at the
bottleneck. **This is a genuinely clean width/depth family** — the ratios above are the design rule.

```
8M   (NTv3_8M_pre)   embed 256  L2  H8  ffn1024 ks32  ds7   -> measured 7.69M   (matches "8M")
100M (NTv3_100M_pre) embed 768  L6  H12 ffn3072 ks64  ds7   -> measured 106.46M (matches note "106.5M")
650M (generative)    embed 1536 L12 H24 ffn6144 ks64  ds7   -> measured 651.83M (matches "650M")
```

Snapshots on disk (all carry the modeling `.py`, gated `auto_map` prefix stripped in the `_loadable`
siblings): `/extra/zhanglab0/INDV/pengchx3/ntv3_local/{8m_pre,100m_pre,100m_post,generative}[_loadable]`.
The lm-head is ~0.0M in every case (token_embed_dim=16), so **total ≈ backbone**; the track head is
`embed_dim×34 + norms` (≈9–52k), negligible.

**Validation method** (reproducible): I built each config with
`AutoModelForMaskedLM.from_config(cfg, trust_remote_code=True)` off the local `8m_pre_loadable`
snapshot and summed `p.numel()`. Native sizes reproduced to <1 % — so the proposed-tier numbers in §3
are trustworthy, not back-of-envelope. Script: `scratchpad/count_params.py` (in this session's scratch).

---

## 3. Proposed new tiers (exact configs + measured params)

Design rule applied: hold family invariants (`ffn=4·embed`, `conv_init_embed_dim=embed`, `ds=7`,
`token_embed_dim=16`), pick `embed_dim` divisible by `heads`, interpolate `num_layers` on the native
2→6→12 trend, keep `head_dim = embed/heads ∈ {32,56,64}`.

| target | **embed_dim** | **layers** | **heads** | **ffn** | **key_size** | head_dim | **measured params** | lands |
|---|---|---|---|---|---|---|---|---|
| **<8M** | 192 | 2 | 6 | 768 | 32 | 32 ✓ | **4.34M** | ~half of 8M, clean point below |
| **~30M** | 448 | 4 | 8 | 1792 | 56 | 56 ✓ | **29.87M** | dead-on 30M |
| **~300M** | 1152 | 9 | 18 | 4608 | 64 | 64 ✓ | **303.05M** | dead-on 300M |

Head divisibility all check: 192/6=32, 448/8=56, 1152/18=64. All instantiated **without error** on CPU
(full model built, not just param math).

Resulting ladder (params, log-spaced ≈1.6–3.5× steps):
```
4.34M -> 7.69M(8M) -> 29.87M -> 106.46M(100M) -> 303.05M -> 651.83M(650M/teacher)
```

Alternative points I measured if you want to shift a tier:
- smaller <8M: `128/2/4/512/32` → **1.95M**; larger <8M: keep 192 → 4.34M (recommended).
- ~30M cheaper/richer: `384/4/6/1536/64` → 21.96M; `512/4/8/2048/64` → 38.99M.
- ~300M: `1024/10/16/4096/64` → 256.28M; `1280/8/20/5120/64` → 347.85M.

Recommended = the three bolded rows (4.34M / 29.87M / 303.05M): each hits its target and sits mid-way
(log scale) between the neighboring native points, giving an even 6-point curve.

---

## 4. Minimal code change

The new tiers are random-init NTv3 configs, so they need a `from_config` path (the current NTv3 path is
`from_pretrained`-only). Cleanest minimal change, all in `code_ntv3/src/model/ntv3_finetune.py`:

1. **Add a registry** (top of file):
```python
# Scaled random-init NTv3-pretrained variants (no InstaDeep ckpt exists at these sizes). Each reuses the
# 8m_pre_loadable snapshot's modeling .py + tokenizer as a TEMPLATE; only the config dims change.
NTV3_TEMPLATE = "/extra/zhanglab0/INDV/pengchx3/ntv3_local/8m_pre_loadable"
NTV3_SCALED_SIZES = {  # embed_dim, num_layers, attention_heads, ffn_embed_dim, key_size
    "ntv3-4m":   dict(embed_dim=192,  num_layers=2, attention_heads=6,  ffn_embed_dim=768,  key_size=32),
    "ntv3-30m":  dict(embed_dim=448,  num_layers=4, attention_heads=8,  ffn_embed_dim=1792, key_size=56),
    "ntv3-300m": dict(embed_dim=1152, num_layers=9, attention_heads=18, ffn_embed_dim=4608, key_size=64),
}
```

2. **Branch in `NTv3PreBigWigModel.__init__`** (or a thin sibling): if `model_name` is a registry key,
   load the template config, apply overrides (`embed_dim`, `conv_init_embed_dim=embed_dim`,
   `num_layers`, `attention_heads`, `ffn_embed_dim`, `key_size`), and build with
   `AutoModelForMaskedLM.from_config(cfg, trust_remote_code=True)` instead of `from_pretrained`. Head +
   forward are byte-identical to the existing PRE student (it reads `self.config.embed_dim`).

3. **Route it in `build_bigwig_model`**: `if model_name in NTV3_SCALED_SIZES: return NTv3PreBigWigModel(model_name, ...)`
   (the branch above handles the from_config load).

4. **Tokenizer in `finetune_ntv3.py`**: `AutoTokenizer.from_pretrained(args.model, ...)` must resolve — map
   a registry key to `NTV3_TEMPLATE` before tokenizer load (one line), since all NTv3 sizes share the
   single-nt tokenizer.

No new architecture code, no new loss/data/metric/trainer code. ~20 lines total. (The BPNet path proves
the pattern: a scalable student already lives behind the same factory + entrypoint.)

**Audit before use:** per lab policy, spawn a subagent to unit-test the new registry branch — assert each
tier builds, param count matches §3, forward returns `bigwig_tracks_logits [B,12288,34]` +
`features [B,12288,embed_dim]`, and head_dim divisibility. Add a docstring to the new branch.

---

## 5. Distillation plan (parallels the classification story)

**Recipe = identical to the existing joint-34 KD runs** (`ntv3_100m_kd_stdmse`, `ntv3_8m_kd_pois`), so
new points are directly comparable to the current 8M/100M ones:
- teacher = **650M** `ntv3_targets/ntv3_ft_faithful_s0/best_model.pth` (the reproduced 0.606-PCC teacher)
- loss = 2-term KD: `w_ce=0.5` (poisson_multinomial GT) + `w_kl=0.5` (`standardized_mse` teacher distill),
  `w_mse=0` (no feature align in joint mode — matches existing runs), `multinomial_weight=5`
- `--sequence_length 32768`, `--mini_batch_size 4`, `--num_accumulation_gradient 8` (eff. batch 32),
  `--num_steps_training 19932`, `--num_steps_warmup 598`, LR 1e-5→5e-5 square decay, `--amp`
- best-val checkpoint select; final test on full test set (`best_model.pth`).

**Joint-34 (recommended)** parallels the classification multi-task scaling curve. **Seeds:** 1 seed for
the first curve (6 points × 1 = enough to see the trend), then 3 seeds on the final figure once the shape
is confirmed — matches how the classification `rebuttal_infra/size*` sweeps were staged.

### The pretraining-confound fix (do NOT skip)
Because 4M/30M/300M are random-init but 8M/100M/650M are pretrained, plotting all six on one axis is not
apples-to-apples. Pick one:
- **(A) Two-line figure (recommended):** report a **"distilled-from-scratch" line** through same-protocol
  random-init points at *all* sizes — i.e. also train random-init 8M, 100M (+ optionally a 650M-arch
  from scratch) with the identical KD recipe — plus the existing **"pretrained+distilled" points** as a
  second series. The scratch line is the clean scaling curve; the gap to the pretrained points *is* the
  pretraining bonus (a nice extra result). Cost: +2 reference runs (random-init 8M, 100M).
- **(B) One line, honest caption:** plot the three new scratch tiers together with a clearly-labeled
  caveat that 8M/100M/650M carry pretraining; use them only as an upper reference, not curve points.

Recommend (A) — it turns the confound into a finding and makes the curve rigorous.

---

## 6. Compute estimate + SLURM

**Dominant cost = the 650M teacher forward, and whether it is cached.** Facts from the code:
- Joint-34 KD currently uses a **LIVE 650M teacher forward every step** (`--teacher` path; the cache is
  logit-only and joint-34 is too big to cache — see below). So per-run wall-clock is
  **teacher-forward-bound** and ≈ constant across *small* student sizes.
- A combined 34-track logit cache would be `64000 windows × 12288 L_out × 34 tracks × 4B = ~107 GB`
  (fp16 ~53 GB) — the trainer loads the cache **fully into RAM**, so this does **not** fit. That is why
  the existing joint runs used a live teacher; per-track caches (`gen/t{idx}.pt`, ~3 GB each) are only
  used by the **specialist** (1-track) runs.

Anchor: the faithful **650M full-FT** is ~28 h on an H100 (voyager) for 19,932 steps (fwd+bwd+optim).
A joint-34 **live-KD** run does student fwd+bwd+optim **+** 650M teacher fwd (no bwd/optim):

| new tier | student cost | + teacher fwd (live) | est. H100 wall-clock / run | mem risk |
|---|---|---|---|---|
| 4.34M | tiny | teacher-bound | **~12–16 h** | none |
| 29.87M | small | teacher-bound | **~13–17 h** | none |
| 303.05M | comparable to teacher | student+teacher | **~24–32 h** | **watch** — see below |

**Total (1 seed, 3 new tiers, joint-34 live):** ≈ 50–65 H100-h. With the §5(A) fix (+random-init 8M,
100M reference runs): +~30 h → ≈ 80–95 H100-h. Three seeds later ≈ 3× the final subset.

**Cheaper option to amortize the teacher across the sweep:** build **one combined 34-track cache** and
**memmap** it (a small trainer change: `np.load(mmap_mode='r')` + per-batch index, instead of the current
full-RAM load). One 650M pass (~3–5 h) then makes every student run **teacher-free** (student-only:
4M/30M ≈ 3–6 h, 300M ≈ 12–18 h). Net saving grows with #sizes×#seeds. Flag this as an optional
follow-on; the faithful live-teacher path needs **zero** trainer change and matches existing points.

**SLURM** (reuse `code_ntv3/slurm/ntv3_finetune.sbatch`, which already handles KD via pass-through args):
- 4M, 30M: **laniakea** (RTX 6000 Ada, 49 GB) is plenty and is our primary node; 24 GB galaxy also fits.
- 300M: **laniakea or voyager (H100, 80 GB)**. At seq_len 32768, eff-batch via accum=8×mbs=4, a 300M
  student + activations + the live 650M teacher resident on the same GPU is the memory risk. Mitigations
  already in-repo: `--amp` (bf16), `expandable_segments`, and if needed drop `--mini_batch_size` to 2 /
  raise `--num_accumulation_gradient` to 16 (eff-batch unchanged). If teacher+300M won't co-reside,
  switch that tier to the **memmap-cache** path (teacher not resident) — the cleanest 300M fix.
- Caps: laniakea ≤8, voyager ≤4, galaxy ≤5. Env, HF-offline, triton-cache, wandb-off-/home are all
  handled by `ntv3_finetune.sbatch`. Preflight + watchdog before any GPU submit (lab policy).

---

## 7. Dry-run (validate before the sweep)

Run the faithful entrypoint's built-in `--dry_run` (one batch through train→val→test, then exit) on the
smallest new tier, **CPU or a single GPU**, to prove the from_config build + head + KD loss + metric all
run for a registered size before spending GPU-h:

```bash
cd /home/pengchx3/text-dna/OmegaGenome_Revise_202606/code_ntv3
SSD=/srv/disk00/sshfs/pengchx3   # or /tmp/galaxy_srv_disk00/pengchx3 on laniakea/voyager
PY=/home/pengchx3/text-dna/OmegaGenome_Revise_202606/code_carbon/.venv_carbon_portable/bin/python
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
HF_TOKEN="$(cat /home/pengchx3/text-dna/huggingface-token-0616.txt)" \
"$PY" -u -m src.train.finetune_ntv3 \
    --data_dir "$SSD/ntv3_benchmark_data" \
    --model ntv3-4m \
    --teacher "$SSD/ntv3_targets/ntv3_ft_faithful_s0/best_model.pth" \
    --teacher_base InstaDeepAI/NTv3_650M_post \
    --kd_w_ce 0.5 --kd_w_kl 0.5 --kd_w_mse 0.0 \
    --kd_gt_loss poisson_multinomial --kd_distill_loss standardized_mse \
    --amp --dry_run --out "$SSD/ntv3_targets/ntv3_4m_dryrun"
```

Expect: `trainable params: 4.3M`, one train step, `[val]` line, a `TEST mean Pearson`, and a written
`ntv3_finetune_result.json` — all error-free. Then repeat with `--model ntv3-30m` / `ntv3-300m` (300M
dry-run on a GPU to also smoke the teacher+student co-residency memory). Only then launch the full sweep.

---

## 8. Data (confirmed)

- 34-track human bigWig benchmark: `/srv/disk00/sshfs/pengchx3/ntv3_benchmark_data` (`= /tmp/galaxy_srv_disk00/...`
  on laniakea/voyager), **40 GB on the galaxy SSD** (genome FASTA + 34 bigWigs + metadata). Loader
  `GenomeBigWigDataset` tiles dense 32,768-bp windows; `--stage_dir` copies node-local at job start.
- Teacher: `ntv3_targets/ntv3_ft_faithful_s0/best_model.pth` (2.6 GB, the reproduced 650M, test PCC 0.606).
- Existing joint points to compare against: `ntv3_targets/ntv3_8m_kd_pois` (test 0.476, saturated),
  `ntv3_targets/ntv3_100m_kd_stdmse` (test 0.545) and baselines `ntv3_8m_baseline` (0.475),
  `ntv3_100m_baseline` (0.518).

---

## 9. Risks / honest caveats

1. **Pretraining confound (biggest).** New tiers are random-init; native points are pretrained. Use the
   §5(A) two-series figure or the curve is not apples-to-apples. Do not ship a naive 6-point line.
2. **8M is already saturated** (KD ≈ baseline ≈ 0.476). The 4M point may sit *below* the from-scratch
   trend if distillation can't compensate for tiny capacity — that is a legitimate curve point, not a bug.
3. **300M memory** with a co-resident live 650M teacher at seq 32768. Mitigate via amp / smaller mbs+more
   accum / the memmap-cache path (§6). Dry-run the 300M tier on-GPU first.
4. **Joint-34 uses a live teacher** → each run pays the 650M forward. If the sweep grows (sizes×seeds),
   invest the one-time memmap-cache change to make all student runs teacher-free.
5. **Not a git repo** at the OmegaGenome_Revise root; commit discipline is via the code_carbon subtree if
   used. Record every run's `result.json` to the experiment doc per lab policy.
```
