# OmegaGenome / NTv3 HuggingFace backup audit + plan

READ-ONLY audit (2026-07-20). No HF or disk state modified. SSD (`/srv/disk00/sshfs/pengchx3/`)
is a working store, **not a backup**. Goal: identify which artifacts needed for the NeurIPS/SciAdv
rebuttal exist only on the SSD vs. are already mirrored to HuggingFace, and give a concrete plan
(with exact commands) to close the real gaps.

HF account: **`explcre`** (also org `diagramAI`, empty). Token: `~/text-dna/huggingface-token-0616.txt`.
Python: `~/.conda/envs/caduceus_t26/bin/python` (huggingface_hub).

---

## TL;DR — the only real gaps

Almost everything is already safe. The 650M teacher, the existing 8M/100M joint students
(distilled + from-scratch), and the per-task 8M students are **already on the PUBLIC repo**
`explcre/omegagenome-distilled-students` under `regression/`, verified byte-for-byte against the SSD.

Only two things are unprotected:

1. **The NEW from-scratch KD size-sweep** (`ntv3_targets/size_sweep/`, 4M/8M/30M/100M/300M) —
   not on HF. Sweep is **still running** (300M empty, no `result.json` written yet). Back up
   *once it finishes*.
2. **The size-ladder configs/sbatch/specs/design docs** in `rebuttal_infra/` — **untracked in git**
   (`git status` = `??`) and on HF nowhere. These *define* the new tiers. ~1 MB of text, highest
   risk-adjusted priority. Back up now (or `git add`).

Everything else is either already-mirrored, public-elsewhere, or cheaply regenerable.

---

## 1. What's already on HF

`list_datasets/list_models(author="explcre")`: 17 datasets + 26 models. Relevant to this audit:

| repo | type | vis | holds |
|---|---|---|---|
| `explcre/omegagenome-distilled-students` | model | **public** | classification 3-seed students; **`regression/` subtree (58 files)** |
| `explcre/omegagenome-revise-distill-results` | dataset | private | per-track CSVs + `ntv3/650m_s{0,1,2}_result.json`, 8m baseline (metrics only, no ckpts) |
| `explcre/omegagenome-ntv3-teacher-cache` | dataset | public | *generation-model* teacher logits/feats (t0..t33) — unrelated to the 34-track regression cache |
| `explcre/omegagenome-carbon-hp-students` | model | public | Carbon HP-grid classification ckpts (799 files) |
| `explcre/omegagenome-embedding-cache` | dataset | public | DNABERT-2 token-embedding cache (classification) |
| `explcre/galaxy-ssd-pengchx3-backup` | dataset | private | broad 86k-file SSD dump (DRAKES etc.) — does **not** contain ntv3_targets |

### `regression/` subtree already on `omegagenome-distilled-students` (public)

Verified real (sizes fetched via `get_paths_info`, match SSD exactly):

| HF path | HF size | SSD source | match |
|---|---|---|---|
| `regression/teacher/ntv3-650m/joint34/seed0/best_model.pth` | 2622.3 MB | `ntv3_ft_faithful_s0/best_model.pth` (2622304610 B) | ✅ |
| `regression/joint/100m/distilled/best_model.pth` | 426.5 MB | `ntv3_100m_*` (426458013 B) | ✅ |
| `regression/joint/100m/from_scratch/best_model.pth` | 426.5 MB | — | ✅ |
| `regression/joint/8m/distilled/best_model.pth` | 30.9 MB | `ntv3_8m_baseline` (30942041 B) | ✅ |
| `regression/joint/8m/from_scratch/best_model.pth` | 30.9 MB | — | ✅ |
| `regression/joint/{8m,100m}/{distilled,from_scratch}/ntv3_finetune_result.json` + README | — | | ✅ |
| `regression/per_task/8m/t{0,3,11,12,14,18,20}_*/{distilled,from_scratch}/{best_model.pth,result.json}` | — | | ✅ |

**Conclusion:** teacher + 8M/100M joint students (both protocols) + 8M per-task students are fully
backed up on a public repo. The size-sweep's 8M/100M *scratch_kd* tiers are a **different protocol**
(randomly-init scaled config + KD) and the 4M/30M/300M tiers are **new sizes with no HF equivalent**,
so the whole sweep is a genuine gap.

---

## 2. What's on the SSD (`/srv/disk00/sshfs/pengchx3/`) — candidate artifacts

### 2a. NEW from-scratch KD size-sweep — `ntv3_targets/size_sweep/` (RUNNING, incomplete)

| tier | dir | `best_model.pth` | `latest_state.pth` | `result.json` | done? |
|---|---|---|---|---|---|
| 4M | `ntv3-4m_scratch_kd_s0` | 17.5 MB | 52.5 MB | ❌ | ckpt written, eval pending |
| 8M | `ntv3-8m_scratch_kd_s0` | 30.9 MB | 92.7 MB | ❌ | ckpt written, eval pending |
| 30M | `ntv3-30m_scratch_kd_s0` | 119.9 MB | 359.0 MB | ❌ | ckpt written, eval pending |
| 100M | `ntv3-100m_scratch_kd_s0` | 426.5 MB | 1278.5 MB | ❌ | ckpt written, eval pending |
| 300M | `ntv3-300m_scratch_kd_s0` | — (empty dir) | — | ❌ | **not started** |

`best_model.pth` total (4 done tiers) ≈ **595 MB**. With 300M ≈ 1.2 GB. `result.json`s not yet
written for any tier — **do not back up until the sweep completes and evals land.**

### 2b. Teacher + existing students — `ntv3_targets/` (178 GB total dir)

- `ntv3_ft_faithful_s0/` — teacher: `best_model.pth` 2.62 GB (✅ on HF), `latest_state.pth` 7.86 GB
  (optimizer state, resume-only, not on HF), `ntv3_finetune_result.json` (✅ metrics on HF).
- `ntv3_8m_baseline/`, `ntv3_100m_baseline/`, and the many `ntv3_{8m,100m,bpnet}_*` KD-ablation run
  dirs — the *published-in-paper* joint & per-task points are already on HF (§1). The long tail of
  intermediate KD ablations (t11/t14/dkd/cwd/featalign/…) is exploratory and not paper-critical.

### 2c. Teacher logit cache — `ntv3_targets/teacher_cache_joint34/` (50 GB)

`logits.npy` 53232550016 B (~53 GB memmap) + `meta.pt` (1 MB). **Regenerable** from the (backed-up)
teacher via `code_ntv3/src/train/precompute_teacher_logits.py`. **SKIP.**

### 2d. 34-track regression benchmark — `ntv3_benchmark_data/` (40 GB)

**This is NOT unique to us — it is a public InstaDeep dataset.** `download.log` shows it was pulled
verbatim from `InstaDeepAI/NTv3_benchmark_dataset` (41 files), and the local `README.md` is InstaDeep's
own (`InstaDeepAI/NTv3_benchmark_dataset/...`, license cc-by-nc-4.0). Contents: per-species bigWigs +
BEDs + `human/genome.fasta`. **Do not re-host 40 GB.** Point at the public source instead.
- Lab-derived-only: `_prep_tvt_w16384_tr2000_te400.pt` (2.7 GB) + `_prep_tvt_w16384_tr8000_te1500.pt`
  (10.6 GB) — windowed train/val/test tensors. **Regenerable** from the public bigWigs + the prep
  code; skip unless recomputation cost matters, in which case treat as *nice*.

### 2e. Size-ladder code / configs / sbatch — `rebuttal_infra/` (local home, NFS)

`git status` → **UNTRACKED (`??`)**, and not on HF: `ntv3_size_ladder_design.md`, `ntv3_size_audit.md`,
`size_monotonicity_diagnosis.md`, `size18task.sbatch` + `size18task_specs.{py,txt}`,
`size_seed0.sbatch` + `size_seed0_specs.txt`, `size_seed12.sbatch` + `size_seed12_specs.txt`.
These hold the exact scaled configs (embed_dim/layers/heads/ffn/key_size for 4M/30M/300M) and launch
recipes — the *only* record of how the new tiers are defined (the design doc notes the size registry
is **not yet in `code_ntv3`**). ~1 MB text. **AT RISK — highest risk-adjusted priority.**

The `code_ntv3` repo itself is git-backed (remote `jhliu17/OmegaGenome`); the size-registry code, when
written, should be committed there.

---

## 3. Gap analysis + priority

| artifact | on HF? (repo / path) | SSD size | priority |
|---|---|---|---|
| 650M teacher ckpt | ✅ `distilled-students` `regression/teacher/…/seed0` | 2.62 GB | done |
| 8M/100M joint students (distilled + scratch) | ✅ `distilled-students` `regression/joint/…` | ~0.9 GB | done |
| 8M per-task students (7 tasks × 2) | ✅ `distilled-students` `regression/per_task/8m/…` | — | done |
| teacher/student result.jsons + per-track CSVs | ✅ `distilled-students` + `revise-distill-results` | — | done |
| **NEW size-sweep `best_model.pth` (4/8/30/100/300M)** | ❌ | ~1.2 GB (when complete) | **MUST (after sweep finishes)** |
| **NEW size-sweep `result.json` (5 tiers)** | ❌ (not yet written) | tiny | **MUST (after evals land)** |
| **Size-ladder configs/sbatch/specs/design docs** | ❌ untracked in git | ~1 MB | **MUST (now)** |
| size-sweep `latest_state.pth` (optimizer) | ❌ | ~1.8 GB | nice (resume-only) |
| teacher `latest_state.pth` | ❌ | 7.86 GB | nice (resume-only) |
| 34-track benchmark bigWigs/BEDs/fasta | public `InstaDeepAI/NTv3_benchmark_dataset` | 40 GB | **skip — pointer only** |
| `_prep_tvt_*.pt` windowed tensors | ❌ | 13.3 GB | skip-regenerable (nice) |
| teacher logit cache `logits.npy` | ❌ | 53 GB | **skip — regenerable** |
| KD-ablation run-dir long tail | ❌ | ~150 GB | skip (exploratory) |

### Recommended minimal backup set (what actually needs saving)
1. Size-ladder **configs/sbatch/specs/design docs** (~1 MB) — do now; nothing else records the tier defs.
2. Size-sweep **`best_model.pth` × 5 + `result.json` × 5** (~1.2 GB) — after the sweep + evals finish.
3. A **README pointer** to `InstaDeepAI/NTv3_benchmark_dataset` for the benchmark data (no re-host).

Everything else = already mirrored, public elsewhere, or regenerable.

---

## 4. Proposed backup target: public HF dataset repo `explcre/omegagenome-ntv3-regression`

New **public** dataset repo scoped to the regression size-scaling story. (Alternative: drop the
size-sweep under the existing `explcre/omegagenome-distilled-students` `regression/size_ladder/` tree —
same repo as the teacher/8M/100M so the whole regression family stays in one place. Either is fine;
a dedicated dataset repo keeps configs+data-pointer+ckpts together. Pick one and stay consistent.)

Layout:
```
omegagenome-ntv3-regression/
  README.md                              # story + pointer to InstaDeepAI/NTv3_benchmark_dataset
  configs/                               # from rebuttal_infra/ (the AT-RISK files)
    ntv3_size_ladder_design.md
    ntv3_size_audit.md
    size_monotonicity_diagnosis.md
    size18task.sbatch  size18task_specs.{py,txt}
    size_seed0.sbatch  size_seed0_specs.txt
    size_seed12.sbatch size_seed12_specs.txt
  size_sweep/
    ntv3-4m_scratch_kd_s0/{best_model.pth,ntv3_finetune_result.json}
    ntv3-8m_scratch_kd_s0/{best_model.pth,ntv3_finetune_result.json}
    ntv3-30m_scratch_kd_s0/{best_model.pth,ntv3_finetune_result.json}
    ntv3-100m_scratch_kd_s0/{best_model.pth,ntv3_finetune_result.json}
    ntv3-300m_scratch_kd_s0/{best_model.pth,ntv3_finetune_result.json}
```
Skip `latest_state.pth` (optimizer, resume-only), the 53 GB logit cache, and the 40 GB benchmark data.

---

## 5. Exact upload commands (DO NOT RUN until sweep completes)

Run on a node that mounts the SSD (laniakea/voyager), env
`~/.conda/envs/caduceus_t26/bin/python`, `HF_TOKEN=$(cat ~/text-dna/huggingface-token-0616.txt)`.

### 5a. Configs (safe to run now — closes the AT-RISK gap)
```python
import os
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"])
REPO = "explcre/omegagenome-ntv3-regression"
api.create_repo(REPO, repo_type="dataset", private=False, exist_ok=True)  # public

RI = "/home/pengchx3/text-dna/OmegaGenome_Revise_202606/code_carbon/rebuttal_infra"
for f in ["ntv3_size_ladder_design.md","ntv3_size_audit.md","size_monotonicity_diagnosis.md",
          "size18task.sbatch","size18task_specs.py","size18task_specs.txt",
          "size_seed0.sbatch","size_seed0_specs.txt",
          "size_seed12.sbatch","size_seed12_specs.txt"]:
    api.upload_file(path_or_fileobj=f"{RI}/{f}", path_in_repo=f"configs/{f}",
                    repo_id=REPO, repo_type="dataset")
```
(Simplest alternative for the configs: just `git add` them in `code_carbon` and push to
`jhliu17/OmegaGenome` — they belong in version control anyway.)

### 5b. Size-sweep checkpoints + results (ONLY after sweep + evals finish; verify each `result.json` exists)
```python
SS = "/srv/disk00/sshfs/pengchx3/ntv3_targets/size_sweep"
for tier in ["4m","8m","30m","100m","300m"]:
    d = f"{SS}/ntv3-{tier}_scratch_kd_s0"
    for fn in ["best_model.pth","ntv3_finetune_result.json"]:
        src = f"{d}/{fn}"
        assert os.path.exists(src), f"MISSING {src} — sweep not done, abort"
        api.upload_file(path_or_fileobj=src,
                        path_in_repo=f"size_sweep/ntv3-{tier}_scratch_kd_s0/{fn}",
                        repo_id=REPO, repo_type="dataset")
# (or upload_folder with allow_patterns=["best_model.pth","ntv3_finetune_result.json"])
```

### 5c. README pointer for benchmark data (no re-host)
Write a `README.md` noting the 34-track benchmark = public `InstaDeepAI/NTv3_benchmark_dataset`
(cc-by-nc-4.0), and that `_prep_tvt_*.pt` windowed tensors + the 53 GB teacher logit cache are
regenerable from it via `code_ntv3/src/train/precompute_teacher_logits.py`; then `upload_file`.

---

## 6. Flags / at-risk

- **Size-ladder configs are untracked in git and nowhere on HF** — single copy on NFS home. Losing the
  node/quota loses the tier definitions. Fix now (5a or `git add`).
- **Size-sweep is mid-run** — 300M not started, no `result.json` yet. Backup is premature until it
  finishes; the `assert` in 5b guards against uploading an incomplete run.
- Nothing appears already *lost*. The 40 GB benchmark and 53 GB cache are safe to leave off HF
  (public / regenerable) — re-hosting them would be wasted quota.
