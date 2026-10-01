# Carbon teacher fine-tuning (stage 1)

Adds Carbon (HuggingFaceBio, open autoregressive gLM, Apache-2.0) as a teacher and a
clean, teacher-agnostic stage-1 fine-tuning path for the NT 18-task benchmark. The same
code fine-tunes any HF sequence-classification gLM — adding a teacher is one config entry.

## What was added

| File | Purpose |
|---|---|
| `src/model/glm.py` | `GLMConfig` gains `input_prefix` + `add_special_tokens` (teacher-specific tokenization; defaults are no-ops) |
| `config/distillation/glm.py` | `carbon_3b` / `carbon_8b` / `carbon_500m` teacher configs |
| `config/distillation/teacher_finetune.py` | `TeacherFinetuneConfig` + `finetune_configs` registry |
| `src/trainer/finetune_trainer.py` | teacher-agnostic engine (HF Trainer; LoRA + bf16 + grad-checkpointing; MCC) |
| `src/train/finetune_teacher.py` | CLI entrypoint (tyro + `@slurm_fn`) |
| `src/train/carbon_reproduce.py` | env/weights validation (sequence-recovery proxy) |
| `slurm/carbon_*.sbatch` | SLURM scripts (A6000 default; H100/3090 options) |
| `pyproject.toml` | `carbon` uv extra |

## Environment

A local env (`.venv_carbon`, ~6 GB) is built on a local disk so imports do not cross a network mount:

```bash
cd code_carbon
UV_PROJECT_ENVIRONMENT=.venv_carbon uv sync --extra carbon
# if uv complains about its cache (SSD symlink unmounted):
#   UV_CACHE_DIR=$PWD/.uv_cache UV_PROJECT_ENVIRONMENT=.venv_carbon uv sync --extra carbon
```

Run everything with `.venv_carbon/bin/python` (no activate needed).

## Usage

GPU jobs go through SLURM — **confirm with the lab before submitting** (GPU-sharing policy).

```bash
# 1. validate the Carbon install + weights
sbatch slurm/carbon_reproduce.sbatch

# 2. single-task smoke test (1 task, 1 epoch) — do this before the full run
sbatch slurm/carbon_finetune.sbatch                 # carbon_3b_debug

# 3. full 18-task teacher fine-tune
sbatch slurm/carbon_finetune.sbatch carbon_3b
```

Outputs: per-task best checkpoint (LoRA adapter) in
`<output_dir>/{task}_finetuned/`, plus a `teacher_finetune_summary.csv` with val/test MCC
per task. That `<output_dir>` is exactly the `teacher_parent_dir` stage-2 distillation reads,
so the fine-tuned Carbon teacher drops straight into the existing distillation pipeline.

CLI overrides (tyro) work too, e.g. fewer tasks / epochs for a quick check:

```bash
.venv_carbon/bin/python -m src.train.finetune_teacher carbon_3b \
    --task-names H3K4me3 enhancers --epochs 5
```

### GPU notes
- **A6000 (laniakea, 48 GB)** — default in the sbatch scripts; preferred.
- **H100 (voyager, 80 GB)** — set `--nodelist=voyager`; limited availability, ask before using.
- **3090 (galaxy, 24 GB)** — set `--nodelist=galaxy`; fallback. Carbon-3B LoRA + bf16 +
  gradient checkpointing fits 24 GB; Carbon-8B wants A6000/H100.

## Adding another teacher (e.g. AIDO.DNA-7B, GENERATOR-1B)

1. Add a `GLMConfig` in `config/distillation/glm.py` (HF id + any `input_prefix`).
2. Add a `TeacherFinetuneConfig` in `config/distillation/teacher_finetune.py` and register it
   in `finetune_configs`:

   ```python
   aido_dna_7b_finetune = replace(
       carbon_3b_finetune, teacher_name="aido_dna_7b",
       teacher_config=aido_dna_7b, output_dir=f"{output_path}/teacher_finetune/aido_dna_7b",
   )
   finetune_configs["aido_dna_7b"] = ("Fine-tune AIDO.DNA-7B on the 18-task benchmark",
                                      aido_dna_7b_finetune)
   ```

Nothing in the engine or CLI changes — `python -m src.train.finetune_teacher aido_dna_7b`.
(AIDO.DNA needs the `aido_dna` uv extra / modelgenerator; Carbon and GENERATOR are plain HF.)
