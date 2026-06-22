# OmegaGenome

## Installation

### Environment

- Install `uv`.
- Set up the `env.toml` file. A template is provided in `env_sample.toml`.

Now, we have the following virtual enviroments:

**default**

This enviroment supports `DNA BERT v2`, `NT`.

To switch into this env, run
```bash
uv sync --extra default
```

**aido_dna**

This enviroment supports `AIDO.DNA`.

To switch into this env, run
```bash
UV_PROJECT_ENVIRONMENT=.venv_aido_dna uv sync --extra aido_dna
```

**caduceus**

This enviroment supports `caduceus`.

To switch into this env, run
```bash
UV_PROJECT_ENVIRONMENT=.venv_caduceus uv sync --extra caduceus
```

To install or uninstall a package in a specific enviroment, please make sure you activate the enviroment first

```bash
#  e.g. this is important when you are running the code as well
source .venv_aido_dna/bin/activate
```

then you should run `uv add [package_name] --optional [enviroment_name] --active`. For example, `uv add modelgenerator --optional aido_dna --active`.

### Data Folder and Output Folder Setup

Replace `[YOUR_USER_NAME]` with your real username on clusters.

```bash
# create data folder and output folder on our storage disk
mkdir -p /extra/zhanglab0/INDV/[YOUR_USER_NAME]/OmegaGenome/data
mkdir -p /extra/zhanglab0/INDV/[YOUR_USER_NAME]/OmegaGenome/output

# use soft links to add them into the project folder
ln -s /extra/zhanglab0/INDV/[YOUR_USER_NAME]/OmegaGenome/data data
ln -s /extra/zhanglab0/INDV/[YOUR_USER_NAME]/OmegaGenome/output output
```
Example for linking NT finetuned models:
```
ln -s \
/extra/zhanglab0/INDV/pengchx3/NT/2b5-multi-species_nucleotide-transformer-finetune-results-lora-epoch10-3-22-revised-r32-fix-num-label-v2 \
./data/finetuned_models/2b5-multi-species_nucleotide-transformer-finetune-results-lora-epoch10-3-22-revised-r32-fix-num-label-v2
```
```
ln -sfnT /extra/zhanglab0/INDV/pengchx3/NT/2b5-multi-species_nucleotide-transformer-finetune-results-lora-ep
och20-10-17-revised-r32-fix-num-label  ./data/finetuned_models/2b5-multi-species_nucleotide-transformer-finetu
ne-results-lora-epoch20-10-17-revised-r32-fix-num-label
```
Example for linking dnabert-2 finetuned models:
```
ln -sfn /extra/zhanglab0/INDV/pengchx3/dnabert2_output_shared ./data/finetuned_models/dnabert2_output_shared
```

Example for linking enformer finetuned models:
```
mkdir -p ./data/finetuned_models/enformer_finetune_results/
ln -sfn /extra/zhanglab0/INDV/pengchx3/enformer_finetune_results  ./data/finetuned_models/enformer_finetune_results/
```
Example for linking caduceus finetuned models
```
mkdir -p "/extra/zhanglab0/INDV/pengchx3/OmegaGenome_different_version/OmegaGenome/data/finetuned_model" \
&& ln -sfnT "/extra/zhanglab0/INDV/pengchx3/caduceus/checkpoints-8-7-bz8" \
            "./data/finetuned_models/caduceus_finetune_results"
```
## Training

### Finetuning

### Distiallation

See help message

```bash
python -m src.train.distill -h
```

Example of doing hyperparameter search

```bash
python -m src.train.distill_hyperparam dna_bert_v2 --task-names H3K27ac H3K4me2 H3K4me3 H3K27me3 H3K36me3 H3K9me3 H4K20me1 enhancers enhancers_types promoter_tata promoter_no_tata
```



## Carbon-3B → BPNet distillation: run, HP grid search, and results table

End-to-end recipe for the Carbon-3B-teacher → 0.12M-BPNet-student distillation campaign (18
NT-benchmark tasks). All commands run from this repo root with the portable venv
(`.venv_carbon_portable/bin/python`, set `PYTHONPATH=.` for direct runs).

### 1. Run the distillation (one task or a config)

Entry point is `src.train.distill <config> [overrides]`. Configs (see `config/distillation/experiments/carbon.py`):

| config | meaning |
|---|---|
| `carbon-raw` | 18-task vanilla distill, **raw** MSE (ce0.5/kl0.5/mse0.2) |
| `carbon-l2norm` | same, **L2-normalized** MSE |
| `carbon-base-raw` | the **HP-sweep** grid def (raw): kl{0,.25,.5,1} × mse{0,1,2,5} × T{.5,1,1.5,2,4} |
| `carbon-smoke` | tiny inline smoke test |

```bash
# (a) GPU job via SLURM — one task, runs inline on the allocated GPU:
sbatch slurm/carbon_distill.sbatch carbon-raw --task-names H3K27me3 --slurm-config.mode run

# (b) override any hyperparameter (this is how the grid sweeps a point):
sbatch slurm/carbon_distill.sbatch carbon-raw --task-names H3K27me3 \
  --distillation-config.weight-ce 0.5 --distillation-config.weight-kl 0.5 \
  --distillation-config.weight-mse 2 --distillation-config.temperature 4.0 \
  --trainer-config.early-stop-patience 100 --slurm-config.mode run

# (c) direct (no SLURM, e.g. CPU smoke):  PYTHONPATH=. .venv_carbon_portable/bin/python -m src.train.distill carbon-smoke
# (d) full help / all overrides:          python -m src.train.distill -h
```
Each run writes `final_summary.json` + `summary.csv` (+ best/epoch checkpoints) under
`$CARBON_OUTPUT_BASE/deploy_120k/<task>/<ts>/<uuid>/<hp_str>/`. The training **seed** is
`--random-state` (default **42**); the HP search leaves it at 42 (fixed-seed search), the final
phase re-runs the best HP at seeds 0/1/2 for significance.

### 2. Hyperparameter grid search (1440 raw combos = 80/task × 18)

```bash
# generate ONLY the not-yet-done combos (resume-aware: skips any with a final_summary.json):
PYTHONPATH=. .venv_carbon_portable/bin/python slurm/gen_hp_specs.py --out hp_specs.txt

# cap-aware auto-submitter (one job/combo, fills per-node slots as they free); nohup for long runs:
nohup bash slurm/auto_submit_specs.sh hp_specs.txt > carbon_pipeline.log 2>&1 &

# per-node caps are LIVE — edit this file and the running submitter picks it up next loop (no restart):
#   slurm/submit_caps.env   ->   LAN_CAP / VOY_CAP / GAL_CAP   (clamped to the node's GPU count)

# self-healing: after the one-pass submitter drains, this re-runs any FAILED holes until 0 remain:
nohup bash slurm/reconcile_hp.sh <submitter_pid> > reconcile_hp.log 2>&1 &
```

**Monitor grid progress (fast — runs the dir-walk ON galaxy-local disk, ~5 s):**
```bash
PYTHONPATH=. .venv_carbon_portable/bin/python slurm/grid_progress.py        # -> RAW grid: N/1440
```

### 3. Get the results tables

```bash
# (A) FULL per-run table: one row per (task × hyperparameters × seed) -> val/test MCC, best_epoch, etc.
#     stdlib-only -> run ON galaxy (local /srv/disk00 = fast) and redirect the CSV here:
ssh galaxy '/home/pengchx3/.local/share/uv/python/cpython-3.11.15-linux-x86_64-gnu/bin/python3.11 \
  '"$PWD"'/slurm/collate_runs.py --base /srv/disk00/sshfs/pengchx3/carbon_distillation' \
  > results/carbon_grid_results.csv
# columns: variant,task,weight_ce,weight_kl,weight_mse,temperature,lr,batch_size,random_state,
#          best_val_mcc,best_test_mcc,final_test_mcc,best_test_f1,best_epoch,total_epochs,early_stopped,path

# (B) BEST hyperparameters per task (selected on VALIDATION MCC, never test):
PYTHONPATH=. .venv_carbon_portable/bin/python slurm/extract_best_hyperparams.py --out best_hyperparams.json
```

### 4. Final phase — 3-seed on the best HP (significance)

```bash
PYTHONPATH=. .venv_carbon_portable/bin/python slurm/gen_3seed_best_specs.py \
  --best best_hyperparams.json --seeds 0 1 2 --out best_3seed_specs.txt   # 18 tasks × 3 = 54 jobs
nohup bash slurm/auto_submit_specs.sh best_3seed_specs.txt > seed3.log 2>&1 &
# per-task mean ± std (reads final_summary.json, selects ONLY random_state in {0,1,2}):
PYTHONPATH=. .venv_carbon_portable/bin/python slurm/aggregate_3seed.py --best best_hyperparams.json
```

See `EXPERIMENTS_carbon_distill.md` for recorded results and the failure post-mortem.

# ruff fix
```
ruff format . && ruff check . --fix
ruff format --check --diff
```
