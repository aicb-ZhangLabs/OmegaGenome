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
Example for linking caduceus finetuned models
```
mkdir -p "/extra/zhanglab0/INDV/pengchx3/OmegaGenome_different_version/OmegaGenome/data/finetuned_model" \
&& ln -sfnT "/extra/zhanglab0/INDV/pengchx3/caduceus/checkpoints-8-7-bz8" \
            "/extra/zhanglab0/INDV/pengchx3/OmegaGenome_different_version/OmegaGenome/data/finetuned_model/caduceus_finetune_results"
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
