# OmegaGenome

## Installation

```bash
uv sync
```

Set up the `env.toml` file. A template is provided in `env_sample.toml`.

## Training

### Finetuning

### Distiallation

```bash
python -m src.train.distill -h
```

```bash
python -m src.train.distill_hyperparam dna_bert_v2 --task-names H3K27ac H3K4me2 H3K4me3 H3K27me3 H3K36me3 H3K9me3 H4K20me1 enhancers enhancers_types promoter_tata promoter_no_tata
```