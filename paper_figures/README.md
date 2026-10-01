# Paper figures

The main-text figures as submitted, the code that draws them, and the tabulated results that code
reads. Everything here runs on a laptop from the bundled tables — no GPU, no cluster, no data
download.

## Main-text figures

| Figure | File | Produced by |
|---|---|---|
| 1. Overview of the distillation framework | [figures/Figure1.pdf](figures/Figure1.pdf) | drawn by hand (vector source, no script) |
| 2. Distilled experts match teacher performance | [figures/Figure2.pdf](figures/Figure2.pdf) | `figure_panels/build/fig2_combined_carbon.py` |
| 3. Teacher accuracy at a fraction of the compute | [figures/Figure3.pdf](figures/Figure3.pdf) | `figure_panels/build/fig4_efficiency_carbon.py` |
| 4. Representation alignment between student and teacher | [figures/Figure4.pdf](figures/Figure4.pdf) | `../analysis/extract_feats_logits.py` then `../analysis/feature_viz_omega_dkd.py` (needs the trained checkpoints) |
| 5. Robustness to method, student size and loss weights | [figures/Figure5.pdf](figures/Figure5.pdf) | `figure_panels/build/fig6_method_size_hp.py` |
| 6. Algorithm workflow | [figures/Figure6.pdf](figures/Figure6.pdf) | drawn by hand (vector source, no script) |

Script names keep the draft figure numbers they were written under (`fig4_…` draws the published
Figure 3, `fig6_…` the published Figure 5); each script's docstring states which figure it became.

## Supplementary figures

`figure_panels/build/` also holds the supplementary panels: the student-size sweep
(`fig6b_size_18task.py`, `fig6c_pertask_smallmultiples.py`, `fig6d_persize_heatmap.py`,
`fig6e_delta_scaling_bars.py`), the hyperparameter analyses (`fig_appendix_hp.py`,
`fig_appendix_hp_aggregated.py`, `fig_hp_atlas_18task.py`), the cross-task transfer matrix
(`fig_s2_crosstask_matrix.py`), the feature-space comparison (`fig_s_dkd_feature_space.py`), the
regression size ladder (`fig_sizeladder.py`) and the per-method comparison
(`method_18task_figs.py`). The scripts in this directory's root are the earlier single-panel
versions and the standalone legend.

## Running them

```bash
uv sync --extra analysis          # matplotlib, seaborn, scipy, pandas
cd paper_figures
python figure_panels/build/fig4_efficiency_carbon.py     # -> output/fig4_efficiency_carbon.pdf
python latency.py                                        # -> output/latency.pdf
```

Run the scripts from this directory: they read `data/` relative to the working directory and write
to `output/` (PDF for the paper, PNG at 300 dpi for quick viewing, plus a preview copy under
`previews/`). Both output directories are created on demand and are not tracked.

The figures use Helvetica Neue where it is installed and fall back to the default sans-serif
otherwise, which can shift text metrics slightly from the submitted PDFs.

## Data

| path | contents |
|---|---|
| `data/model_comparison_*.csv`, `data/formal_5teacher_benchmark.csv` | per-task and 18-task-mean MCC for every teacher, its distilled student and the BPNet baseline |
| `data/efficiency_18task_authoritative.csv`, `data/latency_metrics.csv` | measured GPU/CPU latency and peak memory per model |
| `data/method_comparison_18task.csv`, `data/distillation_method_comparison_enformer.tsv` | distillation-method comparison (OmegaGenome, DKD, DIST, LS) |
| `data/distillation_hyperparameter_comparison_enformer.csv`, `data/distillation_summary_expanded.csv` | loss-weight and temperature sweeps |
| `data/size_18task_seed0_matrix.csv`, `data/size18task_seed0_best_test_mcc.json` | student-size sweep across 18 tasks |
| `data/hp_search_csv/`, `hyperparam_csvs/` | the raw per-run hyperparameter-search exports behind the appendix figures |

These tables are aggregated from the training runs produced by the code in the repository root;
`figure_panels/build/aggregate_method3_18task.py` and `data/_merge_hpfix.py` are the aggregation
steps. The notebooks (`*.ipynb`) are the exploratory versions of the same plots, kept with their
outputs stripped.
