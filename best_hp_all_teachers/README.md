# Best-HP per teacher (NT / DNABERT2 / Caduceus / Enformer)

Per-task best hyperparameters for each teacher's BPNet-distillation HP search, selected by **validation MCC**
(test MCC recorded from that same best-val row). Mirrors the carbon-3B format in
`code_carbon/slurm/extract_best_hyperparams.py` / `best_hyperparams.json`.

## Files
- `make_best_hp.py` — generator. `python make_best_hp.py --csv <wandb_export.csv> --teacher <nt|dnabert2|caduceus|enformer> --out-dir best_hp`
- `verify_best_hp.py` — re-derives from the CSV and asserts each YAML's best_val == CSV max-val and best_test == that row.
- `best_hp/best_hp_<teacher>.{yaml,csv,md}` — 18 tasks each.

## Source (wandb HP-search exports)
`/extra/zhanglab0/INDV/pengchx3/OmegaGenome_different_version/OmegaGenome_11_1_fix_ckpt/OmegaGenome/`:
`{nt,dnabert2}_hyperparam_flat_clean.csv`, `caduceus_hyperparam_1-15_flat_clean.csv`, `enformer_hyperparam_flat_clean.csv`.

## Notes
- All teachers: weight_ce=0.5, distill_method=vanilla, kl_method=kl, lr=1e-4. Swept per task: weight_kl, weight_mse, temperature.
- KNOWN DATA GAP: nt/`splice_sites_all` best-val row has no test MCC (recorded null) — re-run if a test number is needed.
