| task | weight_ce | weight_kl | weight_mse | temperature | distill_method | kl_method | dkd_alpha | dkd_beta | lr | batch_size | epochs | seed | model_size | hidden_dim | best_val_mcc | best_test_mcc | best_epoch | n_candidates |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| H2AFZ | 0.5 | 0 | 5 | 0.5 | vanilla | kl | 1 | 8 | 0.0001 | 32 | 200 | 42 | original |  | 0.5169 | 0.4813 | 200 | 48 |
| H3K27ac | 0.5 | 0 | 1 | 0.5 | vanilla | kl | 1 | 8 | 0.0001 | 32 | 200 | 42 | original |  | 0.4846 | 0.4756 | 200 | 48 |
| H3K27me3 | 0.5 | 0.25 | 0 | 0.5 | vanilla | kl | 1 | 8 | 0.0001 | 32 | 200 | 42 | original |  | 0.5924 | 0.5618 | 200 | 94 |
| H3K36me3 | 0.5 | 1 | 2 | 2 | vanilla | kl | 1 | 8 | 0.0001 | 32 | 200 | 42 | original |  | 0.628 | 0.5831 | 200 | 48 |
| H3K4me1 | 0.5 | 1 | 1 | 2 | vanilla | kl | 1 | 8 | 0.0001 | 32 | 200 | 42 | original |  | 0.4995 | 0.4705 | 200 | 48 |
| H3K4me2 | 0.5 | 1 | 1 | 1.5 | vanilla | kl | 1 | 8 | 0.0001 | 32 | 200 | 42 | original |  | 0.6099 | 0.5372 | 200 | 48 |
| H3K4me3 | 0.5 | 0.5 | 1 | 2 | vanilla | kl | 1 | 8 | 0.0001 | 32 | 200 | 42 | original |  | 0.6983 | 0.5933 | 200 | 48 |
| H3K9ac | 0.5 | 0.5 | 5 | 0.5 | vanilla | kl | 1 | 8 | 0.0001 | 32 | 200 | 42 | original |  | 0.5421 | 0.5054 | 200 | 48 |
| H3K9me3 | 0.5 | 1 | 1 | 4 | vanilla | kl | 1 | 8 | 0.0001 | 32 | 200 | 42 | original |  | 0.4405 | 0.4024 | 200 | 48 |
| H4K20me1 | 0.5 | 1 | 5 | 4 | vanilla | kl | 1 | 8 | 0.0001 | 32 | 200 | 42 | original |  | 0.6234 | 0.6008 | 200 | 48 |
| enhancers | 0.5 | 1 | 2 | 2 | vanilla | kl | 1 | 8 | 0.0001 | 32 | 200 | 42 | original |  | 0.5233 | 0.51 | 200 | 48 |
| enhancers_types | 0.5 | 1 | 5 | 1.5 | vanilla | kl | 1 | 8 | 0.0001 | 32 | 200 | 42 | original |  | 0.5043 | 0.4636 | 200 | 48 |
| promoter_all | 0.5 | 0.5 | 5 | 0.5 | vanilla | kl | 1 | 8 | 0.0001 | 32 | 200 | 42 | original |  | 0.7435 | 0.7344 | 200 | 48 |
| promoter_no_tata | 0.5 | 1 | 5 | 0.5 | vanilla | kl | 1 | 8 | 0.0001 | 32 | 200 | 42 | original |  | 0.7626 | 0.7365 | 200 | 48 |
| promoter_tata | 0.5 | 1 | 1 | 0.5 | vanilla | kl | 1 | 8 | 0.0001 | 32 | 200 | 42 | original |  | 0.8627 | 0.8208 | 200 | 48 |
| splice_sites_acceptors | 0.5 | 0 | 1 | 0.5 | vanilla | kl | 1 | 8 | 0.0001 | 32 | 200 | 42 | original |  | 0.8488 | 0.832 | 200 | 48 |
| splice_sites_all | 0.5 | 1 | 1 | 4 | vanilla | kl | 1 | 8 | 0.0001 | 32 | 200 | 42 | original |  | 0.8714 | 0.854 | 200 | 48 |
| splice_sites_donors | 0.5 | 0 | 1 | 0.5 | vanilla | kl | 1 | 8 | 0.0001 | 32 | 200 | 42 | original |  | 0.9155 | 0.876 | 200 | 48 |
