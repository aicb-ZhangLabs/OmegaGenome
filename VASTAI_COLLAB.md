# vast.ai H100 collaboration — control plane (GOAL-B Carbon)

> Operational coordination only. Sensitive access info is in the gitignored REMOTE_H100_VAST.md.
> Results go to EXPERIMENTS_carbon_distill.md (single source of truth) after merge.

## Division of labor (task partition — disjoint, so no double-runs)
| Machine | Owns (carbon tasks, end-to-end stage1+2+3) |
|---|---|
| **vast.ai H100** | splice_sites_all, splice_sites_acceptors, splice_sites_donors, enhancers, enhancers_types, H3K27ac, H3K4me1 |
| **main cluster** | the other 11 tasks (H2AFZ, H3K27me3, H3K36me3, H3K4me2/me3, H3K9ac/me3, H4K20me1, promoter_all/no_tata/tata) + all GOAL-A NTv3 |

Both sides SKIP any config whose final_summary.json already exists, so the partition guarantees no collisions.

## Run recipe (vast.ai, no SLURM)
- config list: hp_original_stage1.txt (grep the 7 owned tasks)
- launch: local bash loop, 2-3 configs concurrent on the single H100
- writes: carbon_distillation/original/<task>/.../final_summary.json

## Merge protocol
1. rsync vast.ai:carbon_distillation/original/<7 tasks>/ -> galaxy SSD $SSD/carbon_distillation/original/
2. collate: slurm/gen_stage_csvs.py --tag original
3. record summary numbers into EXPERIMENTS_carbon_distill.md

## Sync log (append each merge)
- 2026-06-27: partition defined; vast.ai owns the 7 GPU-starved laggard tasks. Setup pending (SSH key handoff).
