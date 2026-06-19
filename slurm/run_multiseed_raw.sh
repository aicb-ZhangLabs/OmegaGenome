#!/bin/bash
# Multi-seed carbon-raw for significance (P0-3): seeds 1 & 2 (seed 0 already done) -> 3 seeds total.
# Cache is fully populated + self-validating, so these are cache-hit + student-train only (fast).
# Sequential per seed (one submitter at a time); authoritative count_node keeps caps even while the
# l2norm campaign drains concurrently.
cd /home/pengchx3/text-dna/OmegaGenome_Revise_202606/code_carbon
TASKS="H3K27me3 H3K36me3 H4K20me1 H2AFZ H3K27ac H3K4me1 H3K4me2 H3K4me3 H3K9ac H3K9me3 \
       promoter_all promoter_tata promoter_no_tata enhancers enhancers_types \
       splice_sites_all splice_sites_acceptors splice_sites_donors"
for S in 1 2; do
  echo "[multiseed $(date +%H:%M:%S)] launching carbon-raw seed=$S"
  SEED=$S bash slurm/auto_submit_carbon.sh carbon-raw $TASKS > /tmp/auto_carbon_seed$S.log 2>&1
  echo "[multiseed $(date +%H:%M:%S)] seed=$S all submitted"
done
echo "[multiseed $(date +%H:%M:%S)] all multi-seed submitted"
