#!/bin/bash
# Wait for the carbon-raw auto-submitter to finish submitting all its tasks, then launch the
# carbon-l2norm campaign (same 18 tasks). Sequential = one submitter at a time (no cap double-count);
# l2norm hits raw's now-cached teacher precomputes (atomic cache -> any overlap is safe).
cd ${OG_ROOT:-$PWD}
echo "[chain $(date +%H:%M:%S)] waiting for carbon-raw submitter to finish..."
while pgrep -f "auto_submit_carbon.sh carbon-raw" >/dev/null 2>&1; do sleep 120; done
echo "[chain $(date +%H:%M:%S)] raw submitter done -> launching carbon-l2norm"
L2="H3K27me3 H3K36me3 H4K20me1 H2AFZ H3K27ac H3K4me1 H3K4me2 H3K4me3 H3K9ac H3K9me3 \
    promoter_all promoter_tata promoter_no_tata enhancers enhancers_types \
    splice_sites_all splice_sites_acceptors splice_sites_donors"
nohup bash slurm/auto_submit_carbon.sh carbon-l2norm $L2 > /tmp/auto_carbon_l2norm.log 2>&1 &
echo "[chain $(date +%H:%M:%S)] carbon-l2norm submitter launched (PID $!)"
