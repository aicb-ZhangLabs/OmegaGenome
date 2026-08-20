#!/bin/bash
# CPU-only: prefetch a manifest's ENCODE bigWigs to the shared SSD cache so a later GPU eval job
# skips the download and goes straight to inference. Schedules immediately under GPU saturation.
#   sbatch slurm/ntv3_prefetch_bigwigs.sbatch config/distillation/ntv3_native_broad.json
#SBATCH --job-name=ntv3-prefetch
#SBATCH --nodelist=galaxy
#SBATCH --partition=zhanglab.p
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --nice=0
#SBATCH --requeue
#SBATCH --time=3:00:00
#SBATCH --output=slurm-%x-%j.out
set -euo pipefail
REPO=/home/pengchx3/text-dna/OmegaGenome_Revise_202606/code_ntv3; cd "$REPO"
PV=/home/pengchx3/text-dna/OmegaGenome_Revise_202606/code_carbon/.venv_carbon_portable/bin/python
SSD=/tmp/galaxy_srv_disk00/pengchx3
mountpoint -q "$SSD" 2>/dev/null || SSD=/srv/disk00/sshfs/pengchx3
MANIFEST="${1:-config/distillation/ntv3_native_broad.json}"
CUDA_VISIBLE_DEVICES="" "$PV" -c "
import json,sys; from src.data.ground_truth import prepare_bigwigs
m=json.load(open('$MANIFEST')); d='$SSD/ntv3_targets/ground_truth_bigwigs'
p=prepare_bigwigs(m,d); print('cached',sum(x is not None for x in p),'/',len(p),'bigWigs ->',d)
"
echo "=== prefetch done ==="
