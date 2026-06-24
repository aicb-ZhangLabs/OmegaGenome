#!/bin/bash
# Final-phase PREP watcher. Polls grid completion (fast galaxy-local find); once all 1440 RAW combos
# have a final_summary.json, it runs the READ-ONLY prep and STOPS — it does NOT launch the 54 3-seed
# GPU jobs (that stays an explicit human go, to control fair-share burn). Produces:
#   results/carbon_grid_results.csv   (full per-run table, refreshed)
#   best_hyperparams.json             (best-on-VAL per task)
#   best_3seed_specs.txt              (18 tasks x 3 seeds = 54 specs, ready to submit)
#   results/FINAL_PREP_READY.txt      (marker + best-HP summary)
# To then launch the 3-seed jobs:  nohup bash slurm/auto_submit_specs.sh best_3seed_specs.txt &
set -uo pipefail
cd /home/pengchx3/text-dna/OmegaGenome_Revise_202606/code_carbon
PY=.venv_carbon_portable/bin/python
GPY=/home/pengchx3/.local/share/uv/python/cpython-3.11.15-linux-x86_64-gnu/bin/python3.11
GAL=/srv/disk00/sshfs/pengchx3/carbon_distillation
SSH="ssh -o BatchMode=yes -o ConnectTimeout=15 galaxy"

gridcount(){  # unique completed RAW (task,kl,mse,T) via galaxy-local find; echoes an int (0 on failure)
  $SSH "find $GAL -name final_summary.json 2>/dev/null" 2>/dev/null | $PY -c "
import sys,re
r=set()
for l in sys.stdin:
    d=l.split('/')[-2]; m=re.search(r'deploy_120k/([^/]+)/',l)
    if m and 'normalizeFalse' in d and re.search(r'weight_kl([0-9.]+)',d) and re.search(r'weight_mse([0-9.]+)',d) and re.search(r'temperature([0-9.]+)',d):
        r.add((m.group(1),re.search(r'weight_kl([0-9.]+)',d).group(1),re.search(r'weight_mse([0-9.]+)',d).group(1),re.search(r'temperature([0-9.]+)',d).group(1)))
print(len(r))" 2>/dev/null || echo 0
}

echo "[final_prep $(date '+%F %H:%M:%S')] waiting for grid==1440 ..."
while :; do
  n=$(gridcount); n=${n:-0}
  echo "[final_prep $(date '+%F %H:%M:%S')] grid $n/1440"
  [ "$n" -ge 1440 ] 2>/dev/null && break
  sleep 900
done
echo "[final_prep $(date '+%F %H:%M:%S')] GRID COMPLETE — running read-only prep (NO gpu jobs launched)"

# (1) refresh full per-run table
$SSH "$GPY $PWD/slurm/collate_runs.py --base $GAL" > results/carbon_grid_results.csv 2>/dev/null
# (2) best-on-val per task + (3) 3-seed specs
PYTHONPATH=. $PY slurm/extract_best_hyperparams.py --out best_hyperparams.json
PYTHONPATH=. $PY slurm/gen_3seed_best_specs.py --best best_hyperparams.json --seeds 0 1 2 --out best_3seed_specs.txt

{
  echo "FINAL PREP READY @ $(date '+%F %H:%M:%S')"
  echo "grid: 1440/1440 complete; full table -> results/carbon_grid_results.csv"
  echo "best_hyperparams.json + best_3seed_specs.txt ($(grep -vc '^$' best_3seed_specs.txt) specs) generated."
  echo "3-seed GPU jobs NOT launched (awaiting explicit go)."
  echo "to launch:  nohup bash slurm/auto_submit_specs.sh best_3seed_specs.txt > seed3.log 2>&1 &"
} > results/FINAL_PREP_READY.txt
cat results/FINAL_PREP_READY.txt
