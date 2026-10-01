#!/bin/bash
# Durable dispatcher: runs the cap-respecting auto-submitter from a COMPUTE node so it survives
# login-node reaping/round-robin. Regenerates resume-aware gaps (carbon HP + 3-seed) each round,
# submits within caps, sleeps, repeats until both are empty. CPU-only (no GPU).
#SBATCH --job-name=og-dispatch
#SBATCH --partition=zhanglab.p
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=2-00:00:00
#SBATCH --output=slurm/dispatch-%j.out
set -uo pipefail
REPO=${OG_ROOT:-$PWD}
cd "$REPO"
PY=python
export PATH=/pkg/slurm/22.05.3/bin:$PATH

for round in $(seq 1 288); do
  echo "===== round $round $(date) ====="
  "$PY" slurm/gen_carbon_gaps.py        > hp_carbon_remaining.txt   2>>slurm/_dispatch_gapcount.log
  "$PY" slurm/gen_3seed_gaps.py         > threeseed_resume_specs.txt 2>>slurm/_dispatch_gapcount.log
  "$PY" slurm/gen_carbon_3seed_gaps.py  > carbon_3seed_specs.txt     2>>slurm/_dispatch_gapcount.log
  cat hp_carbon_remaining.txt threeseed_resume_specs.txt carbon_3seed_specs.txt > _dispatch_all.txt
  n=$(grep -cvE '^\s*$' _dispatch_all.txt)
  echo "carbonHP=$(grep -cvE '^\s*$' hp_carbon_remaining.txt) 4teach3seed=$(grep -cvE '^\s*$' threeseed_resume_specs.txt) carbon3seed=$(grep -cvE '^\s*$' carbon_3seed_specs.txt) total_gaps=$n"
  if [ "$n" -eq 0 ]; then echo "ALL DONE at round $round"; break; fi
  bash slurm/auto_submit_specs.sh _dispatch_all.txt   # returns after all dispatched within caps
  sleep 300                                            # let jobs run, then re-check gaps
done
echo "===== dispatcher exiting $(date) ====="
