# Vast.ai box access (Carbon-3B HP search / rebuttal)

> **DESTROYED 2026-07-01** — instance terminated by user after full backup (ckpts+curves+33G wandb on lab SSD `box_backup/`, students+curves on HF). IP/port below are DEAD; kept for reference only.


**Connect:**
```bash
ssh -o StrictHostKeyChecking=no -p 26925 root@115.124.123.240
```
- The **port `-p 26925` is required** — plain `ssh root@115.124.123.240` (port 22) fails with
  "Permission denied". My `~/.ssh/id_rsa` IS authorized on the box.
- Vast.ai instances are ephemeral: **IP and port can change on restart.** If it fails, get the
  current `ssh` line from the vast.ai dashboard and update this file + the `sync_box_*.sh` scripts.
- Box hostname (container): `e0635474e19b`. Disk: overlay 180G (~57G free as of 2026-07-01).

**Box layout:**
- `BOX_ROOT=/workspace/rebuttal_nt/run` — results/, output/ (best_model ckpts), wandb/, logs
- Operating guide ON the box: `/etc/vast-agents-guide.md` (also `./AGENTS.md`) — read before acting.

**Sync (box -> lab SSD), already running as loops on the lab node:**
- `slurm/sync_box_r13_artifacts.sh` -> `/tmp/galaxy_srv_disk00/pengchx3/rebuttal_nt/run/`
  (pulls *.csv, output/best_model/**, metadata.json, run_info.txt, wandb/, logs; `rsync --update`)
- `slurm/sync_box_nt_latefuse_results.sh`, `slurm/sync_box_dnabert2_results.sh` (CSV-only loops)
- SSH opts used: `-o ConnectTimeout=25 -o StrictHostKeyChecking=no -o ServerAliveInterval=15 -p 26925`

**HF backup repos (box uploads here directly):**
- `explcre/omegagenome-carbon-hp-students` — full HP-grid ckpts + metrics/ + r13_embedding/
- `explcre/omegagenome-distilled-students` — carbon_bpnet_students/{task}.pt (best per task) + ntv3_kd_students/
