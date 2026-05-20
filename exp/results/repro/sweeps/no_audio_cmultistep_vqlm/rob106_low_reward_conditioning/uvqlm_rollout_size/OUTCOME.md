# ROB-106 UVQLM Rollout Folder Size

Exact CPU-node scan of the Stanage UVQLM rollout folder requested as a ROB-106
follow-up.

- Root: `/mnt/parscratch/users/acp21rjf/l2augment_rollout_uvqmlm`
- Generated UTC: `2026-05-20T20:02:26.278760+00:00`
- External artifact directory: `/mnt/parscratch/users/acp21rjf/symphony-job-artifacts/ROB-106/uvqlm_rollout_size`
- Log: `/mnt/parscratch/users/acp21rjf/symphony-job-artifacts/ROB-106/uvqlm_rollout_size/rob106_uvqlm_size_scan_10244545.log`
- Slurm job: `10244545`
- Queued command: `sbatch scripts/slurm_rob106_uvqlm_size_scan.sbatch`
- Measurement command: `python3 scripts/measure_rollout_tree_size.py --root /mnt/parscratch/users/acp21rjf/l2augment_rollout_uvqmlm --output-dir /mnt/parscratch/users/acp21rjf/symphony-job-artifacts/ROB-106/uvqlm_rollout_size --progress-every 50000`

## Total

| Metric | Value |
| --- | ---: |
| Files | 266408 |
| Directories | 3 |
| Symlinks | 0 |
| Other entries | 0 |
| Apparent bytes | 157005586888 (146.22 GiB) |
| Allocated bytes | 157552103424 (146.73 GiB) |

## Top-Level Breakdown

| Name | Files | Directories | Apparent bytes | Allocated bytes |
| --- | ---: | ---: | ---: | ---: |
| `.` | 0 | 1 | 4096 (4.00 KiB) | 4096 (4.00 KiB) |
| `dev` | 507 | 1 | 554856872 (529.15 MiB) | 555896832 (530.14 MiB) |
| `train` | 265901 | 1 | 156450725920 (145.71 GiB) | 156996202496 (146.21 GiB) |
