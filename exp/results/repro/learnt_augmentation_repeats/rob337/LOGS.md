# ROB-337 Stanage Logs

The per-cell Slurm logs are not committed because they are bulky generated
artifacts. They remain on Stanage at:

```text
/mnt/parscratch/users/acp21rjf/symphony-workspaces-learning-to-augment/ROB-337/exp/results/repro/learnt_augmentation_repeats/rob337/logs/stanage/
```

At sync time this directory was approximately `147M`. The finalizer log is:

```text
/mnt/parscratch/users/acp21rjf/symphony-workspaces-learning-to-augment/ROB-337/exp/results/repro/learnt_augmentation_repeats/rob337/logs/stanage/rob337_stanage_finalizer-10782851.log
```

The committed `queued_jobs.tsv` maps each Slurm job ID to its method, dataset,
epoch, repeat, config path, and result path. Cell log filenames follow:

```text
<method>_<dataset>_epoch<epochs>_lr1e-5_repeat<repeat>-<job_id>.log
```

The retry matrix was submitted with:

```bash
REPO_DIR=/mnt/parscratch/users/acp21rjf/symphony-workspaces-learning-to-augment/ROB-337 \
  scripts/submit_rob337_learnt_augmentation_repeats_stanage.sh
```
