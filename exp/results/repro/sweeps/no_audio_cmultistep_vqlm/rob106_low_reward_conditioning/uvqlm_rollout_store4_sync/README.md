# ROB-106 UVQLM Rollout Store4 Sync

Follow-up artifact path for the callback-backed copy of the Stanage UVQLM
rollout folder to Mimas `/store/store4`.

Source:

```text
stanage.shef.ac.uk:/mnt/parscratch/users/acp21rjf/l2augment_rollout_uvqmlm/
```

Destination:

```text
/store/store4/data/l2augment_rollout_uvqmlm/
```

Wrapper:

```bash
./scripts/launch_rob106_uvqlm_store4_sync.sh
```

Detached launch pattern:

```bash
screen -L -Logfile exp/results/repro/sweeps/no_audio_cmultistep_vqlm/rob106_low_reward_conditioning/uvqlm_rollout_store4_sync/rob106_uvqlm_store4_rsync.screen.log \
  -dmS rob106-uvqlm-store4-sync \
  bash -lc 'cd /exp/exp4/acp21rjf/symphony-workspaces-learning-to-augment/ROB-106 && ./scripts/launch_rob106_uvqlm_store4_sync.sh'
```

The wrapper logs live `/store/store4` free space, runs resumable `rsync` with
`--partial --append-verify`, and posts a Linear callback on every shell exit.
