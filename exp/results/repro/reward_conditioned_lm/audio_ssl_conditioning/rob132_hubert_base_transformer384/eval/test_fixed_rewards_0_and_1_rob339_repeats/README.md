# ROB-339 RAC-MLM Fixed-Reward Repeats

This directory is the ROB-339 result root for two additional RAC-MLM fixed-reward
test-set repeats. It keeps repeat 2 and repeat 3 outputs separate from the
historical repeat 1 artifacts in the sibling
`test_fixed_rewards_0_and_1/` directory so the original ROB-132/ROB-201 rows are
not overwritten.

Launch helper:

```bash
scripts/submit_rob339_rac_mlm_fixed_reward_repeats_stanage.sh
```

The Stanage launcher submits one GPU Slurm job per dataset, fixed reward, epoch,
and new repeat. The finalizer aggregates repeats 1, 2, and 3 into
`rob339_rac_mlm_fixed_reward_repeats.csv`,
`rob339_rac_mlm_fixed_reward_repeats_aggregate.csv`, and `OUTCOME.md`.

Repeat seeds:

- repeat 1: `123456` from the historical result root
- repeat 2: `123457`
- repeat 3: `123458`
