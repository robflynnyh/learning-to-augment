# ROB-124 384-Dropout Reward-Conditioning Evals

This folder groups the useful downstream investigations for the preferred
ROB-124 384-hidden-dim, dropout-0.1 no-audio reward-conditioned mask LM.

## Contents

- `earnings_reward_controls/`: corrected matched Earnings-22 reward-control
  comparison against ROB-120.
- `all_dataset_sampled_reward_0p5_to_1p0/`: completed all-dataset sampled
  reward `[0.5, 1.0]` eval.
- `all_dataset_fixed_rewards_0_and_1/`: completed all-dataset fixed reward
  `1.0` and `0.0` eval.
- `earnings_rmm_lm_rerank/`: RMM proposal plus reward-1 LM CE reranker eval.

The stale initial Earnings result and the cancelled sampled `[0.0, 1.0]`
scaffold were removed from the active tree. See the top-level `OUTCOME.md` for
the consolidated interpretation.
