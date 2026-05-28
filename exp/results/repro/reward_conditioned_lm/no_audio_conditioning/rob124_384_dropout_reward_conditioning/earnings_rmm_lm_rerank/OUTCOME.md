# ROB-124 384-Dropout RMM-LM Rerank Evaluation

## Run Metadata

- Checkpoint: `/store/store5/data/acp21rjf_checkpoints/l2augment/models/reward_conditioned_mask_lm/no_audio_tedlium_per_utterance_384d_dropout0p1_500ep_lr1e3.pt`
- Policy: `RMMRewardConditionedMaskLMReranker`.
- Rerank recipe: generate `15` RMM candidate masks per adaptation step, encode each mask with the mask BVAE, score each VQ sequence with the 384/dropout reward-conditioned mask LM at fixed reward `1.0`, and adapt with the lowest per-candidate CE-loss mask.
- Dataset/split: `earnings22` / `test`
- Adaptation: `epochs=1`, `lr=1e-5`, multistep rollout
- Previous ROB-124 comparison CSV: `/exp/exp4/acp21rjf/symphony-workspaces-learning-to-augment/ROB-124/exp/results/repro/reward_conditioned_lm/no_audio_conditioning/rob124_384_dropout_reward_conditioning/earnings_reward_controls/rob124_384_dropout_earnings_reward_controls.csv`
- ROB-120 comparison CSV: `/exp/exp4/acp21rjf/symphony-workspaces-learning-to-augment/ROB-124/exp/results/repro/reward_conditioned_lm/no_audio_conditioning/old_ablations/rob120_earnings_reward_controls/rob120_earnings_reward_controls.csv`
- Branch: `symphony/ROB-124-384-dropout-mask-lm`
- Commit: `3112b135a23f4ca8dda50a62b541611a69a3d14b`
- Main log: `/exp/exp4/acp21rjf/symphony-workspaces-learning-to-augment/ROB-124/exp/results/repro/reward_conditioned_lm/no_audio_conditioning/rob124_384_dropout_reward_conditioning/earnings_rmm_lm_rerank/logs/rob124_384_dropout_rmm_lm_rerank.log`
- Screen log: `/exp/exp4/acp21rjf/symphony-workspaces-learning-to-augment/ROB-124/exp/results/repro/reward_conditioned_lm/no_audio_conditioning/rob124_384_dropout_reward_conditioning/earnings_rmm_lm_rerank/logs/rob124_384_dropout_rmm_lm_rerank.screen.log`
- Queued command: `screen -L -Logfile /exp/exp4/acp21rjf/symphony-workspaces-learning-to-augment/ROB-124/exp/results/repro/reward_conditioned_lm/no_audio_conditioning/rob124_384_dropout_reward_conditioning/earnings_rmm_lm_rerank/logs/rob124_384_dropout_rmm_lm_rerank.screen.log -dmS rob124-384-dropout-rmm-lm-rerank bash -lc 'cd /exp/exp4/acp21rjf/symphony-workspaces-learning-to-augment/ROB-124 && /store/store5/software/simple-gpu-schedule/with-gpu 1,2 -- scripts/launch_rob124_384_dropout_rmm_lm_rerank.sh'`

## Results

| Method | Status | Original WER | Updated WER | Delta | vs ROB-124 fixed 1.0 | vs ROB-124 best prior | vs ROB-120 fixed 1.0 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| RMM 15 candidates, reward-1 LM CE rerank | complete | 0.235239 | 0.202377 | -0.032862 | 0.006923 | 0.007434 | 0.004758 |

CSV artifact: `/exp/exp4/acp21rjf/symphony-workspaces-learning-to-augment/ROB-124/exp/results/repro/reward_conditioned_lm/no_audio_conditioning/rob124_384_dropout_reward_conditioning/earnings_rmm_lm_rerank/rob124_384_dropout_rmm_lm_rerank.csv`

## Interpretation

The callback-backed eval completed successfully and the 384/dropout checkpoint
is usable as an LM scorer inside the RMM candidate rerank loop. The selected
masks reduced WER from `0.235239` to `0.202377` on the matched Earnings-22 test
adaptation setup.

This rerank recipe does not improve on the prior ROB-124 reward-control evals:
it is `0.006923` absolute WER worse than the ROB-124 fixed reward `1.0`
condition and `0.007434` worse than the best prior ROB-124 condition
(`uniform_0.5_1.0`). It is also `0.004758` worse than the ROB-120 fixed reward
`1.0` baseline. The 384/dropout checkpoint remains the best current capacity
point, but this specific 15-candidate RMM proposal plus reward-1 LM CE-rerank
strategy is not justified over direct reward-conditioned sampling by this eval.
