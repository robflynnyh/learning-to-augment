# ROB-124 384-Dropout Earnings Reward-Control Evaluation

## Run Metadata

- Checkpoint: `/store/store5/data/acp21rjf_checkpoints/l2augment/models/reward_conditioned_mask_lm/no_audio_tedlium_per_utterance_384d_dropout0p1_500ep_lr1e3.pt`
- Policy: `RewardConditionedMaskLM`, `hidden_dim=384`, `dropout=0.1`
- Dataset/split: `earnings22` / `test`
- Adaptation: `epochs=1`, `lr=1e-5`, multistep rollout
- ROB-120 baseline CSV: `/exp/exp4/acp21rjf/symphony-workspaces-learning-to-augment/ROB-124/exp/results/repro/reward_conditioned_lm/no_audio_conditioning/old_ablations/rob120_earnings_reward_controls/rob120_earnings_reward_controls.csv`
- Branch: `symphony/ROB-124-384-dropout-mask-lm`
- Commit: `1d5609134199f89bdf117b9002d671b53d744852`
- Main log: `/exp/exp4/acp21rjf/symphony-workspaces-learning-to-augment/ROB-124/exp/results/repro/reward_conditioned_lm/no_audio_conditioning/rob124_384_dropout_reward_conditioning/earnings_reward_controls/logs/rob124_384_dropout_earnings_reward_controls_corrected_sampling.log`
- Screen log: `/exp/exp4/acp21rjf/symphony-workspaces-learning-to-augment/ROB-124/exp/results/repro/reward_conditioned_lm/no_audio_conditioning/rob124_384_dropout_reward_conditioning/earnings_reward_controls/logs/rob124_384_dropout_earnings_reward_controls_corrected_sampling.screen.log`
- Queued command: `screen -L -Logfile /exp/exp4/acp21rjf/symphony-workspaces-learning-to-augment/ROB-124/exp/results/repro/reward_conditioned_lm/no_audio_conditioning/rob124_384_dropout_reward_conditioning/earnings_reward_controls/logs/rob124_384_dropout_earnings_reward_controls_corrected_sampling.screen.log -dmS rob124-384-dropout-earnings-corrected-sampling bash -lc 'cd /exp/exp4/acp21rjf/symphony-workspaces-learning-to-augment/ROB-124 && /store/store5/software/simple-gpu-schedule/with-gpu 1,2 -- scripts/launch_rob124_384_dropout_earnings_reward_controls_corrected_sampling.sh'`

## Results

| Condition | Status | Original WER | Updated WER | Delta | Relative change | ROB-120 updated WER | Delta vs ROB-120 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| fixed reward 0.0 | complete | 0.235218 | 0.202255 | -0.032964 | -14.01% | 0.200008 | 0.002247 |
| fixed reward 1.0 | complete | 0.235218 | 0.196740 | -0.038478 | -16.36% | 0.197619 | -0.000879 |
| uniform reward [0.0, 1.0] | complete | 0.235218 | 0.196700 | -0.038519 | -16.38% | 0.195801 | 0.000899 |
| uniform reward [0.5, 1.0] | complete | 0.235218 | 0.194433 | -0.040786 | -17.34% | 0.196434 | -0.002001 |

Completed conditions: `4/4`.

## Interpretation

This corrected rerun confirms that the earlier range-labelled rows were
methodologically stale: the corrected path now samples from
`conditioning_reward_range` during adaptation rather than silently falling back
to the default fixed reward.

The best corrected ROB-124 condition is true uniform reward `[0.5, 1.0]` with
updated WER `0.194433`. It is better than the matched ROB-120 `[0.5, 1.0]`
updated WER by `0.002001` absolute WER, and it is also better than the corrected
fixed reward `1.0` ROB-124 row by `0.002307`. This is sufficient evidence to
resume the paused ROB-108-style all-dataset sampled-reward eval for the
384/dropout checkpoint.

CSV artifact:

```text
/exp/exp4/acp21rjf/symphony-workspaces-learning-to-augment/ROB-124/exp/results/repro/reward_conditioned_lm/no_audio_conditioning/rob124_384_dropout_reward_conditioning/earnings_reward_controls/rob124_384_dropout_earnings_reward_controls.csv
```
