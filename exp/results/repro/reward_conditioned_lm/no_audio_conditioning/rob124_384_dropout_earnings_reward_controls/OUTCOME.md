# ROB-124 384-Dropout Earnings Reward-Control Evaluation

## Run Metadata

- Checkpoint: `/store/store5/data/acp21rjf_checkpoints/l2augment/models/reward_conditioned_mask_lm/no_audio_tedlium_per_utterance_384d_dropout0p1_500ep_lr1e3.pt`
- Policy: `RewardConditionedMaskLM`, `hidden_dim=384`, `dropout=0.1`
- Dataset/split: `earnings22` / `test`
- Adaptation: `epochs=1`, `lr=1e-5`, multistep rollout
- ROB-120 baseline CSV: `/exp/exp4/acp21rjf/symphony-workspaces-learning-to-augment/ROB-124/exp/results/repro/reward_conditioned_lm/no_audio_conditioning/rob120_earnings_reward_controls/rob120_earnings_reward_controls.csv`
- Branch: `symphony/ROB-124-384-dropout-mask-lm`
- Commit: `8fd7e5e15f0bf96ee516ee3b12836ad8f74ab4fa`
- Main log: `/exp/exp4/acp21rjf/symphony-workspaces-learning-to-augment/ROB-124/exp/results/repro/reward_conditioned_lm/no_audio_conditioning/rob124_384_dropout_earnings_reward_controls/logs/rob124_384_dropout_earnings_reward_controls.log`
- Screen log: `/exp/exp4/acp21rjf/symphony-workspaces-learning-to-augment/ROB-124/exp/results/repro/reward_conditioned_lm/no_audio_conditioning/rob124_384_dropout_earnings_reward_controls/logs/rob124_384_dropout_earnings_reward_controls.screen.log`
- Queued command: `screen -L -Logfile /exp/exp4/acp21rjf/symphony-workspaces-learning-to-augment/ROB-124/exp/results/repro/reward_conditioned_lm/no_audio_conditioning/rob124_384_dropout_earnings_reward_controls/logs/rob124_384_dropout_earnings_reward_controls.screen.log -dmS rob124-384-dropout-earnings-reward-controls bash -lc 'cd /exp/exp4/acp21rjf/symphony-workspaces-learning-to-augment/ROB-124 && /store/store5/software/simple-gpu-schedule/with-gpu 1,2 -- scripts/launch_rob124_384_dropout_earnings_reward_controls.sh'`

## Results

| Condition | Status | Original WER | Updated WER | Delta | Relative change | ROB-120 updated WER | Delta vs ROB-120 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| fixed reward 0.0 | complete | 0.235218 | 0.201438 | -0.033781 | -14.36% | 0.200008 | 0.001430 |
| fixed reward 1.0 | complete | 0.235198 | 0.195454 | -0.039744 | -16.90% | 0.197619 | -0.002165 |
| uniform reward [0.0, 1.0] | complete | 0.235218 | 0.195474 | -0.039744 | -16.90% | 0.195801 | -0.000327 |
| uniform reward [0.5, 1.0] | complete | 0.235198 | 0.194943 | -0.040255 | -17.12% | 0.196434 | -0.001491 |

Completed conditions: `4/4`.

CSV artifact:

```text
/exp/exp4/acp21rjf/symphony-workspaces-learning-to-augment/ROB-124/exp/results/repro/reward_conditioned_lm/no_audio_conditioning/rob124_384_dropout_earnings_reward_controls/rob124_384_dropout_earnings_reward_controls.csv
```
