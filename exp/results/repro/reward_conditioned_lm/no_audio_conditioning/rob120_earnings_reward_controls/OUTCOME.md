# ROB-120 Earnings Reward-Control Evaluation

## Run Metadata

- Checkpoint: `/store/store5/data/acp21rjf_checkpoints/l2augment/models/reward_conditioned_mask_lm/no_audio_tedlium_per_utterance_resume100_500ep_lr1e3.pt`
- Dataset/split: `earnings22` / `test`
- Adaptation: `epochs=1`, `lr=1e-5`, multistep rollout
- Branch: `symphony/ROB-120-earnings-reward-controls`
- Commit: `ddd349c30acb056cc791788e85e46153fa344c1b`
- Main log: `exp/results/repro/reward_conditioned_lm/no_audio_conditioning/rob120_earnings_reward_controls/logs/rob120_earnings_reward_controls.log`
- Screen log: `exp/results/repro/reward_conditioned_lm/no_audio_conditioning/rob120_earnings_reward_controls/logs/rob120_earnings_reward_controls.screen.log`
- Queued command: `screen -L -Logfile exp/results/repro/reward_conditioned_lm/no_audio_conditioning/rob120_earnings_reward_controls/logs/rob120_earnings_reward_controls.screen.log -dmS rob120-earnings-reward-controls bash -lc 'cd /exp/exp4/acp21rjf/symphony-workspaces-learning-to-augment/ROB-120 && /store/store5/software/simple-gpu-schedule/with-gpu 1,2 -- scripts/launch_rob120_earnings_reward_controls.sh'`

## Results

| Condition | Status | Config | Result | Original WER | Updated WER | Delta | Relative change |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| fixed reward 0.0 | missing | `exp/results/repro/reward_conditioned_lm/no_audio_conditioning/rob120_earnings_reward_controls/RewardConditionedMaskLMReward0/configs/earnings22_test_epoch1_lr1e-5.yaml` | `exp/results/repro/reward_conditioned_lm/no_audio_conditioning/rob120_earnings_reward_controls/RewardConditionedMaskLMReward0/earnings22_test_epoch1_lr1e-5.txt` |  |  |  |  |
| fixed reward 1.0 | missing | `exp/results/repro/reward_conditioned_lm/no_audio_conditioning/rob120_earnings_reward_controls/RewardConditionedMaskLMReward1/configs/earnings22_test_epoch1_lr1e-5.yaml` | `exp/results/repro/reward_conditioned_lm/no_audio_conditioning/rob120_earnings_reward_controls/RewardConditionedMaskLMReward1/earnings22_test_epoch1_lr1e-5.txt` |  |  |  |  |
| uniform reward [0.0, 1.0] | missing | `exp/results/repro/reward_conditioned_lm/no_audio_conditioning/rob120_earnings_reward_controls/RewardConditionedMaskLMUniform0to1/configs/earnings22_test_epoch1_lr1e-5.yaml` | `exp/results/repro/reward_conditioned_lm/no_audio_conditioning/rob120_earnings_reward_controls/RewardConditionedMaskLMUniform0to1/earnings22_test_epoch1_lr1e-5.txt` |  |  |  |  |
| uniform reward [0.5, 1.0] | missing | `exp/results/repro/reward_conditioned_lm/no_audio_conditioning/rob120_earnings_reward_controls/RewardConditionedMaskLMUniform0p5to1/configs/earnings22_test_epoch1_lr1e-5.yaml` | `exp/results/repro/reward_conditioned_lm/no_audio_conditioning/rob120_earnings_reward_controls/RewardConditionedMaskLMUniform0p5to1/earnings22_test_epoch1_lr1e-5.txt` |  |  |  |  |

Completed conditions: `0/4`.

CSV artifact:

```text
exp/results/repro/reward_conditioned_lm/no_audio_conditioning/rob120_earnings_reward_controls/rob120_earnings_reward_controls.csv
```

Residual risk: this is a partial snapshot; inspect the log before treating it as the final comparison.
