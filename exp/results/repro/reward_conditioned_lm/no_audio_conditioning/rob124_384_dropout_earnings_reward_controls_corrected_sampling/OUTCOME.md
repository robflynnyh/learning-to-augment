# ROB-124 384-Dropout Earnings Reward-Control Evaluation

## Run Metadata

- Checkpoint: `/store/store5/data/acp21rjf_checkpoints/l2augment/models/reward_conditioned_mask_lm/no_audio_tedlium_per_utterance_384d_dropout0p1_500ep_lr1e3.pt`
- Policy: `RewardConditionedMaskLM`, `hidden_dim=384`, `dropout=0.1`
- Dataset/split: `earnings22` / `test`
- Adaptation: `epochs=1`, `lr=1e-5`, multistep rollout
- ROB-120 baseline CSV: `exp/results/repro/reward_conditioned_lm/no_audio_conditioning/rob120_earnings_reward_controls/rob120_earnings_reward_controls.csv`
- Branch: `symphony/ROB-124-384-dropout-mask-lm`
- Commit: `4c4ce1c378424df43f7dd7793c1a55899b8d5c23`
- Main log: `exp/results/repro/reward_conditioned_lm/no_audio_conditioning/rob124_384_dropout_earnings_reward_controls_corrected_sampling/logs/rob124_384_dropout_earnings_reward_controls_corrected_sampling.log`
- Screen log: `exp/results/repro/reward_conditioned_lm/no_audio_conditioning/rob124_384_dropout_earnings_reward_controls_corrected_sampling/logs/rob124_384_dropout_earnings_reward_controls_corrected_sampling.screen.log`
- Queued command: `screen -L -Logfile /exp/exp4/acp21rjf/symphony-workspaces-learning-to-augment/ROB-124/exp/results/repro/reward_conditioned_lm/no_audio_conditioning/rob124_384_dropout_earnings_reward_controls_corrected_sampling/logs/rob124_384_dropout_earnings_reward_controls_corrected_sampling.screen.log -dmS rob124-384-dropout-earnings-corrected-sampling bash -lc 'cd /exp/exp4/acp21rjf/symphony-workspaces-learning-to-augment/ROB-124 && /store/store5/software/simple-gpu-schedule/with-gpu 1,2 -- scripts/launch_rob124_384_dropout_earnings_reward_controls_corrected_sampling.sh'`

## Results

| Condition | Status | Original WER | Updated WER | Delta | Relative change | ROB-120 updated WER | Delta vs ROB-120 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| fixed reward 0.0 | missing |  |  |  |  | 0.200008 |  |
| fixed reward 1.0 | missing |  |  |  |  | 0.197619 |  |
| uniform reward [0.0, 1.0] | missing |  |  |  |  | 0.195801 |  |
| uniform reward [0.5, 1.0] | missing |  |  |  |  | 0.196434 |  |

Completed conditions: `0/4`.

CSV artifact:

```text
exp/results/repro/reward_conditioned_lm/no_audio_conditioning/rob124_384_dropout_earnings_reward_controls_corrected_sampling/rob124_384_dropout_earnings_reward_controls.csv
```

Residual risk: this is a partial snapshot; inspect the log before treating it as the final comparison.
