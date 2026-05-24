# ROB-124 Corrected 384-Dropout Earnings Reward-Control Rerun

This result root is a corrected rerun of the ROB-124 Earnings-22 reward-control
comparison after fixing `RewardConditionedMaskLM.augment` so
`conditioning_reward_range` is sampled during adaptation.

Scope:

- Policy checkpoint:
  `/store/store5/data/acp21rjf_checkpoints/l2augment/models/reward_conditioned_mask_lm/no_audio_tedlium_per_utterance_384d_dropout0p1_500ep_lr1e3.pt`
- Policy shape: `RewardConditionedMaskLM`, `hidden_dim=384`, `dropout=0.1`.
- Dataset/split: Earnings-22 test.
- Adaptation: `epochs=1`, `lr=1e-5`, multistep rollout.
- Conditions: fixed reward `0.0`, fixed reward `1.0`, uniform `[0.0, 1.0]`,
  and uniform `[0.5, 1.0]`.
- Result CSV: `rob124_384_dropout_earnings_reward_controls.csv`.
- Outcome: `OUTCOME.md`.

The wrapper is:

```bash
scripts/launch_rob124_384_dropout_earnings_reward_controls_corrected_sampling.sh
```

The detached Mimas command is:

```bash
screen -L -Logfile /exp/exp4/acp21rjf/symphony-workspaces-learning-to-augment/ROB-124/exp/results/repro/reward_conditioned_lm/no_audio_conditioning/rob124_384_dropout_earnings_reward_controls_corrected_sampling/logs/rob124_384_dropout_earnings_reward_controls_corrected_sampling.screen.log -dmS rob124-384-dropout-earnings-corrected-sampling bash -lc 'cd /exp/exp4/acp21rjf/symphony-workspaces-learning-to-augment/ROB-124 && /store/store5/software/simple-gpu-schedule/with-gpu 1,2 -- scripts/launch_rob124_384_dropout_earnings_reward_controls_corrected_sampling.sh'
```
