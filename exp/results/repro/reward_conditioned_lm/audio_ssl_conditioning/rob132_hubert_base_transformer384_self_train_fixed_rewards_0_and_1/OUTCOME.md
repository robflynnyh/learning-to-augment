# ROB-132 Audio SSL Self-Training Fixed-Reward Eval

## Metadata

- Checkpoint: `/store/store5/data/acp21rjf_checkpoints/l2augment/models/reward_conditioned_mask_lm/audio_ssl_hubert_base_tedlium_per_utterance_transformer384_dropout0p1_500ep_lr1e3.pt`
- Policy: `AudioRewardConditionedMaskLM`, HuBERT SSL conditioning, transformer decoder
- Reward control: fixed scalar `conditioning_reward`, no `conditioning_reward_range`
- Dataset: `tedlium`, split `dev`
- Adaptation: `epochs=1` and `epochs=5`, `lr=1e-5`, multistep rollout
- Branch: `symphony/ROB-132-audio-conditioned-vq-mask-lm`
- Commit: `b3d66492d08c696ec169e8f79de0c00ef9394d82`
- Main log: `/exp/exp4/acp21rjf/symphony-workspaces-learning-to-augment/ROB-132/exp/results/repro/reward_conditioned_lm/audio_ssl_conditioning/rob132_hubert_base_transformer384_self_train_fixed_rewards_0_and_1/logs/rob132_audio_ssl_self_train_fixed_rewards_0_and_1_retry1.log`
- Screen log: `/exp/exp4/acp21rjf/symphony-workspaces-learning-to-augment/ROB-132/exp/results/repro/reward_conditioned_lm/audio_ssl_conditioning/rob132_hubert_base_transformer384_self_train_fixed_rewards_0_and_1/logs/rob132_audio_ssl_self_train_fixed_rewards_0_and_1_retry1.screen.log`
- Queued command: `screen -L -Logfile /exp/exp4/acp21rjf/symphony-workspaces-learning-to-augment/ROB-132/exp/results/repro/reward_conditioned_lm/audio_ssl_conditioning/rob132_hubert_base_transformer384_self_train_fixed_rewards_0_and_1/logs/rob132_audio_ssl_self_train_fixed_rewards_0_and_1_retry1.screen.log -dmS rob132-audio-ssl-selftrain-fixed-rewards-r1 bash -lc 'cd /exp/exp4/acp21rjf/symphony-workspaces-learning-to-augment/ROB-132 && /store/store5/software/simple-gpu-schedule/with-gpu 1,2 -- scripts/launch_rob132_audio_ssl_self_train_fixed_rewards.sh'`

Completed cells: `4/4`.

## Results

| Reward | Epochs | Original WER | Updated WER | Abs Delta | Rel Delta % | Status | Result |
| ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1 | 1 | 0.100088 | 0.089532 | -0.010556 | -10.55 | complete | `exp/results/repro/reward_conditioned_lm/audio_ssl_conditioning/rob132_hubert_base_transformer384_self_train_fixed_rewards_0_and_1/AudioRewardConditionedMaskLMReward1p0/tedlium_dev_reward1p0_epoch1_lr1e-5.txt` |
| 1 | 5 | 0.100088 | 0.086880 | -0.013209 | -13.20 | complete | `exp/results/repro/reward_conditioned_lm/audio_ssl_conditioning/rob132_hubert_base_transformer384_self_train_fixed_rewards_0_and_1/AudioRewardConditionedMaskLMReward1p0/tedlium_dev_reward1p0_epoch5_lr1e-5.txt` |
| 0 | 1 | 0.100088 | 0.091633 | -0.008456 | -8.45 | complete | `exp/results/repro/reward_conditioned_lm/audio_ssl_conditioning/rob132_hubert_base_transformer384_self_train_fixed_rewards_0_and_1/AudioRewardConditionedMaskLMReward0p0/tedlium_dev_reward0p0_epoch1_lr1e-5.txt` |
| 0 | 5 | 0.100088 | 0.088372 | -0.011717 | -11.71 | complete | `exp/results/repro/reward_conditioned_lm/audio_ssl_conditioning/rob132_hubert_base_transformer384_self_train_fixed_rewards_0_and_1/AudioRewardConditionedMaskLMReward0p0/tedlium_dev_reward0p0_epoch5_lr1e-5.txt` |

CSV artifact:

```text
/exp/exp4/acp21rjf/symphony-workspaces-learning-to-augment/ROB-132/exp/results/repro/reward_conditioned_lm/audio_ssl_conditioning/rob132_hubert_base_transformer384_self_train_fixed_rewards_0_and_1/rob132_audio_ssl_self_train_fixed_rewards.csv
```
