# ROB-132 Audio SSL Fixed-Reward Self-Training Eval

This result root evaluates the trained ROB-132 audio+reward-conditioned mask LM
checkpoint under fixed reward controls during multistep self-training.

## Matrix

- Dataset: `tedlium`, split `dev`
- Rewards: fixed `conditioning_reward: 1.0` and fixed `conditioning_reward: 0.0`
- Adaptation epochs: `1` and `5`
- Optimizer LR: `1e-5`
- Total cells: `4`

Each generated config sets both
`evaluation.augmentation_config.conditioning_reward` and
`policy.config.default_conditioning_reward` to the same scalar, with no
`conditioning_reward_range`.

## Checkpoints

- Policy:
  `/store/store5/data/acp21rjf_checkpoints/l2augment/models/reward_conditioned_mask_lm/audio_ssl_hubert_base_tedlium_per_utterance_transformer384_dropout0p1_500ep_lr1e3.pt`
- ASR:
  `/store/store5/data/acp21rjf_checkpoints/spotify/rotary_pos_6l_256d_seq_sched/n_seq_sched_2048_rp_1/step_105360.pt`
- Mask VAE:
  `/store/store5/data/acp21rjf_checkpoints/l2augment/models/bvae/bvae_USINGTHISFORNOW_2048gpu.pt`

## Runtime

- Wrapper: `scripts/launch_rob132_audio_ssl_self_train_fixed_rewards.sh`
- Summarizer: `scripts/summarize_rob132_audio_ssl_self_train_fixed_rewards.py`
- Scratch/cache root: `/exp/exp4/acp21rjf/rob132-audio-ssl-scratch`
- Logs: `logs/`

The eval path computes frozen HuBERT features from the corresponding raw
recording segment for each self-training spectrogram chunk, then passes those
features into `AudioRewardConditionedMaskLM.augment()`.

## Pre-Launch Validation

- Config-only wrapper generation produced exactly four configs for the fixed
  reward/epoch matrix.
- The actual wrapper `EXIT` callback path passed with
  `ROB132_SELFTRAIN_CALLBACK_ONLY=1 ROB132_SELFTRAIN_CALLBACK_CHECK_ONLY=1`.
- A GPU smoke through `with-gpu 1,2` completed one TED-LIUM dev recording at
  reward `1.0`, epoch `1`, with callbacks disabled and result saving disabled.
  Smoke log:
  `logs/rob132_audio_ssl_self_train_smoke.log`.

## Completion Checks

```bash
cd /exp/exp4/acp21rjf/symphony-workspaces-learning-to-augment/ROB-132
/store/store5/software/simple-gpu-schedule/with-gpu 1,2 --status
screen -ls | rg 'rob132-audio-ssl-selftrain-fixed-rewards'
tail -100 exp/results/repro/reward_conditioned_lm/audio_ssl_conditioning/rob132_hubert_base_transformer384_self_train_fixed_rewards_0_and_1/logs/rob132_audio_ssl_self_train_fixed_rewards_0_and_1.log
cat exp/results/repro/reward_conditioned_lm/audio_ssl_conditioning/rob132_hubert_base_transformer384_self_train_fixed_rewards_0_and_1/OUTCOME.md
```
