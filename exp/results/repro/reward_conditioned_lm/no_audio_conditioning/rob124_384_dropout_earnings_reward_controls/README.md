# ROB-124 384-Dropout Earnings Reward Controls

Evaluates the ROB-124 384-dimensional dropout no-audio reward-conditioned mask
LM checkpoint on Earnings-22 test adaptation with `lr=1e-5`, matching the
ROB-120 reward-control comparison.

Checkpoint:

```text
/store/store5/data/acp21rjf_checkpoints/l2augment/models/reward_conditioned_mask_lm/no_audio_tedlium_per_utterance_384d_dropout0p1_500ep_lr1e3.pt
```

Required conditions:

- fixed reward `0.0`
- fixed reward `1.0`
- uniform sampled reward `[0.0, 1.0]`
- uniform sampled reward `[0.5, 1.0]`

Launch wrapper:

```bash
screen -L -Logfile exp/results/repro/reward_conditioned_lm/no_audio_conditioning/rob124_384_dropout_earnings_reward_controls/logs/rob124_384_dropout_earnings_reward_controls.screen.log -dmS rob124-384-dropout-earnings-reward-controls bash -lc 'cd /exp/exp4/acp21rjf/symphony-workspaces-learning-to-augment/ROB-124 && /store/store5/software/simple-gpu-schedule/with-gpu 1,2 -- scripts/launch_rob124_384_dropout_earnings_reward_controls.sh'
```

The wrapper generates condition configs under this directory, runs `exp/eval.py`
for each condition, and writes
`rob124_384_dropout_earnings_reward_controls.csv` plus `OUTCOME.md`.

## Queue Handoff

- Queued: 2026-05-22 19:50 UTC
- Screen: `rob124-384-dropout-earnings-reward-controls`
- Queue ticket: `a85da454`
- Pool: `1,2`
- Queued commit: `0f24bb5526c9b97a2c75d9595f8cd6a73c967c27`
- Main log:
  `exp/results/repro/reward_conditioned_lm/no_audio_conditioning/rob124_384_dropout_earnings_reward_controls/logs/rob124_384_dropout_earnings_reward_controls.log`
- Screen log:
  `exp/results/repro/reward_conditioned_lm/no_audio_conditioning/rob124_384_dropout_earnings_reward_controls/logs/rob124_384_dropout_earnings_reward_controls.screen.log`
- Callback target state: `Todo`

## Completion

- Completed: 2026-05-22
- Exit status: `0`
- Completed conditions: `4/4`
- Summary CSV:
  `exp/results/repro/reward_conditioned_lm/no_audio_conditioning/rob124_384_dropout_earnings_reward_controls/rob124_384_dropout_earnings_reward_controls.csv`
- Outcome summary:
  `exp/results/repro/reward_conditioned_lm/no_audio_conditioning/rob124_384_dropout_earnings_reward_controls/OUTCOME.md`

Best ROB-124 condition on this Earnings-22 test pass was uniform sampled reward
`[0.5, 1.0]`, with updated WER `0.194943` versus the matched ROB-120 updated
WER `0.196434`. The fixed reward `0.0` condition was slightly worse than
ROB-120, while fixed reward `1.0` and both uniform reward controls were
slightly better.
