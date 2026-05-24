# ROB-120 Earnings Reward Controls

Evaluates the ROB-117 no-audio reward-conditioned mask LM checkpoint on
Earnings-22 test adaptation with `lr=1e-5`.

Checkpoint:

```text
/store/store5/data/acp21rjf_checkpoints/l2augment/models/reward_conditioned_mask_lm/no_audio_tedlium_per_utterance_resume100_500ep_lr1e3.pt
```

Required conditions:

- fixed reward `0.0`
- fixed reward `1.0`
- uniform sampled reward `[0.0, 1.0]`
- uniform sampled reward `[0.5, 1.0]`

Launch wrapper:

```bash
screen -L -Logfile exp/results/repro/reward_conditioned_lm/no_audio_conditioning/rob120_earnings_reward_controls/logs/rob120_earnings_reward_controls.screen.log -dmS rob120-earnings-reward-controls bash -lc 'cd /exp/exp4/acp21rjf/symphony-workspaces-learning-to-augment/ROB-120 && /store/store5/software/simple-gpu-schedule/with-gpu 1,2 -- scripts/launch_rob120_earnings_reward_controls.sh'
```

The wrapper generates condition configs under this directory, runs `exp/eval.py`
for each condition, and writes `rob120_earnings_reward_controls.csv` plus
`OUTCOME.md`.
