# ROB-124 512-Dim Dropout Outcome

Status: follow-up requested after the completed ROB-124 384/dropout handoff;
prequeue validation passed, and full training is queued on Mimas.

## Scope

The 512-dimensional follow-up keeps the same no-audio reward-conditioned mask
LM family and changes only the GRU hidden size relative to the completed
384/dropout run:

- `hidden_dim: 512`
- `dropout: 0.1`
- same rollout data: `/store/store4/data/l2augment_rollout_uvqmlm/{train,dev}`
- same per-utterance min-max reward normalization
- same W&B/dev-loss early-stopping contract
- same Mimas detached `screen` + `with-gpu 1,2` + callback wrapper discipline

## Comparison Points

- ROB-117 256/no-dropout resumed baseline:
  `/store/store5/data/acp21rjf_checkpoints/l2augment/models/reward_conditioned_mask_lm/no_audio_tedlium_per_utterance_resume100_500ep_lr1e3.pt`
- ROB-117 best available standalone dev loss: `2.653739192269065`
- ROB-124 384/dropout checkpoint:
  `/store/store5/data/acp21rjf_checkpoints/l2augment/models/reward_conditioned_mask_lm/no_audio_tedlium_per_utterance_384d_dropout0p1_500ep_lr1e3.pt`
- ROB-124 384/dropout best logged dev loss: `2.6247272274710913`

## Queued 512-Dim Run

- Full config:
  `exp/configs/reward_conditioned_lm/no_audio_conditioning/tedlium_per_utterance_512d_dropout0p1_500ep_lr1e3.yaml`
- Smoke config:
  `exp/configs/reward_conditioned_lm/no_audio_conditioning/tedlium_per_utterance_512d_dropout0p1_smoke.yaml`
- Wrapper:
  `scripts/launch_rob124_reward_conditioned_mask_lm_training_512d_dropout0p1.sh`
- Result root:
  `exp/results/repro/reward_conditioned_lm/no_audio_conditioning/rob124_512_dropout/`
- Checkpoint:
  `/store/store5/data/acp21rjf_checkpoints/l2augment/models/reward_conditioned_mask_lm/no_audio_tedlium_per_utterance_512d_dropout0p1_500ep_lr1e3.pt`
- Scratch root:
  `/exp/exp4/acp21rjf/rob124-512-scratch`
- Queued command:
  `screen -L -Logfile /exp/exp4/acp21rjf/symphony-workspaces-learning-to-augment/ROB-124/exp/results/repro/reward_conditioned_lm/no_audio_conditioning/rob124_512_dropout/logs/rob124_no_audio_reward_conditioned_mask_lm_512d_dropout0p1_500ep_lr1e3.screen.log -dmS rob124-reward-conditioned-mask-lm-512d-dropout0p1 bash -lc 'cd /exp/exp4/acp21rjf/symphony-workspaces-learning-to-augment/ROB-124 && /store/store5/software/simple-gpu-schedule/with-gpu 1,2 -- scripts/launch_rob124_reward_conditioned_mask_lm_training_512d_dropout0p1.sh'`
- Screen: `rob124-reward-conditioned-mask-lm-512d-dropout0p1`
- Queue ticket: `32c3350a`
- Pool: `1,2`
- Queued commit: `c36c89ee6ea5ef5be0433cd8c404026fc3009c0f`
- Callback target state: `Todo`
- Queue status at handoff: waiting behind one pool `1,2` ticket
- Main log:
  `exp/results/repro/reward_conditioned_lm/no_audio_conditioning/rob124_512_dropout/logs/rob124_no_audio_reward_conditioned_mask_lm_512d_dropout0p1_500ep_lr1e3.log`
- Screen log:
  `exp/results/repro/reward_conditioned_lm/no_audio_conditioning/rob124_512_dropout/logs/rob124_no_audio_reward_conditioned_mask_lm_512d_dropout0p1_500ep_lr1e3.screen.log`

## Validation

Passed before queueing on 2026-05-23:

- `bash -n scripts/launch_rob124_reward_conditioned_mask_lm_training_512d_dropout0p1.sh`
- `bash -ic 'python -m py_compile l2augment/modelling/models.py l2augment/utils/helpers.py exp/train_freq_mask.py exp/results/repro/reward_conditioned_lm/no_audio_conditioning/scripts/post_training_sanity_check.py'`
- config parse check confirmed `hidden_dim: 512`, `dropout: 0.1`, and
  rollout root `/store/store4/data/l2augment_rollout_uvqmlm`
- one-file tiny training smoke under bashrc Python 3.10 / Torch 2.6 with CUDA
  hidden because all GPUs were occupied; log:
  `exp/results/repro/reward_conditioned_lm/no_audio_conditioning/rob124_512_dropout/logs/rob124_smoke_512d_dropout0p1_prequeue_cpu_20260523.log`
- actual wrapper `EXIT` callback check with
  `ROB124_512_CALLBACK_ONLY=1 ROB124_512_CALLBACK_CHECK_ONLY=1`; log:
  `exp/results/repro/reward_conditioned_lm/no_audio_conditioning/rob124_512_dropout/logs/rob124_callback_only_check_20260523.log`
- `git diff --check`

The smoke reported `8.41M` trainable policy parameters and finished cleanly.
The callback check confirmed `/usr/bin/python3.10`, Torch `2.6.0+cu124`, and
the same wrapper callback path that will be used for the queued full run.

After successful training, run the fixed-length reward-control checkpoint-load
sanity check at reward controls `0.0` and `1.0`, then compare dev loss against
both the ROB-117 baseline and completed 384/dropout checkpoint.
