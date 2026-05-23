# ROB-124 512-Dim Dropout Reward-Conditioned Mask LM

This is the 512-dimensional capacity follow-up requested after the completed
ROB-124 384-dimensional dropout run.

## Context

- ROB-117 baseline checkpoint:
  `/store/store5/data/acp21rjf_checkpoints/l2augment/models/reward_conditioned_mask_lm/no_audio_tedlium_per_utterance_resume100_500ep_lr1e3.pt`
- ROB-117 baseline PR: <https://github.com/robflynnyh/learning-to-augment/pull/18>
- ROB-117 merged head: `deddca9d3d78acd3b9734b88b858f9698b0f81a8`
- ROB-117 merge commit: `ddd349c30acb056cc791788e85e46153fa344c1b`
- ROB-124 384/dropout checkpoint:
  `/store/store5/data/acp21rjf_checkpoints/l2augment/models/reward_conditioned_mask_lm/no_audio_tedlium_per_utterance_384d_dropout0p1_500ep_lr1e3.pt`

The active checkout was verified to contain the ROB-117 merge commit before the
384/dropout implementation. The 512 follow-up is on the same PR branch:
`symphony/ROB-124-384-dropout-mask-lm`.

## Config

- Full training config:
  `exp/configs/reward_conditioned_lm/no_audio_conditioning/tedlium_per_utterance_512d_dropout0p1_500ep_lr1e3.yaml`
- Smoke config:
  `exp/configs/reward_conditioned_lm/no_audio_conditioning/tedlium_per_utterance_512d_dropout0p1_smoke.yaml`
- Policy class: `RewardConditionedMaskLM`
- Policy change from the completed 384/dropout run:
  - `hidden_dim: 512`
  - `dropout: 0.1`
- Reward/data contract is unchanged:
  - rollout root: `/store/store4/data/l2augment_rollout_uvqmlm/{train,dev}`
  - reward metric: WER
  - normalization: per-utterance min-max
  - degenerate reward groups: `0.5`
  - no-audio training on saved VQ `generation` sequences

## Queued Run

- Result root:
  `exp/results/repro/reward_conditioned_lm/no_audio_conditioning/rob124_512_dropout/`
- Wrapper:
  `scripts/launch_rob124_reward_conditioned_mask_lm_training_512d_dropout0p1.sh`
- Scratch root:
  `/exp/exp4/acp21rjf/rob124-512-scratch`
- Checkpoint:
  `/store/store5/data/acp21rjf_checkpoints/l2augment/models/reward_conditioned_mask_lm/no_audio_tedlium_per_utterance_512d_dropout0p1_500ep_lr1e3.pt`
- Temporary checkpoint:
  `/store/store5/data/acp21rjf_checkpoints/l2augment/models/reward_conditioned_mask_lm/no_audio_tedlium_per_utterance_512d_dropout0p1_500ep_lr1e3_tmp.pt`

Queue command:

```bash
screen -L -Logfile /exp/exp4/acp21rjf/symphony-workspaces-learning-to-augment/ROB-124/exp/results/repro/reward_conditioned_lm/no_audio_conditioning/rob124_512_dropout/logs/rob124_no_audio_reward_conditioned_mask_lm_512d_dropout0p1_500ep_lr1e3.screen.log -dmS rob124-reward-conditioned-mask-lm-512d-dropout0p1 bash -lc 'cd /exp/exp4/acp21rjf/symphony-workspaces-learning-to-augment/ROB-124 && /store/store5/software/simple-gpu-schedule/with-gpu 1,2 -- scripts/launch_rob124_reward_conditioned_mask_lm_training_512d_dropout0p1.sh'
```

Queued on Mimas at 2026-05-23 17:42 UTC.

- Screen: `rob124-reward-conditioned-mask-lm-512d-dropout0p1`
- Queue ticket: `32c3350a`
- Pool: `1,2`
- Queued commit: `c36c89ee6ea5ef5be0433cd8c404026fc3009c0f`
- Completion callback target state: `Todo`
- Queue status at handoff: waiting behind one pool `1,2` ticket

Expected full-run logs:

- Main log:
  `exp/results/repro/reward_conditioned_lm/no_audio_conditioning/rob124_512_dropout/logs/rob124_no_audio_reward_conditioned_mask_lm_512d_dropout0p1_500ep_lr1e3.log`
- Screen log:
  `exp/results/repro/reward_conditioned_lm/no_audio_conditioning/rob124_512_dropout/logs/rob124_no_audio_reward_conditioned_mask_lm_512d_dropout0p1_500ep_lr1e3.screen.log`

Completion check:

```bash
cd /exp/exp4/acp21rjf/symphony-workspaces-learning-to-augment/ROB-124
/store/store5/software/simple-gpu-schedule/with-gpu --status
screen -ls | rg 'rob124-reward-conditioned-mask-lm-512d-dropout0p1'
tail -80 exp/results/repro/reward_conditioned_lm/no_audio_conditioning/rob124_512_dropout/logs/rob124_no_audio_reward_conditioned_mask_lm_512d_dropout0p1_500ep_lr1e3.screen.log
tail -80 exp/results/repro/reward_conditioned_lm/no_audio_conditioning/rob124_512_dropout/logs/rob124_no_audio_reward_conditioned_mask_lm_512d_dropout0p1_500ep_lr1e3.log
ls -lh /store/store5/data/acp21rjf_checkpoints/l2augment/models/reward_conditioned_mask_lm/no_audio_tedlium_per_utterance_512d_dropout0p1_500ep_lr1e3.pt
```

## Prequeue Validation

Passed before queueing on 2026-05-23:

```bash
bash -n scripts/launch_rob124_reward_conditioned_mask_lm_training_512d_dropout0p1.sh
bash -ic 'python -m py_compile l2augment/modelling/models.py l2augment/utils/helpers.py exp/train_freq_mask.py exp/results/repro/reward_conditioned_lm/no_audio_conditioning/scripts/post_training_sanity_check.py'
CUDA_VISIBLE_DEVICES="" bash -ic 'export TMPDIR=/exp/exp4/acp21rjf/rob124-512-scratch/tmp; mkdir -p "$TMPDIR"; export PYTHONPATH="$PWD:$PWD/exp:/exp/exp4/acp21rjf/long-context-asr:/exp/exp4/acp21rjf/language_modelling${PYTHONPATH:+:$PYTHONPATH}"; python exp/train_freq_mask.py --config exp/configs/reward_conditioned_lm/no_audio_conditioning/tedlium_per_utterance_512d_dropout0p1_smoke.yaml'
ROB124_512_CALLBACK_ONLY=1 ROB124_512_CALLBACK_CHECK_ONLY=1 CALLBACK_TARGET_STATE=Todo scripts/launch_rob124_reward_conditioned_mask_lm_training_512d_dropout0p1.sh
git diff --check
```

The tiny smoke was run with CUDA hidden because all Mimas GPUs were occupied
and pool `1,2` already had a waiting ticket; the full training launch remains
queued through `/store/store5/software/simple-gpu-schedule/with-gpu 1,2`.

Validation artifacts:

- CPU smoke log:
  `exp/results/repro/reward_conditioned_lm/no_audio_conditioning/rob124_512_dropout/logs/rob124_smoke_512d_dropout0p1_prequeue_cpu_20260523.log`
- Callback check log:
  `exp/results/repro/reward_conditioned_lm/no_audio_conditioning/rob124_512_dropout/logs/rob124_callback_only_check_20260523.log`

The smoke reported `8.41M` trainable policy parameters and finished cleanly.
The callback check confirmed the wrapper resolves to `/usr/bin/python3.10` with
Torch `2.6.0+cu124` and exits through the checked callback path.

## Post-Training Validation

If the full run exits successfully, run:

```bash
bash -ic 'export TMPDIR=/exp/exp4/acp21rjf/rob124-512-scratch/tmp; mkdir -p "$TMPDIR"; export PYTHONPATH="$PWD:$PWD/exp:/exp/exp4/acp21rjf/long-context-asr:/exp/exp4/acp21rjf/language_modelling${PYTHONPATH:+:$PYTHONPATH}"; python exp/results/repro/reward_conditioned_lm/no_audio_conditioning/scripts/post_training_sanity_check.py --config exp/configs/reward_conditioned_lm/no_audio_conditioning/tedlium_per_utterance_512d_dropout0p1_500ep_lr1e3.yaml --checkpoint /store/store5/data/acp21rjf_checkpoints/l2augment/models/reward_conditioned_mask_lm/no_audio_tedlium_per_utterance_512d_dropout0p1_500ep_lr1e3.pt --output exp/results/repro/reward_conditioned_lm/no_audio_conditioning/rob124_512_dropout/post_training_sanity_check_512d_dropout0p1_500ep_lr1e3.json'
```

Compare the final dev loss and checkpoint sanity result against:

- ROB-117 resumed baseline best available dev loss: `2.653739192269065`
- ROB-124 384/dropout best logged dev loss: `2.6247272274710913`
