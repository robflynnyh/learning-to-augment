# ROB-124 512-Dim Dropout Outcome

Status: follow-up requested after the completed ROB-124 384/dropout handoff;
training completed successfully, and post-training fixed-length reward-control
sanity passed.

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

## Training Result

- Full config:
  `exp/configs/reward_conditioned_lm/no_audio_conditioning/tedlium_per_utterance_512d_dropout0p1_500ep_lr1e3.yaml`
- Smoke config:
  `exp/configs/reward_conditioned_lm/no_audio_conditioning/tedlium_per_utterance_512d_dropout0p1_smoke.yaml`
- Wrapper:
  `scripts/launch_rob124_reward_conditioned_mask_lm_training_512d_dropout0p1.sh`
- Result root:
  `exp/results/repro/reward_conditioned_lm/no_audio_conditioning/old_ablations/rob124_512_dropout/`
- Checkpoint:
  `/store/store5/data/acp21rjf_checkpoints/l2augment/models/reward_conditioned_mask_lm/no_audio_tedlium_per_utterance_512d_dropout0p1_500ep_lr1e3.pt`
- Scratch root:
  `/exp/exp4/acp21rjf/rob124-512-scratch`
- Queued command:
  `screen -L -Logfile /exp/exp4/acp21rjf/symphony-workspaces-learning-to-augment/ROB-124/exp/results/repro/reward_conditioned_lm/no_audio_conditioning/old_ablations/rob124_512_dropout/logs/rob124_no_audio_reward_conditioned_mask_lm_512d_dropout0p1_500ep_lr1e3.screen.log -dmS rob124-reward-conditioned-mask-lm-512d-dropout0p1 bash -lc 'cd /exp/exp4/acp21rjf/symphony-workspaces-learning-to-augment/ROB-124 && /store/store5/software/simple-gpu-schedule/with-gpu 1,2 -- scripts/launch_rob124_reward_conditioned_mask_lm_training_512d_dropout0p1.sh'`
- Screen: `rob124-reward-conditioned-mask-lm-512d-dropout0p1`
- Queue ticket: `32c3350a`
- Pool: `1,2`
- Run commit recorded by wrapper: `1ed8eb26a38d59b1f9a68ce321ca9fd9f451a3d0`
- Queued implementation commit: `c36c89ee6ea5ef5be0433cd8c404026fc3009c0f`
- Callback target state: `Todo`
- Callback returned status `0` on 2026-05-23 and moved ROB-124 back to
  `Todo` for finalization.
- W&B: `eager-terrain-2169` / `bjjnsv4j`
  (<https://wandb.ai/wobrob101/l2augment/runs/bjjnsv4j>)
- Runtime: bashrc Python `/usr/bin/python3.10`, Torch `2.6.0+cu124`
- GPU assigned by `with-gpu 1,2`: `CUDA_VISIBLE_DEVICES=2`, NVIDIA RTX A4500
- Trainable policy parameters: `8,405,505`
- Checkpoint size: `42M` (`43,943,714` bytes)
- Best logged dev loss: `2.6258603876287285`
- Final validation before rollback: `2.627896092154763`
- Early stopping: patience fired and the training loop restored the best
  previous state before writing the final checkpoint.
- Main log:
  `exp/results/repro/reward_conditioned_lm/no_audio_conditioning/old_ablations/rob124_512_dropout/logs/rob124_no_audio_reward_conditioned_mask_lm_512d_dropout0p1_500ep_lr1e3.log`
- Screen log:
  `exp/results/repro/reward_conditioned_lm/no_audio_conditioning/old_ablations/rob124_512_dropout/logs/rob124_no_audio_reward_conditioned_mask_lm_512d_dropout0p1_500ep_lr1e3.screen.log`

## Validation

Passed before queueing on 2026-05-23:

- `bash -n scripts/launch_rob124_reward_conditioned_mask_lm_training_512d_dropout0p1.sh`
- `bash -ic 'python -m py_compile l2augment/modelling/models.py l2augment/utils/helpers.py exp/train_freq_mask.py exp/results/repro/reward_conditioned_lm/no_audio_conditioning/scripts/post_training_sanity_check.py'`
- config parse check confirmed `hidden_dim: 512`, `dropout: 0.1`, and
  rollout root `/store/store4/data/l2augment_rollout_uvqmlm`
- one-file tiny training smoke under bashrc Python 3.10 / Torch 2.6 with CUDA
  hidden because all GPUs were occupied; log:
  `exp/results/repro/reward_conditioned_lm/no_audio_conditioning/old_ablations/rob124_512_dropout/logs/rob124_smoke_512d_dropout0p1_prequeue_cpu_20260523.log`
- actual wrapper `EXIT` callback check with
  `ROB124_512_CALLBACK_ONLY=1 ROB124_512_CALLBACK_CHECK_ONLY=1`; log:
  `exp/results/repro/reward_conditioned_lm/no_audio_conditioning/old_ablations/rob124_512_dropout/logs/rob124_callback_only_check_20260523.log`
- `git diff --check`

The smoke reported `8.41M` trainable policy parameters and finished cleanly.
The callback check confirmed `/usr/bin/python3.10`, Torch `2.6.0+cu124`, and
the same wrapper callback path used by the full run.

Post-training fixed-length reward-control checkpoint-load sanity passed:

```bash
bash -ic 'export TMPDIR=/exp/exp4/acp21rjf/rob124-512-scratch/tmp; mkdir -p "$TMPDIR"; export PYTHONPATH="$PWD:$PWD/exp:/exp/exp4/acp21rjf/long-context-asr:/exp/exp4/acp21rjf/language_modelling${PYTHONPATH:+:$PYTHONPATH}"; python exp/results/repro/reward_conditioned_lm/no_audio_conditioning/scripts/post_training_sanity_check.py --config exp/configs/reward_conditioned_lm/no_audio_conditioning/tedlium_per_utterance_512d_dropout0p1_500ep_lr1e3.yaml --checkpoint /store/store5/data/acp21rjf_checkpoints/l2augment/models/reward_conditioned_mask_lm/no_audio_tedlium_per_utterance_512d_dropout0p1_500ep_lr1e3.pt --output exp/results/repro/reward_conditioned_lm/no_audio_conditioning/old_ablations/rob124_512_dropout/post_training_sanity_check_512d_dropout0p1_500ep_lr1e3.json'
```

Artifact:
`exp/results/repro/reward_conditioned_lm/no_audio_conditioning/old_ablations/rob124_512_dropout/post_training_sanity_check_512d_dropout0p1_500ep_lr1e3.json`

Result on `/store/store4/data/l2augment_rollout_uvqmlm/dev/AlGore_2009_0.pt`:

- Checkpoint loaded on CPU.
- Reward `0.0` and `1.0` both generated exactly `29` VQ tokens.
- Decoded masks and augmented audio matched `[1, 80, 1042]`.
- Reward `0.0` mask active fraction: `0.30000001192092896`.
- Reward `1.0` mask active fraction: `1.0`.
- Reward `0.0` vs `1.0` greedy generation mismatch: `29/29` tokens.

## Comparison

- ROB-117 best available standalone dev loss: `2.653739192269065`.
- ROB-124 384/dropout best logged dev loss: `2.6247272274710913`.
- ROB-124 512/dropout best logged dev loss: `2.6258603876287285`.
- 512/dropout delta vs ROB-117: `-0.027878804640336745`.
- 512/dropout delta vs 384/dropout: `+0.0011331601576371786`.

The 512/dropout checkpoint is loadable and usable for downstream eval/oracle
comparison, but it does not improve the held-out LM loss over the completed
384/dropout checkpoint. The 384/dropout policy remains the better current
capacity point from LM dev loss and the matched ROB-120-style Earnings eval.
