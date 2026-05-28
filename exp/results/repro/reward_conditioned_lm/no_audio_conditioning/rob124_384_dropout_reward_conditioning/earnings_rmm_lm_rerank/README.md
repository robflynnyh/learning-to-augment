# ROB-124 384-Dropout RMM-LM Rerank

This result root tracks the follow-up requested on 2026-05-23 after the
384/dropout checkpoint was confirmed as the preferred capacity point.

The eval uses the trained 384-dim/dropout no-audio reward-conditioned mask LM:

```text
/store/store5/data/acp21rjf_checkpoints/l2augment/models/reward_conditioned_mask_lm/no_audio_tedlium_per_utterance_384d_dropout0p1_500ep_lr1e3.pt
```

At each Earnings-22 adaptation step the policy generates 15 candidate masks
with the existing RMM proposal distribution, encodes each candidate mask with
the mask BVAE, scores the candidate VQ token sequence with the 384/dropout
reward-conditioned mask LM at fixed reward `1.0`, and adapts with the mask that
has the lowest per-candidate CE loss.

The callback-backed launcher is:

```text
scripts/launch_rob124_384_dropout_rmm_lm_rerank.sh
```

Status: completed successfully on 2026-05-23 via
`screen:rob124-384-dropout-rmm-lm-rerank`, with callback exit status `0`.

Queued full-run command:

```text
screen -L -Logfile /exp/exp4/acp21rjf/symphony-workspaces-learning-to-augment/ROB-124/exp/results/repro/reward_conditioned_lm/no_audio_conditioning/rob124_384_dropout_reward_conditioning/earnings_rmm_lm_rerank/logs/rob124_384_dropout_rmm_lm_rerank.screen.log -dmS rob124-384-dropout-rmm-lm-rerank bash -lc 'cd /exp/exp4/acp21rjf/symphony-workspaces-learning-to-augment/ROB-124 && /store/store5/software/simple-gpu-schedule/with-gpu 1,2 -- scripts/launch_rob124_384_dropout_rmm_lm_rerank.sh'
```

The wrapper writes generated configs, the raw eval text result, a CSV summary,
and `OUTCOME.md` under this directory.

Final result:

```text
Original WER: 0.235239
Updated WER:  0.202377
WER delta:   -0.032862
```

The rerank policy improves substantially over unadapted Earnings-22, but it is
worse than the prior ROB-124 reward-control evals: `+0.006923` absolute WER
versus the ROB-124 fixed reward `1.0` condition, `+0.007434` versus the best
ROB-124 prior condition, and `+0.004758` versus the ROB-120 fixed reward `1.0`
baseline. The 384/dropout checkpoint remains the preferred model, but this
specific RMM/LM CE-rerank proposal selector is not better than direct
reward-conditioned sampling for this matched Earnings comparison.

Prequeue validation:

```text
bash -n scripts/launch_rob124_384_dropout_rmm_lm_rerank.sh
bash -ic 'python -m py_compile l2augment/modelling/models.py scripts/summarize_rob124_384_dropout_rmm_lm_rerank.py exp/eval.py'
ROB124_RERANK_DISABLE_CALLBACK=1 ROB124_RERANK_CONFIG_ONLY=1 scripts/launch_rob124_384_dropout_rmm_lm_rerank.sh
ROB124_RERANK_CALLBACK_ONLY=1 ROB124_RERANK_CALLBACK_CHECK_ONLY=1 scripts/launch_rob124_384_dropout_rmm_lm_rerank.sh
python3 -m json.tool exp/results/repro/reward_conditioned_lm/no_audio_conditioning/rob124_384_dropout_reward_conditioning/earnings_rmm_lm_rerank/preflight_cropped_earnings_rollout.json >/dev/null
git diff --check
```

The cropped CPU preflight used bashrc Python 3.10 / Torch 2.6 with CUDA hidden,
loaded the real ASR checkpoint and 384/dropout LM checkpoint, ran the multistep
rollout path on a 512-frame Earnings crop, and exercised one 15-candidate
RMM/LM-rerank adaptation step. Artifact:

```text
exp/results/repro/reward_conditioned_lm/no_audio_conditioning/rob124_384_dropout_reward_conditioning/earnings_rmm_lm_rerank/preflight_cropped_earnings_rollout.json
```
