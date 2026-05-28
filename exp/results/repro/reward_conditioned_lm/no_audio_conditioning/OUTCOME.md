# No-Audio Reward-Conditioned Mask LM Outcome

This is the consolidated result summary for the no-audio reward-conditioned mask
LM family through ROB-124.

## Current Conclusion

The ROB-124 384-hidden-dim, dropout-0.1 GRU checkpoint is the usable and
preferred model from this ablation set. It improves the ROB-117 256-dim
baseline family on held-out LM loss and gives useful downstream reward-control
signals. The 512/dropout follow-up did not improve held-out LM loss, so larger
GRU capacity is not justified by this result alone.

For downstream eval/oracle comparisons, use the 384/dropout checkpoint with
1-epoch adaptation or dataset-specific reward/epoch choices. Five-epoch
adaptation is not safe as a blanket default because CHiME-6 collapses under
some high-reward settings.

## Key Checkpoints

| Model | Status | Checkpoint |
| --- | --- | --- |
| ROB-117 256d no-dropout baseline | merged baseline | `/store/store5/data/acp21rjf_checkpoints/l2augment/models/reward_conditioned_mask_lm/no_audio_tedlium_per_utterance_resume100_500ep_lr1e3.pt` |
| ROB-124 384d dropout 0.1 | preferred | `/store/store5/data/acp21rjf_checkpoints/l2augment/models/reward_conditioned_mask_lm/no_audio_tedlium_per_utterance_384d_dropout0p1_500ep_lr1e3.pt` |
| ROB-124 512d dropout 0.1 | old ablation | `/store/store5/data/acp21rjf_checkpoints/l2augment/models/reward_conditioned_mask_lm/no_audio_tedlium_per_utterance_512d_dropout0p1_500ep_lr1e3.pt` |

## Folder Map

| Path | Purpose |
| --- | --- |
| `rob124_384_dropout/` | Main 384/dropout training and sanity artifacts. |
| `rob124_384_dropout_reward_conditioning/earnings_reward_controls/` | Corrected matched Earnings-22 reward-control eval. |
| `rob124_384_dropout_reward_conditioning/all_dataset_sampled_reward_0p5_to_1p0/` | Completed all-dataset sampled reward `[0.5, 1.0]` eval. |
| `rob124_384_dropout_reward_conditioning/all_dataset_fixed_rewards_0_and_1/` | Completed all-dataset fixed reward `1.0` and `0.0` eval. |
| `rob124_384_dropout_reward_conditioning/earnings_rmm_lm_rerank/` | Earnings-22 RMM proposal plus reward-1 LM CE reranker eval. |
| `old_ablations/rob120_earnings_reward_controls/` | ROB-120 baseline Earnings reward-control comparison. |
| `old_ablations/rob124_512_dropout/` | ROB-124 512/dropout capacity follow-up. |

Removed roots:

- `rob124_384_dropout_earnings_reward_controls/`: stale initial Earnings run,
  superseded by the corrected sampling rerun.
- `rob124_384_dropout_all_dataset_reward_sampling_0to1/`: cancelled scaffold;
  no GPU eval cells ran, and the corrected request was the fixed-reward sweep.

## Main Results

| Result | Summary | Detailed artifact |
| --- | --- | --- |
| 384/dropout training | Best logged dev loss `2.624727`; post-training reward `0.0`/`1.0` generation sanity passed; `5,124,993` trainable parameters. | `rob124_384_dropout/OUTCOME.md` |
| 512/dropout training | Best logged dev loss `2.625860`, slightly worse than 384/dropout; moved to old ablations. | `old_ablations/rob124_512_dropout/OUTCOME.md` |
| Corrected Earnings reward controls | Best row is true uniform reward `[0.5, 1.0]`, updated WER `0.194433`; this beats the matched ROB-120 row by `0.002001` absolute WER. | `rob124_384_dropout_reward_conditioning/earnings_reward_controls/OUTCOME.md` |
| All-dataset sampled `[0.5, 1.0]` | Completed `10/10` cells; `9/10` improved; the only regression is CHiME-6 at 5 epochs, `0.843620 -> 1.000000` WER. | `rob124_384_dropout_reward_conditioning/all_dataset_sampled_reward_0p5_to_1p0/OUTCOME.md` |
| All-dataset fixed rewards `0.0` and `1.0` | Completed `20/20` cells; fixed reward `0.0` improved `10/10`, fixed reward `1.0` improved `9/10`; the only regression is again CHiME-6 at reward `1.0`, 5 epochs, `0.843620 -> 1.000000` WER. | `rob124_384_dropout_reward_conditioning/all_dataset_fixed_rewards_0_and_1/OUTCOME.md` |
| Earnings RMM reward-1 LM rerank | Updated WER `0.202377`, better than unadapted Earnings-22 but worse than direct reward-conditioned sampling. | `rob124_384_dropout_reward_conditioning/earnings_rmm_lm_rerank/OUTCOME.md` |
| Reward-control average masks | Averaged `10,000` UC-MLM masks plus `10,000` sampled RC-MLM masks at reward `0.0` and `10,000` at reward `1.0` from the 384/dropout checkpoint without storing all masks. The figures plot masked percentage: `0%` means fully retained/unmasked and `100%` means fully suppressed/masked out. Average masked percentage is `49.13%` for UC-MLM, `70.18%` for `RC-MLM (reward=0.0)`, and `33.19%` for `RC-MLM (reward=1.0)`. | `visualizations/reward_conditioned_average_masks_10k/metadata.json` |

## Interpretation

The useful signal is not simply larger model capacity. The 384/dropout model is
the current best capacity/dropout point, while 512/dropout does not improve the
held-out LM objective. Direct reward-conditioned sampling is stronger than the
RMM reranker on the matched Earnings eval.

Across the all-dataset sweeps, 1-epoch adaptation is the robust default. Longer
5-epoch adaptation can help TED-LIUM, Earnings22, Rev16, and TAL, but it is
dataset-sensitive and can fail badly on CHiME-6. Treat reward and epoch
selection as downstream tuning variables rather than a universal setting.

The 10k average-mask visualization gives the most direct qualitative view of
RC-MLM reward conditioning against the UC-MLM baseline: UC-MLM sits between
`RC-MLM (reward=0.0)` and `RC-MLM (reward=1.0)`, while higher reward shifts the
sampled multiplicative masks toward masking less of the spectrogram on the
ROB-124 probe utterance. The figures show masked percentage; the inverse
percentage is the amount retained.
