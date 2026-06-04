# ROB-132 HuBERT Audio+Reward Mask LM

This root groups the ROB-132 audio+reward-conditioned mask LM artifacts for
the HuBERT-base, transformer-384 policy.

## Contents

- `train/`: model design notes, training launch metadata, mapping checks, smoke
  checks, and training logs for the ROB-132 policy checkpoint.
- `eval/dev_tedlium_fixed_rewards_0_and_1/`: TED-LIUM dev fixed-reward
  self-training eval for rewards `1.0` and `0.0`.
- `eval/test_fixed_rewards_0_and_1/`: consolidated test-set fixed-reward eval
  across TED-LIUM, Earnings22, Rev16, TAL, and CHiME-6.

The test eval currently has `16/20` planned cells complete. The missing four
cells are Rev16/TAL epoch-5 runs at rewards `1.0` and `0.0`; they were
intentionally deferred after runtime estimates showed the Stanage jobs were
likely to hit the 4-day limit.
