# ROB-132 Remaining Test Datasets

Follow-up fixed-reward test-set eval for the ROB-132 HuBERT audio+reward
conditioned transformer mask LM.

This root is reserved for the remaining ROB-124-style test datasets not covered
by the first ROB-132 TED-LIUM/Earnings22 handoff:

- Rev16 test
- This American Life test, tagged as `TAL`
- CHiME-6 test

Each dataset is evaluated at fixed conditioning rewards `1.0` and `0.0`, with
test-time self-training epochs `1` and `5`, using `lr=1e-5`. Stanage launches
should use separate GPU jobs per cell plus a finalizer callback.

The June 2026 handoff completed 8 of the 12 planned cells. The four missing
Rev16/TAL 5-epoch cells were intentionally cancelled after runtime estimates
showed they were likely to hit the 4-day Stanage limit. Do not treat those
missing rows as accidental failed cells or rerun them without an explicit new
instruction.
