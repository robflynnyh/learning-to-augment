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
