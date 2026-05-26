# ROB-108 Test Policy Evaluations

This directory is the durable result root for ROB-108 test-split policy evals.

The launch wrapper is `scripts/launch_rob108_test_policy_evals.sh`. It generates
configs under per-method `configs/` directories, writes per-cell `.txt` result
files, and refreshes `OUTCOME.md` plus `rob108_test_policy_evals.csv`.

Initial expected cells:

- Datasets: `tedlium`, `earnings22`, `chime6`, `rev16`, `TAL`
- Split: `test` for every dataset
- Policy methods: `RFM`, `RMM`, `UFMR`, `UVQLM`
- Baseline method: `NoAug`
- Repeats: `1`
- One-epoch LRs: `1e-5`, `3e-5`
- Five-epoch LRs: `1e-5`

The CSV/Markdown schema includes `repeat` and `seed` columns so later repeat
fills can append rows without changing the result contract.
