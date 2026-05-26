# ROB-108 Test Policy Evaluations

This directory holds ROB-108-specific README and aggregate artifacts. Raw
per-cell outputs remain in the shared `exp/results/repro/` tree under method
directories such as `RFM/`, `RMM/`, `UFMR/`, and `UVQLM/`.

The launch wrapper is `scripts/launch_rob108_test_policy_evals.sh`. It generates
configs under top-level per-method `configs/` directories, writes per-cell
`.txt` result files, and refreshes the aggregate `ROB-108_OUTCOME.md` plus
`rob108_test_policy_evals.csv` in this issue-specific directory.

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
