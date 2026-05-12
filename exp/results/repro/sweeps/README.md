# Reproduction Sweeps

This directory is for issue-specific reproduction sweeps that are separate from
the historical paper-result tree.

## ROB-80 TED-LIUM Dev Policy LR Sweep

`scripts/launch_rob80_tedlium_policy_sweep.sh` runs the TED-LIUM dev sweep for
`RFM`, `RMM`, and `UFMR` under:

- `exp/results/repro/sweeps/RFM/`
- `exp/results/repro/sweeps/RMM/`
- `exp/results/repro/sweeps/UFMR/`

The sweep evaluates learning rates `5e-6`, `1e-5`, and `2e-5` for both `1` and
`5` adaptation epochs. The launched wrapper writes generated YAML configs under
each method's `configs/` folder and writes the final result table to
`exp/results/repro/sweeps/ROB-80_OUTCOME.md`.
