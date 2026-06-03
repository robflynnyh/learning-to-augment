# ROB-177 UFMR Candidate-Repeat Ablation

This directory holds the ROB-177 launcher outputs and aggregate artifacts for
the UFMR candidate-repeat ablation on Earnings22 and TED-LIUM.

The launch wrapper is `scripts/launch_rob177_ufmr_repeat_ablation.sh`. It
generates per-cell configs under `results/configs/`, writes per-cell result
logs under `results/`, and refreshes `ROB-177_OUTCOME.md` plus
`rob177_ufmr_repeat_ablation.csv` in this directory. The generated config YAMLs
are intentionally ignored in Git because the wrapper recreates them from the
single matrix in the script.

Scope:

- Dataset/split: `earnings22` / `test`, `tedlium` / `test`
- Method: `UFMR`
- ASR context: 2048 sequence-length checkpoint
- Adaptation: epoch `1`, LR `1e-5`
- Candidate repeats: `1 2 5 10 15 20 40 100 200 1000`
- Seed trials per candidate-repeat setting: `123456 123457 123458`
- Default references: existing repeat-15 ROB-108 UFMR Earnings22 and TED-LIUM
  epoch-1 LR-1e-5 results when present

`candidate_repeats` means the number of candidate frequency masks UFMR samples
and scores inside `evaluation.augmentation_config.repeats`; it is not the
experiment seed-repeat dimension used by some older sweep tables.

For Earnings22 `candidate_repeats` values completed before the three-trial
follow-up (`2 5 10 20 40 100 200`), the legacy no-seed filename is reused as
the `123456` trial to avoid rerunning an already completed matching cell. New
Earnings22 repeat-`1`/`15` cells and all TED-LIUM cells use seed-tagged
filenames.
