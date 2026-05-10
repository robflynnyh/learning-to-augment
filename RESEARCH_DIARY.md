# Research Diary

Keep concise dated notes for project changes that future agents need in order to
reproduce results, interpret metrics, or avoid known failure modes.

## 2026-05-10

- Added Symphony project wiring for the Learn-to-Augment Linear project,
  including split instruction files, detached-job callbacks, wrapper templates,
  and a blocked follow-up issue helper.
- Added ROB-62 standard 16384 RMM/RFM evaluation scaffolding under
  `exp/results/repro/policy/ROB-62_standard_16384/`. The checked-in
  `RMM_eval/multiepoch` configs were confirmed unsafe because they still use
  `FrequencyMaskingRanker`, `5e-6`, and RFM save paths; use
  `scripts/launch_rob62_standard_eval.sh` for the corrected `9e-5` run.
