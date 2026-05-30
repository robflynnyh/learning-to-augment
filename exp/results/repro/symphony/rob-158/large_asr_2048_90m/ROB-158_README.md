# ROB-158 UFMR Large-ASR Evaluation

This directory holds the ROB-158-specific aggregate artifacts for testing
whether the learned UFMR augmentation pattern transfers to a larger ASR model,
plus the follow-up RFM comparison on that same larger ASR model.

The launch wrapper is `scripts/launch_rob158_ufmr_large_asr_eval.sh`. It
generates per-cell configs under `results/UFMR/configs/`, writes per-cell
`.txt` result files under `results/UFMR/`, and refreshes
`ROB-158_OUTCOME.md` plus `rob158_ufmr_large_asr_eval.csv`.

Experiment contract:

- ASR checkpoint:
  `/store/store5/data/acp21rjf_checkpoints/SAP_LCASR/n_seq_sched_2048_rp_1/step_105360.pt`
- ASR checkpoint inventory: `n_seq_sched_2048_rp_1`, repeat `1`, 2048
  centiseconds, approximately 90M parameters.
- Policy: `UnconditionalFrequencyMaskingRanker`
- UFMR checkpoint:
  `/store/store5/data/acp21rjf_checkpoints/l2augment/ufmr/test_wer/model.pt`
- Datasets: `tedlium`, `earnings22`, `chime6`, `rev16`, `TAL`
- Split: `test` for every dataset
- Repeats: `1`
- One-epoch LRs: `1e-5`, `3e-5`
- Five-epoch LRs: `1e-5`
- Candidate masks per step: `15`

The comparison target is the existing ROB-108 UFMR repro output under
`exp/results/repro/symphony/rob-108/`. The intended variable is the ASR model
size; policy, datasets, repeat, candidate count, and LR/epoch cells are kept the
same.

## RFM Follow-Up

The follow-up launcher is `scripts/launch_rob158_rfm_large_asr_eval.sh`. It
uses the same ASR checkpoint, datasets, split, repeat, and result root, but
generates configs under `results/RFM/configs/` and writes per-cell files under
`results/RFM/`.

RFM follows the latest Linear clarification:

- Policy: `FrequencyMaskingRanker`
- One-epoch LRs: `1e-5`
- Five-epoch LRs: `1e-5`
- Dropped cells: `3e-5` RFM trials

The comparison helper is `scripts/compare_rob158_large_asr_policy_evals.py`.
After RFM results are available, it writes:

- `rob158_vs_rob108_rfm_comparison.csv`
- `rob158_large_asr_ufmr_vs_rfm_comparison.csv`

Final interpretation lives in `ROB-158_OUTCOME.md` for UFMR and
`ROB-158_RFM_OUTCOME.md` for the RFM follow-up plus UFMR-vs-RFM comparison.
