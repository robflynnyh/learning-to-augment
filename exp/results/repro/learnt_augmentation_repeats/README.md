# Learnt Augmentation Repeat Evaluations

This directory holds follow-up repeats for learnt augmentation WER table cells
whose original ROB-108 artifacts only had one completed run.

ROB-337 targets:

- Methods: `UFMR`, `RFM`, `RMM`, and `UVQLM` (the UC-MLM table row).
- Datasets: TED-LIUM, Earnings-22, Rev16, TAL, and CHiME-6 test splits.
- Adaptation settings: `epochs=1` and `epochs=5`, `lr=1e-5`, 6L/2048 ASR.
- Repeats: existing ROB-108 repeat 1 plus new ROB-337 repeats 2 and 3.

Expected generated artifacts live under `rob337/`:

- `rob337/<method>/configs/*_repeat{2,3}.yaml`
- `rob337/<method>/*_repeat{2,3}.txt`
- `rob337/queued_jobs.tsv`
- `rob337/rob337_learnt_augmentation_repeats.csv`
- `rob337/rob337_learnt_augmentation_repeats_aggregate.csv`
- `rob337/OUTCOME.md`

The Stanage launch entry point is:

```bash
scripts/submit_rob337_learnt_augmentation_repeats_stanage.sh
```
