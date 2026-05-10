# ROB-62 RMM Standard 16384 Results

This directory is for the ROB-62 standard non-oracle policy evaluation comparing
random mixed masking (RMM) to random frequency masking (RFM).

Requested setup:

- ASR checkpoint: `/store/store5/data/acp21rjf_checkpoints/SAP_LCASR/n_seq_sched_16384_rp_1/step_105360.pt`
- ASR context: `16384`
- Adaptation learning rate: `9e-5`
- Evaluation epochs: `5`
- Rollout: `exp/eval.py`
- RFM policy: `FrequencyMaskingRanker`
- RMM policy: `MixedMaskingRanker`

The checked-in `exp/configs/configs_in_paper/RMM_eval/multiepoch/*.yaml`
configs are not used for this run because they still point at RFM behavior and
RFM result paths.

Run wrapper:

```bash
scripts/launch_rob62_standard_eval.sh
```

The wrapper writes generated configs and result text files under this directory:

```text
configs/
  RFM/
  RMM/
RFM/
RMM/
logs/
```

Mimas dataset availability:

- TED-LIUM, Earnings-22, Rev16, and CHiME-6 are available under `/store/store4/data`.
- TAL is not mirrored on this Mimas host. The wrapper only includes TAL when
  `INCLUDE_TAL=1` and `L2A_TAL_DIR` points at an existing TAL directory.

After results are available, generate the comparison table with:

```bash
python3 exp/results/repro/policy/ROB-62_standard_16384/summarize_results.py
```
