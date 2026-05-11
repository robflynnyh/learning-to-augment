# ROB-62 RMM Result-Repo 2048 Single-Epoch Results

This directory is for the corrected ROB-62 non-oracle policy evaluation
comparing random mixed masking (RMM) to random frequency masking (RFM).

Corrected setup from the existing result-repo configs:

- ASR checkpoint: `/store/store5/data/acp21rjf_checkpoints/spotify/rotary_pos_6l_256d_seq_sched/n_seq_sched_2048_rp_1/step_105360.pt`
- ASR context: `2048`
- Adaptation learning rate: `5e-6`
- Evaluation epochs: `1`
- Rollout: `exp/eval.py`
- RFM policy: `FrequencyMaskingRanker`
- RMM policy: `MixedMaskingRanker`

The earlier ROB-62 `16384` / `9e-5` scaffold was superseded by Robert's
2026-05-10 Linear correction and should not be mixed into this comparison.

Run wrapper:

```bash
scripts/launch_rob62_result_repo_eval.sh
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
python3 exp/results/repro/policy/ROB-62_result_repo_2048_1epoch/summarize_results.py
```

Completed outputs from the 2026-05-11 Mimas run:

- Raw result summaries: `RFM/{tedlium,e22,rev16,chime6}.txt` and
  `RMM/{tedlium,e22,rev16,chime6}.txt`.
- Comparison tables: `comparison.csv` and `comparison.md`.
- Local run logs, not committed: `logs/run.log` and `logs/queued-screen.log`.
- TAL status: not run because TAL was not mirrored on this Mimas host.
