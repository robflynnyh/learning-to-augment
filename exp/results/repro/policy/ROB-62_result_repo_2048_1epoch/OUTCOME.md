# ROB-62 Outcome

The corrected result-repo RMM/RFM comparison completed on Mimas on 2026-05-11.

Setup:

- ASR checkpoint: `/store/store5/data/acp21rjf_checkpoints/spotify/rotary_pos_6l_256d_seq_sched/n_seq_sched_2048_rp_1/step_105360.pt`
- ASR context: `2048`
- Adaptation learning rate: `5e-6`
- Evaluation epochs: `1`
- RFM policy: `FrequencyMaskingRanker`
- RMM policy: `MixedMaskingRanker`

Command path:

```bash
screen -L -Logfile exp/results/repro/policy/ROB-62_result_repo_2048_1epoch/logs/queued-screen.log -dmS rob62_rmm_result_repo_2048_1epoch bash -lc 'cd /exp/exp4/acp21rjf/symphony-workspaces-learning-to-augment/ROB-62 && LOG_PATH=exp/results/repro/policy/ROB-62_result_repo_2048_1epoch/logs/run.log SCREEN_NAME=rob62_rmm_result_repo_2048_1epoch /store/store5/software/simple-gpu-schedule/with-gpu 1,2 -- scripts/launch_rob62_result_repo_eval.sh'
```

Summary:

| dataset | RFM updated WER | RMM updated WER | RMM - RFM |
| --- | --- | --- | --- |
| TED-LIUM | 0.079284 | 0.077725 | -0.001559 |
| Earnings-22 | 0.198333 | 0.198109 | -0.000225 |
| Rev16 | 0.165039 | 0.164513 | -0.000526 |
| CHiME-6 | 0.664152 | 0.814522 | 0.150370 |
| TAL | missing | missing | missing |

The complete traceable table is in `comparison.csv` and `comparison.md`. TAL was
not run because the dataset was not mirrored on this Mimas host.

Validation:

```bash
python3 exp/results/repro/policy/ROB-62_result_repo_2048_1epoch/summarize_results.py
git diff --check
```
