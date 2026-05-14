# Research Diary

Keep concise dated notes for project changes that future agents need in order to
reproduce results, interpret metrics, or avoid known failure modes.

## 2026-05-10

- Added Symphony project wiring for the Learn-to-Augment Linear project,
  including split instruction files, detached-job callbacks, wrapper templates,
  and a blocked follow-up issue helper.
- Replaced the superseded ROB-62 `16384` / `9e-5` scaffold with the corrected
  result-repo setup under
  `exp/results/repro/policy/ROB-62_result_repo_2048_1epoch/`. This uses the
  mirrored Spotify 2048 checkpoint, `5e-6`, one evaluation epoch, RFM
  `FrequencyMaskingRanker`, and RMM `MixedMaskingRanker`; use
  `scripts/launch_rob62_result_repo_eval.sh` for the queued run.

## 2026-05-11

- Completed the ROB-62 corrected result-repo comparison on Mimas for TED-LIUM,
  Earnings-22, Rev16, and CHiME-6. Small result summaries and comparison tables
  are under `exp/results/repro/policy/ROB-62_result_repo_2048_1epoch/`; TAL
  remains missing because it was not mirrored on this host.
- Added ROB-71 grid-style config support via `exp/run_config_grid.py` and
  converted the repetitive oracle eval YAML fanout to `tedlium_grid.yaml`
  configs that materialize one-run YAMLs under ignored `.generated/`
  directories at launch time.
- Followed up ROB-71 by converting the remaining repro configs under
  `exp/results/repro/` to grid YAMLs and adding paired-case/product grid support
  for runs that sweep repeats over matched `lr` / `single_step_lr` settings.
- Added ROB-73 unconditional VQ mask LM sample rendering under
  `exp/results/repro/unconditional_lm/ROB-73_sample/`, using the trained UMLM
  and BVAE checkpoints cached from Stanage into the local Mimas checkpoint cache.
- Extended the ROB-73 render to 10 unconditional VQ mask LM samples with an
  overview grid, per-sample PNG/PDF masks, toy-spectrogram visualizations, and
  per-sample VQ-code metadata.
- Reorganized `exp/results/` for ROB-74: older reference artifacts now live
  under `exp/results/historical_results/`, while newer resumed-work outputs
  remain under `exp/results/repro/`. Treat the historical numbers as reference
  material until they are reverified.

## 2026-05-12

- Added the ROB-80 TED-LIUM dev policy LR sweep wrapper for RFM, RMM, and UFMR under
  `scripts/launch_rob80_tedlium_policy_sweep.sh`. It writes generated configs
  and results under `exp/results/repro/sweeps/`, covers adaptation LRs `5e-6`,
  `1e-5`, and `2e-5` for `1` and `5` epochs, and summarizes completed cells via
  `scripts/summarize_rob80_tedlium_sweep.py`.
- Completed the ROB-80 TED-LIUM dev sweep: all 18 cells finished. The best
  updated WERs were RFM `0.087653` at `1e-5` / 5 epochs, RMM `0.086051` at
  `5e-6` / 5 epochs, and UFMR `0.087985` at `2e-5` / 5 epochs. The summary
  table and CSV are in `exp/results/repro/sweeps/`.
- Prepared a ROB-80 UFMR-only higher-LR follow-up after the initial best UFMR
  point landed at the highest tested LR. The wrapper and summarizer now include
  UFMR `4e-5`, `8e-5`, and `1.6e-4` for both 1 and 5 epochs while leaving the
  RFM/RMM grids unchanged.
- Completed the ROB-80 UFMR higher-LR follow-up: all 24 TED-LIUM dev cells are
  now summarized in `exp/results/repro/sweeps/ROB-80_OUTCOME.md`. The highest
  UFMR LR (`1.6e-4`) diverged badly at 5 epochs, so subsequent segmented-dev
  sweeps should stay centered on the original `5e-6`, `1e-5`, `2e-5` range.
- Added a ROB-80 TED-LIUM segmented dev wrapper,
  `scripts/launch_rob80_tedlium_segmented_policy_sweep.sh`, which reuses the
  policy sweep runner with `dataset: tedlium3_segmented_data`, `split: dev`,
  and a separate result root under `exp/results/repro/sweeps/segmented_dev/`.

## 2026-05-13

- Investigated the failed ROB-80 segmented dev queue. The wrapper did start on
  Mimas and acquired GPU 2, but `exp/eval.py` could not consume
  `tedlium3_segmented_data` utterance-list outputs, and the failure callback
  resolved `scripts/callbacks/...` relative to `exp/`. The segmented wrapper now
  runs `oracle_eval.py` with `rollout_setting: policy`, and the callback path is
  rooted at the repository directory.
- Completed the ROB-80 TED-LIUM segmented dev follow-up: all 18 RFM/RMM/UFMR
  cells finished for `5e-6`, `1e-5`, and `2e-5` at 1 and 5 epochs. The best
  segmented updated WERs were RFM `0.097601` at `5e-6` / 5 epochs, RMM
  `0.097325` at `1e-5` / 5 epochs, and UFMR `0.096717` at `1e-5` / 5 epochs;
  the table and CSV are under `exp/results/repro/sweeps/segmented_dev/`.
- Updated Symphony rules for ROB-83 to forbid `/tmp` on Mimas for working files,
  logs, downloads, result staging, callbacks, and experiment scratch space. Use
  durable repo-local paths, `exp/results/`, or suitable `/store/...` locations
  for artifacts that later Symphony turns may need to inspect.

## 2026-05-14

- Prepared the ROB-80 no-audio CMultiStepVQLM follow-up after the user selected
  that policy family. The Mimas launcher writes generated configs and results
  under `exp/results/repro/sweeps/no_audio_cmultistep_vqlm/`, uses
  `ConditionalMultiStepMaskGenerator` with `condition_on_audio: false`, and
  sweeps TED-LIUM dev learning rates `5e-6`, `1e-5`, and `2e-5` for 1 and 5
  adaptation epochs. The initial `no_audio_modelgpu_big.pt` smoke failed
  because that checkpoint predates the current signal-conditioned model
  modules; the queued follow-up uses the compatible
  `CMultiStepMLM/no_audio_modelsignals.pt` checkpoint.
- Completed the ROB-80 no-audio CMultiStepVQLM TED-LIUM dev follow-up: all 6
  cells finished for `5e-6`, `1e-5`, and `2e-5` at 1 and 5 epochs. The best
  updated WER was `0.087322` at `1e-5` / 5 epochs; the table, CSV, generated
  configs, and per-cell result files are under
  `exp/results/repro/sweeps/no_audio_cmultistep_vqlm/`.
