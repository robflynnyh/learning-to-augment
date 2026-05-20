# Reproduction Sweeps

This directory is for issue-specific reproduction sweeps that are separate from
the historical paper-result tree.

## ROB-80 TED-LIUM Dev Policy LR Sweep

`scripts/launch_rob80_tedlium_policy_sweep.sh` runs the TED-LIUM dev sweep for
`RFM`, `RMM`, and `UFMR` under:

- `exp/results/repro/sweeps/RFM/`
- `exp/results/repro/sweeps/RMM/`
- `exp/results/repro/sweeps/UFMR/`

The initial sweep evaluates learning rates `5e-6`, `1e-5`, and `2e-5` for both
`1` and `5` adaptation epochs. The UFMR follow-up extends the UFMR-only grid to
`4e-5`, `8e-5`, and `1.6e-4` because the initial best UFMR result was at the
highest tested learning rate. The launched wrapper writes generated YAML configs
under each method's `configs/` folder and writes the final result table to
`exp/results/repro/sweeps/ROB-80_OUTCOME.md`.

`scripts/launch_rob80_tedlium_segmented_policy_sweep.sh` runs the requested
TED-LIUM segmented dev follow-up with the same base LR grid (`5e-6`, `1e-5`,
`2e-5`) and epochs (`1`, `5`) under:

- `exp/results/repro/sweeps/segmented_dev/RFM/`
- `exp/results/repro/sweeps/segmented_dev/RMM/`
- `exp/results/repro/sweeps/segmented_dev/UFMR/`

The segmented follow-up summary is written to
`exp/results/repro/sweeps/segmented_dev/ROB-80_SEGMENTED_OUTCOME.md`.

`scripts/launch_rob80_tedlium_noaudio_cmultistep_sweep.sh` runs the requested
no-audio CMultiStepVQLM follow-up on TED-LIUM dev under:

- `exp/results/repro/sweeps/no_audio_cmultistep_vqlm/CMultiStepVQLM/`

This follow-up uses `ConditionalMultiStepMaskGenerator` with
`condition_on_audio: false`, the locally cached
`CMultiStepMLM/no_audio_modelsignals.pt` checkpoint, and the same centered
learning-rate grid (`5e-6`, `1e-5`, `2e-5`) for `1` and `5` adaptation epochs.
The older `no_audio_modelgpu_big.pt` checkpoint exists in the cache for
provenance but does not load into the current signal-conditioned model class.
The summary is written to
`exp/results/repro/sweeps/no_audio_cmultistep_vqlm/ROB-80_NOAUDIO_CMULTISTEP_OUTCOME.md`.

## ROB-82 TED-LIUM UVQLM LR Sweep

`scripts/launch_rob82_tedlium_uvqlm_sweep.sh` runs UVQLM as a separate policy
family from UFMR. It uses the same centered LR grid (`5e-6`, `1e-5`, `2e-5`),
epochs (`1`, `5`), and two-repeat reporting contract used for the finalized
ROB-80 sweep tables.

Outputs are kept under `exp/results/repro/sweeps/uvqlm/`:

- `tedlium_dev/UVQLM/` for TED-LIUM dev via `exp/eval.py`
- `segmented_dev/UVQLM/` for TED-LIUM segmented dev via `exp/oracle_eval.py`

The segmented-dev path intentionally uses `oracle_eval.py` with
`rollout_setting: policy`, because that entrypoint consumes
`tedlium3_segmented_data` utterance lists. `exp/eval.py` expects each
`process_fn` to return a two-value `(audio, text)` pair.
