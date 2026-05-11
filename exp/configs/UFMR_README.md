# UFMR configs — `UFMR.yaml` & `UFMR_wer.yaml`

This note documents *exactly* how the two **Unconditional Frequency-Mask Ranker** (UFMR) policies in this directory are trained, end-to-end. It only covers `UFMR.yaml` and `UFMR_wer.yaml`; the third file `UFMR_wer_cer.yaml` is included only as a contrast (it is the joint-objective variant).

## The model

Both configs train the same architecture:

```yaml
policy:
  lr: 1e-4
  class: UnconditionalFrequencyMaskingRanker
```

`UnconditionalFrequencyMaskingRanker` (`l2augment/modelling/models.py:405`) is a tiny ranker over **frequency masks**:

- A SpecAugment masker generates random frequency masks (`n_freq_masks=6`, `freq_mask_param=34`, time-masking off, zero-masking).
- The mask (a 80-dim vector — one bit per mel-bin) is fed through a `SwiGlu(80 → 1, expansion 3)` MLP that scores it.
- At inference, `learnt_augmentation` samples `repeats` random masks per utterance and picks the one with the highest score; that mask is applied (zeroing the masked mel bins) before running the frozen ASR model.

The model has **no audio conditioning** — it ranks masks on the mask alone (hence "Unconditional").

## The data the policy is trained on (rollouts)

The ranker is trained *offline* on a directory of pre-computed **rollouts**. Each rollout is a `.pt` file written by `exp/generate.py`:

```
generation:
  save_dir: "/mnt/parscratch/users/acp21rjf/l2augment_rollout_test/"
```

A rollout file contains, per audio utterance:

- `audio` — the input mel-spec (only loaded if `dataset.load_audio: true`; UFMR has it `false`),
- `mask` — a batch of candidate frequency masks (each 80-dim) that were tried,
- `reward` — the WER (and optionally CER) **before** vs **after** running the frozen ASR model with that mask applied. `data.py:124` chunks this into `before, after`; `decrease_measurement` controls whether the reward is `before - after` (absolute) or the percentage change.

`prepare_data` (in `exp/train_freq_mask.py`) just walks `<save_dir>/train/*.pt` and `<save_dir>/dev/*.pt` and feeds them to the dataset class.

## The dataset wrapper — `CustomDataset` (`l2augment/utils/data.py:16`)

This is the default class used (`dataset_class: 'default'`). For each rollout it:

1. Computes the per-mask reward as the WER/CER **decrease** caused by that mask (`before − after`).
2. If `cer_weight` and `wer_weight` are set (only `UFMR_wer.yaml` does this — see below), splits the reward into the CER and WER channels, standardises each channel independently, and recombines as `cer*cer_weight + wer*wer_weight`.
3. Standardises rewards (zero-mean, unit-std), with optional clamping (`clamp_min` / `clamp_max`).
4. Returns `{ 'reward', 'masks', ... }` to the dataloader; audio is omitted because `load_audio: false`.

## Training loop — `exp/train_freq_mask.py`

`main(config)` does:

1. `wandb.init(project="l2augment")`.
2. Builds a `UnconditionalFrequencyMaskingRanker` (via `load_rl_models`); resumes from `model_save_path` if it already exists.
3. Builds `CustomDataset` over `train/` and `dev/` rollout dirs and wraps each in a `DataLoader` (batch 84, 12 workers, prefetch 6).
4. Optimiser: `MADGRAD` at `policy.lr = 1e-4`.
5. Calls `train_policy(...)`:
   - **Per epoch**: first runs the dev split in eval mode to compute average validation loss, logs it as `avg_val_loss`.
   - Early-stops with a patience counter (`tolerance`, default 2): if `avg_val_loss` increases or is NaN for 2 consecutive epochs, the policy is rolled back to the last-best state-dict and training stops.
   - Then trains on the train split for one epoch (MSE loss between predicted score and per-mask reward).
   - Writes a *temporary* checkpoint to `tmp_model_save_path` after each epoch.
   - Hard cap at `training.epochs` (100 here).
6. After the loop the final policy is written to `training.model_save_path`.

The loss is set in `TrainableFrequencyMaskingRanker.forward_pass` (`models.py:384`):
```python
loss = nn.functional.mse_loss(score, rewards, reduction='mean')
```
i.e. **MSE between the ranker's scalar score and the standardised reward** for every mask in the batch (`loss_type` defaults to `'mse'`; the alternative `'mult'` does `-mean(score * rewards)`).

## What the `validation:` / `evaluation:` blocks do

These blocks are *not* used by the SGD step — they describe a downstream **test-time-adaptation** evaluation that is run on the dev set as a sanity check via `eval.py`. The relevant knobs:

- `dataset` / `split` — which corpus to TTA on,
- `use_cer` — score with CER instead of WER,
- `augmentation_config.repeats` — how many candidate masks to draw per utterance at inference,
- `optim_args.lr` — learning rate of the **inner ASR adaptation step** (5e-6).

The training loop in `train_freq_mask.py` doesn't actually call `eval.py` itself (the import is there for future use); these blocks are consumed when you run `python eval.py --config <UFMR cfg>` afterwards.

## What's different between the two configs

| Field | `UFMR.yaml` | `UFMR_wer.yaml` |
|---|---|---|
| `training.model_save_path` | `.../l2augment_model/ufm/test_cer/model.pt` | `.../l2augment_model/ufm/test_wer/model.pt` |
| `training.tmp_model_save_path` | `.../ufm/test_cer/tmp_model.pt` | `.../ufm/test_wer/tmp_model.pt` |
| `validation.dataset` / `split` | `tedlium / dev` | `tedlium / dev` |
| `validation.use_cer` | `true` | `true` |
| `evaluation.dataset` / `split` | `earnings22 / test` | `tedlium / dev` |
| `evaluation.use_cer` | `false` | `false` |
| `evaluation.augmentation_config.repeats` | `10` | `15` |
| `dataset.cer_weight` / `wer_weight` | *not set* (defaults: 0.0 / 1.0 in `CustomDataset.__init__`) | **explicit** `0.0 / 1.0` |
| Everything else (model, optimiser, batch size, epochs, ASR ckpt, rollout dir, policy class, lr) | identical | identical |

So the two are functionally near-identical — both train the same WER-only ranker on the same rollouts with the same hyperparameters. The differences are essentially cosmetic / bookkeeping:

- **Save destination** — separate folders so the two runs don't overwrite each other.
- **Defaulted vs. explicit reward weighting** — `UFMR_wer.yaml` makes the WER-only objective explicit; `UFMR.yaml` just inherits it from `CustomDataset`'s defaults.
- **Reference eval** — `UFMR.yaml` is set up to eval on Earnings-22 test (10 repeats) while `UFMR_wer.yaml` evals on TEDLIUM dev (15 repeats). This is the metric you'd see in `results/historical_results/UFMR/single_epoch/{e22,tedlium_dev,...}.txt`.

> Note: the `model_save_path` for `UFMR.yaml` is named `test_cer/model.pt` despite the run being WER-only; the directory name is misleading. The actual CER+WER joint-objective config is `UFMR_wer_cer.yaml` (`cer_weight: 0.5, wer_weight: 0.5`), which writes to a separate folder.

## Reproducing a UFMR training run

1. **Frozen ASR checkpoint** — must exist at:
   ```
   /mnt/parscratch/users/acp21rjf/spotify/rotary_pos_6l_256d_seq_sched/n_seq_sched_2048_rp_1/step_105360.pt
   ```
2. **Rollouts** — populate `train/` and `dev/` under
   ```
   /mnt/parscratch/users/acp21rjf/l2augment_rollout_test/
   ```
   by running `python generate.py --config <some generation cfg>` against the same ASR checkpoint. (UFMR doesn't ship its own dedicated generation YAML; any frequency-mask rollout dataset with WER/CER rewards is compatible.)
3. **Train**:
   ```bash
   sbatch --export=CONFIG=./configs/UFMR_wer.yaml launch_scripts/train_cpu.sh
   # or, for the H100 partition, edit train.sh to point at the config and:
   sbatch launch_scripts/train.sh
   ```
   Both launch scripts ultimately do `python train_freq_mask.py --config <yaml>`.
4. The final policy lands at `training.model_save_path`.
5. **Evaluate** with `python eval.py --config <eval cfg>` — the matching paper-eval YAMLs live in `configs/configs_in_paper/UFRM/UFRM_eval/{singlestep,singleepoch,multiepoch}/`.

## Hyperparameter cheat-sheet

| Knob | Value | Where |
|---|---|---|
| Policy class | `UnconditionalFrequencyMaskingRanker` | `policy.class` |
| Policy lr | 1e-4 | `policy.lr` |
| Optimiser | MADGRAD | `train_freq_mask.py:157` |
| Batch size | 84 | `training.batch_size` |
| Max epochs | 100 | `training.epochs` |
| Early-stop patience | 2 (default) | `training.tolerance` (not set; default applies) |
| Loss | MSE(score, standardised reward) | `models.py:397` |
| Reward signal | WER decrease (absolute) per mask | `data.py:124-145` |
| Mask family | `SpecAugment(n_freq_masks=6, freq_mask_param=34, zero_masking=True)` | `models.py:115` |
| Random seed | 1234 | `training.random_seed` |
