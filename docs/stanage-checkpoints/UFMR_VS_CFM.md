# UFMR vs CFM

This note compares the two closest frequency-mask-ranker setups in Learn-to-Augment:

- **UFMR** — `UnconditionalFrequencyMaskingRanker`
- **CFM** — `ConditionalFrequencyMaskingRanker`

Short version: **CFM is the closest “UFMR + audio conditioning” experiment**. It uses the same style of 80-dimensional frequency masks as UFMR, but scores each candidate mask together with an embedding of the current audio.

## High-level idea

### UFMR

UFMR is the simple unconditional baseline.

It samples candidate frequency masks, scores each mask, and applies the best one. The score is based only on the mask itself:

```text
mask[80] -> ranker -> scalar score
```

It does **not** condition on:

- the audio being augmented,
- previous masks,
- previous adaptation state,
- or any sequence history.

Despite that simplicity, this was the line that worked quite well.

### CFM

CFM keeps the same basic frequency-mask-ranker setup, but adds audio conditioning:

```text
audio -> VAE/audio encoder -> audio embedding
mask[80] -> mask encoder -> mask embedding
concat(audio embedding, mask embedding) -> ranker -> scalar score
```

So CFM asks: **given this particular audio example, which frequency mask should I choose?**

It is the most direct audio-conditioned analogue of UFMR found in the repo/checkpoints.

## Mask type

Both UFMR and CFM use the same kind of mask.

The base class `FrequencyMaskingRanker` constructs masks with:

```python
SpecAugment(n_time_masks=0, n_freq_masks=6, freq_mask_param=34, zero_masking=True)
```

So the mask is:

```text
80-dimensional frequency mask
1 = keep mel bin
0 = mask mel bin
```

There is no time masking in these rankers, and no 2D time-frequency mask.

At evaluation/augmentation time, both methods sample `repeats` candidate masks and choose the highest-scoring one.

## Architecture comparison

| Aspect | UFMR | CFM |
|---|---|---|
| Policy class | `UnconditionalFrequencyMaskingRanker` | `ConditionalFrequencyMaskingRanker` |
| Candidate mask | 80-dim frequency mask | 80-dim frequency mask |
| Uses audio? | no | yes |
| Mask encoder | small MLP/SwiGLU directly to scalar | SwiGLU `80 -> 80` mask embedding |
| Audio encoder | none | `SingleStateVariationalAutoEncoder` |
| Score head | mask-only MLP | MLP over concat(audio embedding, mask embedding) |
| Typical loss | MSE(score, reward) | MSE(score, reward) |
| Reward source | rollout WER/CER improvement | rollout WER/CER improvement |

Relevant CFM code shape:

```python
z_audio = self.vae(audio, lengths=lengths, mode='compress')
z_audio = self.encode_audio(z_audio)
z_mask = self.encode_mask(mask)
z = torch.cat((z_audio, z_mask), dim=-1)
score = self.predict(z)
```

## Training data / reward setup

Both are trained as offline rankers over precomputed rollout files.

The training script is:

```text
exp/train_freq_mask.py
```

It reads rollout files from:

```yaml
generation.save_dir: <rollout-root>
```

and expects:

```text
<rollout-root>/train/*.pt
<rollout-root>/dev/*.pt
```

Each rollout file contains candidate masks and reward information from ASR evaluation/adaptation. The dataset turns those into training examples: mask(s), optional audio, and reward.

The training objective is usually:

```text
MSE(predicted_score, rollout_reward)
```

There is also a `mult` loss variant for some UFMR runs, but the main UFMR/CFM comparison is MSE-based.

The optimizer in `train_freq_mask.py` is:

```python
MADGRAD(policy.parameters(), lr=config['policy']['lr'])
```

The loop evaluates validation loss on the dev rollout split, trains one epoch, writes `tmp_model_save_path`, and finally saves `model_save_path`. It has early stopping via validation loss patience (`training.tolerance`, default 2).

## UFMR configs/checkpoints

Main configs:

```text
exp/configs/UFMR.yaml
exp/configs/UFMR_wer.yaml
exp/configs/UFMR_wer_cer.yaml
```

Paper/eval configs:

```text
exp/configs/configs_in_paper/UFRM/UFRM_eval/singlestep/
exp/configs/configs_in_paper/UFRM/UFRM_eval/singleepoch/
exp/configs/configs_in_paper/UFRM/UFRM_eval/multiepoch/
```

Known Stanage checkpoint variants:

```text
/mnt/parscratch/users/acp21rjf/l2augment_model/ufm/test_wer/model.pt
/mnt/parscratch/users/acp21rjf/l2augment_model/ufm/test_wer/tmp_model.pt
/mnt/parscratch/users/acp21rjf/l2augment_model/ufm/test_cer/model.pt
/mnt/parscratch/users/acp21rjf/l2augment_model/ufm/test_cer/tmp_model.pt
/mnt/parscratch/users/acp21rjf/l2augment_model/ufm/test_wer_cer/model.pt
/mnt/parscratch/users/acp21rjf/l2augment_model/ufm/test_wer_cer/tmp_model.pt
/mnt/parscratch/users/acp21rjf/l2augment_model/ufm/mseloss/model.pt
/mnt/parscratch/users/acp21rjf/l2augment_model/ufm/mseloss2e1/model.pt
/mnt/parscratch/users/acp21rjf/l2augment_model/ufm/multloss/model.pt
```

`test_wer` is the one directly referenced by the checked-in paper eval configs.

## CFM configs/checkpoints

Main config:

```text
exp/configs/conditional_freq_mask.yaml
```

Known Stanage checkpoints:

```text
/mnt/parscratch/users/acp21rjf/l2augment_model/cfm/tmp_model.pt
/mnt/parscratch/users/acp21rjf/l2augment_model/cfmsimple/tmp_model.pt
```

The config says the intended CFM final checkpoint path was:

```text
/mnt/parscratch/users/acp21rjf/l2augment_model/cfm/model.pt
```

but the inventory only found:

```text
/mnt/parscratch/users/acp21rjf/l2augment_model/cfm/tmp_model.pt
```

So CFM looks more exploratory/incomplete than UFMR from the available artifacts.

Important CFM config fields:

```yaml
policy:
  lr: 1e-4
  class: ConditionalFrequencyMaskingRanker
  config:
    loss_type: mse
    latent_dim: 256
    mel_bins: 80
    vae_config:
      input_dim: 80
      hidden_dim: 128
      latent_dim: 256
      layers: 6
      min_input_size: 256
    vae_state_dict_path: /mnt/parscratch/users/acp21rjf/l2augment_model/ssvae/model.pt

training:
  batch_size: 84
  epochs: 3
  model_save_path: /mnt/parscratch/users/acp21rjf/l2augment_model/cfm/model.pt
  tmp_model_save_path: /mnt/parscratch/users/acp21rjf/l2augment_model/cfm/tmp_model.pt

dataset:
  load_audio: true
  clamp_min: -50
  clamp_max: 50

collate_function: 1dmask_and_audio

generation:
  save_dir: /mnt/parscratch/users/acp21rjf/l2augment_rollout_2e1/
```

The key differences from UFMR are `dataset.load_audio: true`, `collate_function: 1dmask_and_audio`, and the VAE state dict used to embed audio.

## Reproducing UFMR training on Stanage

From Stanage:

```bash
ssh acp21rjf@stanage.shef.ac.uk
cd /users/acp21rjf/learning-to-augment/exp/launch_scripts
```

Train the WER UFMR variant:

```bash
sbatch --export=CONFIG=/users/acp21rjf/learning-to-augment/exp/configs/UFMR_wer.yaml ./train_cpu.sh
```

Train the other checked-in variants:

```bash
sbatch --export=CONFIG=/users/acp21rjf/learning-to-augment/exp/configs/UFMR.yaml ./train_cpu.sh
sbatch --export=CONFIG=/users/acp21rjf/learning-to-augment/exp/configs/UFMR_wer_cer.yaml ./train_cpu.sh
```

`train_cpu.sh` ultimately runs:

```bash
python train_freq_mask.py --config $CONFIG
```

Expected prerequisites:

- ASR checkpoint exists at the config's `checkpointing.asr_model` path.
- Rollouts exist under the config's `generation.save_dir`, with `train/` and `dev/` subdirectories.
- Conda env exists at `/mnt/parscratch/users/acp21rjf/conda/main`.

For UFMR, the key rollout root is documented in `exp/configs/UFMR_README.md` as:

```text
/mnt/parscratch/users/acp21rjf/l2augment_rollout_test/
```

## Reproducing CFM training on Stanage

From Stanage:

```bash
ssh acp21rjf@stanage.shef.ac.uk
cd /users/acp21rjf/learning-to-augment/exp/launch_scripts
```

Run CFM training:

```bash
sbatch --export=CONFIG=/users/acp21rjf/learning-to-augment/exp/configs/conditional_freq_mask.yaml ./train_cpu.sh
```

This also runs:

```bash
python train_freq_mask.py --config $CONFIG
```

Expected prerequisites:

- ASR checkpoint:

```text
/mnt/parscratch/users/acp21rjf/spotify/rotary_pos_6l_256d_seq_sched/n_seq_sched_2048_rp_1/step_105360.pt
```

- Audio VAE / single-state VAE checkpoint:

```text
/mnt/parscratch/users/acp21rjf/l2augment_model/ssvae/model.pt
```

- Rollouts under:

```text
/mnt/parscratch/users/acp21rjf/l2augment_rollout_2e1/train/*.pt
/mnt/parscratch/users/acp21rjf/l2augment_rollout_2e1/dev/*.pt
```

CFM uses `dataset.load_audio: true`, so its rollout files must include/load audio as well as masks/rewards.

## Evaluating UFMR on Stanage

The standard launcher is:

```bash
cd /users/acp21rjf/learning-to-augment/exp/launch_scripts
sbatch --export=CONFIG=/users/acp21rjf/learning-to-augment/exp/configs/configs_in_paper/UFRM/UFRM_eval/singleepoch/tedlium.yaml ./run_eval_cpu.sh
```

To launch the checked-in UFMR paper evals:

```bash
cd /users/acp21rjf/learning-to-augment/exp/launch_scripts

for mode in singlestep singleepoch multiepoch; do
  for cfg in /users/acp21rjf/learning-to-augment/exp/configs/configs_in_paper/UFRM/UFRM_eval/$mode/*.yaml; do
    sbatch --export=CONFIG="$cfg" ./run_eval_cpu.sh
  done
done
```

These configs directly point at:

```text
/mnt/parscratch/users/acp21rjf/l2augment_model/ufm/test_wer/model.pt
```

To evaluate a different UFMR checkpoint, copy/patch one of those eval configs and change:

```yaml
training.model_save_path: /mnt/parscratch/users/acp21rjf/l2augment_model/ufm/<VARIANT>/model.pt
training.tmp_model_save_path: /mnt/parscratch/users/acp21rjf/l2augment_model/ufm/<VARIANT>/tmp_model.pt
```

Also update `evaluation.save_path` so results do not overwrite the `test_wer` result.

## Evaluating CFM on Stanage

CFM does not have the same curated paper eval directory as UFMR.

The closest available config is:

```text
exp/configs/conditional_freq_mask.yaml
```

It includes an `evaluation:` block:

```yaml
evaluation:
  dataset: tedlium
  split: test
  use_cer: false
  augmentation_config:
    repeats: 10
    use_random: false
```

So the basic direct eval command is:

```bash
cd /users/acp21rjf/learning-to-augment/exp/launch_scripts
sbatch --export=CONFIG=/users/acp21rjf/learning-to-augment/exp/configs/conditional_freq_mask.yaml ./run_eval_cpu.sh
```

However, note the current config points `training.model_save_path` at:

```text
/mnt/parscratch/users/acp21rjf/l2augment_model/cfm/model.pt
```

and the discovered checkpoint inventory only found:

```text
/mnt/parscratch/users/acp21rjf/l2augment_model/cfm/tmp_model.pt
```

So, to evaluate the discovered CFM checkpoint, make a patched config that points both model fields at `tmp_model.pt` and writes a unique result path:

```bash
cd /users/acp21rjf/learning-to-augment
mkdir -p exp/configs/patched_eval/cfm

python - <<'PY'
from omegaconf import OmegaConf
src='exp/configs/conditional_freq_mask.yaml'
dst='exp/configs/patched_eval/cfm/tedlium_tmp_model.yaml'
cfg=OmegaConf.load(src)
cfg.training.model_save_path='/mnt/parscratch/users/acp21rjf/l2augment_model/cfm/tmp_model.pt'
cfg.training.tmp_model_save_path='/mnt/parscratch/users/acp21rjf/l2augment_model/cfm/tmp_model.pt'
cfg.evaluation.save_path='/users/acp21rjf/learning-to-augment/exp/results/CFM/tedlium_tmp_model.txt'
OmegaConf.save(cfg, dst)
print(dst)
PY

cd exp/launch_scripts
sbatch --export=CONFIG=/users/acp21rjf/learning-to-augment/exp/configs/patched_eval/cfm/tedlium_tmp_model.yaml ./run_eval_cpu.sh
```

To evaluate `cfmsimple/tmp_model.pt`, use the same patch style but change the checkpoint path and probably confirm the architecture/config first, because no exact checked-in config was found for `cfmsimple`.

## Evaluating UFMR locally on mimas

UFMR has a helper for local/mimas eval using the copied `/store/store5` checkpoints:

```bash
cd /exp/exp4/acp21rjf/learning-to-augment/exp/launch_scripts
./run_eval_mimas.sh ../configs/configs_in_paper/UFRM/UFRM_eval/singleepoch/tedlium.yaml test_wer
```

Other copied UFMR variants can be selected by changing the second argument, e.g.:

```bash
./run_eval_mimas.sh ../configs/configs_in_paper/UFRM/UFRM_eval/singleepoch/tedlium.yaml test_cer
./run_eval_mimas.sh ../configs/configs_in_paper/UFRM/UFRM_eval/singleepoch/tedlium.yaml test_wer_cer
```

There is no equivalent local helper for CFM yet, and CFM checkpoints were not copied locally in the current checkpoint transfer.

## Why CFM looks less complete than UFMR

This is based on artifacts, not model quality:

- UFMR has multiple final `model.pt` checkpoints; CFM inventory only found `tmp_model.pt`.
- UFMR has curated paper eval config directories; CFM has only `conditional_freq_mask.yaml`.
- UFMR has a dedicated README; CFM does not.
- UFMR checkpoints were copied locally; CFM checkpoints were not.
- UFMR was trained for 100 epochs in the main configs; CFM config says 3 epochs.

So the safe wording is:

> CFM is the direct audio-conditioned analogue of UFMR, but the available artifacts suggest it was exploratory or not promoted to the main evaluated checkpoint set.

## Quick comparison table

| Question | UFMR | CFM |
|---|---|---|
| Same 80-dim frequency masks? | yes | yes |
| Conditions on audio? | no | yes |
| Conditions on previous masks? | no | no |
| Directly analogous? | baseline | UFMR + audio conditioning |
| Main config | `exp/configs/UFMR_wer.yaml` | `exp/configs/conditional_freq_mask.yaml` |
| Main eval configs | `configs_in_paper/UFRM/UFRM_eval/...` | `conditional_freq_mask.yaml`, patched if using `tmp_model.pt` |
| Main checkpoint found | `ufm/test_wer/model.pt` | `cfm/tmp_model.pt` |
| Training script | `exp/train_freq_mask.py` | `exp/train_freq_mask.py` |
| Loss | MSE by default | MSE |
| Reward source | rollout WER/CER improvement | rollout WER/CER improvement |
| Status | main/simple successful line | exploratory direct audio-conditioned analogue |
