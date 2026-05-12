# Evaluation Launch Commands for Stanage Checkpoints

This companion note maps the high-level checkpoint families in `CHECKPOINT_SUMMARY_STANAGE.md` to the evaluation configs and launch commands that exist in the repo.

The emphasis here is practical: **which config should I run to evaluate which checkpoint family?** Where the repo has no obvious eval config for a checkpoint, that is marked **unknown / needs a patched config**.

## Assumed Stanage layout

The existing configs use Stanage paths such as:

```text
/users/acp21rjf/learning-to-augment
/mnt/parscratch/users/acp21rjf/l2augment_model/...
/mnt/parscratch/users/acp21rjf/spotify/...
```

The standard Stanage launch pattern is:

```bash
ssh acp21rjf@stanage.shef.ac.uk
cd /users/acp21rjf/learning-to-augment/exp/launch_scripts
sbatch --export=CONFIG=/users/acp21rjf/learning-to-augment/exp/configs/<CONFIG>.yaml ./run_eval_cpu.sh
```

`run_eval_cpu.sh` runs:

```bash
python eval.py --config $CONFIG
```

from the `exp/` directory after activating `/mnt/parscratch/users/acp21rjf/conda/main`.

For config directories with many datasets, launch all YAMLs with:

```bash
cd /users/acp21rjf/learning-to-augment/exp/launch_scripts
for cfg in /users/acp21rjf/learning-to-augment/exp/configs/<DIR>/*.yaml; do
  sbatch --export=CONFIG="$cfg" ./run_eval_cpu.sh
done
```

## 1. UFMR — unconditional frequency-mask ranker

Checkpoint family:

```text
/mnt/parscratch/users/acp21rjf/l2augment_model/ufm/*
```

Main policy class:

```text
UnconditionalFrequencyMaskingRanker
```

High-level role: unconditional baseline that samples/ranks frequency masks without using audio or mask history. This worked quite well and is the main simple policy family.

### Existing eval configs

UFMR has three evaluation regimes:

```text
exp/configs/configs_in_paper/UFRM/UFRM_eval/singlestep/
exp/configs/configs_in_paper/UFRM/UFRM_eval/singleepoch/
exp/configs/configs_in_paper/UFRM/UFRM_eval/multiepoch/
```

Available datasets/configs:

- `singlestep`: `TAL.yaml`, `chime6.yaml`, `e22.yaml`, `rev16.yaml`, `tedlium.yaml`, `tedlium_dev.yaml`
- `singleepoch`: `TAL.yaml`, `chime6.yaml`, `e22.yaml`, `rev16.yaml`, `tedlium.yaml`, `tedlium_dev.yaml`
- `multiepoch`: `TAL.yaml`, `chime6.yaml`, `e22.yaml`, `e22_move_to_random.yaml`, `rev16.yaml`, `tedlium.yaml`, `tedlium_dev.yaml`

The checked-in configs point at:

```text
/mnt/parscratch/users/acp21rjf/l2augment_model/ufm/test_wer/model.pt
```

so they directly evaluate the `ufm/test_wer` checkpoint unless patched.

### Launch all UFMR evals on Stanage

```bash
cd /users/acp21rjf/learning-to-augment/exp/launch_scripts

for cfg in /users/acp21rjf/learning-to-augment/exp/configs/configs_in_paper/UFRM/UFRM_eval/singlestep/*.yaml; do
  sbatch --export=CONFIG="$cfg" ./run_eval_cpu.sh
done

for cfg in /users/acp21rjf/learning-to-augment/exp/configs/configs_in_paper/UFRM/UFRM_eval/singleepoch/*.yaml; do
  sbatch --export=CONFIG="$cfg" ./run_eval_cpu.sh
done

for cfg in /users/acp21rjf/learning-to-augment/exp/configs/configs_in_paper/UFRM/UFRM_eval/multiepoch/*.yaml; do
  sbatch --export=CONFIG="$cfg" ./run_eval_cpu.sh
done
```

### Launch one UFMR eval

```bash
cd /users/acp21rjf/learning-to-augment/exp/launch_scripts
sbatch --export=CONFIG=/users/acp21rjf/learning-to-augment/exp/configs/configs_in_paper/UFRM/UFRM_eval/singleepoch/tedlium.yaml ./run_eval_cpu.sh
```

### Evaluating other UFMR variants

For these checkpoints:

```text
ufm/test_cer/model.pt
ufm/test_wer/model.pt
ufm/test_wer_cer/model.pt
ufm/mseloss/model.pt
ufm/mseloss2e1/model.pt
ufm/multloss/model.pt
ufm/test/tmp_model.pt
```

use the same UFMR eval configs, but patch:

```yaml
training.model_save_path: /mnt/parscratch/users/acp21rjf/l2augment_model/ufm/<VARIANT>/model.pt
training.tmp_model_save_path: /mnt/parscratch/users/acp21rjf/l2augment_model/ufm/<VARIANT>/tmp_model.pt
```

For `ufm/test`, only `tmp_model.pt` was found, so point both fields at `tmp_model.pt`.

Example patch flow:

```bash
cd /users/acp21rjf/learning-to-augment
mkdir -p exp/configs/patched_eval/ufmr_test_wer_cer
python - <<'PY'
from omegaconf import OmegaConf
src='exp/configs/configs_in_paper/UFRM/UFRM_eval/singleepoch/tedlium.yaml'
dst='exp/configs/patched_eval/ufmr_test_wer_cer/tedlium.yaml'
cfg=OmegaConf.load(src)
cfg.training.model_save_path='/mnt/parscratch/users/acp21rjf/l2augment_model/ufm/test_wer_cer/model.pt'
cfg.training.tmp_model_save_path='/mnt/parscratch/users/acp21rjf/l2augment_model/ufm/test_wer_cer/tmp_model.pt'
cfg.evaluation.save_path='/users/acp21rjf/learning-to-augment/exp/results/historical_results/UFMR/test_wer_cer/singleepoch/tedlium.txt'
OmegaConf.save(cfg, dst)
print(dst)
PY

cd exp/launch_scripts
sbatch --export=CONFIG=/users/acp21rjf/learning-to-augment/exp/configs/patched_eval/ufmr_test_wer_cer/tedlium.yaml ./run_eval_cpu.sh
```

### Mimas/local UFMR eval helper

Only UFMR has a local mimas helper that rewrites Stanage checkpoint paths to `/store/store5`:

```bash
cd /exp/exp4/acp21rjf/learning-to-augment/exp/launch_scripts
./run_eval_mimas.sh ../configs/configs_in_paper/UFRM/UFRM_eval/singleepoch/tedlium.yaml test_wer
```

Variants accepted by that helper are directory names under:

```text
/store/store5/data/acp21rjf_checkpoints/l2augment/ufmr/
```

For example:

```bash
./run_eval_mimas.sh ../configs/configs_in_paper/UFRM/UFRM_eval/singleepoch/tedlium.yaml test_wer
./run_eval_mimas.sh ../configs/configs_in_paper/UFRM/UFRM_eval/singleepoch/tedlium.yaml test_cer
./run_eval_mimas.sh ../configs/configs_in_paper/UFRM/UFRM_eval/singleepoch/tedlium.yaml test_wer_cer
```

## 2. CMultiStepMLM — conditional multi-step mask generator

Checkpoint family:

```text
/mnt/parscratch/users/acp21rjf/l2augment_model/CMultiStepMLM/*
```

Main policy class:

```text
ConditionalMultiStepMaskGenerator
```

High-level role: multi-step generator conditioned on previous mask-generation state; some variants also condition on audio, while `no_audio_*` variants intentionally disable audio conditioning.

### Main audio-conditioned eval configs

These point at:

```text
/mnt/parscratch/users/acp21rjf/l2augment_model/CMultiStepMLM/curbest.pt
```

Configs:

```text
exp/configs/configs_in_paper/conditional_multistep_mask_lm/eval_main/e22.yaml
exp/configs/configs_in_paper/conditional_multistep_mask_lm/eval_main/ted_dev.yaml
exp/configs/configs_in_paper/conditional_multistep_mask_lm/eval_main/ted_test.yaml
```

Launch all:

```bash
cd /users/acp21rjf/learning-to-augment/exp/launch_scripts
for cfg in /users/acp21rjf/learning-to-augment/exp/configs/configs_in_paper/conditional_multistep_mask_lm/eval_main/*.yaml; do
  sbatch --export=CONFIG="$cfg" ./run_eval_cpu.sh
done
```

Launch one:

```bash
cd /users/acp21rjf/learning-to-augment/exp/launch_scripts
sbatch --export=CONFIG=/users/acp21rjf/learning-to-augment/exp/configs/configs_in_paper/conditional_multistep_mask_lm/eval_main/ted_test.yaml ./run_eval_cpu.sh
```

### No-audio big eval configs

These point at:

```text
/mnt/parscratch/users/acp21rjf/l2augment_model/CMultiStepMLM/no_audio_modelgpu_big.pt
```

Configs:

```text
exp/configs/configs_in_paper/conditional_multistep_mask_lm/eval_no_audio_big/e22.yaml
exp/configs/configs_in_paper/conditional_multistep_mask_lm/eval_no_audio_big/ted_dev.yaml
exp/configs/configs_in_paper/conditional_multistep_mask_lm/eval_no_audio_big/ted_test.yaml
```

Launch all:

```bash
cd /users/acp21rjf/learning-to-augment/exp/launch_scripts
for cfg in /users/acp21rjf/learning-to-augment/exp/configs/configs_in_paper/conditional_multistep_mask_lm/eval_no_audio_big/*.yaml; do
  sbatch --export=CONFIG="$cfg" ./run_eval_cpu.sh
done
```

### Other CMultiStepMLM checkpoints

For these:

```text
modelcpu.pt
tmp_modelcpu.pt
no_audio_modelcpu.pt
no_audio_modelcpu2.pt
no_audio_modelsignals.pt
no_audio_tmp_*.pt
```

no dedicated eval config directory was found. Use the closest config (`eval_main` for audio-conditioned, `eval_no_audio_big` for no-audio), then patch:

```yaml
training.model_save_path: /mnt/parscratch/users/acp21rjf/l2augment_model/CMultiStepMLM/<CHECKPOINT>.pt
training.tmp_model_save_path: /mnt/parscratch/users/acp21rjf/l2augment_model/CMultiStepMLM/<TMP_CHECKPOINT>.pt
policy.config.condition_on_audio: true/false as appropriate
```

Important: for `no_audio_*` checkpoints, keep/patch `policy.config.condition_on_audio: false`.

## 3. Multi-step frequency-mask ranker

Checkpoint family:

```text
/mnt/parscratch/users/acp21rjf/l2augment_model/multistep_FM_ranker/*
```

Main policy class:

```text
MultiStepMaskRanker
```

High-level role: sequential ranker that conditions on previously chosen masks / recurrent state rather than just scoring each mask independently.

Existing eval configs:

```text
exp/configs/configs_in_paper/multistep_FM_ranker/eval/e22.yaml
exp/configs/configs_in_paper/multistep_FM_ranker/eval/e22_5epoch.yaml
exp/configs/configs_in_paper/multistep_FM_ranker/eval/ted_dev.yaml
exp/configs/configs_in_paper/multistep_FM_ranker/eval/ted_dev5epoch.yaml
exp/configs/configs_in_paper/multistep_FM_ranker/eval/ted_test.yaml
exp/configs/configs_in_paper/multistep_FM_ranker/eval/ted_test5_epoch.yaml
```

These configs point primarily at:

```text
/mnt/parscratch/users/acp21rjf/l2augment_model/multistep_FM_ranker/modelcpu2.pt
```

Launch all:

```bash
cd /users/acp21rjf/learning-to-augment/exp/launch_scripts
for cfg in /users/acp21rjf/learning-to-augment/exp/configs/configs_in_paper/multistep_FM_ranker/eval/*.yaml; do
  sbatch --export=CONFIG="$cfg" ./run_eval_cpu.sh
done
```

Launch one:

```bash
cd /users/acp21rjf/learning-to-augment/exp/launch_scripts
sbatch --export=CONFIG=/users/acp21rjf/learning-to-augment/exp/configs/configs_in_paper/multistep_FM_ranker/eval/ted_test.yaml ./run_eval_cpu.sh
```

For `modelcpu.pt`, `modelcpul2.pt`, `tmp_modelcpuaudio.pt`, etc., patch `training.model_save_path` to the desired checkpoint and ensure `policy.config` matches the checkpoint's embedded setup if present.

## 4. CM — conditional masking policy

Checkpoint family:

```text
/mnt/parscratch/users/acp21rjf/l2augment_model/cm/test/*
```

Main policy class:

```text
ConditionalMaskingPolicy
```

High-level role: audio-conditioned policy that directly predicts a Bernoulli mask over the mel/time grid.

Existing eval configs:

```text
exp/configs/configs_in_paper/CM_eval/singlestep/TAL.yaml
exp/configs/configs_in_paper/CM_eval/singlestep/chime6.yaml
exp/configs/configs_in_paper/CM_eval/singlestep/e22.yaml
exp/configs/configs_in_paper/CM_eval/singlestep/rev16.yaml
exp/configs/configs_in_paper/CM_eval/singlestep/tedlium.yaml
exp/configs/configs_in_paper/CM_eval/singlestep/tedlium_dev.yaml
exp/configs/configs_in_paper/CM_eval/singleepoch/tedlium.yaml
```

These point at:

```text
/mnt/parscratch/users/acp21rjf/l2augment_model/cm/test/model.pt
```

Launch all singlestep CM evals:

```bash
cd /users/acp21rjf/learning-to-augment/exp/launch_scripts
for cfg in /users/acp21rjf/learning-to-augment/exp/configs/configs_in_paper/CM_eval/singlestep/*.yaml; do
  sbatch --export=CONFIG="$cfg" ./run_eval_cpu.sh
done
```

Launch singleepoch TEDLIUM:

```bash
cd /users/acp21rjf/learning-to-augment/exp/launch_scripts
sbatch --export=CONFIG=/users/acp21rjf/learning-to-augment/exp/configs/configs_in_paper/CM_eval/singleepoch/tedlium.yaml ./run_eval_cpu.sh
```

## 5. UMLM — unconditional VQ mask language model

Checkpoint family:

```text
/mnt/parscratch/users/acp21rjf/l2augment_model/UMLM/*
```

Main policy class:

```text
UnconditionalMaskGenerator
```

High-level role: unconditional mask generator using the binary/mask VAE codebook. It does not condition on the audio.

Existing eval configs are under oracle eval:

```text
exp/configs/configs_in_paper/oracle_eval/unconditional_vq_lm/
```

Top-level configs:

```text
tedlium_5.yaml
tedlium_5_2.yaml
tedlium_5_3.yaml
tedlium_5_4.yaml
tedlium_10.yaml
tedlium_10_2.yaml
tedlium_10_3.yaml
tedlium_10_4.yaml
```

Additional subdirectories:

```text
exp/configs/configs_in_paper/oracle_eval/unconditional_vq_lm/4e-2/*.yaml
exp/configs/configs_in_paper/oracle_eval/unconditional_vq_lm/8e-2/*.yaml
```

Many of these point at:

```text
/mnt/parscratch/users/acp21rjf/l2augment_model/UMLM/tmp_modelgpu.pt
```

Launch top-level UMLM oracle evals:

```bash
cd /users/acp21rjf/learning-to-augment/exp/launch_scripts
for cfg in /users/acp21rjf/learning-to-augment/exp/configs/configs_in_paper/oracle_eval/unconditional_vq_lm/*.yaml; do
  sbatch --export=CONFIG="$cfg" ./run_eval_cpu.sh
done
```

Launch one:

```bash
cd /users/acp21rjf/learning-to-augment/exp/launch_scripts
sbatch --export=CONFIG=/users/acp21rjf/learning-to-augment/exp/configs/configs_in_paper/oracle_eval/unconditional_vq_lm/tedlium_10.yaml ./run_eval_cpu.sh
```

For `UMLM/modelgpu.pt` or `UMLM/tmp_modelcpu.pt`, patch `training.model_save_path` and `training.tmp_model_save_path` to the desired checkpoint.

## 6. MLM — conditional mask language model

Checkpoint family:

```text
/mnt/parscratch/users/acp21rjf/l2augment_model/MLM/*
```

Main policy class:

```text
ConditionalMaskGenerator
```

High-level role: audio-conditioned mask generator using audio VAE + mask BVAE/codebook.

Training configs exist:

```text
exp/configs/configs_in_paper/conditional_mask_lm/MLM.yaml
```

But I did **not** find a dedicated eval directory for this exact `MLM/modelcpu.pt` family. It may be evaluated through a generation/oracle flow or may have been superseded by `CMultiStepMLM`.

Status: **unknown / needs a patched eval config**.

Likely starting point if evaluating manually: copy a nearby mask-generator eval config, such as a `conditional_multistep_mask_lm/eval_main/*.yaml`, then change:

```yaml
policy.class: ConditionalMaskGenerator
training.model_save_path: /mnt/parscratch/users/acp21rjf/l2augment_model/MLM/modelcpu.pt
training.tmp_model_save_path: /mnt/parscratch/users/acp21rjf/l2augment_model/MLM/tmp_modelcpu.pt
```

But this should be tested; I would not treat it as a confirmed launch recipe.

## 7. CFM / cfmsimple — conditional frequency-mask rankers

Checkpoint families:

```text
/mnt/parscratch/users/acp21rjf/l2augment_model/cfm/tmp_model.pt
/mnt/parscratch/users/acp21rjf/l2augment_model/cfmsimple/tmp_model.pt
```

Main policy class:

```text
ConditionalFrequencyMaskingRanker
```

High-level role: UFMR-like frequency-mask ranker, but conditioned on audio through a VAE.

Training/config file found:

```text
exp/configs/conditional_freq_mask.yaml
```

However, no dedicated paper-style eval directory was found for CFM. The config itself includes the policy/training/checkpoint fields and may be runnable with `eval.py` if it contains a valid `evaluation` block for the desired dataset; otherwise it needs patching.

Status: **unknown / needs a patched eval config**.

Possible starting command after confirming/patching the config:

```bash
cd /users/acp21rjf/learning-to-augment/exp/launch_scripts
sbatch --export=CONFIG=/users/acp21rjf/learning-to-augment/exp/configs/conditional_freq_mask.yaml ./run_eval_cpu.sh
```

## 8. DT — decision-transformer-style conditional masking policy

Checkpoint family:

```text
/mnt/parscratch/users/acp21rjf/l2augment_model/dt/test/*
```

Main policy class:

```text
DTConditionalMaskingPolicy
```

High-level role: audio-conditioned, reward-conditioned mask-code generator.

Training configs found:

```text
exp/configs/configs_in_paper/DT_train/DT.yaml
exp/configs/configs_in_paper/DT_train/test.yaml
```

Launch script found:

```text
exp/launch_scripts/DTtest.sh
```

I did **not** find a dedicated `DT_eval` directory. `dt/test/best_model.pt` is likely the selected checkpoint, but evaluation launch provenance is unknown.

Status: **unknown / needs a patched eval config**.

Likely starting point: use the DT training config as the base only if it has/gets an `evaluation` block compatible with `eval.py`, or adapt a CM/MLM eval config to `DTConditionalMaskingPolicy` and point at:

```yaml
training.model_save_path: /mnt/parscratch/users/acp21rjf/l2augment_model/dt/test/best_model.pt
training.tmp_model_save_path: /mnt/parscratch/users/acp21rjf/l2augment_model/dt/test/tmp_model.pt
```

## 9. VAE / BVAE / audio autoencoder support checkpoints

Checkpoint families:

```text
l2augment_model/autoenc_audio/*
l2augment_model/bvae/*
l2augment_model/vae/*
l2augment_model/vae_tmp/*
l2augment_model/ssvae/*
```

These are support models, not usually evaluated directly by `eval.py` as augmentation policies.

They are referenced by downstream policy configs as:

```yaml
policy.config.audio_vae_state_dict_path: ...
policy.config.mask_vae_state_dict_path: ...
policy.config.vae_state_dict_path: ...
```

Examples:

- `autoenc_audio/model_gpu.pt` is used by `CMultiStepMLM` eval configs.
- `bvae/bvae_USINGTHISFORNOW_2048gpu.pt` is used by UMLM and CMultiStepMLM configs.
- `vae/model.pt` is used by CM eval configs.

Direct evaluation command: **not applicable / unknown** unless the goal is reconstruction-quality evaluation, which is not represented in the discovered `eval.py` configs.

## 10. Baselines / controls: NoAug, RFM, RMM

These are not learned checkpoint families from the Stanage inventory, but they are useful comparison evals.

### NoAug

Configs:

```text
exp/configs/configs_in_paper/NoAug_eval/*.yaml
```

Launch:

```bash
cd /users/acp21rjf/learning-to-augment/exp/launch_scripts
for cfg in /users/acp21rjf/learning-to-augment/exp/configs/configs_in_paper/NoAug_eval/*.yaml; do
  sbatch --export=CONFIG="$cfg" ./run_eval_cpu.sh
done
```

### RFM / random frequency masking

Configs:

```text
exp/configs/configs_in_paper/RFM_eval/singlestep/*.yaml
exp/configs/configs_in_paper/RFM_eval/singleepoch/*.yaml
exp/configs/configs_in_paper/RFM_eval/multiepoch/*.yaml
```

Launch singleepoch:

```bash
cd /users/acp21rjf/learning-to-augment/exp/launch_scripts
for cfg in /users/acp21rjf/learning-to-augment/exp/configs/configs_in_paper/RFM_eval/singleepoch/*.yaml; do
  sbatch --export=CONFIG="$cfg" ./run_eval_cpu.sh
done
```

### RMM / random mixed masking

Configs:

```text
exp/configs/configs_in_paper/RMM_eval/singlestep/*.yaml
exp/configs/configs_in_paper/RMM_eval/singleepoch/*.yaml
exp/configs/configs_in_paper/RMM_eval/multiepoch/*.yaml
```

Launch singleepoch:

```bash
cd /users/acp21rjf/learning-to-augment/exp/launch_scripts
for cfg in /users/acp21rjf/learning-to-augment/exp/configs/configs_in_paper/RMM_eval/singleepoch/*.yaml; do
  sbatch --export=CONFIG="$cfg" ./run_eval_cpu.sh
done
```

## Quick lookup table

| Checkpoint family | Direct eval config exists? | Primary config dir |
|---|---|---|
| `ufm/test_wer` | yes | `configs_in_paper/UFRM/UFRM_eval/{singlestep,singleepoch,multiepoch}` |
| other `ufm/*` variants | yes, with patch | same UFMR eval configs, patch checkpoint path |
| `CMultiStepMLM/curbest.pt` | yes | `conditional_multistep_mask_lm/eval_main` |
| `CMultiStepMLM/no_audio_modelgpu_big.pt` | yes | `conditional_multistep_mask_lm/eval_no_audio_big` |
| other `CMultiStepMLM/*` | partial, with patch | nearest CMultiStep eval config |
| `multistep_FM_ranker/modelcpu2.pt` | yes | `multistep_FM_ranker/eval` |
| other `multistep_FM_ranker/*` | yes, with patch | `multistep_FM_ranker/eval` |
| `cm/test/model.pt` | yes | `CM_eval/{singlestep,singleepoch}` |
| `UMLM/tmp_modelgpu.pt` | yes | `oracle_eval/unconditional_vq_lm` |
| other `UMLM/*` | yes, with patch | `oracle_eval/unconditional_vq_lm` |
| `MLM/modelcpu.pt` | unknown | no dedicated eval config found |
| `cfm/tmp_model.pt` | unknown/partial | `conditional_freq_mask.yaml` may need patching |
| `cfmsimple/tmp_model.pt` | unknown | no exact eval config found |
| `dt/test/best_model.pt` | unknown | no dedicated eval config found |
| VAE/BVAE/audio autoencoders | not direct eval policies | used as support checkpoints |
| `l2augment_value/model.pt` | unknown | no eval config found |

## Safety note

Most configs write results under `/users/acp21rjf/learning-to-augment/exp/results/...`. Before launching a large sweep, check whether that result file already exists if overwriting matters.
