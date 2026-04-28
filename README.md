# learning-to-augment

Code for **learning augmentation patterns for test-time adaptation** of automatic speech recognition (ASR) models.

The idea: rather than hand-designing augmentations (e.g. SpecAugment) for self-supervised test-time adaptation, train a *policy network* that selects/generates augmentations specialised for adapting a frozen ASR model to unseen test audio.

## Overview

Given a pre-trained ASR model and unlabelled test audio, the workflow is:

1. **Generate rollouts** — run the ASR model under various augmentations and record outcomes (e.g. WER, CER, teacher logits).
2. **Train a policy** — supervised / contrastive / RL-style training of an augmentation policy on the rollouts.
3. **Evaluate** — perform test-time adaptation using the learnt policy and measure WER/CER on the target domain.

Different policy classes implement different augmentation families (frequency masking rankers, mask-LM generators, VAEs, decision-transformer-style conditional generators, etc.).

## Repository layout

```
.
├── l2augment/              # importable Python package
│   ├── modelling/          # policy / VAE / ranker model definitions
│   ├── rollout/            # rollout functions (cpu / gpu / multistep / oracle)
│   └── utils/              # data, datasets, collate, masks, teacher logits, helpers
├── exp/                    # experiment scripts, configs, launch scripts, results
└── setup.py
```

The `l2augment` package is the library; everything in `exp/` is the research-experiment harness that drives it. See [`exp/README.md`](exp/README.md) for details on the experiments.

## Installation

This codebase depends on [`lcasr`](https://github.com/robflynnyh/lcasr) (long-context ASR utilities — tokeniser, audio processing, model loading, WER scoring) which must be installed separately.

```bash
# inside an environment with PyTorch
pip install -e .
# plus dependencies referenced from scripts:
#   lcasr, omegaconf, einops, madgrad, wandb, tqdm, matplotlib
```

`setup.py` declares only `torch` and `numpy`; install the rest as needed.

## Quick start

A typical experiment run looks like:

```bash
cd exp
# 1. (optional) generate rollouts / teacher logits
python generate.py        --config configs/UFMR.yaml
# 2. train a policy
python train.py           --config configs/UFMR.yaml
# 3. evaluate on the target test set
python eval.py            --config configs/configs_in_paper/UFRM/UFRM_eval/singlestep/...
```

Configs are YAML and loaded with OmegaConf; `configs/configs_in_paper/` mirrors the experiment structure used in the paper. SLURM launch scripts live in `exp/launch_scripts/`.

## Policy classes (l2augment/modelling/models.py)

| Class | Role |
|-------|------|
| `NoAugmentationPolicy` | Baseline — no augmentation |
| `FrequencyMaskingRanker`, `TrainableFrequencyMaskingRanker`, `UnconditionalFrequencyMaskingRanker` | Rank/learn frequency-mask augmentations (RFM / UFMR) |
| `MultiStepMaskRanker`, `MixedMaskingRanker` | Multi-step and mixed mask rankers |
| `ConditionalFrequencyMaskingRanker`, `ConditionalMaskingPolicy`, `ConditionalMaskGenerator`, `ConditionalMultiStepMaskGenerator` | Input-conditional mask generation (CM / MLM family) |
| `UnconditionalMaskGenerator` | Unconditional mask generation |
| `DTConditionalMaskingPolicy` | Decision-transformer-style conditional policy (DT) |
| `VariationalAutoEncoder`, `VQVariationalAutoEncoder`, `BinaryVariationalAutoEncoder`, `SingleStateVariationalAutoEncoder` | VAE / VQ-VAE generators over augmentations or audio |

## Rollout functions (l2augment/rollout/)

- `singlestep.py` — single augmentation step rollout
- `cpu_multistep.py` / `cpu_multistep_oracle.py` — multi-step CPU rollouts (and oracle variant for upper-bound evaluation)
- `cpu.py`, `cpu_loss.py`, `cpu_test.py` — CPU rollout variants
- `gpu_eval.py`, `gpu_parellel.py` — GPU evaluation / parallel rollout

The rollout function is selected at eval time via `evaluation.rollout_fn` in the config.

## Citation / contact

Author: Rob Flynn — `rjflynn2@sheffield.ac.uk`
License: Apache 2.0
