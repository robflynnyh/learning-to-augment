# exp/ — experiment harness

This directory contains the scripts, configs and launch helpers used to run the experiments behind the *learning-to-augment* project. The reusable code lives in the top-level `l2augment` package; everything here is a thin driver around it.

## Layout

```
exp/
├── train.py                  # train a policy (generic)
├── train_freq_mask.py        # train frequency-masking policies (UFMR / CM-MLM / RFM …)
├── train_with_rollouts.py    # train while collecting fresh rollouts on the fly
├── train_vae.py              # train (VQ/Binary/Standard) VAE generators
├── generate.py               # generate rollouts (augmentation outcomes / WERs)
├── generate_search.py        # generate rollouts via search over augmentations
├── generate_teacher_logits.py # cache teacher (frozen ASR) logits for distillation
├── eval.py                   # test-time-adaptation evaluation of a learnt policy
├── oracle_eval.py            # oracle (upper-bound) evaluation
├── segment.py                # audio segmentation helper
├── run_sweep.py              # wandb sweep driver
├── bin/                      # auxiliary entry points / older scripts
├── configs/                  # active YAML configs (see below)
├── sweep_configs/            # wandb sweep YAMLs (PL.yaml, Ranker.yaml)
├── launch_scripts/           # SLURM / shell launchers for the SHEFFIELD cluster
└── results/                  # checkpoints, logs and the figure notebook
```

## Scripts

All scripts take `--config path/to/file.yaml` and load the YAML with OmegaConf.

| Script | Purpose |
|---|---|
| `generate.py` | Roll the ASR model out under augmentations and dump per-utterance results to `generation.save_dir`. Used to build the offline training set for ranker / policy training. |
| `generate_search.py` | Variant that searches over augmentation parameters during rollout generation. |
| `generate_teacher_logits.py` | Forward the frozen ASR model on the un-augmented audio and cache the logits. Required by distillation-style policy losses. |
| `train.py` | Generic policy-training loop. Loads a `Policy` class (chosen via `policy.class` in the config), trains on saved rollouts, logs to wandb. |
| `train_freq_mask.py` | The main training entry point for the frequency-masking family (UFMR, CM-MLM, RFM, RMM, etc.). |
| `train_freq_mask_loop.py` *(in `bin/`)* | Variant that interleaves training with rollout generation. |
| `train_with_rollouts.py` | On-policy training — collects rollouts each step instead of from disk. |
| `train_vae.py` | Trains the VAE / VQ-VAE / Binary-VAE generators in `l2augment.modelling.models`. |
| `eval.py` | Loads a trained policy and an ASR checkpoint, runs the configured rollout function (`singlestep` / `multistep`), reports WER / CER. |
| `oracle_eval.py` | Same evaluation framework but uses an oracle (ground-truth-aware) policy — gives the upper bound that learnt policies are compared against. |
| `run_sweep.py` | Launches a wandb sweep using one of the YAMLs in `sweep_configs/`. |
| `run_config_grid.py` | Expands one grid-style YAML into ordinary per-run configs and launches them sequentially, with Mimas `with-gpu`, or through Slurm. |
| `segment.py` | Pre-segments long audio for evaluation / generation. |

## Configs

`configs/` holds the canonical YAMLs:

| File | What it configures |
|---|---|
| `example.yaml` | Minimal template / starting point. |
| `UFMR.yaml`, `UFMR_wer.yaml`, `UFMR_wer_cer.yaml` | Unconditional Frequency Mask Ranker — train / eval recipes. WER- and WER+CER-objective variants. |
| `RFM.yaml` | Ranking Frequency Masks. |
| `RMM.yaml` | Ranking Multi-step Masks. |
| `conditional_freq_mask.yaml` | Input-conditional frequency-mask policy. |
| `generation_CM.yaml`, `generation_test.yaml` | Rollout-generation configs for the conditional-mask family. |
| `vae.yaml` | VAE generator training. |

A config typically has these sections:

```yaml
checkpointing:        # where to load the frozen ASR model from
training:             # device, batch size, epochs, model_save_path
validation:           # dataset / split / repeats / optim args used for in-loop val
evaluation:           # dataset / split / rollout_fn / repeats used by eval.py
dataset:              # rollout dataset options (load_audio, clamping, …)
policy:               # policy class (must match a name in models.py) + lr
generation:           # save_dir for rollouts produced by generate.py
```

Grid-style configs are supported for runs that only differ by a small number of
parameters. Put shared settings at the top level and add a `grid:` block:

```yaml
evaluation:
  search_repeats: 1
  optim_args:
    lr: 1e-6

grid:
  name: oracle_repeats
  id_template: "repeats_{evaluation.search_repeats}"
  id_path: evaluation.id
  axes:
    evaluation.search_repeats: [1, 2, 3, 4]
```

Materialize and inspect the per-run YAMLs without launching:

```bash
python exp/run_config_grid.py \
  --grid-config exp/configs/configs_in_paper/oracle_eval/RMM/tedlium_grid.yaml \
  --materialize-only
```

Launch sequentially on the current machine:

```bash
python exp/run_config_grid.py \
  --grid-config exp/configs/configs_in_paper/oracle_eval/RMM/tedlium_grid.yaml \
  --entrypoint "python oracle_eval.py --config {config}" \
  --workdir exp
```

Launch separate queued jobs on Mimas:

```bash
python exp/run_config_grid.py \
  --grid-config exp/configs/configs_in_paper/oracle_eval/RMM/tedlium_grid.yaml \
  --mode mimas \
  --parallel \
  --gpu-pool 1,2 \
  --entrypoint "python oracle_eval.py --config {config}" \
  --workdir exp
```

Launch separate Slurm jobs on Stanage or another Slurm cluster:

```bash
python exp/run_config_grid.py \
  --grid-config exp/configs/configs_in_paper/oracle_eval/RMM/tedlium_grid.yaml \
  --mode slurm \
  --sbatch-script exp/launch_scripts/run_eval_oracle_cpu.sh
```

Generated one-run configs are written under `.generated/` next to the grid YAML
by default and are intentionally ignored by Git.

### `configs/configs_in_paper/`

Frozen, paper-faithful copies of every experiment in the publication. Organised by method × phase:

```
configs_in_paper/
├── UFRM/UFRM_eval/{singlestep,singleepoch,multiepoch}/   # Unconditional Frequency Ranker — eval
├── RFM_eval/{singlestep,singleepoch,multiepoch}/         # Ranking Frequency Masks — eval
├── RMM_eval/                                              # Ranking Multi-step Masks — eval
├── CM_train/CM.yaml,  CM_eval/                           # Conditional-Mask policy
├── conditional_mask_lm/, conditional_multistep_mask_lm/  # Mask-LM generator family
├── unconditional_mask_lm/                                 # Unconditional Mask-LM
├── multistep_FM_ranker/                                   # Multi-step FM ranker
├── audio_vae_train/, binary_vae_train/                    # VAE training recipes
├── DT_train/                                              # Decision-Transformer-style policy
├── NoAug_eval/                                            # No-augmentation baseline
└── oracle_eval/{RFM, RMM, unconditional_vq_lm}/           # Oracle upper bounds
```

If you want to reproduce a paper number, start from the matching folder here.

### `sweep_configs/`

- `Ranker.yaml` — sweep over ranker training hyper-parameters.
- `PL.yaml` — sweep over the policy-loss / pseudo-label training setup.

Drive a sweep with `run_sweep.py`.

## Launch scripts (`launch_scripts/`)

Cluster launchers for Stanage (SLURM). Naming convention:

- `train.sh`, `train_h100.sh`, `train_h1002.sh`, `train_cpu*.sh` — training jobs on different partitions.
- `train_vae.sh`, `train_vae_audio.sh`, `train_cpu_vae*.sh` — VAE training.
- `train_masklm_{cpu,gpu}.sh` — mask-LM training.
- `eval_cpu.sh`, `run_eval_cpu.sh`, `run_eval_oracle_cpu.sh` — evaluation jobs.
- `gen_teacher_logits.sh` — pre-compute teacher logits.
- `job*.sh`, `job_search*.sh`, `job_val*.sh` — generic job templates and (validation-)search jobs.
- `start_sweep_cpu.sh`, `start_agent_cpu.sh` — wandb sweep / agent launchers.
- `DTtest.sh`, `jobtest.sh`, `runjobtest.sh` — debugging / smoke-test launchers.

Each `.sh` is a SLURM batch script: it activates the conda env, `cd`s into `exp/`, and runs the relevant Python entry point with a config path. Edit the `--config` line to point at the experiment you want to run.

## Results (`results/`)

Per-method directories (`UFMR`, `RFM`, `RMM`, `NoAug`, `UVQLM`, `CMultiStepVQLM`, `multistep_FM_ranker`) hold checkpoints / metric dumps written by training and evaluation runs. Aggregate plots and the figure notebook live in `results/figures/` and `results/vis.ipynb`.

## Typical workflow

1. Pick (or write) a config under `configs/`.
2. If the policy needs offline data, run `generate.py` (and optionally `generate_teacher_logits.py`) to populate `generation.save_dir`.
3. Train: `python train_freq_mask.py --config <cfg>` (or the appropriate `train_*.py`).
4. Evaluate: `python eval.py --config <eval_cfg>`; compare to `oracle_eval.py` for an upper bound and to `NoAug_eval/` for a lower bound.
5. Inspect results in `results/<method>/` and `results/vis.ipynb`.

On the cluster, replace step 3/4 with `sbatch launch_scripts/<script>.sh`.
