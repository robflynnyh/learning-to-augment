#!/usr/bin/env python3
"""ROB-132 full-config GPU prefix check for candidate microbatch training."""

from __future__ import annotations

import argparse
import os

import torch
from madgrad import MADGRAD
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from exp.train_freq_mask import prepare_data
from l2augment.utils.collate_functions import collate_functions_dict
from l2augment.utils.datasets import dataset_classes_dict
from l2augment.utils.helpers import load_rl_models


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--dev-batches", type=int, default=2)
    parser.add_argument("--train-batches", type=int, default=20)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    print(f"python_executable={os.sys.executable}")
    print(f"torch_version={torch.__version__}")
    print(f"cuda_visible_devices={os.environ.get('CUDA_VISIBLE_DEVICES')}")
    print(f"cuda_available={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"cuda_device={torch.cuda.get_device_name(0)}")
    print(f"batch_size={config.training.batch_size}")
    print(f"policy_training_step={config.training.get('policy_training_step', False)}")
    print(f"candidate_microbatch_size={config.policy.config.get('candidate_microbatch_size')}")

    policy = load_rl_models(config).to(device)
    policy.device = device
    optim = MADGRAD(policy.parameters(), lr=config.policy.lr)

    dataset_class = dataset_classes_dict[config.get("dataset_class", "default")]
    collate_fn = collate_functions_dict[config.get("collate_function", "default")]
    dataset_config = config.get("dataset", {})
    train_dataset = dataset_class(prepare_data(config, "train"), **dataset_config, logger=print)
    dev_dataset = dataset_class(prepare_data(config, "dev"), **dataset_config, logger=print)
    loader_kwargs = {
        "batch_size": config.training.batch_size,
        "num_workers": 0,
        "collate_fn": collate_fn,
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    dev_loader = DataLoader(dev_dataset, shuffle=False, **loader_kwargs)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    policy.eval()
    seen = 0
    for batch_idx, batch in enumerate(dev_loader):
        if batch is None:
            continue
        with torch.no_grad():
            loss, _ = policy.forward_pass(batch, device)
        peak_gb = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
        print(
            f"dev_batch={batch_idx} loss={loss.item():.6f} "
            f"rows={batch['generations'].shape[0]} "
            f"unique_audio={tuple(batch['audio_features'].shape)} "
            f"peak_gb={peak_gb:.3f}"
        )
        seen += 1
        if seen >= args.dev_batches:
            break
    if seen < args.dev_batches:
        raise RuntimeError(f"Only saw {seen} dev batches; expected {args.dev_batches}")

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    policy.train()
    seen = 0
    for batch_idx, batch in enumerate(train_loader):
        if batch is None:
            continue
        loss, _ = policy.training_step(batch, device, optim)
        peak_gb = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
        print(
            f"train_batch={batch_idx} loss={loss.item():.6f} "
            f"rows={batch['generations'].shape[0]} "
            f"unique_audio={tuple(batch['audio_features'].shape)} "
            f"peak_gb={peak_gb:.3f}"
        )
        seen += 1
        if seen >= args.train_batches:
            break
    if seen < args.train_batches:
        raise RuntimeError(f"Only saw {seen} train batches; expected {args.train_batches}")


if __name__ == "__main__":
    main()
