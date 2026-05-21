#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

import torch

SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = next(parent for parent in SCRIPT_PATH.parents if (parent / "l2augment" / "modelling" / "models.py").exists())
sys.path.insert(0, str(REPO_ROOT))
VECTOR_QUANTIZE_SITE = Path("/store/store4/software/bin/anaconda3/envs/flash_attn_pytorch2/lib/python3.9/site-packages")
if VECTOR_QUANTIZE_SITE.exists():
    sys.path.append(str(VECTOR_QUANTIZE_SITE))

from l2augment.modelling.models import RewardConditionedMaskLM
from l2augment.utils.collate_functions import RewardConditionedMaskLM_fn
from l2augment.utils.datasets import RewardConditionedMaskLMDataset


def rollout_files(root, split, limit):
    files = sorted((Path(root) / split).glob("*.pt"))
    if limit is not None:
        files = files[:limit]
    if len(files) == 0:
        raise FileNotFoundError(f"No rollout files found for split={split} under {root}")
    return [str(path) for path in files]


def dataset_stats(files):
    dataset = RewardConditionedMaskLMDataset(files)
    stats = {
        "files": len(files),
        "samples": 0,
        "degenerate_reward_groups": 0,
        "finite_normalized_rewards": True,
        "finite_raw_rewards": True,
        "generation_lengths": [],
    }
    items = []
    for idx in range(len(dataset)):
        item = dataset[idx]
        if item is None:
            raise RuntimeError(f"Dataset returned None for {files[idx]}")
        items.append(item)
        stats["samples"] += int(item["generation"].shape[0])
        stats["degenerate_reward_groups"] += int(item["degenerate_reward_group"])
        stats["finite_normalized_rewards"] &= bool(torch.isfinite(item["reward"]).all())
        stats["finite_raw_rewards"] &= bool(torch.isfinite(item["raw_reward"]).all())
        stats["generation_lengths"].append(int(item["generation_length"]))
    return dataset, items, stats


def build_model(args):
    model = RewardConditionedMaskLM(
        hidden_dim=args.hidden_dim,
        default_conditioning_reward=0.0,
        reward_encoder="timestep",
        sample_generation=False,
        mask_vae_state_dict_path=args.bvae_checkpoint,
        mask_vae_config={
            "latent_dim": 128,
            "codebook_size": 2048,
            "use_vq": True,
        },
    )
    return model.cpu()


def run_model_smoke(model, item, target_steps):
    batch = RewardConditionedMaskLM_fn([item])
    loss, losses = model.forward_pass(batch, torch.device("cpu"))
    if not torch.isfinite(loss):
        raise RuntimeError(f"Non-finite CE loss: {loss.item()}")
    loss.backward()

    grad_tensors = [param.grad for param in model.parameters() if param.requires_grad and param.grad is not None]
    if len(grad_tensors) == 0:
        raise RuntimeError("No trainable gradients produced")
    if not all(torch.isfinite(grad).all() for grad in grad_tensors):
        raise RuntimeError("Non-finite gradient produced")

    generations = {}
    for reward in (-1.0, 1.0):
        generated = model.generate(
            conditioning_reward=reward,
            sample=False,
            target_prediction_steps=target_steps,
            target_output_length=None,
            device="cpu",
        )
        if generated is False:
            raise RuntimeError(f"Fixed-length generation failed for reward={reward}")
        _, tokens = generated
        if int(tokens.numel()) != target_steps:
            raise RuntimeError(f"Expected {target_steps} VQ tokens for reward={reward}, got {tokens.numel()}")
        generations[str(reward)] = tokens.tolist()

    return {
        "loss": float(loss.item()),
        "logged_losses": {key: float(value.item()) for key, value in losses.items()},
        "fixed_length_generation_steps": target_steps,
        "fixed_length_generations": generations,
    }


def run_augment_smoke(model, rollout_path):
    try:
        rollout = torch.load(rollout_path, weights_only=True)
    except Exception:
        rollout = torch.load(rollout_path, weights_only=False)
    audio = rollout["audio"].to(dtype=torch.float32)
    if audio.ndim == 2:
        audio = audio.unsqueeze(0)
    if audio.size(0) != 1:
        audio = audio[:1]

    expected_steps = int(model.mask_enc.calc_downsampled_length(torch.tensor([audio.size(-1)])).item())
    augmented, mask, metadata = model.augment(audio, conditioning_reward=1.0, sample=False)
    generation = metadata["generation"]

    if int(generation.numel()) != expected_steps:
        raise RuntimeError(f"Expected augment generation length {expected_steps}, got {generation.numel()}")
    if metadata["generation_length"] != expected_steps:
        raise RuntimeError("Augment metadata did not record the audio-derived generation length")
    if mask.shape[-1] != audio.shape[-1]:
        raise RuntimeError(f"Expected mask length {audio.shape[-1]}, got {mask.shape[-1]}")
    if augmented.shape != audio.shape:
        raise RuntimeError(f"Expected augmented audio shape {tuple(audio.shape)}, got {tuple(augmented.shape)}")

    return {
        "source_path": str(rollout_path),
        "audio_frames": int(audio.shape[-1]),
        "audio_derived_generation_length": expected_steps,
        "metadata_generation_length": int(metadata["generation_length"]),
        "generated_tokens": int(generation.numel()),
        "mask_shape": list(mask.shape),
        "augmented_shape": list(augmented.shape),
        "conditioning_reward": float(metadata["conditioning_reward"]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollout-root", default="/store/store4/data/l2augment_rollout_uvqmlm")
    parser.add_argument("--bvae-checkpoint", default="/store/store5/data/acp21rjf_checkpoints/l2augment/models/bvae/bvae_USINGTHISFORNOW_2048gpu.pt")
    parser.add_argument("--train-files", type=int, default=2)
    parser.add_argument("--dev-files", type=int, default=2)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--target-steps", type=int, default=7)
    parser.add_argument("--seed", type=int, default=114)
    parser.add_argument("--output", default="exp/results/repro/reward_conditioned_lm/no_audio_conditioning/smoke/smoke_reward_conditioned_mask_lm.json")
    args = parser.parse_args()
    torch.manual_seed(args.seed)

    train_files = rollout_files(args.rollout_root, "train", args.train_files)
    dev_files = rollout_files(args.rollout_root, "dev", args.dev_files)

    _, train_items, train_stats = dataset_stats(train_files)
    _, _, dev_stats = dataset_stats(dev_files)

    model = build_model(args)
    model_stats = run_model_smoke(model, train_items[0], args.target_steps)
    augment_stats = run_augment_smoke(model, dev_files[0])

    result = {
        "train_stats": train_stats,
        "dev_stats": dev_stats,
        "model_smoke": model_stats,
        "augment_smoke": augment_stats,
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
