#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

import torch
import yaml


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = next(
    parent
    for parent in SCRIPT_PATH.parents
    if (parent / "l2augment" / "modelling" / "models.py").exists()
)
sys.path.insert(0, str(REPO_ROOT))

from l2augment.modelling.models import RewardConditionedMaskLM


def load_config(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def build_model(config):
    policy_config = dict(config["policy"]["config"])
    return RewardConditionedMaskLM(**policy_config)


def load_audio(rollout_path):
    rollout = torch.load(rollout_path)
    audio = rollout["audio"].to(dtype=torch.float32)
    if audio.ndim == 2:
        audio = audio.unsqueeze(0)
    return audio[:1]


def generate_one(model, audio, reward, seed):
    torch.manual_seed(seed)
    augmented, mask, metadata = model.augment(audio, conditioning_reward=reward, sample=True)
    generation = metadata["generation"].detach().cpu().to(dtype=torch.long)
    mask = mask.detach().cpu().to(dtype=torch.float32)
    expected_steps = int(model.mask_enc.calc_downsampled_length(torch.tensor([audio.size(-1)])).item())
    if int(generation.numel()) != expected_steps:
        raise RuntimeError(f"reward={reward}: expected {expected_steps} tokens, got {generation.numel()}")
    if mask.shape[-1] != audio.shape[-1]:
        raise RuntimeError(f"reward={reward}: expected mask length {audio.shape[-1]}, got {mask.shape[-1]}")
    if augmented.shape != audio.shape:
        raise RuntimeError(f"reward={reward}: expected augmented shape {tuple(audio.shape)}, got {tuple(augmented.shape)}")
    return {
        "generation": generation,
        "mask": mask,
        "summary": {
            "conditioning_reward": float(reward),
            "seed": int(seed),
            "audio_frames": int(audio.shape[-1]),
            "expected_generation_steps": expected_steps,
            "generated_tokens": int(generation.numel()),
            "generation": [int(item) for item in generation.tolist()],
            "first_12_tokens": [int(item) for item in generation[:12].tolist()],
            "mask_shape": list(mask.shape),
            "mask_active_fraction": float(mask.mean().item()),
            "mask_min": float(mask.min().item()),
            "mask_max": float(mask.max().item()),
            "metadata_generation_length": int(metadata["generation_length"]),
            "metadata_target_output_length": int(metadata["target_output_length"]),
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="exp/configs/reward_conditioned_lm/no_audio_conditioning/tedlium_per_utterance_resume100_500ep_lr1e3.yaml",
    )
    parser.add_argument(
        "--checkpoint",
        default="/store/store5/data/acp21rjf_checkpoints/l2augment/models/reward_conditioned_mask_lm/no_audio_tedlium_per_utterance_resume100_500ep_lr1e3.pt",
    )
    parser.add_argument(
        "--rollout",
        default="/store/store4/data/l2augment_rollout_uvqmlm/dev/AlGore_2009_0.pt",
    )
    parser.add_argument("--samples-per-reward", type=int, default=5)
    parser.add_argument("--reward", action="append", type=float, default=[0.0, 1.0])
    parser.add_argument("--seed", type=int, default=20260522)
    parser.add_argument(
        "--output",
        default="exp/results/repro/reward_conditioned_lm/no_audio_conditioning/post_training_10_sampled_masks_reward_0_vs_1.json",
    )
    parser.add_argument(
        "--tensor-output",
        default="exp/results/repro/reward_conditioned_lm/no_audio_conditioning/post_training_10_sampled_masks_reward_0_vs_1.pt",
    )
    args = parser.parse_args()

    if args.samples_per_reward < 1:
        raise ValueError("--samples-per-reward must be positive")

    config = load_config(args.config)
    model = build_model(config).cpu()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    audio = load_audio(args.rollout)
    sample_summaries = []
    masks = []
    generations = []
    rewards = []
    seeds = []

    with torch.no_grad():
        for reward_index, reward in enumerate(args.reward):
            for sample_index in range(args.samples_per_reward):
                seed = args.seed + reward_index * 1000 + sample_index
                generated = generate_one(model, audio, reward, seed)
                sample_summaries.append(
                    {
                        "sample_index": int(sample_index),
                        **generated["summary"],
                    }
                )
                masks.append(generated["mask"])
                generations.append(generated["generation"])
                rewards.append(float(reward))
                seeds.append(int(seed))

    token_length = {int(generation.numel()) for generation in generations}
    if len(token_length) != 1:
        raise RuntimeError(f"generated token lengths differ: {sorted(token_length)}")

    reward_groups = {}
    for summary in sample_summaries:
        reward_key = f"{summary['conditioning_reward']:.6g}"
        reward_groups.setdefault(reward_key, []).append(summary)

    result = {
        "config": args.config,
        "checkpoint": args.checkpoint,
        "rollout": args.rollout,
        "checkpoint_bytes": int(Path(args.checkpoint).stat().st_size),
        "model_total_parameters": int(model.total_parameters()),
        "sample": True,
        "base_seed": int(args.seed),
        "samples_per_reward": int(args.samples_per_reward),
        "rewards": [float(reward) for reward in args.reward],
        "total_masks": len(sample_summaries),
        "tensor_output": args.tensor_output,
        "reward_group_summary": {
            reward: {
                "count": len(items),
                "mask_active_fraction_min": min(item["mask_active_fraction"] for item in items),
                "mask_active_fraction_mean": sum(item["mask_active_fraction"] for item in items) / len(items),
                "mask_active_fraction_max": max(item["mask_active_fraction"] for item in items),
            }
            for reward, items in reward_groups.items()
        },
        "samples": sample_summaries,
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    tensor_output = Path(args.tensor_output)
    tensor_output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "config": args.config,
            "checkpoint": args.checkpoint,
            "rollout": args.rollout,
            "rewards": torch.tensor(rewards, dtype=torch.float32),
            "seeds": torch.tensor(seeds, dtype=torch.long),
            "generations": torch.stack(generations),
            "masks": torch.stack(masks).to(dtype=torch.float16),
            "sample_summaries": sample_summaries,
        },
        tensor_output,
    )

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
