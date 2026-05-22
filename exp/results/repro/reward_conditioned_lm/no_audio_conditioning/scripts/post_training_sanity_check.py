#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

import torch
import yaml


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = next(parent for parent in SCRIPT_PATH.parents if (parent / "l2augment" / "modelling" / "models.py").exists())
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


def run_reward_check(model, audio, reward, sample):
    expected_steps = int(model.mask_enc.calc_downsampled_length(torch.tensor([audio.size(-1)])).item())
    augmented, mask, metadata = model.augment(audio, conditioning_reward=reward, sample=sample)
    generation = metadata["generation"]
    if int(generation.numel()) != expected_steps:
        raise RuntimeError(f"reward={reward}: expected {expected_steps} tokens, got {generation.numel()}")
    if mask.shape[-1] != audio.shape[-1]:
        raise RuntimeError(f"reward={reward}: expected mask length {audio.shape[-1]}, got {mask.shape[-1]}")
    if augmented.shape != audio.shape:
        raise RuntimeError(f"reward={reward}: expected augmented shape {tuple(audio.shape)}, got {tuple(augmented.shape)}")
    return {
        "conditioning_reward": float(reward),
        "sample": bool(sample),
        "audio_frames": int(audio.shape[-1]),
        "expected_generation_steps": expected_steps,
        "generated_tokens": int(generation.numel()),
        "first_12_tokens": [int(item) for item in generation[:12].tolist()],
        "mask_shape": list(mask.shape),
        "mask_active_fraction": float(mask.to(dtype=torch.float32).mean().item()),
        "augmented_shape": list(augmented.shape),
        "metadata_generation_length": int(metadata["generation_length"]),
        "metadata_target_output_length": int(metadata["target_output_length"]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="exp/configs/reward_conditioned_lm/no_audio_conditioning/tedlium_per_utterance.yaml")
    parser.add_argument("--checkpoint", default="/store/store5/data/acp21rjf_checkpoints/l2augment/models/reward_conditioned_mask_lm/no_audio_tedlium_per_utterance.pt")
    parser.add_argument("--rollout", default="/store/store4/data/l2augment_rollout_uvqmlm/dev/AlGore_2009_0.pt")
    parser.add_argument("--output", default="exp/results/repro/reward_conditioned_lm/no_audio_conditioning/post_training_sanity_check.json")
    parser.add_argument("--sample", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    model = build_model(config).cpu()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    audio = load_audio(args.rollout)
    checks = [run_reward_check(model, audio, reward, args.sample) for reward in (0.0, 1.0)]
    result = {
        "config": args.config,
        "checkpoint": args.checkpoint,
        "rollout": args.rollout,
        "checkpoint_bytes": int(Path(args.checkpoint).stat().st_size),
        "model_total_parameters": int(model.total_parameters()),
        "reward_checks": checks,
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
