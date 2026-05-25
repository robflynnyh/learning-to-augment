#!/usr/bin/env python3
"""Small checkpoint/generation sanity check for ROB-132 audio SSL mask LM."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from omegaconf import OmegaConf

from l2augment.utils.helpers import load_rl_models, load_model as load_policy
from l2augment.utils.datasets import AudioRewardConditionedMaskLMDataset


def cache_path_for_rollout(cache_root: Path, rollout_path: Path) -> Path:
    return cache_root / rollout_path.parent.name / rollout_path.name


def load_audio_features(config, rollout_path: Path, cache_root: Path | None):
    if cache_root is not None:
        cache_path = cache_path_for_rollout(cache_root, rollout_path)
        cached = torch.load(cache_path, map_location="cpu", weights_only=False)
        return cached["ssl_features"], str(cache_path), int(cached.get("target_steps", cached["ssl_features"].shape[0]))

    dataset = AudioRewardConditionedMaskLMDataset([str(rollout_path)], **config.get("dataset", {}))
    item = dataset[0]
    if item is None:
        raise RuntimeError(f"Failed to load audio-conditioned item for {rollout_path}")
    return item["audio_features"], "on_the_fly", int(item["generation_length"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--cache-root", type=Path, default=None)
    parser.add_argument("--rollout", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    policy = load_rl_models(config)
    if args.checkpoint is not None:
        load_policy(policy, config, path=args.checkpoint)
    policy = policy.to(args.device)
    policy.eval()

    results = []
    for rollout_path in args.rollout:
        audio_features, feature_source, target_steps = load_audio_features(config, rollout_path, args.cache_root)
        audio_features = audio_features.to(args.device)
        per_reward = {}
        for reward in (0.0, 1.0):
            generated = policy.generate(
                audio_features=audio_features,
                conditioning_reward=reward,
                sample=False,
                target_prediction_steps=target_steps,
                device=args.device,
            )
            if generated is False:
                per_reward[str(reward)] = {"ok": False}
                continue
            mask, tokens = generated
            per_reward[str(reward)] = {
                "ok": True,
                "tokens": int(tokens.numel()),
                "mask_shape": list(mask.shape),
                "token_prefix": tokens[:10].detach().cpu().tolist(),
            }
        results.append({
            "rollout": str(rollout_path),
            "feature_source": feature_source,
            "rewards": per_reward,
        })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as handle:
        json.dump(results, handle, indent=2)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
