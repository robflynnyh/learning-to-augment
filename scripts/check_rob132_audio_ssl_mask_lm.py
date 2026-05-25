#!/usr/bin/env python3
"""Small checkpoint/generation sanity check for ROB-132 audio SSL mask LM."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from omegaconf import OmegaConf

from l2augment.utils.helpers import load_rl_models, load_model as load_policy


def cache_path_for_rollout(cache_root: Path, rollout_path: Path) -> Path:
    return cache_root / rollout_path.parent.name / rollout_path.name


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--cache-root", type=Path, required=True)
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
        cached = torch.load(cache_path_for_rollout(args.cache_root, rollout_path), map_location="cpu", weights_only=False)
        audio_features = cached["ssl_features"].to(args.device)
        per_reward = {}
        for reward in (0.0, 1.0):
            generated = policy.generate(
                audio_features=audio_features,
                conditioning_reward=reward,
                sample=False,
                target_prediction_steps=int(cached["target_steps"]),
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
            "cache": str(cache_path_for_rollout(args.cache_root, rollout_path)),
            "rewards": per_reward,
        })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as handle:
        json.dump(results, handle, indent=2)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
