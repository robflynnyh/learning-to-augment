#!/usr/bin/env python3
"""Render ROB-117 reward-controlled mask samples as PDF/PNG artifacts."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR",
    "/exp/exp4/acp21rjf/rob117-scratch/matplotlib",
)

import matplotlib.pyplot as plt
import numpy as np
import torch


DEFAULT_INPUT = Path(
    "exp/results/repro/reward_conditioned_lm/no_audio_conditioning/"
    "post_training_10_sampled_masks_reward_0_vs_1.pt"
)
DEFAULT_OUTPUT_DIR = Path(
    "exp/results/repro/reward_conditioned_lm/no_audio_conditioning/visualizations"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render sampled reward-conditioned masks from a tensor bundle."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def save_mask(mask: np.ndarray, title: str, path_base: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 3.2), constrained_layout=True)
    ax.imshow(
        mask,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        cmap="gray",
        vmin=0,
        vmax=1,
    )
    ax.set_xlabel("time frame")
    ax.set_ylabel("mel bin")
    ax.set_title(title)
    for suffix in (".png", ".pdf"):
        fig.savefig(path_base.with_suffix(suffix), dpi=180)
    plt.close(fig)


def save_grid(masks: list[np.ndarray], rewards: list[float], path_base: Path) -> None:
    cols = 5
    rows = int(np.ceil(len(masks) / cols))
    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(3.2 * cols, 2.0 * rows),
        constrained_layout=True,
    )
    axes = np.asarray(axes).reshape(-1)
    for idx, ax in enumerate(axes):
        if idx >= len(masks):
            ax.axis("off")
            continue
        ax.imshow(
            masks[idx],
            aspect="auto",
            origin="lower",
            interpolation="nearest",
            cmap="gray",
            vmin=0,
            vmax=1,
        )
        ax.set_title(f"sample {idx:02d}, reward {rewards[idx]:.1f}")
        ax.set_xlabel("time")
        ax.set_ylabel("mel")
    for suffix in (".png", ".pdf"):
        fig.savefig(path_base.with_suffix(suffix), dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if not args.input.exists():
        raise FileNotFoundError(f"Missing tensor bundle: {args.input}")

    bundle = torch.load(args.input, map_location="cpu", weights_only=False)
    masks_tensor = bundle["masks"].to(dtype=torch.float32)
    rewards_tensor = bundle["rewards"].to(dtype=torch.float32)
    if masks_tensor.ndim != 4 or masks_tensor.shape[1] != 1:
        raise ValueError(f"expected masks with shape [N, 1, mel, time], got {tuple(masks_tensor.shape)}")
    if rewards_tensor.numel() != masks_tensor.shape[0]:
        raise ValueError("reward count does not match mask count")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    samples_dir = args.output_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    masks = [masks_tensor[idx, 0].numpy() for idx in range(masks_tensor.shape[0])]
    rewards = [float(item) for item in rewards_tensor.tolist()]
    sample_summaries = []
    for idx, (mask, reward) in enumerate(zip(masks, rewards, strict=True)):
        reward_name = str(reward).replace(".", "p")
        path_base = samples_dir / f"sample_{idx:02d}_reward_{reward_name}_mask"
        save_mask(
            mask,
            title=f"Reward-conditioned mask sample {idx:02d} (reward {reward:.1f})",
            path_base=path_base,
        )
        sample_summaries.append(
            {
                "sample_index": idx,
                "conditioning_reward": reward,
                "mask_path_png": str(path_base.with_suffix(".png").relative_to(args.output_dir)),
                "mask_path_pdf": str(path_base.with_suffix(".pdf").relative_to(args.output_dir)),
                "mask_shape": list(mask.shape),
                "mask_active_fraction": float(mask.mean()),
            }
        )

    grid_base = args.output_dir / "reward_conditioned_mask_samples_grid"
    save_grid(masks, rewards, grid_base)

    metadata = {
        "input": str(args.input),
        "grid_path_png": "reward_conditioned_mask_samples_grid.png",
        "grid_path_pdf": "reward_conditioned_mask_samples_grid.pdf",
        "num_samples": len(sample_summaries),
        "samples": sample_summaries,
    }
    metadata_path = args.output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    print(f"Wrote artifacts to {args.output_dir}")
    print(
        f"num_samples={len(masks)} mask_shape={masks[0].shape} "
        f"active_fraction_range="
        f"{min(item['mask_active_fraction'] for item in sample_summaries):.4f}-"
        f"{max(item['mask_active_fraction'] for item in sample_summaries):.4f}"
    )


if __name__ == "__main__":
    main()
