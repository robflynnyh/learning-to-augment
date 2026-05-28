#!/usr/bin/env python3
"""Average reward-conditioned mask samples without storing every sample."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR",
    "/exp/exp4/acp21rjf/rob117-scratch/matplotlib",
)

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
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


DEFAULT_OUTPUT_DIR = Path(
    "exp/results/repro/reward_conditioned_lm/no_audio_conditioning/"
    "visualizations/reward_conditioned_average_masks_10k"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate reward-conditioned masks and store only streaming averages "
            "for each requested reward."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "exp/configs/reward_conditioned_lm/no_audio_conditioning/"
            "tedlium_per_utterance_384d_dropout0p1_500ep_lr1e3.yaml"
        ),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "/store/store5/data/acp21rjf_checkpoints/l2augment/models/"
            "reward_conditioned_mask_lm/"
            "no_audio_tedlium_per_utterance_384d_dropout0p1_500ep_lr1e3.pt"
        ),
    )
    parser.add_argument(
        "--rollout",
        type=Path,
        default=Path("/store/store4/data/l2augment_rollout_uvqmlm/dev/AlGore_2009_0.pt"),
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--samples-per-reward", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--reward", action="append", type=float, default=[0.0, 1.0])
    parser.add_argument("--seed", type=int, default=20260528)
    parser.add_argument(
        "--device",
        default="auto",
        choices=("auto", "cpu", "cuda"),
        help="Use cuda when available by default.",
    )
    return parser.parse_args()


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def build_model(config: dict) -> RewardConditionedMaskLM:
    policy_config = dict(config["policy"]["config"])
    return RewardConditionedMaskLM(**policy_config)


def load_audio_length(rollout_path: Path) -> int:
    rollout = torch.load(rollout_path, map_location="cpu", weights_only=False)
    audio = rollout["audio"].to(dtype=torch.float32)
    if audio.ndim == 2:
        audio = audio.unsqueeze(0)
    return int(audio[:1].shape[-1])


@torch.no_grad()
def generate_mask_batch(
    model: RewardConditionedMaskLM,
    *,
    reward: float,
    batch_size: int,
    target_output_length: int,
    target_prediction_steps: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    reward_tensor = torch.full((batch_size,), float(reward), dtype=torch.float32, device=device)
    seq = model.reward_encoder(reward_tensor.unsqueeze(-1)).unsqueeze(1)
    outputs = []
    h_n = None

    for _ in range(target_prediction_steps):
        z, h_n = model.decoder(seq, h_n)
        pred = model.prediction(z)
        pred[..., model.codebook_size] = -torch.finfo(pred.dtype).max
        probs = pred.softmax(-1)
        idx = torch.multinomial(probs.squeeze(1), 1).squeeze(-1)
        outputs.append(idx)
        seq = model.embeddings(idx.unsqueeze(1))

    tokens = torch.stack(outputs, dim=1)
    mask_latent = model.mask_enc.VQ.codebook[tokens].transpose(-1, -2)
    mask_h = model.mask_enc.latent_to_hidden(mask_latent)
    mask_h = model.mask_enc.rnn_out(mask_h) + mask_h
    mask_h = model.mask_enc.decoder(mask_h)
    mask_h = torch.nn.functional.interpolate(
        mask_h,
        size=target_output_length,
        mode="linear",
        align_corners=False,
    )
    masks = model.mask_enc.output(mask_h).sigmoid()
    masks = torch.round(masks, decimals=0).to(dtype=torch.float32)
    return masks, tokens


def reward_label(reward: float) -> str:
    return str(float(reward)).replace(".", "p").replace("-", "m")


def as_percent(value: float) -> float:
    return float(value) * 100.0


def save_mask(mask: np.ndarray, title: str, path_base: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 3.2), constrained_layout=True)
    im = ax.imshow(
        mask,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        cmap="viridis",
        vmin=0,
        vmax=1,
    )
    ax.set_xlabel("time frame")
    ax.set_ylabel("mel bin")
    ax.set_title(title)
    colorbar = fig.colorbar(im, ax=ax, label="kept / unmasked frames")
    colorbar.ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    for suffix in (".png", ".pdf"):
        fig.savefig(path_base.with_suffix(suffix), dpi=180)
    plt.close(fig)


def save_difference(diff: np.ndarray, title: str, path_base: Path) -> None:
    max_abs = max(float(np.abs(diff).max()), 1e-6)
    fig, ax = plt.subplots(figsize=(10, 3.2), constrained_layout=True)
    im = ax.imshow(
        diff,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        cmap="coolwarm",
        vmin=-max_abs,
        vmax=max_abs,
    )
    ax.set_xlabel("time frame")
    ax.set_ylabel("mel bin")
    ax.set_title(title)
    colorbar = fig.colorbar(im, ax=ax, label="keep-rate change, reward 1.0 - 0.0")
    colorbar.ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    for suffix in (".png", ".pdf"):
        fig.savefig(path_base.with_suffix(suffix), dpi=180)
    plt.close(fig)


def save_grid(averages: dict[str, np.ndarray], rewards: list[float], path_base: Path) -> None:
    cols = len(rewards)
    fig, axes = plt.subplots(1, cols, figsize=(5.0 * cols, 3.0), constrained_layout=True)
    axes = np.asarray(axes).reshape(-1)
    for ax, reward in zip(axes, rewards, strict=True):
        label = reward_label(reward)
        mask = averages[label]
        im = ax.imshow(
            mask,
            aspect="auto",
            origin="lower",
            interpolation="nearest",
            cmap="viridis",
            vmin=0,
            vmax=1,
        )
        ax.set_title(f"reward {reward:.1f} keep %")
        ax.set_xlabel("time")
        ax.set_ylabel("mel")
        colorbar = fig.colorbar(im, ax=ax, label="kept / unmasked")
        colorbar.ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    for suffix in (".png", ".pdf"):
        fig.savefig(path_base.with_suffix(suffix), dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.samples_per_reward < 1:
        raise ValueError("--samples-per-reward must be positive")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")

    config = load_config(args.config)
    model = build_model(config).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    model.mask_enc.eval()

    audio_length = load_audio_length(args.rollout)
    target_prediction_steps = int(
        model.mask_enc.calc_downsampled_length(torch.tensor([audio_length], device=device)).item()
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    averages: dict[str, np.ndarray] = {}
    summary: dict[str, dict] = {}

    for reward_index, reward in enumerate(args.reward):
        label = reward_label(reward)
        torch.manual_seed(args.seed + reward_index * 100_000)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(args.seed + reward_index * 100_000)

        running_average = None
        active_min = float("inf")
        active_max = float("-inf")
        active_sum = 0.0
        example_tokens = []
        generated = 0

        while generated < args.samples_per_reward:
            current_batch = min(args.batch_size, args.samples_per_reward - generated)
            masks, tokens = generate_mask_batch(
                model,
                reward=reward,
                batch_size=current_batch,
                target_output_length=audio_length,
                target_prediction_steps=target_prediction_steps,
                device=device,
            )
            masks_cpu = masks.detach().cpu()
            if running_average is None:
                running_average = torch.zeros(masks_cpu.shape[1:], dtype=torch.float64)
            running_average += masks_cpu.sum(dim=0).to(dtype=torch.float64) / args.samples_per_reward

            active_fractions = masks_cpu.mean(dim=(1, 2))
            active_min = min(active_min, float(active_fractions.min().item()))
            active_max = max(active_max, float(active_fractions.max().item()))
            active_sum += float(active_fractions.sum().item())
            if len(example_tokens) < 3:
                for row in tokens.detach().cpu()[: 3 - len(example_tokens)]:
                    example_tokens.append([int(item) for item in row.tolist()])
            generated += current_batch
            print(
                f"reward={reward:.1f} generated={generated}/{args.samples_per_reward}",
                flush=True,
            )

        if running_average is None:
            raise RuntimeError(f"reward={reward}: no masks were generated")
        average = running_average.numpy().astype(np.float32)
        average_keep_fraction = float(average.mean())
        sample_keep_fraction_mean = active_sum / args.samples_per_reward
        averages[label] = average
        path_base = args.output_dir / f"average_reward_{label}_mask"
        save_mask(average, f"Average sampled keep mask, reward {reward:.1f}", path_base)
        summary[label] = {
            "conditioning_reward": float(reward),
            "samples": int(args.samples_per_reward),
            "average_mask_shape": list(average.shape),
            "mask_value_semantics": "1.0 keeps/unmasks the time-frequency bin; 0.0 suppresses/masks it out",
            "average_keep_fraction": average_keep_fraction,
            "average_keep_percentage": as_percent(average_keep_fraction),
            "average_masked_out_fraction": 1.0 - average_keep_fraction,
            "average_masked_out_percentage": as_percent(1.0 - average_keep_fraction),
            "sample_keep_fraction_min": active_min,
            "sample_keep_percentage_min": as_percent(active_min),
            "sample_keep_fraction_mean": sample_keep_fraction_mean,
            "sample_keep_percentage_mean": as_percent(sample_keep_fraction_mean),
            "sample_keep_fraction_max": active_max,
            "sample_keep_percentage_max": as_percent(active_max),
            "sample_masked_out_percentage_mean": as_percent(1.0 - sample_keep_fraction_mean),
            "average_active_fraction": average_keep_fraction,
            "sample_active_fraction_min": active_min,
            "sample_active_fraction_mean": sample_keep_fraction_mean,
            "sample_active_fraction_max": active_max,
            "example_token_sequences": example_tokens,
            "average_mask_png": path_base.with_suffix(".png").name,
            "average_mask_pdf": path_base.with_suffix(".pdf").name,
        }

    if {reward_label(0.0), reward_label(1.0)}.issubset(averages):
        diff = averages[reward_label(1.0)] - averages[reward_label(0.0)]
        diff_base = args.output_dir / "average_reward_1p0_minus_0p0_mask"
        save_difference(diff, "Average keep-mask difference, reward 1.0 minus 0.0", diff_base)
        difference_artifacts = {
            "difference_png": diff_base.with_suffix(".png").name,
            "difference_pdf": diff_base.with_suffix(".pdf").name,
            "difference_semantics": "positive values mean reward 1.0 keeps more of the bin than reward 0.0",
            "keep_fraction_difference_mean": float(diff.mean()),
            "keep_percentage_point_difference_mean": as_percent(float(diff.mean())),
            "keep_fraction_difference_min": float(diff.min()),
            "keep_percentage_point_difference_min": as_percent(float(diff.min())),
            "keep_fraction_difference_max": float(diff.max()),
            "keep_percentage_point_difference_max": as_percent(float(diff.max())),
            "difference_mean": float(diff.mean()),
            "difference_min": float(diff.min()),
            "difference_max": float(diff.max()),
        }
    else:
        difference_artifacts = None

    grid_base = args.output_dir / "average_reward_0p0_vs_1p0_grid"
    save_grid(averages, [float(reward) for reward in args.reward], grid_base)

    npz_path = args.output_dir / "average_reward_controlled_masks_10k.npz"
    np.savez_compressed(
        npz_path,
        **{f"reward_{label}": average for label, average in averages.items()},
    )

    metadata = {
        "config": str(args.config),
        "checkpoint": str(args.checkpoint),
        "rollout": str(args.rollout),
        "checkpoint_bytes": int(args.checkpoint.stat().st_size),
        "model_total_parameters": int(model.total_parameters()),
        "device": str(device),
        "base_seed": int(args.seed),
        "samples_per_reward": int(args.samples_per_reward),
        "batch_size": int(args.batch_size),
        "target_output_length": int(audio_length),
        "target_prediction_steps": int(target_prediction_steps),
        "mask_value_semantics": "decoded masks are multiplicative keep masks: 1.0 keeps/unmasks a bin, 0.0 suppresses/masks it out",
        "plot_units": "percent kept / unmasked; masked-out percentage is 100 minus this value",
        "memory_policy": "streaming average; only one generated batch and the running averages are retained",
        "grid_png": grid_base.with_suffix(".png").name,
        "grid_pdf": grid_base.with_suffix(".pdf").name,
        "average_masks_npz": npz_path.name,
        "difference": difference_artifacts,
        "rewards": summary,
    }
    metadata_path = args.output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
